"""
Ball detector — supports two interchangeable backends:

  * detector=yolo (default): run YOLOv8 on the cached RealSense color image,
    project cloud points into the image plane, keep points inside any
    sports-ball bbox. Pretrained weights auto-download to
    ~/.config/Ultralytics on first run. One-time setup inside the distrobox:
        pip install -r src/roboball_perception/requirements.txt

  * detector=hsv: per-point HSV color mask on the depth+color point cloud's
    `rgb` field, AND'd with the workspace bounding box. No external weights.
    Tune the bounds with `--tune` (see below).

Switch live without rebuilding:
    ros2 param set /ball_detector detector hsv
    ros2 param set /ball_detector detector yolo

Both backends publish:
    /ball_pose        geometry_msgs/PointStamped — centroid in target_frame
    /filtered_points  sensor_msgs/PointCloud2    — surviving points

Tune HSV bounds:
  Live, against the lab RealSense (run inside the distrobox while the camera
  is publishing):
    ros2 run roboball_perception ball_detector --tune \\
        --topic /camera/camera/color/image_raw
  Offline, on a laptop (no ROS):
    python3 ball_detector.py --tune --camera 0
    python3 ball_detector.py --tune --image ball.jpg
  On a running detector node, push values directly:
    ros2 param set /ball_detector hsv_lower1 '[h, s, v]'
"""

import sys
from typing import Optional

import cv2
import numpy as np

from roboball_perception.hsv_filter import HSVRange, hsv_mask_from_packed_rgb

# ROS imports are deferred so this file can be run as a tuner on a laptop
# without ROS installed (`python3 ball_detector.py --tune ...`).
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from rcl_interfaces.msg import SetParametersResult
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from geometry_msgs.msg import PointStamped
    from std_msgs.msg import Header
    from cv_bridge import CvBridge
    import sensor_msgs_py.point_cloud2 as pc2
    from tf2_ros import Buffer, TransformListener, TransformException
    from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
    from scipy.spatial.transform import Rotation as R
    _HAVE_ROS = True
except ImportError:
    _HAVE_ROS = False
    Node = object  # so the class definition below still parses


SPORTS_BALL_CLASS_ID = 32  # COCO


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')

        # ---- backend selection -----------------------------------------
        self.detector_mode = str(
            self.declare_parameter('detector', 'yolo').value
        ).lower()
        if self.detector_mode not in ('yolo', 'hsv'):
            self.get_logger().warn(
                f"Unknown detector '{self.detector_mode}', falling back to 'yolo'."
            )
            self.detector_mode = 'yolo'

        # ---- frames + workspace box (shared by both backends) ----------
        self.target_frame = self.declare_parameter('target_frame', 'base_link').value
        self.color_frame = self.declare_parameter(
            'color_frame', 'camera_color_optical_frame'
        ).value
        # HSV path uses depth-optical as the cloud's frame (matches the
        # /depth/color/points output); kept configurable for sim variants.
        self.source_frame = self.declare_parameter(
            'source_frame', 'camera_depth_optical_frame'
        ).value

        self.min_x = float(self.declare_parameter('min_x', -0.80).value)
        self.max_x = float(self.declare_parameter('max_x',  0.80).value)
        self.min_y = float(self.declare_parameter('min_y', -0.80).value)
        self.max_y = float(self.declare_parameter('max_y',  0.80).value)
        self.min_z = float(self.declare_parameter('min_z',  0.00).value)
        self.max_z = float(self.declare_parameter('max_z',  2.00).value)

        # ---- YOLO params ------------------------------------------------
        weights = str(self.declare_parameter('yolo_weights', 'yolov8n.pt').value)
        self.conf_thresh = float(self.declare_parameter('yolo_conf', 0.25).value)
        self.class_id = int(
            self.declare_parameter('yolo_class', SPORTS_BALL_CLASS_ID).value
        )
        self.min_inliers = int(self.declare_parameter('min_inliers', 20).value)

        # ---- HSV params -------------------------------------------------
        # Defaults carried over from commit 8ea3c1e — tuned for the lab beach
        # ball under lab lighting (red/orange wedge: H≈0-12, high S, V>=80).
        # Re-tune with --tune if the ball or lighting changes. Set hsv_upper2
        # to all-zeros to disable the second range (used for red wraparound
        # when lower1 already starts at H=0).
        self.hsv_lower1 = list(self.declare_parameter('hsv_lower1', [0, 120, 80]).value)
        self.hsv_upper1 = list(self.declare_parameter('hsv_upper1', [12, 255, 255]).value)
        self.hsv_lower2 = list(self.declare_parameter('hsv_lower2', [0, 0, 0]).value)
        self.hsv_upper2 = list(self.declare_parameter('hsv_upper2', [0, 0, 0]).value)
        self.min_color_points = int(self.declare_parameter('min_color_points', 20).value)

        self.add_on_set_parameters_callback(self._on_parameter_update)

        # ---- backend-specific init -------------------------------------
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_bgr: Optional[np.ndarray] = None
        self.color_K: Optional[np.ndarray] = None
        self.image_w = 0
        self.image_h = 0
        self.model = None
        if self.detector_mode == 'yolo':
            self._load_yolo(weights)

        # Subscriptions are shared. Image + camera_info are only consumed by
        # the YOLO path, but keeping them subscribed lets the user toggle
        # backends at runtime via `ros2 param set`.
        self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self._on_image, qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info',
            self._on_camera_info, 10,
        )
        self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points',
            self._on_pointcloud, 10,
        )

        self.ball_pose_pub = self.create_publisher(PointStamped, '/ball_pose', 1)
        self.filtered_points_pub = self.create_publisher(PointCloud2, '/filtered_points', 1)

        self.get_logger().info(
            f"Ball detector up. backend={self.detector_mode}, "
            f"target={self.target_frame}, "
            f"workspace x=[{self.min_x},{self.max_x}] y=[{self.min_y},{self.max_y}] "
            f"z=[{self.min_z},{self.max_z}]."
        )

    # ----------------------------------------------------------- yolo init

    def _load_yolo(self, weights: str):
        try:
            from ultralytics import YOLO
        except ImportError as ex:
            self.get_logger().error(
                f"Cannot load YOLO ({ex}). Either install ultralytics "
                "(`pip install -r src/roboball_perception/requirements.txt`) "
                "or run with `detector:=hsv`."
            )
            raise
        self.get_logger().info(f"Loading YOLO weights: {weights}")
        self.model = YOLO(weights)

    # --------------------------------------------------------- subscriptions

    def _on_camera_info(self, msg: 'CameraInfo'):
        if self.color_K is None:
            self.color_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.image_w = int(msg.width)
            self.image_h = int(msg.height)
            self.get_logger().info(
                f"Color intrinsics ready: {self.image_w}x{self.image_h}, "
                f"fx={self.color_K[0,0]:.1f} fy={self.color_K[1,1]:.1f}"
            )

    def _on_image(self, msg: 'Image'):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as ex:
            self.get_logger().warn(f"cv_bridge failed: {ex}", throttle_duration_sec=5.0)

    def _on_pointcloud(self, msg: 'PointCloud2'):
        if self.detector_mode == 'yolo':
            self._handle_yolo(msg)
        else:
            self._handle_hsv(msg)

    # ================================================================ YOLO

    def _handle_yolo(self, msg: 'PointCloud2'):
        if self.latest_bgr is None:
            self.get_logger().warn('No color image yet.', throttle_duration_sec=2.0)
            return
        if self.color_K is None:
            self.get_logger().warn('No camera_info yet.', throttle_duration_sec=2.0)
            return
        if self.model is None:
            # User toggled to yolo but weights never loaded.
            try:
                self._load_yolo('yolov8n.pt')
            except ImportError:
                return

        bboxes = self._detect_bboxes(self.latest_bgr)
        if not bboxes:
            self.get_logger().warn(
                f"YOLO: no class-{self.class_id} detections.",
                throttle_duration_sec=2.0,
            )
            return

        try:
            tf_color = self.tf_buffer.lookup_transform(
                self.color_frame, msg.header.frame_id, Time())
            tf_base = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, Time())
        except TransformException as ex:
            self.get_logger().warn(f"TF lookup failed: {ex}", throttle_duration_sec=2.0)
            return

        try:
            raw = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception as ex:
            self.get_logger().warn(f"cloud read failed: {ex}", throttle_duration_sec=5.0)
            return
        pts_src = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float64)
        if pts_src.size == 0:
            return

        pts_color = _apply_transform(tf_color, pts_src)
        pts_base = _apply_transform(tf_base, pts_src)

        z = pts_color[:, 2]
        in_front = z > 0.0
        u = self.color_K[0, 0] * pts_color[:, 0] / np.where(in_front, z, 1.0) + self.color_K[0, 2]
        v = self.color_K[1, 1] * pts_color[:, 1] / np.where(in_front, z, 1.0) + self.color_K[1, 2]
        in_image = in_front & (u >= 0) & (u < self.image_w) & (v >= 0) & (v < self.image_h)

        bbox_mask = np.zeros(len(pts_src), dtype=bool)
        for x1, y1, x2, y2 in bboxes:
            bbox_mask |= in_image & (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)

        ws_mask = (
            (pts_base[:, 0] >= self.min_x) & (pts_base[:, 0] <= self.max_x)
            & (pts_base[:, 1] >= self.min_y) & (pts_base[:, 1] <= self.max_y)
            & (pts_base[:, 2] >= self.min_z) & (pts_base[:, 2] <= self.max_z)
        )
        mask = bbox_mask & ws_mask

        n_inliers = int(mask.sum())
        if n_inliers < self.min_inliers:
            self.get_logger().warn(
                f"Only {n_inliers} inliers (< {self.min_inliers}) — skipping.",
                throttle_duration_sec=2.0,
            )
            return

        filtered = pts_base[mask].astype(np.float32)
        self._publish(filtered, msg.header.stamp)

    def _detect_bboxes(self, bgr: np.ndarray):
        results = self.model.predict(
            bgr, conf=self.conf_thresh, classes=[self.class_id], verbose=False,
        )
        bboxes = []
        for r in results:
            if r.boxes is None or r.boxes.xyxy is None:
                continue
            for b in r.boxes.xyxy.cpu().numpy():
                bboxes.append(b.astype(int))
        return bboxes

    # ================================================================= HSV

    def _handle_hsv(self, msg: 'PointCloud2'):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, Time())
        except TransformException as ex:
            self.get_logger().warn(
                f"TF lookup {msg.header.frame_id} -> {self.target_frame} failed: {ex}",
                throttle_duration_sec=2.0,
            )
            return

        # do_transform_cloud handles the rgb field for us — simpler than the
        # YOLO path's manual matrix math.
        transformed = do_transform_cloud(msg, tf)

        try:
            raw = pc2.read_points(
                transformed, field_names=('x', 'y', 'z', 'rgb'), skip_nans=True,
            )
        except (ValueError, AssertionError) as ex:
            self.get_logger().warn(
                f"Cloud has no rgb field ({ex}); HSV backend can't run.",
                throttle_duration_sec=5.0,
            )
            return

        points = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float32)
        ws_mask = (
            (points[:, 0] >= self.min_x) & (points[:, 0] <= self.max_x)
            & (points[:, 1] >= self.min_y) & (points[:, 1] <= self.max_y)
            & (points[:, 2] >= self.min_z) & (points[:, 2] <= self.max_z)
        )
        color_mask = hsv_mask_from_packed_rgb(raw['rgb'], self._hsv_ranges())
        mask = ws_mask & color_mask

        n = int(mask.sum())
        if n == 0:
            self.get_logger().warn(
                'No points survived box+HSV filters — check TF, box, and HSV params.',
                throttle_duration_sec=2.0,
            )
            return
        if n < self.min_color_points:
            self.get_logger().warn(
                f"Only {n} color-matching points (< {self.min_color_points}) — skipping.",
                throttle_duration_sec=2.0,
            )
            return

        filtered = points[mask]
        self._publish(filtered, msg.header.stamp)

    def _hsv_ranges(self):
        ranges = [HSVRange(self.hsv_lower1, self.hsv_upper1)]
        if any(v > 0 for v in self.hsv_upper2):
            ranges.append(HSVRange(self.hsv_lower2, self.hsv_upper2))
        return ranges

    # ============================================================ shared

    def _publish(self, filtered: np.ndarray, stamp):
        header = Header(frame_id=self.target_frame, stamp=stamp)
        self.filtered_points_pub.publish(
            pc2.create_cloud_xyz32(header, filtered.tolist())
        )
        centroid = np.mean(filtered, axis=0)
        ball_pose = PointStamped()
        # Use the cloud's stamp so the predictor sees the actual sensor time.
        ball_pose.header.stamp = stamp
        ball_pose.header.frame_id = self.target_frame
        ball_pose.point.x = float(centroid[0])
        ball_pose.point.y = float(centroid[1])
        ball_pose.point.z = float(centroid[2])
        self.ball_pose_pub.publish(ball_pose)

    # ------------------------------------------------------------ params

    def _on_parameter_update(self, params):
        bounds = {
            'min_x': self.min_x, 'max_x': self.max_x,
            'min_y': self.min_y, 'max_y': self.max_y,
            'min_z': self.min_z, 'max_z': self.max_z,
        }
        hsv_arrays = {
            'hsv_lower1': self.hsv_lower1, 'hsv_upper1': self.hsv_upper1,
            'hsv_lower2': self.hsv_lower2, 'hsv_upper2': self.hsv_upper2,
        }
        new_conf = self.conf_thresh
        new_class = self.class_id
        new_min_inliers = self.min_inliers
        new_min_color_points = self.min_color_points
        new_detector = self.detector_mode

        for p in params:
            if p.name in bounds and p.type_ == Parameter.Type.DOUBLE:
                bounds[p.name] = float(p.value)
            elif p.name in hsv_arrays and p.type_ == Parameter.Type.INTEGER_ARRAY:
                vals = list(p.value)
                if len(vals) != 3 or any(v < 0 or v > 255 for v in vals):
                    return SetParametersResult(
                        successful=False,
                        reason=f"{p.name} must be 3 elements in [0, 255]",
                    )
                hsv_arrays[p.name] = vals
            elif p.name == 'yolo_conf' and p.type_ == Parameter.Type.DOUBLE:
                if not 0.0 <= p.value <= 1.0:
                    return SetParametersResult(
                        successful=False, reason='yolo_conf must be in [0, 1]')
                new_conf = float(p.value)
            elif p.name == 'yolo_class' and p.type_ == Parameter.Type.INTEGER:
                new_class = int(p.value)
            elif p.name == 'min_inliers' and p.type_ == Parameter.Type.INTEGER:
                new_min_inliers = int(p.value)
            elif p.name == 'min_color_points' and p.type_ == Parameter.Type.INTEGER:
                new_min_color_points = int(p.value)
            elif p.name == 'detector' and p.type_ == Parameter.Type.STRING:
                v = str(p.value).lower()
                if v not in ('yolo', 'hsv'):
                    return SetParametersResult(
                        successful=False, reason="detector must be 'yolo' or 'hsv'")
                new_detector = v

        if (bounds['min_x'] > bounds['max_x']
                or bounds['min_y'] > bounds['max_y']
                or bounds['min_z'] > bounds['max_z']):
            return SetParametersResult(successful=False, reason='min_* must be <= max_*')

        self.min_x, self.max_x = bounds['min_x'], bounds['max_x']
        self.min_y, self.max_y = bounds['min_y'], bounds['max_y']
        self.min_z, self.max_z = bounds['min_z'], bounds['max_z']
        self.hsv_lower1 = hsv_arrays['hsv_lower1']
        self.hsv_upper1 = hsv_arrays['hsv_upper1']
        self.hsv_lower2 = hsv_arrays['hsv_lower2']
        self.hsv_upper2 = hsv_arrays['hsv_upper2']
        self.conf_thresh = new_conf
        self.class_id = new_class
        self.min_inliers = new_min_inliers
        self.min_color_points = new_min_color_points
        if new_detector != self.detector_mode:
            self.get_logger().info(
                f"Detector backend switched: {self.detector_mode} -> {new_detector}"
            )
            self.detector_mode = new_detector
            if new_detector == 'yolo' and self.model is None:
                try:
                    self._load_yolo('yolov8n.pt')
                except ImportError:
                    pass
        return SetParametersResult(successful=True)


def _apply_transform(tf, pts: np.ndarray) -> np.ndarray:
    """Apply a TransformStamped to (N, 3) points."""
    t = tf.transform.translation
    q = tf.transform.rotation
    rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return pts @ rot.T + np.array([t.x, t.y, t.z])


# =================================================================== HSV tuner

_TUNER_TRACKBARS = [('H lo', 0, 179), ('H hi', 179, 179),
                    ('S lo', 0, 255), ('S hi', 255, 255),
                    ('V lo', 0, 255), ('V hi', 255, 255)]


def _make_tuner_window() -> str:
    win = 'hsv_tuner'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for name, init, maxval in _TUNER_TRACKBARS:
        cv2.createTrackbar(name, win, init, maxval, lambda _x: None)
    return win


def _render_tuner_frame(win: str, frame: np.ndarray) -> int:
    """Read trackbars, draw mask preview, handle keys. Returns the cv2 keycode."""
    vals = [cv2.getTrackbarPos(n, win) for n, _, _ in _TUNER_TRACKBARS]
    lower = np.array([vals[0], vals[2], vals[4]], dtype=np.uint8)
    upper = np.array([vals[1], vals[3], vals[5]], dtype=np.uint8)

    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    view = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), masked])
    h, w = view.shape[:2]
    if w > 1280:
        view = cv2.resize(view, (1280, int(h * 1280 / w)))

    cv2.imshow(win, view)
    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        lo, hi = [int(x) for x in lower], [int(x) for x in upper]
        print(f"\nros2 param set /ball_detector hsv_lower1 '{lo}'")
        print(f"ros2 param set /ball_detector hsv_upper1 '{hi}'\n")
    return key


def _run_tuner_local(camera, image_path) -> int:
    """Tune against a webcam or a still image. No ROS needed."""
    if image_path is not None:
        frame_static = cv2.imread(image_path)
        if frame_static is None:
            print(f'could not read {image_path}', file=sys.stderr)
            return 1
        cap = None
    else:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            print(f'could not open camera {camera}', file=sys.stderr)
            return 1
        frame_static = None

    win = _make_tuner_window()
    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            frame = frame_static.copy()
        key = _render_tuner_frame(win, frame)
        if key in (ord('q'), 27):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


def _run_tuner_ros(topic: str) -> int:
    """Tune against a live ROS Image topic (e.g. the lab RealSense)."""
    if not _HAVE_ROS:
        print('rclpy not available — cannot use --topic.', file=sys.stderr)
        return 1

    rclpy.init()
    node = rclpy.create_node('hsv_tuner')
    bridge = CvBridge()
    latest = {'frame': None}

    def _cb(msg):
        try:
            latest['frame'] = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            node.get_logger().warn(f'cv_bridge failed: {e}')

    node.create_subscription(Image, topic, _cb, qos_profile_sensor_data)
    node.get_logger().info(f'hsv_tuner subscribed to {topic} — waiting for frames')

    win = _make_tuner_window()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            if latest['frame'] is None:
                continue
            key = _render_tuner_frame(win, latest['frame'])
            if key in (ord('q'), 27):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0


def main(args=None):
    argv = sys.argv[1:] if args is None else list(args)

    if '--tune' in argv:
        import argparse
        # Strip the `--ros-args ...` tail that `ros2 run` appends.
        if '--ros-args' in argv:
            argv = argv[:argv.index('--ros-args')]
        parser = argparse.ArgumentParser(prog='ball_detector --tune')
        parser.add_argument('--tune', action='store_true')
        src = parser.add_mutually_exclusive_group(required=True)
        src.add_argument('--topic', type=str,
                         help='ROS Image topic (e.g. /camera/camera/color/image_raw)')
        src.add_argument('--camera', type=int, help='Webcam index, e.g. 0')
        src.add_argument('--image', type=str, help='Path to a still image')
        ns = parser.parse_args(argv)
        if ns.topic is not None:
            sys.exit(_run_tuner_ros(ns.topic))
        sys.exit(_run_tuner_local(ns.camera, ns.image))

    if not _HAVE_ROS:
        print('rclpy not available — only --tune mode works without ROS.',
              file=sys.stderr)
        sys.exit(1)

    rclpy.init(args=args)
    node = BallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
