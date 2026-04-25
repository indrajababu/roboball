"""
Ball detector — seeded from lab5/perception/process_pointcloud.py.

Pipeline:
  1. Subscribe to RealSense depth point cloud.
  2. Transform the cloud into `target_frame` (default `base_link`).
  3. Filter by a workspace bounding box AND by color (HSV mask applied to
     each point's `rgb` field — RealSense already aligns color to depth).
  4. Publish the centroid of the surviving points as `/ball_pose` (PointStamped)
     and the filtered cloud as `/filtered_points` for RViz debug.

Disable the color stage with `enable_hsv:=false` to fall back to the pure
geometric filter (useful while validating the TF + workspace box).

Tune HSV bounds — three options:
  Live, against the lab RealSense (run inside the distrobox while the camera
  is publishing):
    ros2 run roboball_perception ball_detector --tune \
        --topic /camera/camera/color/image_raw
  Offline, on a laptop (no ROS):
    python3 ball_detector.py --tune --camera 0
    python3 ball_detector.py --tune --image ball.jpg
  On a running detector node, push values directly:
    ros2 param set /ball_detector hsv_lower1 '[h, s, v]'
"""

import sys

import cv2
import numpy as np

from roboball_perception.hsv_filter import HSVRange, hsv_mask_from_packed_rgb

# ROS-only imports are deferred so this file can be run as a tuner on a
# laptop without ROS installed (`python3 ball_detector.py --tune ...`).
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from rcl_interfaces.msg import SetParametersResult
    from sensor_msgs.msg import Image, PointCloud2
    from geometry_msgs.msg import PointStamped
    from cv_bridge import CvBridge
    import sensor_msgs_py.point_cloud2 as pc2
    from tf2_ros import Buffer, TransformListener, TransformException
    from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
    _HAVE_ROS = True
except ImportError:
    _HAVE_ROS = False
    Node = object  # so the class definition below still parses


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')

        self.target_frame = self.declare_parameter('target_frame', 'base_link').value
        self.source_frame = self.declare_parameter('source_frame', 'camera_depth_optical_frame').value

        # Workspace bounding box (in target_frame). Tune in RViz with
        # `ros2 param set /ball_detector <name> <value>`.
        self.min_x = float(self.declare_parameter('min_x', -0.80).value)
        self.max_x = float(self.declare_parameter('max_x',  0.80).value)
        self.min_y = float(self.declare_parameter('min_y', -0.80).value)
        self.max_y = float(self.declare_parameter('max_y',  0.80).value)
        self.min_z = float(self.declare_parameter('min_z',  0.00).value)
        self.max_z = float(self.declare_parameter('max_z',  2.00).value)

        # HSV color filter. Defaults are tuned for a yellow ball; re-tune
        # with `ball_detector --tune --topic ...` against the actual beach
        # ball + lighting. Set hsv_upper2 to all-zeros to disable the second
        # range (red wraps H=0/180, so red balls need both).
        self.enable_hsv = bool(self.declare_parameter('enable_hsv', True).value)
        self.hsv_lower1 = list(self.declare_parameter('hsv_lower1', [20, 100, 100]).value)
        self.hsv_upper1 = list(self.declare_parameter('hsv_upper1', [35, 255, 255]).value)
        self.hsv_lower2 = list(self.declare_parameter('hsv_lower2', [0, 0, 0]).value)
        self.hsv_upper2 = list(self.declare_parameter('hsv_upper2', [0, 0, 0]).value)
        self.min_color_points = int(self.declare_parameter('min_color_points', 20).value)

        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10,
        )

        self.ball_pose_pub = self.create_publisher(PointStamped, '/ball_pose', 1)
        self.filtered_points_pub = self.create_publisher(PointCloud2, '/filtered_points', 1)

        self.get_logger().info(
            f'Ball detector up. Target frame: {self.target_frame}. '
            f'Workspace box x=[{self.min_x},{self.max_x}], '
            f'y=[{self.min_y},{self.max_y}], z=[{self.min_z},{self.max_z}]. '
            f'HSV {"ON" if self.enable_hsv else "OFF"}: '
            f'lower1={self.hsv_lower1} upper1={self.hsv_upper1}'
        )

    def _hsv_ranges(self) -> list[HSVRange]:
        ranges = [HSVRange(self.hsv_lower1, self.hsv_upper1)]
        if any(v > 0 for v in self.hsv_upper2):
            ranges.append(HSVRange(self.hsv_lower2, self.hsv_upper2))
        return ranges

    def pointcloud_callback(self, msg: 'PointCloud2'):
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform {self.source_frame} to {self.target_frame}: {ex}',
                throttle_duration_sec=2.0,
            )
            return

        transformed_cloud = do_transform_cloud(msg, tf)

        # Read xyz always; rgb only if HSV is enabled (avoids the per-point
        # decode cost when the team has the color stage off).
        if self.enable_hsv:
            field_names = ('x', 'y', 'z', 'rgb')
        else:
            field_names = ('x', 'y', 'z')

        try:
            raw = pc2.read_points(transformed_cloud, field_names=field_names, skip_nans=True)
        except (ValueError, AssertionError) as ex:
            # Some RealSense configs publish without an `rgb` field. Fall back
            # to xyz-only on the next frame.
            if self.enable_hsv:
                self.get_logger().warn(
                    f'Cloud has no rgb field ({ex}); disabling HSV stage.',
                    throttle_duration_sec=5.0,
                )
                self.enable_hsv = False
                return
            raise

        points = np.column_stack(
            (raw['x'], raw['y'], raw['z'])
        ).astype(np.float32, copy=False)

        # Workspace box filter.
        mask = (
            (points[:, 0] >= self.min_x) & (points[:, 0] <= self.max_x)
            & (points[:, 1] >= self.min_y) & (points[:, 1] <= self.max_y)
            & (points[:, 2] >= self.min_z) & (points[:, 2] <= self.max_z)
        )

        # HSV color filter, AND'd with the box.
        if self.enable_hsv:
            color_mask = hsv_mask_from_packed_rgb(raw['rgb'], self._hsv_ranges())
            mask &= color_mask

        filtered = points[mask]

        if filtered.size == 0:
            self.get_logger().warn(
                'No points survived box+HSV filters — check TF, box, and HSV params.',
                throttle_duration_sec=2.0,
            )
            return

        if self.enable_hsv and filtered.shape[0] < self.min_color_points:
            self.get_logger().warn(
                f'Only {filtered.shape[0]} color-matching points (< {self.min_color_points}); '
                'likely a false positive — skipping this frame.',
                throttle_duration_sec=2.0,
            )
            return

        filtered_cloud = pc2.create_cloud_xyz32(
            transformed_cloud.header,
            filtered.tolist(),
        )
        self.filtered_points_pub.publish(filtered_cloud)

        centroid = np.mean(filtered, axis=0)

        ball_pose = PointStamped()
        ball_pose.header.stamp = self.get_clock().now().to_msg()
        ball_pose.header.frame_id = self.target_frame
        ball_pose.point.x = float(centroid[0])
        ball_pose.point.y = float(centroid[1])
        ball_pose.point.z = float(centroid[2])
        self.ball_pose_pub.publish(ball_pose)

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
        new_enable_hsv = self.enable_hsv
        new_min_color_points = self.min_color_points

        for p in params:
            if p.name in bounds and p.type_ == Parameter.Type.DOUBLE:
                bounds[p.name] = float(p.value)
            elif p.name in hsv_arrays and p.type_ == Parameter.Type.INTEGER_ARRAY:
                vals = list(p.value)
                if len(vals) != 3 or any(v < 0 or v > 255 for v in vals):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be a 3-element array with values in [0, 255]',
                    )
                hsv_arrays[p.name] = vals
            elif p.name == 'enable_hsv' and p.type_ == Parameter.Type.BOOL:
                new_enable_hsv = bool(p.value)
            elif p.name == 'min_color_points' and p.type_ == Parameter.Type.INTEGER:
                new_min_color_points = int(p.value)

        if bounds['min_x'] > bounds['max_x'] \
                or bounds['min_y'] > bounds['max_y'] \
                or bounds['min_z'] > bounds['max_z']:
            return SetParametersResult(successful=False, reason='min_* must be <= max_*')

        self.min_x, self.max_x = bounds['min_x'], bounds['max_x']
        self.min_y, self.max_y = bounds['min_y'], bounds['max_y']
        self.min_z, self.max_z = bounds['min_z'], bounds['max_z']
        self.hsv_lower1 = hsv_arrays['hsv_lower1']
        self.hsv_upper1 = hsv_arrays['hsv_upper1']
        self.hsv_lower2 = hsv_arrays['hsv_lower2']
        self.hsv_upper2 = hsv_arrays['hsv_upper2']
        self.enable_hsv = new_enable_hsv
        self.min_color_points = new_min_color_points
        return SetParametersResult(successful=True)


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


def _run_tuner_local(camera: int | None, image_path: str | None) -> int:
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
    latest: dict[str, np.ndarray | None] = {'frame': None}

    def _cb(msg: 'Image') -> None:
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
        # Strip the `--ros-args ...` tail that `ros2 run` appends, plus any
        # trailing args after it. Keeps argparse from choking when launched
        # via `ros2 run roboball_perception ball_detector --tune --topic ...`.
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
