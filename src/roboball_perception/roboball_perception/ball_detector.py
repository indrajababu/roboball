"""
Ball detector — YOLO variant (yolo branch).

Pipeline:
  1. Cache latest RealSense color image + color camera intrinsics.
  2. On each depth+color point cloud, run YOLOv8n on the cached color image.
     Filter detections to a single COCO class (default 32 = sports ball).
  3. Project every cloud point into the color image plane using the color
     intrinsics. Keep only points whose pixels fall inside any detection bbox.
  4. AND with the workspace bounding box in `target_frame` (default `base_link`)
     as a safety filter, then publish the centroid as `/ball_pose` and the
     surviving points as `/filtered_points`.

Weights auto-download to ~/.config/Ultralytics on first run.
Dependency (one-time, inside the distrobox): `pip install ultralytics`.
"""

import numpy as np
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
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO


SPORTS_BALL_CLASS_ID = 32  # COCO


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')

        self.target_frame = self.declare_parameter('target_frame', 'base_link').value
        self.color_frame = self.declare_parameter('color_frame', 'camera_color_optical_frame').value

        self.min_x = float(self.declare_parameter('min_x', -0.80).value)
        self.max_x = float(self.declare_parameter('max_x',  0.80).value)
        self.min_y = float(self.declare_parameter('min_y', -0.80).value)
        self.max_y = float(self.declare_parameter('max_y',  0.80).value)
        self.min_z = float(self.declare_parameter('min_z',  0.00).value)
        self.max_z = float(self.declare_parameter('max_z',  2.00).value)

        weights = str(self.declare_parameter('yolo_weights', 'yolov8n.pt').value)
        self.conf_thresh = float(self.declare_parameter('yolo_conf', 0.25).value)
        self.class_id = int(self.declare_parameter('yolo_class', SPORTS_BALL_CLASS_ID).value)
        self.min_inliers = int(self.declare_parameter('min_inliers', 20).value)

        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.get_logger().info(f'Loading YOLO weights: {weights}')
        self.model = YOLO(weights)
        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_bgr = None
        self.color_K = None
        self.image_w = 0
        self.image_h = 0

        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self._on_camera_info, 10)
        self.create_subscription(PointCloud2, '/camera/camera/depth/color/points',
                                 self._on_pointcloud, 10)

        self.ball_pose_pub = self.create_publisher(PointStamped, '/ball_pose', 1)
        self.filtered_points_pub = self.create_publisher(PointCloud2, '/filtered_points', 1)

        self.get_logger().info(
            f'YOLO ball detector up. target={self.target_frame}, color={self.color_frame}, '
            f'class={self.class_id}, conf>={self.conf_thresh}, min_inliers={self.min_inliers}, '
            f'workspace x=[{self.min_x},{self.max_x}] y=[{self.min_y},{self.max_y}] '
            f'z=[{self.min_z},{self.max_z}].'
        )

    # --------------------------------------------------------- subscriptions

    def _on_camera_info(self, msg: CameraInfo):
        if self.color_K is None:
            self.color_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.image_w = int(msg.width)
            self.image_h = int(msg.height)
            self.get_logger().info(
                f'Color intrinsics ready: {self.image_w}x{self.image_h}, '
                f'fx={self.color_K[0,0]:.1f} fy={self.color_K[1,1]:.1f}'
            )

    def _on_image(self, msg: Image):
        try:
            self.latest_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as ex:
            self.get_logger().warn(f'cv_bridge failed: {ex}', throttle_duration_sec=5.0)

    def _on_pointcloud(self, msg: PointCloud2):
        if self.latest_bgr is None:
            self.get_logger().warn('No color image yet.', throttle_duration_sec=2.0)
            return
        if self.color_K is None:
            self.get_logger().warn('No camera_info yet.', throttle_duration_sec=2.0)
            return

        bboxes = self._detect_bboxes(self.latest_bgr)
        if not bboxes:
            self.get_logger().warn(
                f'YOLO: no class-{self.class_id} detections.',
                throttle_duration_sec=2.0,
            )
            return

        try:
            tf_color = self.tf_buffer.lookup_transform(
                self.color_frame, msg.header.frame_id, Time())
            tf_base = self.tf_buffer.lookup_transform(
                self.target_frame, msg.header.frame_id, Time())
        except TransformException as ex:
            self.get_logger().warn(f'TF lookup failed: {ex}', throttle_duration_sec=2.0)
            return

        try:
            raw = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception as ex:
            self.get_logger().warn(f'cloud read failed: {ex}', throttle_duration_sec=5.0)
            return
        pts_src = np.column_stack((raw['x'], raw['y'], raw['z'])).astype(np.float64)
        if pts_src.size == 0:
            return

        pts_color = _apply_transform(tf_color, pts_src)
        pts_base = _apply_transform(tf_base, pts_src)

        # Project into color image plane
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
                f'Only {n_inliers} inliers (< {self.min_inliers}) — skipping.',
                throttle_duration_sec=2.0,
            )
            return

        filtered = pts_base[mask].astype(np.float32)
        header = Header(frame_id=self.target_frame, stamp=msg.header.stamp)
        self.filtered_points_pub.publish(pc2.create_cloud_xyz32(header, filtered.tolist()))

        centroid = np.mean(filtered, axis=0)
        ball_pose = PointStamped()
        ball_pose.header.stamp = self.get_clock().now().to_msg()
        ball_pose.header.frame_id = self.target_frame
        ball_pose.point.x = float(centroid[0])
        ball_pose.point.y = float(centroid[1])
        ball_pose.point.z = float(centroid[2])
        self.ball_pose_pub.publish(ball_pose)

    # ---------------------------------------------------------------- YOLO

    def _detect_bboxes(self, bgr: np.ndarray):
        results = self.model.predict(
            bgr,
            conf=self.conf_thresh,
            classes=[self.class_id],
            verbose=False,
        )
        bboxes = []
        for r in results:
            if r.boxes is None or r.boxes.xyxy is None:
                continue
            for b in r.boxes.xyxy.cpu().numpy():
                bboxes.append(b.astype(int))
        return bboxes

    # ------------------------------------------------------------ params

    def _on_parameter_update(self, params):
        bounds = {
            'min_x': self.min_x, 'max_x': self.max_x,
            'min_y': self.min_y, 'max_y': self.max_y,
            'min_z': self.min_z, 'max_z': self.max_z,
        }
        new_conf = self.conf_thresh
        new_class = self.class_id
        new_min_inliers = self.min_inliers

        for p in params:
            if p.name in bounds and p.type_ == Parameter.Type.DOUBLE:
                bounds[p.name] = float(p.value)
            elif p.name == 'yolo_conf' and p.type_ == Parameter.Type.DOUBLE:
                if not 0.0 <= p.value <= 1.0:
                    return SetParametersResult(
                        successful=False, reason='yolo_conf must be in [0, 1]')
                new_conf = float(p.value)
            elif p.name == 'yolo_class' and p.type_ == Parameter.Type.INTEGER:
                new_class = int(p.value)
            elif p.name == 'min_inliers' and p.type_ == Parameter.Type.INTEGER:
                new_min_inliers = int(p.value)

        if (bounds['min_x'] > bounds['max_x']
                or bounds['min_y'] > bounds['max_y']
                or bounds['min_z'] > bounds['max_z']):
            return SetParametersResult(successful=False, reason='min_* must be <= max_*')

        self.min_x, self.max_x = bounds['min_x'], bounds['max_x']
        self.min_y, self.max_y = bounds['min_y'], bounds['max_y']
        self.min_z, self.max_z = bounds['min_z'], bounds['max_z']
        self.conf_thresh = new_conf
        self.class_id = new_class
        self.min_inliers = new_min_inliers
        return SetParametersResult(successful=True)


def _apply_transform(tf, pts: np.ndarray) -> np.ndarray:
    """Apply a TransformStamped to (N, 3) points."""
    t = tf.transform.translation
    q = tf.transform.rotation
    rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return pts @ rot.T + np.array([t.x, t.y, t.z])


def main(args=None):
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
