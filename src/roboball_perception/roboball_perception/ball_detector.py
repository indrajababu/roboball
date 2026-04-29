"""
YOLO-based ball detector.

Pipeline:
  1. Subscribe to the aligned RGB image, the aligned depth image, and camera_info.
  2. Run YOLOv8n on each RGB frame, filter for `target_class` (default 32 = COCO
     "sports ball") above `conf_threshold`.
  3. Pick the highest-confidence detection. Sample depth at the bbox center
     (median over an NxN patch for robustness) to recover z.
  4. Back-project (u, v, z) to a 3D point in `camera_color_optical_frame`
     using the camera intrinsics, then transform to `target_frame`
     (default `base_link`) via TF.
  5. Publish PointStamped on `/ball_pose` and an annotated debug image on
     `/ball_detector/debug_image`.

Required RealSense launch args:
  pointcloud.enable:=true            (kept for downstream consumers)
  align_depth.enable:=true           (this node's depth source)
  rgb_camera.color_profile:=1920x1080x30
"""

from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point

from ultralytics import YOLO


COCO_SPORTS_BALL = 32
DEFAULT_MODEL_PATH = str(Path.home() / '.cache' / 'ultralytics' / 'yolov8n.pt')


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')

        self.target_frame = self.declare_parameter('target_frame', 'base_link').value
        self.camera_frame = self.declare_parameter(
            'camera_frame', 'camera_color_optical_frame'
        ).value

        model_path = self.declare_parameter('model_path', DEFAULT_MODEL_PATH).value
        self.conf_threshold = float(self.declare_parameter('conf_threshold', 0.30).value)
        self.imgsz = int(self.declare_parameter('imgsz', 640).value)
        self.target_class = int(self.declare_parameter('target_class', COCO_SPORTS_BALL).value)
        device_param = str(self.declare_parameter('device', 'auto').value)
        self.device = self._resolve_device(device_param)

        self.min_x = float(self.declare_parameter('min_x', -1.5).value)
        self.max_x = float(self.declare_parameter('max_x',  1.5).value)
        self.min_y = float(self.declare_parameter('min_y', -1.5).value)
        self.max_y = float(self.declare_parameter('max_y',  1.5).value)
        self.min_z = float(self.declare_parameter('min_z',  0.0).value)
        self.max_z = float(self.declare_parameter('max_z',  3.0).value)

        self.depth_patch = int(self.declare_parameter('depth_patch', 5).value)
        self.min_depth_m = float(self.declare_parameter('min_depth_m', 0.20).value)
        self.max_depth_m = float(self.declare_parameter('max_depth_m', 4.00).value)

        self.publish_debug = bool(self.declare_parameter('publish_debug', True).value)

        self.bridge = CvBridge()
        self.latest_depth: np.ndarray | None = None
        self.K: np.ndarray | None = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(f'Loading YOLO from {model_path} on {self.device}...')
        self.model = YOLO(model_path)
        # One warm-up inference so the first real frame doesn't pay the JIT cost.
        _ = self.model.predict(
            np.zeros((480, 640, 3), dtype=np.uint8),
            conf=self.conf_threshold,
            classes=[self.target_class],
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        self.get_logger().info('YOLO ready.')

        self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self._on_depth,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self._on_color,
            qos_profile_sensor_data,
        )

        self.ball_pose_pub = self.create_publisher(PointStamped, '/ball_pose', 1)
        self.debug_pub = self.create_publisher(Image, '/ball_detector/debug_image', 1)

        self.get_logger().info(
            f'Ball detector (YOLO) up. target_frame={self.target_frame}, '
            f'class={self.target_class}, conf>={self.conf_threshold}, '
            f'imgsz={self.imgsz}, device={self.device}.'
        )

    @staticmethod
    def _resolve_device(device_param: str) -> str:
        if device_param != 'auto':
            return device_param
        try:
            import torch
            return 'cuda:0' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'

    def _on_camera_info(self, msg: CameraInfo):
        if self.K is not None:
            return
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.get_logger().info(
            f'Camera intrinsics: fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, '
            f'cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}, frame={msg.header.frame_id}'
        )

    def _on_depth(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'depth bridge failed: {e}', throttle_duration_sec=2.0)

    def _on_color(self, msg: Image):
        if self.K is None:
            self.get_logger().warn(
                'Waiting for /camera/camera/color/camera_info...',
                throttle_duration_sec=2.0,
            )
            return
        if self.latest_depth is None:
            self.get_logger().warn(
                'Waiting for /camera/camera/aligned_depth_to_color/image_raw — '
                'is align_depth.enable:=true set on rs_launch?',
                throttle_duration_sec=2.0,
            )
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'color bridge failed: {e}', throttle_duration_sec=2.0)
            return

        result = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=[self.target_class],
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            if self.publish_debug:
                self._publish_debug(frame, None)
            return

        idx = int(np.argmax(boxes.conf.cpu().numpy()))
        x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().tolist()
        conf = float(boxes.conf[idx].cpu().numpy())
        u = int(round((x1 + x2) * 0.5))
        v = int(round((y1 + y2) * 0.5))

        z_m = self._depth_at(u, v)
        if z_m is None:
            self.get_logger().warn(
                f'No valid depth at bbox center ({u},{v}).',
                throttle_duration_sec=2.0,
            )
            if self.publish_debug:
                self._publish_debug(frame, ((x1, y1, x2, y2), conf, None))
            return

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x_cam = (u - cx) * z_m / fx
        y_cam = (v - cy) * z_m / fy

        ball_in_cam = PointStamped()
        ball_in_cam.header.stamp = msg.header.stamp
        ball_in_cam.header.frame_id = self.camera_frame
        ball_in_cam.point.x = float(x_cam)
        ball_in_cam.point.y = float(y_cam)
        ball_in_cam.point.z = float(z_m)

        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, self.camera_frame, Time()
            )
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform {self.camera_frame} -> {self.target_frame}: {ex}',
                throttle_duration_sec=2.0,
            )
            return

        ball_in_target = do_transform_point(ball_in_cam, tf)
        p = ball_in_target.point

        if not (self.min_x <= p.x <= self.max_x
                and self.min_y <= p.y <= self.max_y
                and self.min_z <= p.z <= self.max_z):
            self.get_logger().warn(
                f'Ball at ({p.x:.2f},{p.y:.2f},{p.z:.2f}) outside workspace box; rejecting.',
                throttle_duration_sec=2.0,
            )
            if self.publish_debug:
                self._publish_debug(frame, ((x1, y1, x2, y2), conf, None))
            return

        ball_in_target.header.stamp = self.get_clock().now().to_msg()
        self.ball_pose_pub.publish(ball_in_target)

        if self.publish_debug:
            self._publish_debug(frame, ((x1, y1, x2, y2), conf, (p.x, p.y, p.z)))

    def _depth_at(self, u: int, v: int) -> float | None:
        d = self.latest_depth
        if d is None:
            return None
        h, w = d.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None
        half = max(0, self.depth_patch // 2)
        u0, u1 = max(0, u - half), min(w, u + half + 1)
        v0, v1 = max(0, v - half), min(h, v + half + 1)
        patch = d[v0:v1, u0:u1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        # RealSense aligned depth is uint16 millimeters.
        z_m = float(np.median(valid.astype(np.float32))) * 0.001
        if not (self.min_depth_m <= z_m <= self.max_depth_m):
            return None
        return z_m

    def _publish_debug(self, frame: np.ndarray, det):
        if det is None:
            label = 'no detection'
        else:
            (x1, y1, x2, y2), conf, xyz = det
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            if xyz is None:
                label = f'ball {conf:.2f}  no depth'
            else:
                label = f'ball {conf:.2f}  ({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f})'
            cv2.putText(
                frame, label, (int(x1), max(0, int(y1) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        except Exception:
            pass


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
