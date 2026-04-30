#!/usr/bin/env python3
"""
Visualize ball detection: live camera image with centroid overlay (left)
and 3D filtered point cloud (right).

Works for both YOLO and HSV detector backends.

Usage:
  python3 visualize_ball_detection.py

Requirements:
    pip install matplotlib numpy opencv-python
    ROS deps: rclpy sensor_msgs_py cv_bridge
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import PointStamped
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock


class BallDetectionVisualizer(Node):
    def __init__(self):
        super().__init__('ball_detection_visualizer')
        self.centroid = None   # 3-element float64 in target_frame
        self.points = None     # Nx3 float64 in target_frame
        self.bgr = None        # latest color image
        self.color_K = None    # 3x3 intrinsics
        self.lock = Lock()
        self.bridge = CvBridge()

        self.create_subscription(PointStamped, '/ball_pose', self._cb_pose, 10)
        self.create_subscription(PointCloud2, '/filtered_points', self._cb_points, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw',
                                 self._cb_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info',
                                 self._cb_info, 10)

    def _cb_pose(self, msg):
        with self.lock:
            self.centroid = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)

    def _cb_points(self, msg):
        with self.lock:
            pts = list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
            if pts:
                arr = np.array(pts)
                if arr.ndim == 1 and arr.dtype.names is not None:
                    arr = np.column_stack([arr['x'], arr['y'], arr['z']])
                self.points = np.asarray(arr, dtype=np.float64)
            else:
                self.points = None

    def _cb_image(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.lock:
                self.bgr = bgr
        except Exception:
            pass

    def _cb_info(self, msg):
        with self.lock:
            if self.color_K is None:
                self.color_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def get_data(self):
        with self.lock:
            return self.points, self.centroid, self.bgr, self.color_K


def _project_centroid(centroid, K):
    """Project a 3D point in camera_color_optical_frame onto image pixels."""
    if centroid is None or K is None or centroid[2] <= 0:
        return None
    u = int(K[0, 0] * centroid[0] / centroid[2] + K[0, 2])
    v = int(K[1, 1] * centroid[1] / centroid[2] + K[1, 2])
    return u, v


def main():
    rclpy.init()
    node = BallDetectionVisualizer()
    plt.ion()
    fig = plt.figure(figsize=(14, 6))
    ax_img = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection='3d')

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        points, centroid, bgr, K = node.get_data()

        # ---- left: camera image with centroid dot ----------------------
        ax_img.clear()
        ax_img.axis('off')
        ax_img.set_title('Camera + Centroid (red dot)')
        if bgr is not None:
            display = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
            px = _project_centroid(centroid, K)
            if px is not None:
                h, w = display.shape[:2]
                if 0 <= px[0] < w and 0 <= px[1] < h:
                    cv2.circle(display, px, 18, (255, 0, 0), -1)
                    cv2.circle(display, px, 20, (255, 255, 255), 2)
            ax_img.imshow(display)
            if centroid is None:
                ax_img.text(0.02, 0.97, 'No /ball_pose', color='red',
                            transform=ax_img.transAxes, va='top', fontsize=9)
        else:
            ax_img.text(0.5, 0.5, 'Waiting for camera image…',
                        ha='center', va='center', transform=ax_img.transAxes)

        # ---- right: 3D point cloud ------------------------------------
        ax_3d.clear()
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('3D Filtered Points')
        has_artist = False
        if points is not None:
            ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c='b', s=1, label='Filtered Points')
            has_artist = True
        if centroid is not None:
            ax_3d.scatter([centroid[0]], [centroid[1]], [centroid[2]],
                          c='r', s=80, label='Ball Centroid')
            has_artist = True
        else:
            ax_3d.text2D(0.02, 0.95, 'No /ball_pose yet', transform=ax_3d.transAxes)
        if has_artist:
            ax_3d.legend(loc='upper right')

        plt.pause(0.05)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
