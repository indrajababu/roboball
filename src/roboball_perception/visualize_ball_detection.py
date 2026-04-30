#!/usr/bin/env python3
"""
Visualize filtered 3D point cloud for Roboball detection.

Usage:
  python3 visualize_ball_detection.py

Requirements:
    pip install matplotlib numpy
    ROS deps: rclpy sensor_msgs_py

Run this script inside your ROS 2 environment.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import matplotlib.pyplot as plt
from threading import Lock

class BallDetectionVisualizer(Node):
    def __init__(self):
        super().__init__('ball_detection_visualizer')
        self.centroid = None
        self.points = None
        self.lock = Lock()
        self.create_subscription(PointStamped, '/ball_pose', self.ball_pose_cb, 10)
        self.create_subscription(PointCloud2, '/filtered_points', self.filtered_points_cb, 10)

    def ball_pose_cb(self, msg):
        with self.lock:
            self.centroid = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)

    def filtered_points_cb(self, msg):
        with self.lock:
            pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if pts:
                arr = np.array(pts)
                # Normalize to shape (N, 3) for plotting.
                if arr.ndim == 1:
                    if arr.dtype.names is not None:
                        arr = np.column_stack([arr['x'], arr['y'], arr['z']])
                    else:
                        arr = np.array([np.asarray(p, dtype=np.float64) for p in pts], dtype=np.float64)
                self.points = np.asarray(arr, dtype=np.float64)
            else:
                self.points = None

    def get_data(self):
        with self.lock:
            return self.points, self.centroid

def main():
    rclpy.init()
    node = BallDetectionVisualizer()
    plt.ion()
    fig = plt.figure(figsize=(8, 6))
    ax_3d = fig.add_subplot(111, projection='3d')
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        points, centroid = node.get_data()

        ax_3d.clear()
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('3D Filtered Points + Ball Centroid')
        has_artist = False
        if points is not None:
            ax_3d.scatter(points[:,0], points[:,1], points[:,2], c='b', s=1, label='Filtered Points')
            has_artist = True
        if centroid is not None:
            ax_3d.scatter([centroid[0]], [centroid[1]], [centroid[2]], c='r', s=60, label='Ball Centroid')
            has_artist = True
        else:
            ax_3d.text2D(0.02, 0.95, 'No /ball_pose published yet', transform=ax_3d.transAxes)
        if points is None and centroid is None:
            ax_3d.text2D(0.02, 0.90, 'No /filtered_points yet', transform=ax_3d.transAxes)
        if has_artist:
            ax_3d.legend(loc='upper right')
        plt.pause(0.05)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
