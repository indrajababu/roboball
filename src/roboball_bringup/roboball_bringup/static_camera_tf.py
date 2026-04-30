"""
Static TF: `base_link` -> `camera_color_optical_frame`.

Subscribes to `/aruco_markers`, waits for the marker on the robot base,
then computes and latches a static `base_link` -> `camera_color_optical_frame`
transform using:

    T_base_to_cam = inv(T_cam_to_marker @ G_MARKER_TO_BASE)

where T_cam_to_marker comes from the live ArUco detection and G_MARKER_TO_BASE
is the course-provided fixed offset (marker frame -> base_link).

Publishing `base_link` -> `camera_color_optical_frame` (rather than the old
`ar_marker_N` -> `base_link`) avoids the two-parent conflict: MoveIt already
publishes `world` -> `base_link`, so `base_link` cannot also be a child of the
marker frame.  Making the camera a child of `base_link` joins both trees into
one connected tree.

Set `marker_number` to a non-negative integer to wait for a specific ID.
Default (-1) latches onto the first marker detected.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R

from ros2_aruco_interfaces.msg import ArucoMarkers


# Course-provided fixed offset: T_marker_to_base (marker frame -> base_link).
G_MARKER_TO_BASE = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.16],
    [0.0, 1.0, 0.0, -0.13],
    [0.0, 0.0, 0.0, 1.0],
])


class StaticCameraTransform(Node):
    def __init__(self):
        super().__init__('static_camera_tf')

        self.marker_number = int(self.declare_parameter('marker_number', -1).value)

        self.br = StaticTransformBroadcaster(self)
        self._latched_id: int | None = None
        self.transform: TransformStamped | None = None
        self.timer = None

        if self.marker_number >= 0:
            self.get_logger().info(
                f'Explicit mode: waiting for ar_marker_{self.marker_number} '
                f'on /aruco_markers...'
            )
        else:
            self.get_logger().info(
                'Auto-detect mode: waiting for first marker on /aruco_markers...'
            )
        self.create_subscription(ArucoMarkers, '/aruco_markers', self._on_markers, 10)

    def _on_markers(self, msg: ArucoMarkers):
        if self._latched_id is not None:
            return
        if not msg.marker_ids:
            return
        for i, mid in enumerate(msg.marker_ids):
            mid = int(mid)
            if self.marker_number >= 0 and mid != self.marker_number:
                continue
            self._latch(mid, msg.poses[i])
            break

    def _latch(self, marker_id: int, marker_pose_in_camera):
        self._latched_id = marker_id

        # Build T_cam_to_marker (4x4) from the detected pose.
        p = marker_pose_in_camera.position
        q_in = marker_pose_in_camera.orientation
        rot = R.from_quat([q_in.x, q_in.y, q_in.z, q_in.w]).as_matrix()
        T_cam_to_marker = np.eye(4)
        T_cam_to_marker[:3, :3] = rot
        T_cam_to_marker[:3, 3] = [p.x, p.y, p.z]

        # T_cam_to_base = T_cam_to_marker @ G_MARKER_TO_BASE
        T_cam_to_base = T_cam_to_marker @ G_MARKER_TO_BASE

        # T_base_to_cam = inv(T_cam_to_base)
        T_base_to_cam = np.linalg.inv(T_cam_to_base)

        self.transform = TransformStamped()
        self.transform.header.frame_id = 'base_link'
        self.transform.child_frame_id = 'camera_color_optical_frame'
        self.transform.transform.translation.x = float(T_base_to_cam[0, 3])
        self.transform.transform.translation.y = float(T_base_to_cam[1, 3])
        self.transform.transform.translation.z = float(T_base_to_cam[2, 3])
        q_out = R.from_matrix(T_base_to_cam[:3, :3]).as_quat()  # [x, y, z, w]
        self.transform.transform.rotation.x = float(q_out[0])
        self.transform.transform.rotation.y = float(q_out[1])
        self.transform.transform.rotation.z = float(q_out[2])
        self.transform.transform.rotation.w = float(q_out[3])

        self.get_logger().info(
            f'Latched marker {marker_id}. '
            f'Broadcasting base_link -> camera_color_optical_frame.'
        )

        if self.timer is None:
            self.timer = self.create_timer(0.5, self._broadcast)
        self._broadcast()

    def _broadcast(self):
        if self.transform is None:
            return
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)


def main():
    rclpy.init()
    node = StaticCameraTransform()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
