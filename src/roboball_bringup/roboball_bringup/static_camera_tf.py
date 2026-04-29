"""
Static TF: `ar_marker_<base_id>` -> `base_link`.

By default, subscribes to `/aruco_markers` and latches onto the **first marker
ID seen** — so whatever marker is taped to the robot base will work without
having to know its ID up front. Set `marker_number` to a non-negative integer
to skip auto-detect and broadcast immediately for a specific ID.

The transform from marker frame to `base_link` is the course-provided fixed
offset:

    G = np.array([
        [-1, 0, 0,  0.0],
        [ 0, 0, 1,  0.16],
        [ 0, 1, 0, -0.13],
        [ 0, 0, 0,  1.0],
    ])

Once latched, the marker ID stays fixed for the lifetime of the node. To
re-latch onto a different marker, restart the node.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R

from ros2_aruco_interfaces.msg import ArucoMarkers


# Course-provided fixed offset: marker frame -> base_link.
G_MARKER_TO_BASE = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.16],
    [0.0, 1.0, 0.0, -0.13],
    [0.0, 0.0, 0.0, 1.0],
])


class StaticCameraTransform(Node):
    def __init__(self):
        super().__init__('static_camera_tf')

        # marker_number < 0 -> auto-detect (latch on first /aruco_markers detection).
        # marker_number >= 0 -> use that exact ID, skip auto-detect.
        marker_number = int(self.declare_parameter('marker_number', -1).value)

        self.br = StaticTransformBroadcaster(self)
        self._latched_id: int | None = None
        self.transform: TransformStamped | None = None
        self.timer = None

        if marker_number >= 0:
            self.get_logger().info(f'Explicit mode: pinning to ar_marker_{marker_number}.')
            self._latch(marker_number)
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
        self._latch(int(msg.marker_ids[0]))

    def _latch(self, marker_id: int):
        self._latched_id = marker_id
        parent_frame = f'ar_marker_{marker_id}'
        child_frame = 'base_link'

        self.transform = TransformStamped()
        self.transform.header.frame_id = parent_frame
        self.transform.child_frame_id = child_frame
        self.transform.transform.translation.x = float(G_MARKER_TO_BASE[0, 3])
        self.transform.transform.translation.y = float(G_MARKER_TO_BASE[1, 3])
        self.transform.transform.translation.z = float(G_MARKER_TO_BASE[2, 3])
        q = R.from_matrix(G_MARKER_TO_BASE[:3, :3]).as_quat()  # [x, y, z, w]
        self.transform.transform.rotation.x = float(q[0])
        self.transform.transform.rotation.y = float(q[1])
        self.transform.transform.rotation.z = float(q[2])
        self.transform.transform.rotation.w = float(q[3])

        self.get_logger().info(
            f'Latched marker {marker_id}. Broadcasting {parent_frame} -> {child_frame}.'
        )

        # StaticTransformBroadcaster latches, but we re-publish occasionally so
        # downstream nodes that start late still pick it up cleanly.
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
