"""
Static TF: `ar_marker_<n>` -> `base_link`.

This uses the course-provided transform from the ArUco marker mounted at the
robot base to the UR7e `base_link`:

G = np.array([
    [-1, 0, 0,  0.0],
    [ 0, 0, 1,  0.16],
    [ 0, 1, 0, -0.13],
    [ 0, 0, 0,  1.0]
])

With this broadcaster running, and an ArUco detector publishing camera->marker
TF, the camera/base relationship is inferred through the TF tree.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R


class StaticCameraTransform(Node):
    def __init__(self):
        super().__init__('static_camera_tf')

        marker_number = int(self.declare_parameter('marker_number', 5).value)
        child_frame = f'ar_marker_{marker_number}'
        parent_frame = 'base_link'

        # G is the original ar_marker -> base_link transform; invert it so we
        # can publish base_link -> ar_marker (base_link already has a parent
        # from the robot URDF, so it cannot be a child of ar_marker).
        G = np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.16],
            [0.0, 1.0, 0.0, -0.13],
            [0.0, 0.0, 0.0, 1.0],
        ])
        G_inv = np.linalg.inv(G)

        self.br = StaticTransformBroadcaster(self)

        self.transform = TransformStamped()
        self.transform.header.frame_id = parent_frame
        self.transform.child_frame_id = child_frame
        self.transform.transform.translation.x = float(G_inv[0, 3])
        self.transform.transform.translation.y = float(G_inv[1, 3])
        self.transform.transform.translation.z = float(G_inv[2, 3])
        q = R.from_matrix(G_inv[:3, :3]).as_quat()  # [x, y, z, w]
        self.transform.transform.rotation.x = float(q[0])
        self.transform.transform.rotation.y = float(q[1])
        self.transform.transform.rotation.z = float(q[2])
        self.transform.transform.rotation.w = float(q[3])

        self.get_logger().info(
            f'Broadcasting {parent_frame} -> {child_frame}\n'
            f'marker_number={marker_number}\nG =\n{G}\nq = {q}'
        )

        # StaticTransformBroadcaster latches, but we re-publish occasionally so
        # downstream nodes that start late still pick it up cleanly.
        self.timer = self.create_timer(0.5, self._broadcast)
        self._broadcast()

    def _broadcast(self):
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
