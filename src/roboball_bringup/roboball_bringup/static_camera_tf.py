"""
Static TF: `base_link` -> `camera_color_optical_frame` (externally-mounted camera).

Seeded from lab5/planning/static_tf_transform.py. The lab version broadcast
`wrist_3_link -> camera_depth_optical_frame`; for Roboball the camera is fixed
in the environment, so we broadcast `base_link -> camera_color_optical_frame`
instead.

Procedure to fill in `G`:
  1. Place an ArUco tag at a measured pose relative to `base_link` on the table.
  2. Run lab7's aruco_node against the RealSense stream to get the tag pose in
     the camera frame (call it `T_cam_tag`).
  3. You know `T_base_tag` from step 1 (hand-measured).
  4. Compose: `T_base_cam = T_base_tag @ inv(T_cam_tag)`.
  5. Paste the resulting 4x4 into `G` below.
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

        # TODO(step 3): replace with the measured base_link -> camera transform.
        # Placeholder: camera 1.0 m in front of base_link, 0.7 m above the table,
        # looking down and slightly back toward the robot.
        G = np.eye(4)
        G[0, 3] = 1.0     # x (m) — in front of robot
        G[1, 3] = 0.0     # y (m)
        G[2, 3] = 0.7     # z (m) — above table
        # Rotate so the camera optical axis (+z_optical) points roughly back
        # toward the robot and slightly down. Replace with your calibration.
        G[:3, :3] = R.from_euler('xyz', [np.pi, 0.0, 0.0]).as_matrix()

        parent_frame = self.declare_parameter('parent_frame', 'base_link').value
        child_frame = self.declare_parameter('child_frame', 'camera_color_optical_frame').value

        self.br = StaticTransformBroadcaster(self)

        self.transform = TransformStamped()
        self.transform.header.frame_id = parent_frame
        self.transform.child_frame_id = child_frame
        self.transform.transform.translation.x = float(G[0, 3])
        self.transform.transform.translation.y = float(G[1, 3])
        self.transform.transform.translation.z = float(G[2, 3])
        q = R.from_matrix(G[:3, :3]).as_quat()  # [x, y, z, w]
        self.transform.transform.rotation.x = float(q[0])
        self.transform.transform.rotation.y = float(q[1])
        self.transform.transform.rotation.z = float(q[2])
        self.transform.transform.rotation.w = float(q[3])

        self.get_logger().info(
            f'Broadcasting {parent_frame} -> {child_frame}\nG =\n{G}\nq = {q}'
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
