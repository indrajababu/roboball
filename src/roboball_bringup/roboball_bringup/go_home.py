"""
Go-to-home — adapted from lab3_ur7e/joint_control/joint_pos_controller.py.

Publishes a single JointTrajectory point via the validator topic so the
TrajectoryValidator (lab3_ur7e) bound-checks before forwarding to the UR
controller. Run this at the start of every session to park the arm at a
known-good pose before doing anything else.

Usage:
  ros2 run roboball_bringup go_home
"""

import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# Same safe home used in lab3_ur7e/validate_trajectory.py (validator's
# self.valid_joint_positions). Joint-name ordering matches that validator.
HOME_JOINT_NAMES = [
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
    'shoulder_pan_joint',
]
HOME_POSITIONS = [-1.7394, -2.4058, 1.0243, 1.5489, -4.8575, 3.1717]


class GoHome(Node):
    def __init__(self):
        super().__init__('go_home')
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_validated', 10)

        traj = JointTrajectory()
        traj.joint_names = HOME_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = list(HOME_POSITIONS)
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 5
        traj.points.append(point)

        # Wait for the validate_trajectory subscriber to connect before publishing.
        self.get_logger().info('Waiting for /joint_trajectory_validated subscriber...')
        while self.pub.get_subscription_count() == 0:
            time.sleep(0.1)
        self.pub.publish(traj)
        self.get_logger().info(f'Published home target: {HOME_POSITIONS}')


def main(args=None):
    rclpy.init(args=args)
    node = GoHome()
    rclpy.spin_once(node)
    # Give DDS a moment to flush the message before the process exits.
    time.sleep(1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
