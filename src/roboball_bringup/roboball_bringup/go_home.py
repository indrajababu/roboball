#!/usr/bin/env python3
"""
Go-to-home for UR7e.

Publishes a single JointTrajectory point to /joint_trajectory_validated.
This does NOT use IK. It directly commands a known-good joint-space pose.

This version uses the joint ordering observed from your validator / echo:

  shoulder_lift_joint
  elbow_joint
  wrist_1_joint
  wrist_2_joint
  wrist_3_joint
  shoulder_pan_joint

Usage:
  ros2 run roboball_bringup go_home
"""

import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


# Joint order matches your validator / observed trajectory message.
HOME_JOINT_NAMES = [
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "shoulder_pan_joint",
]

# Home pose copied from the known-good joint-space state you pasted.
# This is NOT an IK target. These are direct joint angles in radians.
HOME_POSITIONS = [
    -2.9633223019041957,  # shoulder_lift_joint
    -1.7467526197433472,  # elbow_joint
    1.5880986887165527,   # wrist_1_joint
    1.5541517734527588,   # wrist_2_joint
    -4.671005074177877,   # wrist_3_joint
    3.1735517978668213,   # shoulder_pan_joint
]


class GoHome(Node):
    def __init__(self):
        super().__init__("go_home")

        self.pub = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_validated",
            10,
        )

        self.get_logger().info("Waiting for subscriber on /joint_trajectory_validated...")

        while rclpy.ok() and self.pub.get_subscription_count() == 0:
            self.get_logger().info(
                "No subscriber yet. Is validate_trajectory running?"
            )
            time.sleep(0.5)

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.frame_id = "base_link"
        traj.joint_names = list(HOME_JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = list(HOME_POSITIONS)
        point.velocities = [0.0] * len(HOME_POSITIONS)
        point.accelerations = [0.0] * len(HOME_POSITIONS)
        point.time_from_start.sec = 5
        point.time_from_start.nanosec = 0

        traj.points.append(point)

        self.pub.publish(traj)

        self.get_logger().info("Published direct joint-space home trajectory.")
        self.get_logger().info(f"joint_names={traj.joint_names}")
        self.get_logger().info(f"positions={point.positions}")
        self.get_logger().info("No IK was used.")


def main(args=None):
    rclpy.init(args=args)

    node = GoHome()

    # Spin briefly so DDS has time to publish before shutdown.
    end_time = time.time() + 1.0
    while rclpy.ok() and time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()