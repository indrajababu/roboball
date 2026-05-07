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
"""
At time 1778146688.346375015
- Translation: [-0.436, 0.176, 0.055]
- Rotation: in Quaternion (xyzw) [-0.680, 0.001, 0.733, -0.017]
- Rotation: in RPY (radian) [0.317, 1.493, -2.801]
- Rotation: in RPY (degree) [18.154, 85.531, -160.495]
- Matrix:
 -0.073  0.024 -0.997 -0.436
 -0.026 -0.999 -0.023  0.176
 -0.997  0.024  0.074  0.055
  0.000  0.000  0.000  1.000

  HOME_POSITIONS = [
    -2.316411157647604,   # shoulder_lift_joint
    -1.810107946395874,  # elbow_joint
    4.19159265,   # wrist_1_joint
    5.211,   # wrist_2_joint
    -1.5386202971087855,  # wrist_3_joint
    5.799105644226074,    # shoulder_pan_joint
]


"""
# Home pose copied from the known-good joint-space state you pasted.
# This is NOT an IK target. These are direct joint angles in radians.
HOME_POSITIONS = [
    -2.316411157647604,   # shoulder_lift_joint
    -1.810107946395874,  # elbow_joint
    1.0502943235584716,   # wrist_1_joint
    2.0732617378234863,   # wrist_2_joint
    -1.5386202971087855,  # wrist_3_joint
    5.799105644226074,    # shoulder_pan_joint
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

        # Wait for DDS to fully establish the connection before publishing.
        time.sleep(0.5)

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.frame_id = "base_link"
        traj.joint_names = list(HOME_JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = list(HOME_POSITIONS)
        point.velocities = [0.0] * len(HOME_POSITIONS)
        point.accelerations = [0.0] * len(HOME_POSITIONS)
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 300000000

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