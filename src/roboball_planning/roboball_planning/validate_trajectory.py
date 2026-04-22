#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory


class TrajectoryValidator(Node):
    def __init__(self):
        super().__init__('trajectory_validator')

        self.valid_joint_names = [
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'shoulder_pan_joint',
        ]

        self.valid_joint_positions = [-1.9836, -1.6802, -1.1001, 1.5647, -3.4556, 4.4115]

        self.create_subscription(
            JointTrajectory,
            '/joint_trajectory_validated',
            self.joint_trajectory_callback,
            10
        )

        self.pub = self.create_publisher(
            JointTrajectory,
            '/scaled_joint_trajectory_controller/joint_trajectory',
            10
        )

        self.get_logger().info('Trajectory validator node is running!')

    def joint_trajectory_callback(self, msg: JointTrajectory):
        if msg.joint_names != self.valid_joint_names:
            self.get_logger().error('Joint names do not match expected joint names!')
            return

        if len(msg.points) != 1:
            self.get_logger().error('Trajectory should only have one point!')
            return

        point = msg.points[0]
        target_positions = point.positions

        if any(abs(valid_joint_pos - joint_pos) > 0.5 for valid_joint_pos, joint_pos in zip(self.valid_joint_positions, target_positions)):
            self.get_logger().error('Joint positions may be unsafe!')
            return


        if any(joint_vel != 0 for joint_vel in point.velocities):
            self.get_logger().error('Joint velocities should be zero!')
            return

        self.pub.publish(msg)
        self.get_logger().info(f'Published target positions: {list(target_positions)}')


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryValidator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
