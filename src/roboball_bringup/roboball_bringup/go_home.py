"""
Go-to-home — moves the arm to a known Cartesian home pose via IK.

Calls /compute_ik for the target XYZ, then publishes the resulting joint
trajectory through /joint_trajectory_validated so the TrajectoryValidator
bound-checks before forwarding to the UR controller.

Usage:
  ros2 run roboball_bringup go_home
"""

import time

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


HOME_XYZ = (0.57, 0.17, 0)
# Paddle-face-up orientation measured from tf2_echo base_link tool0 at park pose.
HOME_QUAT = (-0.007, 0.699, 0.0, 0.715)  # (qx, qy, qz, qw)

# Must match validate_trajectory.py valid_joint_names exactly (order matters).
VALIDATOR_JOINT_ORDER = [
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
    'shoulder_pan_joint',
]
# validate_trajectory.py valid_joint_positions — used as IK seed to bias the
# solution into the validator's ±0.5 rad window.
VALIDATOR_SEED = [-1.9836, -1.6802, -1.1001, 1.5647, -3.4556, 4.4115]


class GoHome(Node):
    def __init__(self):
        super().__init__('go_home')
        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self._latest_joint_state = None
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_validated', 10)

    def _on_joint_state(self, msg: JointState):
        self._latest_joint_state = msg

    def run(self) -> bool:
        self.get_logger().info('Waiting for /compute_ik service...')
        if not self._ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('/compute_ik not available')
            return False

        # Collect a seed from live joint states.
        deadline = time.monotonic() + 3.0
        while self._latest_joint_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._latest_joint_state is None:
            self.get_logger().error('No /joint_states received — cannot seed IK')
            return False

        seed = JointState()
        seed.name = list(VALIDATOR_JOINT_ORDER)
        seed.position = list(VALIDATOR_SEED)

        x, y, z = HOME_XYZ
        qx, qy, qz, qw = HOME_QUAT

        req = GetPositionIK.Request()
        req.ik_request.group_name = 'ur_manipulator'
        req.ik_request.ik_link_name = 'wrist_3_link'
        req.ik_request.pose_stamped.header.frame_id = 'base_link'
        req.ik_request.pose_stamped.pose.position.x = float(x)
        req.ik_request.pose_stamped.pose.position.y = float(y)
        req.ik_request.pose_stamped.pose.position.z = float(z)
        req.ik_request.pose_stamped.pose.orientation.x = float(qx)
        req.ik_request.pose_stamped.pose.orientation.y = float(qy)
        req.ik_request.pose_stamped.pose.orientation.z = float(qz)
        req.ik_request.pose_stamped.pose.orientation.w = float(qw)
        req.ik_request.robot_state.joint_state = seed
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = Duration(sec=0, nanosec=int(0.5 * 1e9))

        future = self._ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done() or future.result() is None:
            self.get_logger().error('IK service call failed or timed out')
            return False

        result = future.result()
        if result.error_code.val != result.error_code.SUCCESS:
            self.get_logger().error(f'IK failed, code={result.error_code.val}')
            return False

        names = list(result.solution.joint_state.name)
        vals = list(result.solution.joint_state.position)
        name_to_sol = dict(zip(names, vals))

        traj = JointTrajectory()
        traj.joint_names = list(VALIDATOR_JOINT_ORDER)
        point = JointTrajectoryPoint()
        point.positions = [float(name_to_sol.get(n, 0.0)) for n in VALIDATOR_JOINT_ORDER]
        point.velocities = [0.0] * 6
        point.time_from_start.sec = 5
        traj.points.append(point)

        self.pub.publish(traj)
        self.get_logger().info(
            f'Going home: xyz={HOME_XYZ} joints={[f"{v:.4f}" for v in point.positions]}'
        )
        return True


def main(args=None):
    rclpy.init(args=args)
    node = GoHome()
    try:
        node.run()
        time.sleep(1.0)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
