"""
Strike planner — Node 3 in the Roboball stack.

Subscribes to `/strike_target` (roboball_msgs/StrikeTarget). For each target:
  1. Calls MoveIt `/compute_ik` via `IKPlanner.compute_ik` to solve for joint
     angles at the impact pose.
  2. Builds a `LinearTrajectory` from the current end-effector position to the
     impact point, with `total_time = time_to_impact`.
  3. Hands the joint-goal to the executor (trajectory action by default; PID
     velocity controller once you've switched controllers for live juggling).

Starting state: the outer flow is wired up. The two places you'll actually
edit are marked `TODO`. Before the predictor is finished you can exercise
this file by publishing a hand-crafted StrikeTarget with `ros2 topic pub`.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from roboball_msgs.msg import StrikeTarget
from roboball_planning.ik import IKPlanner
from roboball_planning.controller import UR7eTrajectoryController


class StrikePlanner(Node):
    def __init__(self):
        super().__init__('strike_planner')

        self.joint_state = None
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.create_subscription(StrikeTarget, '/strike_target', self._on_target, 10)

        self.ik_planner = IKPlanner()
        self.controller = UR7eTrajectoryController(self)

        self._busy = False
        self.get_logger().info('Strike planner up. Waiting for /strike_target...')

    def _on_joint_state(self, msg: JointState):
        self.joint_state = msg

    def _on_target(self, msg: StrikeTarget):
        if self._busy:
            self.get_logger().debug('Strike in flight, dropping new target.')
            return
        if self.joint_state is None:
            self.get_logger().warn('No joint state yet, dropping target.')
            return

        p = msg.impact_pose.position
        q = msg.impact_pose.orientation

        ik_solution = self.ik_planner.compute_ik(
            self.joint_state,
            p.x, p.y, p.z,
            qx=q.x, qy=q.y, qz=q.z, qw=q.w,
        )
        if ik_solution is None:
            self.get_logger().error('IK failed for strike target.')
            return

        time_to_impact = msg.time_to_impact.sec + msg.time_to_impact.nanosec * 1e-9
        # Guard: MoveIt+action need a non-zero duration.
        if time_to_impact < 0.2:
            self.get_logger().warn(
                f'time_to_impact={time_to_impact:.3f}s too short — clamping to 0.2s'
            )
            time_to_impact = 0.2

        traj = self._build_single_point_trajectory(ik_solution, time_to_impact)

        self._busy = True
        ok = self.controller.execute_joint_trajectory(traj)
        self._busy = False

        # TODO(step 7): replace the single-point action-based execution above
        # with a LinearTrajectory + PID velocity loop once the
        # forward_velocity_controller is active. See
        # roboball_planning/controller.py:PIDJointVelocityController and
        # roboball_planning/trajectories.py:LinearTrajectory.
        if not ok:
            self.get_logger().error('Trajectory execution failed.')

    @staticmethod
    def _build_single_point_trajectory(joint_state: JointState, total_time: float) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = list(joint_state.name)
        point = JointTrajectoryPoint()
        point.positions = list(joint_state.position)
        point.velocities = [0.0] * len(joint_state.name)
        sec = int(total_time)
        point.time_from_start = Duration(sec=sec, nanosec=int((total_time - sec) * 1e9))
        traj.points.append(point)
        return traj


def main(args=None):
    rclpy.init(args=args)
    node = StrikePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
