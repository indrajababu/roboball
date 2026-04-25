"""
Strike planner — Node 3 in the Roboball stack.

For each `/strike_target`:
  1. Look up current paddle pose via TF (`base_link` -> `tool0`).
  2. Build a `LinearTrajectory` from current pose to impact pose over `time_to_impact`.
  3. Pre-sample IK at NUM_WAYPOINTS evenly-spaced times; finite-difference for joint
     velocities; pack into a JointTrajectory.
  4. A 10 Hz timer interpolates that trajectory and drives the arm via
     `PIDJointVelocityController`, publishing to `/forward_velocity_controller/commands`.

Activates `forward_velocity_controller` on startup and reverts to
`scaled_joint_trajectory_controller` on shutdown so `go_home` keeps working.

Velocity-control loop and controller-swap pattern adapted from
lab7/visual_servoing/main.py.
"""

import subprocess

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener, TransformException

from roboball_msgs.msg import StrikeTarget
from roboball_planning.ik import IKPlanner
from roboball_planning.controller import PIDJointVelocityController
from roboball_planning.trajectories import LinearTrajectory


JOINT_ORDER = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

NUM_WAYPOINTS = 10
CONTROL_PERIOD_S = 0.1
BASE_FRAME = 'base_link'
EE_FRAME = 'tool0'


class StrikePlanner(Node):
    def __init__(self):
        super().__init__('strike_planner')

        # PID gains seeded from lab7/visual_servoing/main.py:58-60.
        Kp = 0.2 * np.array([0.4, 2.0, 1.7, 1.5, 2.0, 2.0])
        Kd = 0.01 * np.array([2.0, 1.0, 2.0, 0.5, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.4, 1.4, 1.0, 0.6, 0.6])
        self.pid = PIDJointVelocityController(self, Kp, Ki, Kd)

        self.ik_planner = IKPlanner()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.joint_state = None
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)
        self.create_subscription(StrikeTarget, '/strike_target', self._on_target, 10)

        self.velocity_pub = self.create_publisher(
            Float64MultiArray, '/forward_velocity_controller/commands', 10
        )

        self._active_traj = None
        self._active_start = None
        self._interp_index = 0
        self._busy = False

        self._switch_controller(
            activate='forward_velocity_controller',
            deactivate='scaled_joint_trajectory_controller',
        )

        self.create_timer(CONTROL_PERIOD_S, self._control_tick)

        self.get_logger().info('Strike planner up. Waiting for /strike_target...')

    # -------------------------------------------------------------- subscriptions

    def _on_joint_state(self, msg: JointState):
        self.joint_state = msg

    def _on_target(self, msg: StrikeTarget):
        if self._busy:
            self.get_logger().debug('Strike in flight, dropping new target.')
            return
        if self.joint_state is None:
            self.get_logger().warn('No joint state yet, dropping target.')
            return

        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, EE_FRAME, Time())
        except TransformException as ex:
            self.get_logger().warn(f'TF lookup {BASE_FRAME}->{EE_FRAME} failed: {ex}')
            return

        start_xyz = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ])
        impact_xyz = np.array([
            msg.impact_pose.position.x,
            msg.impact_pose.position.y,
            msg.impact_pose.position.z,
        ])
        q = msg.impact_pose.orientation

        time_to_impact = msg.time_to_impact.sec + msg.time_to_impact.nanosec * 1e-9
        if time_to_impact < 0.2:
            self.get_logger().warn(
                f'time_to_impact={time_to_impact:.3f}s too short — clamping to 0.2s'
            )
            time_to_impact = 0.2

        cart_traj = LinearTrajectory(start_xyz, impact_xyz, time_to_impact)
        joint_traj = self._build_joint_trajectory(cart_traj, q.x, q.y, q.z, q.w)
        if joint_traj is None:
            return

        self._active_traj = joint_traj
        self._active_start = self.get_clock().now()
        self._interp_index = 0
        self._busy = True
        self.get_logger().info(
            f'Strike armed: {NUM_WAYPOINTS} waypoints over {time_to_impact:.2f}s '
            f'from {start_xyz} to {impact_xyz}.'
        )

    # ---------------------------------------------------------- trajectory build

    def _build_joint_trajectory(self, cart_traj, qx, qy, qz, qw):
        times = np.linspace(0.0, cart_traj.total_time, NUM_WAYPOINTS)
        positions = []
        for i, t in enumerate(times):
            pose = cart_traj.target_pose(t)
            sol = self.ik_planner.compute_ik(
                self.joint_state,
                pose[0], pose[1], pose[2],
                qx=qx, qy=qy, qz=qz, qw=qw,
            )
            if sol is None:
                self.get_logger().error(
                    f'IK failed at waypoint {i + 1}/{NUM_WAYPOINTS} '
                    f'(xyz={pose[:3]}). Aborting strike.'
                )
                return None
            positions.append(_reorder_positions(sol, JOINT_ORDER))

        positions = np.array(positions)
        velocities = _finite_diff(positions, times)

        joint_traj = JointTrajectory()
        joint_traj.joint_names = list(JOINT_ORDER)
        for t, p, v in zip(times, positions, velocities):
            point = JointTrajectoryPoint()
            point.positions = p.tolist()
            point.velocities = v.tolist()
            sec = int(t)
            point.time_from_start = Duration(sec=sec, nanosec=int((t - sec) * 1e9))
            joint_traj.points.append(point)
        return joint_traj

    # ---------------------------------------------------------------- 10 Hz loop

    def _control_tick(self):
        if not self._busy or self._active_traj is None or self.joint_state is None:
            return

        elapsed = (self.get_clock().now() - self._active_start).nanoseconds * 1e-9
        total_time = _last_time(self._active_traj)

        if elapsed >= total_time:
            self._publish_velocity(np.zeros(6))
            self.pid.integral_error = np.zeros(6)
            self._busy = False
            self._active_traj = None
            self.get_logger().info(f'Strike complete (t={elapsed:.2f}s).')
            return

        target_pos, target_vel, self._interp_index = _interpolate(
            self._active_traj, elapsed, self._interp_index
        )
        current_pos, current_vel = _current_joint_vector(self.joint_state, JOINT_ORDER)

        cmd = self.pid.step_control(target_pos, target_vel, current_pos, current_vel)
        self._publish_velocity(cmd)

    def _publish_velocity(self, cmd):
        msg = Float64MultiArray()
        msg.data = list(map(float, cmd))
        self.velocity_pub.publish(msg)

    # ------------------------------------------------------------ controller mgmt

    def _switch_controller(self, activate, deactivate):
        cmd = ['ros2', 'control', 'switch_controllers',
               '--deactivate', deactivate, '--activate', activate]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.get_logger().info(f'Activated {activate}, deactivated {deactivate}.')
            else:
                self.get_logger().warn(
                    f'Controller switch returned {result.returncode}: '
                    f'{result.stderr.strip()}'
                )
        except Exception as e:
            self.get_logger().error(f'Controller switch failed: {e}')

    def restore_default_controller(self):
        self._publish_velocity(np.zeros(6))
        self._switch_controller(
            activate='scaled_joint_trajectory_controller',
            deactivate='forward_velocity_controller',
        )


# ------------------------------------------------------------------- helpers

def _reorder_positions(joint_state: JointState, order):
    name_to_pos = dict(zip(joint_state.name, joint_state.position))
    return np.array([name_to_pos[n] for n in order])


def _current_joint_vector(joint_state: JointState, order):
    name_to_pos = dict(zip(joint_state.name, joint_state.position))
    name_to_vel = (dict(zip(joint_state.name, joint_state.velocity))
                   if joint_state.velocity else {})
    pos = np.array([name_to_pos[n] for n in order])
    vel = np.array([name_to_vel.get(n, 0.0) for n in order])
    return pos, vel


def _finite_diff(positions, times):
    n = len(times)
    velocities = np.zeros_like(positions)
    if n < 2:
        return velocities
    velocities[0] = (positions[1] - positions[0]) / (times[1] - times[0])
    velocities[-1] = (positions[-1] - positions[-2]) / (times[-1] - times[-2])
    for i in range(1, n - 1):
        velocities[i] = (positions[i + 1] - positions[i - 1]) / (times[i + 1] - times[i - 1])
    return velocities


def _last_time(joint_traj: JointTrajectory) -> float:
    last = joint_traj.points[-1].time_from_start
    return last.sec + last.nanosec * 1e-9


def _interpolate(joint_traj: JointTrajectory, t, current_index):
    """Adapted from lab7/visual_servoing/main.py:506-562."""
    eps = 1e-4
    max_index = len(joint_traj.points) - 1

    cur_t = (joint_traj.points[current_index].time_from_start.sec
             + joint_traj.points[current_index].time_from_start.nanosec * 1e-9)
    if cur_t > t:
        current_index = 0

    while (current_index < max_index and
           joint_traj.points[current_index + 1].time_from_start.sec
           + joint_traj.points[current_index + 1].time_from_start.nanosec * 1e-9
           < t + eps):
        current_index += 1

    if current_index < max_index:
        t_lo = (joint_traj.points[current_index].time_from_start.sec
                + joint_traj.points[current_index].time_from_start.nanosec * 1e-9)
        t_hi = (joint_traj.points[current_index + 1].time_from_start.sec
                + joint_traj.points[current_index + 1].time_from_start.nanosec * 1e-9)
        p_lo = np.array(joint_traj.points[current_index].positions)
        p_hi = np.array(joint_traj.points[current_index + 1].positions)
        v_lo = np.array(joint_traj.points[current_index].velocities)
        v_hi = np.array(joint_traj.points[current_index + 1].velocities)
        alpha = (t - t_lo) / (t_hi - t_lo) if t_hi != t_lo else 0.0
        target_pos = p_lo + alpha * (p_hi - p_lo)
        target_vel = v_lo + alpha * (v_hi - v_lo)
    else:
        target_pos = np.array(joint_traj.points[current_index].positions)
        target_vel = np.array(joint_traj.points[current_index].velocities)

    return target_pos, target_vel, current_index


def main(args=None):
    rclpy.init(args=args)
    node = StrikePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_default_controller()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
