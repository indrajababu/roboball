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
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener, TransformException

from roboball_msgs.msg import StrikeTarget
from roboball_planning.strike_objectives import ObjectiveConfig, StrikeObjectivePolicy
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

NUM_WAYPOINTS = 5
CONTROL_PERIOD_S = 0.1
BASE_FRAME = 'base_link'
EE_FRAME = 'tool0'


class StrikePlanner(Node):
    def __init__(self):
        super().__init__('strike_planner')
        self._cb_group = ReentrantCallbackGroup()
        objective_mode = str(self.declare_parameter('objective_mode', 'intercept').value)
        self.apply_objective = bool(self.declare_parameter('apply_objective', False).value)
        self.use_current_tool_orientation = bool(
            self.declare_parameter('use_current_tool_orientation', True).value
        )
        self.num_waypoints = int(self.declare_parameter('num_waypoints', NUM_WAYPOINTS).value)
        self.max_xy_step = float(self.declare_parameter('max_xy_step', 0.12).value)
        self.max_z_drop = float(self.declare_parameter('max_z_drop', 0.02).value)
        self.min_exec_time = float(self.declare_parameter('min_exec_time', 0.12).value)
        self.ik_budget = float(self.declare_parameter('ik_budget', 0.08).value)
        self.ik_timeout = float(self.declare_parameter('ik_timeout', 0.15).value)
        blend_gain = float(self.declare_parameter('objective_blend_gain', 0.35).value)
        center_xy = tuple(self.declare_parameter('objective_center_xy', [0.45, 0.0]).value)
        zone_min_xy = tuple(self.declare_parameter('objective_zone_min_xy', [0.2, -0.3]).value)
        zone_max_xy = tuple(self.declare_parameter('objective_zone_max_xy', [0.75, 0.3]).value)
        human_xy = tuple(self.declare_parameter('objective_human_xy', [0.9, 0.0]).value)
        circle_center_xy = tuple(
            self.declare_parameter('objective_circle_center_xy', [0.45, 0.0]).value
        )
        circle_radius = float(self.declare_parameter('objective_circle_radius', 0.2).value)
        circle_speed = float(self.declare_parameter('objective_circle_speed_rad_s', 0.7).value)

        self.objective = StrikeObjectivePolicy(
            ObjectiveConfig(
                mode=objective_mode,
                center_xy=(float(center_xy[0]), float(center_xy[1])),
                zone_min_xy=(float(zone_min_xy[0]), float(zone_min_xy[1])),
                zone_max_xy=(float(zone_max_xy[0]), float(zone_max_xy[1])),
                human_xy=(float(human_xy[0]), float(human_xy[1])),
                circle_center_xy=(float(circle_center_xy[0]), float(circle_center_xy[1])),
                circle_radius=circle_radius,
                circle_speed_rad_s=circle_speed,
                blend_gain=blend_gain,
            )
        )

        # PID gains seeded from lab7/visual_servoing/main.py:58-60.
        Kp = 0.2 * np.array([0.4, 2.0, 1.7, 1.5, 2.0, 2.0])
        Kd = 0.01 * np.array([2.0, 1.0, 2.0, 0.5, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.4, 1.4, 1.0, 0.6, 0.6])
        self.pid = PIDJointVelocityController(self, Kp, Ki, Kd)

        self.ik_planner = IKPlanner()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.joint_state = None
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10,
            callback_group=self._cb_group,
        )
        self.create_subscription(
            StrikeTarget, '/strike_target', self._on_target, 10,
            callback_group=self._cb_group,
        )

        self.velocity_pub = self.create_publisher(
            Float64MultiArray, '/forward_velocity_controller/commands', 10
        )

        self._active_traj = None
        self._active_start = None
        self._interp_index = 0
        self._busy = False
        self._planning = False
        self._lock = threading.Lock()

        self._switch_controller(
            activate='forward_velocity_controller',
            deactivate='scaled_joint_trajectory_controller',
        )

        self.create_timer(CONTROL_PERIOD_S, self._control_tick, callback_group=self._cb_group)

        self.get_logger().info(
            f'Strike planner up. objective_mode={self.objective.config.mode}. '
            f'num_waypoints={self.num_waypoints}, ik_budget={self.ik_budget:.2f}s, '
            f'ik_timeout={self.ik_timeout:.2f}s. '
            'Waiting for /strike_target...'
        )

    # -------------------------------------------------------------- subscriptions

    def _on_joint_state(self, msg: JointState):
        with self._lock:
            self.joint_state = msg

    def _on_target(self, msg: StrikeTarget):
        with self._lock:
            if self._planning:
                self.get_logger().debug('IK build in progress, dropping superseded target.')
                return
            self._planning = True
            joint_state = self.joint_state

        if joint_state is None:
            with self._lock:
                self._planning = False
            self.get_logger().warn('No joint state yet, dropping target.')
            return

        try:
            self._plan_target(msg, joint_state)
        finally:
            with self._lock:
                self._planning = False

    def _plan_target(self, msg: StrikeTarget, joint_state: JointState):
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
        if self.apply_objective:
            impact_xyz = self.objective.apply(
                impact_xyz,
                now_sec=self.get_clock().now().nanoseconds * 1e-9,
            )

        # Keep per-strike displacement inside a reachable neighborhood around
        # current pose; this prevents impossible IK requests from noisy targets.
        xy_delta = impact_xyz[:2] - start_xyz[:2]
        xy_norm = float(np.linalg.norm(xy_delta))
        if xy_norm > self.max_xy_step and xy_norm > 1e-9:
            impact_xyz[:2] = start_xyz[:2] + (xy_delta / xy_norm) * self.max_xy_step

        min_allowed_z = start_xyz[2] - self.max_z_drop
        if impact_xyz[2] < min_allowed_z:
            impact_xyz[2] = min_allowed_z

        if self.use_current_tool_orientation:
            qx = tf.transform.rotation.x
            qy = tf.transform.rotation.y
            qz = tf.transform.rotation.z
            qw = tf.transform.rotation.w
        else:
            q = msg.impact_pose.orientation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w

        predicted_tti = msg.time_to_impact.sec + msg.time_to_impact.nanosec * 1e-9
        msg_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        msg_age = max(0.0, now_sec - msg_stamp) if msg_stamp > 0.0 else 0.0
        remaining_to_impact = predicted_tti - msg_age
        exec_time = remaining_to_impact - self.ik_budget
        if exec_time < self.min_exec_time:
            self.get_logger().warn(
                f'target too late: predicted={predicted_tti:.3f}s age={msg_age:.3f}s '
                f'budget={self.ik_budget:.3f}s remaining_exec={exec_time:.3f}s'
            )
            return

        build_start = time.monotonic()
        cart_traj = LinearTrajectory(start_xyz, impact_xyz, exec_time)
        joint_traj = self._build_joint_trajectory(
            cart_traj, joint_state, qx, qy, qz, qw
        )
        if joint_traj is None:
            return

        build_elapsed = time.monotonic() - build_start
        actual_remaining = remaining_to_impact - build_elapsed
        if actual_remaining < self.min_exec_time:
            self.get_logger().warn(
                f'IK finished too late: build={build_elapsed:.3f}s '
                f'actual_remaining={actual_remaining:.3f}s'
            )
            return

        # If IK took longer than the reserved budget, begin partway through
        # the trajectory so the final waypoint still aligns with impact time.
        elapsed_offset = max(0.0, cart_traj.total_time - actual_remaining)
        active_start = self.get_clock().now() - RclpyDuration(seconds=elapsed_offset)
        with self._lock:
            self._active_traj = joint_traj
            self._active_start = active_start
            self._interp_index = 0
            self._busy = True
        self.get_logger().info(
            f'Strike updated: {self.num_waypoints} waypoints over {exec_time:.2f}s '
            f'(age={msg_age:.3f}s, IK={build_elapsed:.3f}s) from {start_xyz} to {impact_xyz}. '
            f'limits: max_xy_step={self.max_xy_step:.3f}, max_z_drop={self.max_z_drop:.3f}'
        )

    # ---------------------------------------------------------- trajectory build

    def _build_joint_trajectory(self, cart_traj, joint_state, qx, qy, qz, qw):
        times = np.linspace(0.0, cart_traj.total_time, self.num_waypoints)
        positions = []
        ik_seed_state = joint_state
        start_pose = cart_traj.target_pose(0.0)
        end_pose = cart_traj.target_pose(cart_traj.total_time)
        self.get_logger().info(
            'IK build start: '
            f'waypoints={self.num_waypoints}, total_time={cart_traj.total_time:.3f}s, '
            f'start=({start_pose[0]:.4f},{start_pose[1]:.4f},{start_pose[2]:.4f}), '
            f'end=({end_pose[0]:.4f},{end_pose[1]:.4f},{end_pose[2]:.4f}), '
            f'quat=({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f})'
        )
        for i, t in enumerate(times):
            if i == 0:
                positions.append(_reorder_positions(joint_state, JOINT_ORDER))
                continue
            pose = cart_traj.target_pose(t)
            sol = self.ik_planner.compute_ik(
                ik_seed_state,
                pose[0], pose[1], pose[2],
                qx=qx, qy=qy, qz=qz, qw=qw,
                timeout_sec=self.ik_timeout,
            )
            if sol is None:
                self.get_logger().error(
                    f'IK failed at waypoint {i + 1}/{self.num_waypoints} '
                    f'(t={t:.3f}s, xyz={pose[:3]}). Aborting strike. '
                    f'quat=({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f}), '
                    f'joint_state={_current_joint_summary(joint_state, JOINT_ORDER)}'
                )
                return None
            ik_seed_state = sol
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
        with self._lock:
            busy = self._busy
            active_traj = self._active_traj
            active_start = self._active_start
            interp_index = self._interp_index
            joint_state = self.joint_state

        if not busy or active_traj is None or active_start is None or joint_state is None:
            return

        elapsed = (self.get_clock().now() - active_start).nanoseconds * 1e-9
        total_time = _last_time(active_traj)

        if elapsed >= total_time:
            self._publish_velocity(np.zeros(6))
            self.pid.integral_error = np.zeros(6)
            with self._lock:
                self._busy = False
                self._active_traj = None
            self.get_logger().info(f'Strike complete (t={elapsed:.2f}s).')
            return

        target_pos, target_vel, next_index = _interpolate(
            active_traj, elapsed, interp_index
        )
        with self._lock:
            if self._active_traj is active_traj:
                self._interp_index = next_index
        current_pos, current_vel = _current_joint_vector(joint_state, JOINT_ORDER)

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


def _current_joint_summary(joint_state: JointState, order):
    if joint_state is None:
        return 'none'
    name_to_pos = dict(zip(joint_state.name, joint_state.position))
    values = [f'{n}={name_to_pos.get(n, float("nan")):.3f}' for n in order]
    return ', '.join(values)


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
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_default_controller()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
