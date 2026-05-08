#!/usr/bin/env python3
"""Continuous horizontal ball tracker with one bounded vertical pop per cycle."""

import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformException, TransformListener

from roboball_msgs.msg import BallState
from roboball_planning.controller import PIDJointVelocityController
from roboball_planning.ik import IKPlanner


JOINT_ORDER = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

BASE_FRAME = 'base_link'
EE_FRAME = 'tool0'
IK_LINK_NAME = 'tool0'
INCH_TO_M = 0.0254
DEFAULT_PADDLE_OFFSET_TOOL0 = [0.0, 0.0, 7.0 * INCH_TO_M]
DEFAULT_BALL_TO_PADDLE_OFFSET = [-0.082, 0.094, 0.116]


class HorizontalPopTracker(Node):
    """Track ball XY at a locked height, popping once when the ball gets close."""

    TRACK = 'track'
    POP = 'pop'
    RECOVER = 'recover'

    def __init__(self):
        super().__init__('horizontal_pop_tracker')
        self._cb_group = ReentrantCallbackGroup()

        self.control_period = float(self.declare_parameter('control_period', 0.02).value)
        self.pop_height = min(
            float(self.declare_parameter('pop_height', 9.0 * INCH_TO_M).value),
            6.0 * INCH_TO_M,
        )
        self.max_vertical_rise = min(
            float(self.declare_parameter('max_vertical_rise', 9.0 * INCH_TO_M).value),
            6.0 * INCH_TO_M,
        )
        self.pop_trigger_clearance = float(
            self.declare_parameter('pop_trigger_clearance', 17.5 * INCH_TO_M).value
        )
        self.pop_duration = float(
            self.declare_parameter('pop_duration', 0.16).value
        )
        self.pop_hold_duration = float(
            self.declare_parameter('pop_hold_duration', 0.01).value
        )
        self.pop_duration = float(
            self.declare_parameter('pop_duration', 0.12).value
        )
        self.recovery_duration = float(
            self.declare_parameter('recovery_duration', 0.5).value
        )
        
        self.pop_rearm_hysteresis = float(
            self.declare_parameter('pop_rearm_hysteresis', 0.02).value
        )
        self.lock_contact_height = bool(
            self.declare_parameter('lock_contact_height', True).value
        )
        self.min_pop_interval = float(
            self.declare_parameter('min_pop_interval', 0.10).value
        )
        self.descending_velocity_threshold = float(
            self.declare_parameter('descending_velocity_threshold', -0.03).value
        )
        self.hit_velocity_gain = float(
            self.declare_parameter('hit_velocity_gain', 2).value
        )
        self.min_hit_velocity = float(
            self.declare_parameter('min_hit_velocity', 0.05).value
        )
        self.max_hit_velocity = float(
            self.declare_parameter('max_hit_velocity', 3.0).value
        )
        self._pop_paddle_velocity = 0.0
        configured_contact_height = float(
            self.declare_parameter('contact_height', float('nan')).value
        )
        self.contact_height = (
            configured_contact_height
            if np.isfinite(configured_contact_height)
            else None
        )
        self.min_body_clearance_radius = float(
            self.declare_parameter('min_body_clearance_radius', 0.35).value
        )
        self.ball_timeout = float(self.declare_parameter('ball_timeout', 0.35).value)
        self.ik_timeout = float(self.declare_parameter('ik_timeout', 0.06).value)
        self.max_joint_speed = float(
            self.declare_parameter('max_joint_speed', 1.0).value
        )
        self.max_ik_joint_delta = float(
            self.declare_parameter('max_ik_joint_delta', np.pi / 4.0).value
        )
        self.ee_frame = str(self.declare_parameter('ee_frame', EE_FRAME).value)
        self.ik_link_name = str(self.declare_parameter('ik_link_name', IK_LINK_NAME).value)
        self.paddle_offset_tool0 = np.array(
            self.declare_parameter(
                'paddle_offset_tool0',
                DEFAULT_PADDLE_OFFSET_TOOL0,
            ).value,
            dtype=np.float64,
        )
        self.ball_to_paddle_offset = np.array(
            self.declare_parameter(
                'ball_to_paddle_offset',
                DEFAULT_BALL_TO_PADDLE_OFFSET,
            ).value,
            dtype=np.float64,
        )
        self.ball_to_paddle_offset_margin = np.array(
            self.declare_parameter(
                'ball_to_paddle_offset_margin',
                [0.0, 0.0, 0.0],
            ).value,
            dtype=np.float64,
        )

        Kp = 2 * np.array([2.0, 2.0, 1.7, 1.5, 2.0, 2.0])
        Kd = 0.01 * np.array([2.0, 1.0, 2.0, 0.5, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.4, 1.4, 1.0, 0.6, 0.6])
        self.pid = PIDJointVelocityController(
            self,
            Kp,
            Ki,
            Kd,
            dt=self.control_period,
            max_integral_error=0.5,
        )
        self.pid_vertical = PIDJointVelocityController(
            self,
            Kp * 5,
            Ki,
            Kd,
            dt=self.control_period,
            max_integral_error=0.5,
        )
        self.ik_planner = IKPlanner()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.joint_state = None
        self.ball_state = None
        self.ball_rx_time = None

        #Adding logging of when we last contacted
        self.last_contact_time = None
        self._nominal_paddle_z = self.contact_height
        self._locked_tool_quat = None
        self._initial_target_theta = None
        self._last_target_xy = None
        self._state = self.TRACK
        self.rhythm_tuning = None
        self._pop_start_time = None
        self._recover_start_time = None
        self._pop_armed = True
        
        self._last_pop_time = -1e9
        self._ceiling_active = False
        self._lock = threading.Lock()
        self._tick_lock = threading.Lock()

        self.create_subscription(
            JointState,
            '/joint_states',
            self._on_joint_state,
            10,
            callback_group=self._cb_group,
        )
        self.create_subscription(
            BallState,
            '/ball_state',
            self._on_ball_state,
            10,
            callback_group=self._cb_group,
        )
        self.velocity_pub = self.create_publisher(
            Float64MultiArray,
            '/forward_velocity_controller/commands',
            10,
        )

        self._switch_controller(
            activate='forward_velocity_controller',
            deactivate='scaled_joint_trajectory_controller',
        )
        self.create_timer(
            self.control_period,
            self._control_tick,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f'Horizontal pop tracker up. pop_height={self.pop_height:.3f}m, '
            f'max_vertical_rise={self.max_vertical_rise:.3f}m, '
            f'pop_trigger_clearance={self.pop_trigger_clearance:.3f}m, '
            f'pop_hold_duration={self.pop_hold_duration:.3f}s, '
            f'recovery_duration={self.recovery_duration:.3f}s, '
            f'lock_contact_height={self.lock_contact_height}, '
            f'contact_height={self.contact_height}, '
            f'min_body_clearance_radius={self.min_body_clearance_radius:.3f}m, '
            f'max_joint_speed={self.max_joint_speed:.3f}rad/s, '
            f'max_ik_joint_delta={self.max_ik_joint_delta:.3f}rad. '
            'Z target is fixed except during the bounded pop.'
        )

    def _on_joint_state(self, msg):
        with self._lock:
            self.joint_state = msg

    def _on_ball_state(self, msg):
        with self._lock:
            self.ball_state = msg
            self.ball_rx_time = time.monotonic()

            #Add logging for last reached spot

    def _control_tick(self):
        if not self._tick_lock.acquire(blocking=False):
            return
        try:
            self._control_tick_impl()
        finally:
            self._tick_lock.release()

    def _control_tick_impl(self):
        t00 = time.perf_counter()
        with self._lock:
            joint_state = self.joint_state
            ball_state = self.ball_state
            ball_rx_time = self.ball_rx_time

        if joint_state is None:
            return

        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, self.ee_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(
                f'TF lookup {BASE_FRAME}->{self.ee_frame} failed: {ex}',
                throttle_duration_sec=1.0,
            )
            self._publish_velocity(np.zeros(6))
            return

        tool_xyz = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ], dtype=np.float64)
        tool_quat = (
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        )
        current_paddle_xyz = _tool_to_paddle_xyz(
            tool_xyz,
            tool_quat,
            self.paddle_offset_tool0,
        )

        if self._nominal_paddle_z is None:
            self._nominal_paddle_z = float(current_paddle_xyz[2])
            self.get_logger().info(
                f'Locked paddle height at {self._nominal_paddle_z:.3f}m.'
            )
        if self._locked_tool_quat is None:
            self._locked_tool_quat = tool_quat
        if self._initial_target_theta is None:
            self._initial_target_theta = float(np.arctan2(
                current_paddle_xyz[1], current_paddle_xyz[0]
            ))
            self.get_logger().info(
                f'Locked initial radial angle at {self._initial_target_theta:.3f}rad.'
            )

        now_mono = time.monotonic()
        over_ceiling = self._enforce_vertical_ceiling(
            now_mono=now_mono,
            current_paddle_z=float(current_paddle_xyz[2]),
        )

        ball_valid = (
            ball_state is not None
            and ball_state.fit_valid
            and ball_rx_time is not None
            and now_mono - ball_rx_time <= self.ball_timeout
        )

        target_xy = current_paddle_xyz[:2]
        if self._last_target_xy is not None:
            target_xy = self._last_target_xy.copy()
        if ball_valid:
            ball_pos = np.array([
                ball_state.position.x,
                ball_state.position.y,
                ball_state.position.z,
            ], dtype=np.float64)
            ball_vel = np.array([
                ball_state.velocity.x,
                ball_state.velocity.y,
                ball_state.velocity.z,
            ], dtype=np.float64)
            target_xy = (
                ball_pos[:2]
                - self.ball_to_paddle_offset[:2]
                - self.ball_to_paddle_offset_margin[:2]
            )
            target_xy = _clamp_xy_away_from_body(
                target_xy,
                self.min_body_clearance_radius,
            )
            self._last_target_xy = target_xy.copy()
            self._update_pop_state(now_mono, ball_pos, ball_vel)
        else:
            self._finish_pop_state_without_ball(now_mono)

        target_z = self._target_paddle_z(
            now_mono,
            current_paddle_z=float(current_paddle_xyz[2]),
            over_ceiling=over_ceiling,
        )

        target_paddle_xyz = np.array([
            target_xy[0],
            target_xy[1],
            target_z,
        ], dtype=np.float64)
        theta_target = float(np.arctan2(target_xy[1], target_xy[0]))
        delta_theta = theta_target - self._initial_target_theta
        q_target = _rotate_quat_about_world_z(self._locked_tool_quat, delta_theta)
        tool_target_xyz = _paddle_to_tool_xyz(
            target_paddle_xyz,
            q_target,
            self.paddle_offset_tool0,
        )
        qx, qy, qz, qw = q_target
        t0 = time.perf_counter()
        ik_solution = self.ik_planner.compute_ik(
            joint_state,
            tool_target_xyz[0],
            tool_target_xyz[1],
            tool_target_xyz[2],
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
            ik_link_name=self.ik_link_name,
            timeout_sec=self.ik_timeout,
        )
        if ik_solution is None:
            self._publish_velocity(np.zeros(6))
            return
        ik_ms = (time.perf_counter() - t0) * 1000.0
        self.get_logger().info(
            f"iK took {ik_ms: .1f} ms "
        )
        current_pos, current_vel = _current_joint_vector(joint_state, JOINT_ORDER)
        target_pos = _reorder_positions(ik_solution, JOINT_ORDER)
        target_delta = _shortest_joint_delta(target_pos, current_pos)
        max_delta = float(np.max(np.abs(target_delta)))
        if max_delta > self.max_ik_joint_delta:
            self.get_logger().warn(
                f'Rejecting IK branch jump: max_joint_delta={max_delta:.3f}rad '
                f'> limit={self.max_ik_joint_delta:.3f}rad.',
                throttle_duration_sec=1.0,
            )
            self._reset_pid_integrators()
            self._publish_velocity(np.zeros(6))
            return
        target_pos = current_pos + target_delta
        target_vel = np.zeros(6)        
        active_pid = self.pid_vertical if self._state in (self.POP, self.RECOVER) else self.pid
        cmd = active_pid.step_control(target_pos, target_vel, current_pos, current_vel)
        cmd = np.clip(cmd, -self.max_joint_speed, self.max_joint_speed)
        self._publish_velocity(cmd)
        control_ms = (time.perf_counter() - t00) * 1000.0
        self.get_logger().info(
            f"iK took {control_ms: .1f} ms "
        )

    def _update_pop_state(self, now_mono, ball_pos, ball_vel):
        ball_vel_z = float(ball_vel[2])
        ball_clearance = float(ball_pos[2] - self._nominal_paddle_z)
        
        if not self._pop_armed:
            rearm_clearance = self.pop_trigger_clearance + self.pop_rearm_hysteresis
            if ball_clearance >= rearm_clearance:
                self._pop_armed = True
                self.get_logger().info(
                    f'POP re-armed: ball_clearance={ball_clearance:.3f}m.'
                )

        if self._state == self.TRACK:
            #Let's change this so that we trigger by speed

            ready_by_height = ball_clearance <= self.pop_trigger_clearance
            ready_by_velocity = ball_vel_z <= self.descending_velocity_threshold
            ready_by_time = now_mono - self._last_pop_time >= self.min_pop_interval

            if self._pop_armed and ready_by_height and ready_by_velocity and ready_by_time:
                incoming_speed = max(0.0, -ball_vel_z)
                self._pop_paddle_velocity = float(np.clip(
                    self.hit_velocity_gain * incoming_speed,
                    self.min_hit_velocity,
                    self.max_hit_velocity,
                ))
                self._state = self.POP
                self._pop_start_time = now_mono
                self._pop_armed = False
                self._last_pop_time = now_mono
                self._reset_pid_integrators()
                self.get_logger().info(
                    f'POP start: ball_clearance={ball_clearance:.3f}m, '
                    f'ball_vel_z={ball_vel_z:.3f}m/s, '
                    f'pop_paddle_velocity={self._pop_paddle_velocity:.3f}m/s.'
                )
            return

        if self._state == self.POP:
            if now_mono - self._pop_start_time >= self.pop_duration:
                self._state = self.RECOVER

                self._recover_start_time = now_mono
                self._reset_pid_integrators()
                self.get_logger().info('POP complete; recovering to locked height.')
            return

        if self._state == self.RECOVER:
            if now_mono - self._recover_start_time >= self.recovery_duration:
                self._state = self.TRACK
                self._recover_start_time = None
                self._reset_pid_integrators()
                self.get_logger().info('Recovery complete; tracking XY at locked height.')

    def _finish_pop_state_without_ball(self, now_mono):
        if self._state == self.POP and now_mono - self._pop_start_time >= self.pop_duration:
            self._state = self.RECOVER
            self._recover_start_time = now_mono
            self._reset_pid_integrators()
        elif (
            self._state == self.RECOVER
            and now_mono - self._recover_start_time >= self.recovery_duration
        ):
            self._state = self.TRACK
            self._recover_start_time = None
            self._reset_pid_integrators()

    def _reset_pid_integrators(self):
        self.pid.integral_error = np.zeros(6)
        self.pid_vertical.integral_error = np.zeros(6)


    def _enforce_vertical_ceiling(self, now_mono, current_paddle_z):
        ceiling_z = self._nominal_paddle_z + self.max_vertical_rise
        if current_paddle_z < ceiling_z:
            self._ceiling_active = False
            return False

        if not self._ceiling_active:
            self.get_logger().warn(
                f'Vertical ceiling hit: paddle_z={current_paddle_z:.3f}m '
                f'>= ceiling_z={ceiling_z:.3f}m. Forcing recovery to '
                f'{self._nominal_paddle_z:.3f}m.',
                throttle_duration_sec=0.5,
            )
        self._ceiling_active = True
        if self._state != self.RECOVER:
            self._state = self.RECOVER
            self._recover_start_time = now_mono
            self._reset_pid_integrators()
        return True
    

    def _smooth_bump(self, progress):
        progress = np.clip(float(progress), 0.0, 1.0)
        return np.sin(progress * np.pi)


    def reset_to_recover(self):
        self._state = self.RECOVER
        self.last_contact_time = time.monotonic()

    def smooth_bump(self, progress):
        progress = np.clip(float(progress), 0.0, 1.0)
        return np.sin(progress * np.pi)


    def _target_paddle_z(self, now_mono, current_paddle_z, over_ceiling=False):
        if over_ceiling:
            return self._nominal_paddle_z

        ceiling_z = self._nominal_paddle_z + self.max_vertical_rise

        if self._state == self.POP:
            if self._pop_start_time is None:
                return self._nominal_paddle_z
            
            time_elapsed = now_mono - self._pop_start_time
            scale = self._smooth_bump(time_elapsed / self.pop_duration)

            time_based_hit = self._nominal_paddle_z + self.pop_height * scale
            return min(
                time_based_hit,
                ceiling_z,
            )

        return self._nominal_paddle_z

    def _publish_velocity(self, cmd):
        msg = Float64MultiArray()
        msg.data = list(map(float, cmd))
        self.velocity_pub.publish(msg)

    def _switch_controller(self, activate, deactivate):
        cmd = [
            'ros2',
            'control',
            'switch_controllers',
            '--deactivate',
            deactivate,
            '--activate',
            activate,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.get_logger().info(f'Activated {activate}, deactivated {deactivate}.')
            else:
                self.get_logger().warn(
                    f'Controller switch returned {result.returncode}: '
                    f'{result.stderr.strip()}'
                )
        except Exception as exc:
            self.get_logger().error(f'Controller switch failed: {exc}')

    def restore_default_controller(self):
        self._publish_velocity(np.zeros(6))
        self._switch_controller(
            activate='scaled_joint_trajectory_controller',
            deactivate='forward_velocity_controller',
        )


def _reorder_positions(joint_state, order):
    name_to_pos = dict(zip(joint_state.name, joint_state.position))
    return np.array([name_to_pos[n] for n in order], dtype=np.float64)


def _current_joint_vector(joint_state, order):
    name_to_pos = dict(zip(joint_state.name, joint_state.position))
    name_to_vel = (
        dict(zip(joint_state.name, joint_state.velocity))
        if joint_state.velocity
        else {}
    )
    pos = np.array([name_to_pos[n] for n in order], dtype=np.float64)
    vel = np.array([name_to_vel.get(n, 0.0) for n in order], dtype=np.float64)
    return pos, vel


def _shortest_joint_delta(target_pos, current_pos):
    delta = np.asarray(target_pos, dtype=np.float64) - np.asarray(
        current_pos,
        dtype=np.float64,
    )
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _tool_to_paddle_xyz(tool_xyz, tool_quat, paddle_offset_tool0):
    return np.asarray(tool_xyz, dtype=np.float64) + (
        _quat_to_rot(*tool_quat) @ np.asarray(paddle_offset_tool0, dtype=np.float64)
    )


def _paddle_to_tool_xyz(paddle_xyz, tool_quat, paddle_offset_tool0):
    return np.asarray(paddle_xyz, dtype=np.float64) - (
        _quat_to_rot(*tool_quat) @ np.asarray(paddle_offset_tool0, dtype=np.float64)
    )


def _clamp_xy_away_from_body(xy, min_radius):
    min_radius = max(0.0, float(min_radius))
    point = np.asarray(xy, dtype=np.float64).copy()
    if min_radius <= 0.0:
        return point
    radius = float(np.linalg.norm(point))
    if radius >= min_radius:
        return point
    if radius < 1e-9:
        return np.array([min_radius, 0.0], dtype=np.float64)
    return point * (min_radius / radius)


def _quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _rotate_quat_about_world_z(q, delta_theta):
    half = 0.5 * float(delta_theta)
    q_dz = (0.0, 0.0, np.sin(half), np.cos(half))
    return _quat_multiply(q_dz, q)


def _quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def main(args=None):
    rclpy.init(args=args)
    node = HorizontalPopTracker()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.restore_default_controller()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()