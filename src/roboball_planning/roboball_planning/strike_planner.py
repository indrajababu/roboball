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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from geometry_msgs.msg import Point, Pose, PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray

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

NUM_WAYPOINTS = 2
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
        paddle_offset = self.declare_parameter(
            'paddle_contact_offset_xyz',
            [-0.094, -0.057, 0.145],
        ).value
        self.paddle_contact_offset = np.array(
            [float(paddle_offset[0]), float(paddle_offset[1]), float(paddle_offset[2])],
            dtype=np.float64,
        )
        self.num_waypoints = int(self.declare_parameter('num_waypoints', NUM_WAYPOINTS).value)
        self.max_xy_step = float(self.declare_parameter('max_xy_step', 0.12).value)
        self.max_z_drop = float(self.declare_parameter('max_z_drop', 0.02).value)
        self.min_exec_time = float(self.declare_parameter('min_exec_time', 0.12).value)
        self.ik_budget = float(self.declare_parameter('ik_budget', 0.08).value)
        self.ik_timeout = float(self.declare_parameter('ik_timeout', 0.25).value)
        self.publish_debug_markers = bool(
            self.declare_parameter('publish_debug_markers', True).value
        )
        self.debug_marker_frame = str(
            self.declare_parameter('debug_marker_frame', 'base_link').value
        )
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

        self.debug_marker_pub = None
        self.debug_pose_pub = None
        if self.publish_debug_markers:
            debug_qos = QoSProfile(depth=1)
            debug_qos.reliability = ReliabilityPolicy.RELIABLE
            debug_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
            self.debug_marker_pub = self.create_publisher(
                MarkerArray, '/strike_planner/debug_markers', debug_qos
            )
            self.debug_pose_pub = self.create_publisher(
                PoseArray, '/strike_planner/debug_waypoint_poses', debug_qos
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
            f'paddle_contact_offset={self.paddle_contact_offset.tolist()}. '
            f'publish_debug_markers={self.publish_debug_markers}. '
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

        start_tool_xyz = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ])
        current_qx = tf.transform.rotation.x
        current_qy = tf.transform.rotation.y
        current_qz = tf.transform.rotation.z
        current_qw = tf.transform.rotation.w
        current_tool_rot = _quat_to_rot(current_qx, current_qy, current_qz, current_qw)
        start_contact_xyz = start_tool_xyz + current_tool_rot @ self.paddle_contact_offset

        impact_contact_xyz = np.array([
            msg.impact_pose.position.x,
            msg.impact_pose.position.y,
            msg.impact_pose.position.z,
        ])
        raw_contact_xyz = impact_contact_xyz.copy()
        if self.apply_objective:
            impact_contact_xyz = self.objective.apply(
                impact_contact_xyz,
                now_sec=self.get_clock().now().nanoseconds * 1e-9,
            )

        # Clamp the paddle contact point, then convert back to the tool0 pose
        # that IK actually solves. This keeps the contact surface, not the
        # flange origin, aligned with the predicted ball target.
        xy_delta = impact_contact_xyz[:2] - start_contact_xyz[:2]
        xy_norm = float(np.linalg.norm(xy_delta))
        if xy_norm > self.max_xy_step and xy_norm > 1e-9:
            impact_contact_xyz[:2] = start_contact_xyz[:2] + (
                xy_delta / xy_norm
            ) * self.max_xy_step

        min_allowed_z = start_contact_xyz[2] - self.max_z_drop
        if impact_contact_xyz[2] < min_allowed_z:
            impact_contact_xyz[2] = min_allowed_z

        if self.use_current_tool_orientation:
            qx = current_qx
            qy = current_qy
            qz = current_qz
            qw = current_qw
        else:
            q = msg.impact_pose.orientation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w

        desired_tool_rot = _quat_to_rot(qx, qy, qz, qw)
        impact_tool_xyz = impact_contact_xyz - desired_tool_rot @ self.paddle_contact_offset

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
        cart_traj = LinearTrajectory(start_tool_xyz, impact_tool_xyz, exec_time)
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
            f'(age={msg_age:.3f}s, IK={build_elapsed:.3f}s) '
            f'tool {start_tool_xyz} -> {impact_tool_xyz}; '
            f'contact {start_contact_xyz} -> raw {raw_contact_xyz}, '
            f'clamped {impact_contact_xyz}. '
            f'offset_tool0_to_contact={self.paddle_contact_offset.tolist()}, '
            f'limits: max_xy_step={self.max_xy_step:.3f}, max_z_drop={self.max_z_drop:.3f}'
        )

    # ---------------------------------------------------------- trajectory build

    def _build_joint_trajectory(self, cart_traj, joint_state, qx, qy, qz, qw):
        times = np.linspace(0.0, cart_traj.total_time, self.num_waypoints)
        positions = []
        cart_points = [cart_traj.target_pose(t) for t in times]
        waypoint_quats = []
        waypoint_success = [True] * len(times)
        seed_state = joint_state
        start_pose = cart_points[0]
        end_pose = cart_points[-1]
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
                waypoint_quats.append((qx, qy, qz, qw))
                continue
            pose = cart_points[i]

            req_qx, req_qy, req_qz, req_qw = qx, qy, qz, qw
            sol = self.ik_planner.compute_ik(
                seed_state,
                pose[0], pose[1], pose[2],
                qx=req_qx, qy=req_qy, qz=req_qz, qw=req_qw,
                timeout_sec=self.ik_timeout,
            )
            waypoint_quats.append((req_qx, req_qy, req_qz, req_qw))
            if sol is None:
                waypoint_success[i] = False
                self._publish_debug_visuals(
                    cart_points,
                    waypoint_quats,
                    waypoint_success,
                    failed_index=i,
                )
                self.get_logger().error(
                    f'IK failed at waypoint {i + 1}/{self.num_waypoints} '
                    f'(t={t:.3f}s, xyz={pose[:3]}). Aborting strike. '
                    f'quat=({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f}), '
                    f'joint_state={_current_joint_summary(joint_state, JOINT_ORDER)}'
                )
                return None
            positions.append(_reorder_positions(sol, JOINT_ORDER))
            seed_state = sol

        self._publish_debug_visuals(
            cart_points,
            waypoint_quats,
            waypoint_success,
            failed_index=None,
        )

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

    # ---------------------------------------------------------- RViz debugging

    def _publish_debug_visuals(self, cart_points, waypoint_quats, waypoint_success, failed_index):
        if self.debug_marker_pub is None or self.debug_pose_pub is None:
            return

        # Transform base_link points into the RViz fixed frame so markers render
        # regardless of whether fixed frame is 'world' or 'base_link'.
        target_frame = self.debug_marker_frame
        pts_in_target = cart_points  # default: same frame, no transform needed
        rot = None
        trans = None
        if target_frame != BASE_FRAME:
            try:
                tf_t = self.tf_buffer.lookup_transform(
                    target_frame, BASE_FRAME, Time()
                )
                q = tf_t.transform.rotation
                t = tf_t.transform.translation
                rot = _quat_to_rot(q.x, q.y, q.z, q.w)
                trans = np.array([t.x, t.y, t.z])
                pts_in_target = [
                    rot @ np.array([float(p[0]), float(p[1]), float(p[2])]) + trans
                    for p in cart_points
                ]
            except TransformException:
                self.get_logger().warn(
                    f'Debug markers: TF lookup {BASE_FRAME}->{target_frame} failed; '
                    f'publishing in {BASE_FRAME} instead.',
                    throttle_duration_sec=5.0,
                )
                target_frame = BASE_FRAME
                pts_in_target = cart_points

        stamp = Time().to_msg()

        pose_array = PoseArray()
        pose_array.header.frame_id = target_frame
        pose_array.header.stamp = stamp
        for pt, quat in zip(pts_in_target, waypoint_quats):
            p = Pose()
            p.position.x = float(pt[0])
            p.position.y = float(pt[1])
            p.position.z = float(pt[2])
            p.orientation.x = float(quat[0])
            p.orientation.y = float(quat[1])
            p.orientation.z = float(quat[2])
            p.orientation.w = float(quat[3])
            pose_array.poses.append(p)
        self.debug_pose_pub.publish(pose_array)

        markers = MarkerArray()

        clear = Marker()
        clear.header.frame_id = target_frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        path = Marker()
        path.header.frame_id = target_frame
        path.header.stamp = stamp
        path.ns = 'strike_path'
        path.id = 1
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.scale.x = 0.008
        path.color.r = 0.05
        path.color.g = 0.75
        path.color.b = 1.00
        path.color.a = 1.00
        for pt in pts_in_target:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = float(pt[2])
            path.points.append(p)
        markers.markers.append(path)

        waypoints = Marker()
        waypoints.header.frame_id = target_frame
        waypoints.header.stamp = stamp
        waypoints.ns = 'strike_waypoints'
        waypoints.id = 2
        waypoints.type = Marker.SPHERE_LIST
        waypoints.action = Marker.ADD
        waypoints.scale.x = 0.025
        waypoints.scale.y = 0.025
        waypoints.scale.z = 0.025

        for idx, pt in enumerate(pts_in_target):
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = float(pt[2])
            waypoints.points.append(p)

            color = _rgba(0.20, 0.85, 0.20, 1.00)
            if failed_index is not None and idx == failed_index:
                color = _rgba(0.95, 0.10, 0.10, 1.00)
            elif not waypoint_success[idx]:
                color = _rgba(0.95, 0.10, 0.10, 1.00)
            elif idx == len(pts_in_target) - 1:
                color = _rgba(1.00, 0.70, 0.10, 1.00)
            waypoints.colors.append(color)

        markers.markers.append(waypoints)

        if failed_index is not None:
            failed_pt = pts_in_target[failed_index]
            text = Marker()
            text.header.frame_id = target_frame
            text.header.stamp = stamp
            text.ns = 'strike_failure'
            text.id = 3
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.scale.z = 0.05
            text.color.r = 1.0
            text.color.g = 0.2
            text.color.b = 0.2
            text.color.a = 1.0
            text.pose.position.x = float(failed_pt[0])
            text.pose.position.y = float(failed_pt[1])
            text.pose.position.z = float(failed_pt[2] + 0.06)
            text.pose.orientation.w = 1.0
            text.text = f'IK FAIL wp {failed_index + 1}/{len(pts_in_target)}'
            markers.markers.append(text)

        self.debug_marker_pub.publish(markers)


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


def _rgba(r, g, b, a):
    c = Marker().color
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


def _quat_to_rot(x, y, z, w):
    """Quaternion (x, y, z, w) → 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


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
