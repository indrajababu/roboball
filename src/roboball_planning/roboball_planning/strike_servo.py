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
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import Marker, MarkerArray

from roboball_msgs.msg import StrikeTarget
from roboball_planning.strike_objectives import ObjectiveConfig, StrikeObjectivePolicy
from roboball_planning.ik import IKPlanner
from roboball_planning.controller import PIDJointVelocityController


JOINT_ORDER = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

CONTROL_PERIOD_S = 0.03
BASE_FRAME = 'base_link'
EE_FRAME = 'tool0'


class StrikePlanner(Node):
    def __init__(self):
        super().__init__('strike_planner')
        self._cb_group = ReentrantCallbackGroup()
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
        self.max_xy_step = float(self.declare_parameter('max_xy_step', 0.12).value)
        self.max_z_drop = float(self.declare_parameter('max_z_drop', 0.02).value)

        self.min_exec_time = float(self.declare_parameter('min_exec_time', 0.04).value)

        self.ik_budget = float(self.declare_parameter('ik_budget', 0.09).value)
        self.ik_timeout = float(self.declare_parameter('ik_timeout', 0.06).value)

        self.publish_debug_markers = bool(
            self.declare_parameter('publish_debug_markers', True).value
        )
        self.debug_marker_frame = str(
            self.declare_parameter('debug_marker_frame', 'base_link').value
        )

        # PID gains seeded from lab7/visual_servoing/main.py:58-60.
        Kp = 0.2 * np.array([0.4, 2.0, 1.7, 1.5, 2.0, 2.0])
        Kd = 0.01 * np.array([2.0, 1.0, 2.0, 0.5, 0.8, 0.8])
        Ki = 0.01 * np.array([1.4, 1.4, 1.4, 1.0, 0.6, 0.6])
        self.Kq = 0.2 * np.array([0.4, 2.0, 1.7, 1.5, 2.0, 2.0])

        self.pid = PIDJointVelocityController(self, Kp, Ki, Kd)

        self.ik_planner = IKPlanner()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.simple_motion = True

        self._busy = False
        self._lock = threading.Lock()

        self.joint_state = None
        self.strike_target = None

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

        self._switch_controller(
            activate='forward_velocity_controller',
            deactivate='scaled_joint_trajectory_controller',
        )

        self.cmd = None

        self.create_timer(CONTROL_PERIOD_S, self._control_tick, callback_group=self._cb_group)

        self.get_logger().info(
            f'ik_timeout={self.ik_timeout:.2f}s. '
            f'paddle_contact_offset={self.paddle_contact_offset.tolist()}. '
            f'publish_debug_markers={self.publish_debug_markers}. '
            'Waiting for /strike_target...'
        )

    # -------------------------------------------------------------- subscriptions

    def _on_joint_state(self, msg: JointState):
        #check for recency?
        with self._lock:
            self.joint_state = msg

    def _on_target(self, msg: StrikeTarget):
        with self._lock:
            self.strike_target = msg

    def _predict(self, msg: StrikeTarget, joint_state: JointState):
        #Take the striketarget
        #Take the jointstate
        #Use ik to find the joint_states to move to, meaning we calculate the target_velocity, target_position for the robot
        #Return the target_velocity, target_position
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
        current_tool_rot = _quat_to_rot(current_qx, current_qy, current_qz, current_qw) # Where our EE is currently
        start_contact_xyz = start_tool_xyz + current_tool_rot @ self.paddle_contact_offset # Where our tool is currently

        impact_contact_xyz = np.array([
            msg.impact_pose.position.x,
            msg.impact_pose.position.y,
            msg.impact_pose.position.z,
        ])

        raw_contact_xyz = impact_contact_xyz.copy()

        xy_delta = impact_contact_xyz[:2] - start_contact_xyz[:2]
        xy_norm = float(np.linalg.norm(xy_delta))
        if xy_norm > self.max_xy_step and xy_norm > 1e-9:
            impact_contact_xyz[:2] = start_contact_xyz[:2] + (
                xy_delta / xy_norm # Clamping logic to make the impact the size of self.max_xy_step
            ) * self.max_xy_step

        min_allowed_z = start_contact_xyz[2] - self.max_z_drop 
        if impact_contact_xyz[2] < min_allowed_z: # Prevents robot from moving lower than the min_allowed_z value
            impact_contact_xyz[2] = min_allowed_z 

        if self.use_current_tool_orientation:
            qx = current_qx
            qy = current_qy
            qz = current_qz
            qw = current_qw
        else:
            q = msg.impact_pose.orientation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w

        desired_tool_rot = _quat_to_rot(qx, qy, qz, qw) # Where our EE should be
        impact_tool_xyz = impact_contact_xyz - desired_tool_rot @ self.paddle_contact_offset # Where our tool should be

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

        target_x, target_y, target_z = impact_tool_xyz[0], impact_tool_xyz[1], impact_tool_xyz[2], 

        req_qx, req_qy, req_qz, req_qw = qx, qy, qz, qw
        sol = self.ik_planner.compute_ik(
            current_joint_state=joint_state,
            x=target_x, y=target_y, z=target_z,
            qx=req_qx, qy=req_qy, qz=req_qz, qw=req_qw,
            timeout_sec=self.ik_timeout,
        )

        return sol

    # ---------------------------------------------------------------- 10 Hz loop

    def _control_tick(self):
        with self._lock:

            if self._busy or self.joint_state is None or self.strike_target is None:
                return
            self._busy = True
            joint_state = self.joint_state
            target = self.strike_target

        current_pos, current_vel = _current_joint_vector(joint_state, JOINT_ORDER)
        target_joint_states = self._predict(target, joint_state)
        arr_target_joint_states = np.array([target_joint_states.position[i] for i in range(len(JOINT_ORDER))])

        if self.simple_motion:
            cmd = self.Kq * (arr_target_joint_states - joint_state)
        else:
            #How do we calculate target_pos and target_vel? 
            target_pos = _reorder_positions(target_joint_states, JOINT_ORDER)
            target_vel = (arr_target_joint_states - joint_state) / CONTROL_PERIOD_S #Apply finite diff here?
            cmd = self.pid.step_control(target_pos, target_vel, current_pos, current_vel, CONTROL_PERIOD_S)

        self._publish_velocity(cmd)

        with self._lock:
            self._busy = False


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
