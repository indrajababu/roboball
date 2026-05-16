import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState


JOINT_ORDER = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


class IKProbe(Node):
    def __init__(self):
        super().__init__('ik_probe')
        self._latest_joint_state = None
        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.create_subscription(JointState, '/joint_states', self._on_joint_state, 10)

    def _on_joint_state(self, msg: JointState):
        self._latest_joint_state = msg

    def wait_for_services(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self._ik_client.wait_for_service(timeout_sec=0.25):
                return True
            self.get_logger().info('Waiting for /compute_ik service...')
        return False

    def wait_for_joint_state(self, timeout_sec: float) -> JointState | None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self._latest_joint_state is not None:
                return self._latest_joint_state
            rclpy.spin_once(self, timeout_sec=0.1)
        return None

    def call_ik(
        self,
        seed_state: JointState,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        timeout_sec: float,
        response_wait_sec: float,
        avoid_collisions: bool,
    ):
        req = GetPositionIK.Request()
        req.ik_request.group_name = 'ur_manipulator'
        req.ik_request.ik_link_name = 'tool0'
        req.ik_request.pose_stamped.header.frame_id = 'base_link'
        req.ik_request.pose_stamped.pose.position.x = float(x)
        req.ik_request.pose_stamped.pose.position.y = float(y)
        req.ik_request.pose_stamped.pose.position.z = float(z)
        req.ik_request.pose_stamped.pose.orientation.x = float(qx)
        req.ik_request.pose_stamped.pose.orientation.y = float(qy)
        req.ik_request.pose_stamped.pose.orientation.z = float(qz)
        req.ik_request.pose_stamped.pose.orientation.w = float(qw)
        req.ik_request.robot_state.joint_state = seed_state
        req.ik_request.avoid_collisions = bool(avoid_collisions)
        req.ik_request.timeout = Duration(
            sec=int(timeout_sec),
            nanosec=int((timeout_sec % 1.0) * 1e9),
        )

        future = self._ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=response_wait_sec)
        if not future.done():
            self.get_logger().error(
                f'/compute_ik did not respond within {response_wait_sec:.2f}s '
                f'(requested IK timeout={timeout_sec:.2f}s).'
            )
            return None
        return future.result()


def _normalize_quat(qx: float, qy: float, qz: float, qw: float):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        raise ValueError('Quaternion norm is zero')
    return qx / n, qy / n, qz / n, qw / n


def _make_seed_from_joint_state(msg: JointState) -> JointState:
    out = JointState()
    out.name = list(JOINT_ORDER)
    name_to_pos = dict(zip(msg.name, msg.position))
    out.position = [float(name_to_pos[n]) for n in JOINT_ORDER]
    return out


def _make_seed_from_positions(positions: list[float]) -> JointState:
    out = JointState()
    out.name = list(JOINT_ORDER)
    out.position = [float(v) for v in positions]
    return out


def _parse_seed(seed_csv: str | None) -> list[float] | None:
    if seed_csv is None:
        return None
    values = [float(v.strip()) for v in seed_csv.split(',') if v.strip()]
    if len(values) != 6:
        raise ValueError('--seed must provide exactly 6 comma-separated values')
    return values


def _parse_xyz(csv: str | None) -> list[float] | None:
    if csv is None:
        return None
    values = [float(v.strip()) for v in csv.split(',') if v.strip()]
    if len(values) != 3:
        raise ValueError('expected exactly 3 comma-separated values')
    return values


def _quat_to_rot(x: float, y: float, z: float, w: float):
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def _apply_contact_offset(x, y, z, qx, qy, qz, qw, offset):
    rot = _quat_to_rot(qx, qy, qz, qw)
    tool = [
        x - sum(rot[row][col] * offset[col] for col in range(3))
        for row in range(3)
    ]
    return tool


def main(args=None):
    parser = argparse.ArgumentParser(description='Direct IK probe for /compute_ik')
    parser.add_argument('--x', type=float, required=True)
    parser.add_argument('--y', type=float, required=True)
    parser.add_argument('--z', type=float, required=True)
    parser.add_argument('--qx', type=float, required=True)
    parser.add_argument('--qy', type=float, required=True)
    parser.add_argument('--qz', type=float, required=True)
    parser.add_argument('--qw', type=float, required=True)
    parser.add_argument('--timeout', type=float, default=0.25)
    parser.add_argument('--response-wait', type=float, default=3.0)
    parser.add_argument('--service-wait', type=float, default=10.0)
    parser.add_argument('--joint-state-wait', type=float, default=2.0)
    parser.add_argument('--seed', type=str, default=None)
    parser.add_argument(
        '--paddle-contact-offset',
        type=str,
        default=None,
        help='Treat --x/--y/--z as paddle contact target and convert to tool0 using dx,dy,dz in tool0 frame',
    )
    parser.add_argument('--avoid-collisions', action='store_true')
    parsed = parser.parse_args(args=args)

    rclpy.init()
    node = IKProbe()

    try:
        if not node.wait_for_services(parsed.service_wait):
            node.get_logger().error('/compute_ik service not available')
            return 2

        seed_vals = _parse_seed(parsed.seed)
        if seed_vals is None:
            joint_msg = node.wait_for_joint_state(parsed.joint_state_wait)
            if joint_msg is None:
                node.get_logger().error('No /joint_states received; provide --seed explicitly')
                return 2
            seed = _make_seed_from_joint_state(joint_msg)
            node.get_logger().info('Using current /joint_states as IK seed')
        else:
            seed = _make_seed_from_positions(seed_vals)
            node.get_logger().info('Using explicit --seed values as IK seed')

        qx, qy, qz, qw = _normalize_quat(parsed.qx, parsed.qy, parsed.qz, parsed.qw)
        target_x, target_y, target_z = parsed.x, parsed.y, parsed.z
        offset = _parse_xyz(parsed.paddle_contact_offset)
        if offset is not None:
            target_x, target_y, target_z = _apply_contact_offset(
                parsed.x, parsed.y, parsed.z, qx, qy, qz, qw, offset
            )
            node.get_logger().info(
                'Converted paddle contact target '
                f'({parsed.x:.4f},{parsed.y:.4f},{parsed.z:.4f}) with offset={offset} '
                f'to tool0 target ({target_x:.4f},{target_y:.4f},{target_z:.4f})'
            )
        result = node.call_ik(
            seed,
            target_x,
            target_y,
            target_z,
            qx,
            qy,
            qz,
            qw,
            parsed.timeout,
            parsed.response_wait,
            parsed.avoid_collisions,
        )
        if result is None:
            node.get_logger().error('IK service call returned no response')
            return 1

        code = int(result.error_code.val)
        node.get_logger().info(
            f'IK response code={code} for tool0 pose=({target_x:.4f},{target_y:.4f},{target_z:.4f}) '
            f'quat=({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f}) timeout={parsed.timeout:.3f}s '
            f'avoid_collisions={parsed.avoid_collisions}'
        )

        if code != result.error_code.SUCCESS:
            return 1

        names = list(result.solution.joint_state.name)
        vals = list(result.solution.joint_state.position)
        name_to_pos = dict(zip(names, vals))
        ordered = [f'{n}={name_to_pos.get(n, float("nan")):.6f}' for n in JOINT_ORDER]
        node.get_logger().info('IK solution joints: ' + ', '.join(ordered))
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
