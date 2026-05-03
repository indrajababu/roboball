"""
Trajectory predictor — Node 2 in the Roboball stack.

Subscribes to `/ball_pose` (PointStamped), keeps a rolling buffer of recent
samples, fits a ballistic model (x,y linear; z with gravity), and publishes:

  /ball_state    (roboball_msgs/BallState)   — filtered current state
  /strike_target (roboball_msgs/StrikeTarget) — where/when to hit it

The predictor is intentionally low-latency: publish as soon as a small
ballistic fit is possible, and let downstream planning account for message age.
"""

from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PointStamped

from roboball_msgs.msg import BallState, StrikeTarget


GRAVITY = 9.81  # m/s^2, +z up in base_link
DEFAULT_BALL_RADIUS = 0.033
DEFAULT_BALL_TO_PADDLE_OFFSET = [-0.082, 0.094, 0.116]
DEFAULT_BALL_TO_PADDLE_OFFSET_MARGIN = [0.0, 0.0, 0.015]


class TrajectoryPredictor(Node):
    def __init__(self):
        super().__init__('trajectory_predictor')

        self.buffer_size = int(self.declare_parameter('buffer_size', 12).value)
        self.min_samples = int(self.declare_parameter('min_samples', 4).value)
        self.strike_height = float(self.declare_parameter('strike_height', 0.60).value)
        self.ball_radius = float(self.declare_parameter('ball_radius', DEFAULT_BALL_RADIUS).value)
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
                DEFAULT_BALL_TO_PADDLE_OFFSET_MARGIN,
            ).value,
            dtype=np.float64,
        )
        self.target_paddle_under_ball = bool(
            self.declare_parameter('target_paddle_under_ball', True).value
        )
        self.stationary_start_enabled = bool(
            self.declare_parameter('stationary_start_enabled', True).value
        )
        self.stationary_start_speed_threshold = float(
            self.declare_parameter('stationary_start_speed_threshold', 0.08).value
        )
        self.stationary_start_position_tolerance = float(
            self.declare_parameter('stationary_start_position_tolerance', 0.025).value
        )
        self.stationary_start_cooldown = float(
            self.declare_parameter('stationary_start_cooldown', 1.0).value
        )
        self._last_stationary_start_pub_time = -1e9
        self._stationary_start_armed = True
        self.publish_only_when_descending = bool(
            self.declare_parameter('publish_only_when_descending', True).value
        )
        self.descending_velocity_threshold = float(
            self.declare_parameter('descending_velocity_threshold', -0.05).value
        )

        self.samples: "deque[tuple[float, np.ndarray]]" = deque(maxlen=self.buffer_size)

        self.ball_sub = self.create_subscription(
            PointStamped, '/ball_pose', self.ball_callback, 10
        )
        self.state_pub = self.create_publisher(BallState, '/ball_state', 10)
        self.target_pub = self.create_publisher(StrikeTarget, '/strike_target', 10)

        self.get_logger().info(
            f'Trajectory predictor up. buffer_size={self.buffer_size}, '
            f'min_samples={self.min_samples}, strike_height={self.strike_height} m, '
            f'ball_radius={self.ball_radius} m, '
            f'ball_to_paddle_offset={self.ball_to_paddle_offset.tolist()}, '
            f'ball_to_paddle_offset_margin={self.ball_to_paddle_offset_margin.tolist()}, '
            f'target_paddle_under_ball={self.target_paddle_under_ball}, '
            f'stationary_start_enabled={self.stationary_start_enabled}, '
            f'stationary_start_speed_threshold={self.stationary_start_speed_threshold} m/s, '
            f'stationary_start_position_tolerance={self.stationary_start_position_tolerance} m, '
            f'stationary_start_cooldown={self.stationary_start_cooldown} s, '
            f'stationary_start_armed={self._stationary_start_armed}, '
            f'publish_only_when_descending={self.publish_only_when_descending}, '
            f'descending_velocity_threshold={self.descending_velocity_threshold} m/s'
        )

    def ball_callback(self, msg: PointStamped):
        t = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pos = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)
        self.samples.append((t, pos))

        if len(self.samples) < self.min_samples:
            return

        fit = self._fit_ballistic()
        if fit is None:
            return

        pos_now, vel_now, t_impact, impact_xyz, vel_impact = fit

        state = BallState()
        state.header = msg.header
        state.position.x, state.position.y, state.position.z = pos_now.tolist()
        state.velocity.x, state.velocity.y, state.velocity.z = vel_now.tolist()
        state.fit_valid = True
        self.state_pub.publish(state)

        stationary, raw_speed, raw_spread = self._stationary_start_status()
        if stationary:
            self._last_stationary_start_pub_time = self._now_sec()
            self._stationary_start_armed = False
            self._publish_target(
                msg.header,
                ball_center_xyz=pos_now,
                time_to_impact=0.0,
                vel_at_impact=np.zeros(3, dtype=np.float64),
                reason=f'stationary_start raw_speed={raw_speed:.3f} raw_spread={raw_spread:.3f}',
            )
            return

        if t_impact is None:
            return

        if (
            self.publish_only_when_descending
            and vel_now[2] > self.descending_velocity_threshold
        ):
            self.get_logger().debug(
                f'Suppressing strike target while ball is not descending fast enough: '
                f'vz={vel_now[2]:.3f} m/s threshold={self.descending_velocity_threshold:.3f}',
                throttle_duration_sec=0.5,
            )
            return

        self._publish_target(
            msg.header,
            ball_center_xyz=impact_xyz,
            time_to_impact=t_impact,
            vel_at_impact=vel_impact,
            reason='ballistic',
        )

    def _stationary_start_status(self):
        if not self.stationary_start_enabled:
            return False, float('inf'), float('inf')
        raw_speed, raw_spread = self._raw_motion_summary()
        if raw_speed > self.stationary_start_speed_threshold:
            if not self._stationary_start_armed:
                self.get_logger().info(
                    f'Stationary start re-armed: raw_speed={raw_speed:.3f} m/s.',
                    throttle_duration_sec=1.0,
                )
            self._stationary_start_armed = True
            return False, raw_speed, raw_spread
        if raw_spread > self.stationary_start_position_tolerance:
            if not self._stationary_start_armed:
                self.get_logger().info(
                    f'Stationary start re-armed: raw_spread={raw_spread:.3f} m.',
                    throttle_duration_sec=1.0,
                )
            self._stationary_start_armed = True
            return False, raw_speed, raw_spread
        if not self._stationary_start_armed:
            return False, raw_speed, raw_spread
        if (
            self._now_sec() - self._last_stationary_start_pub_time
            < self.stationary_start_cooldown
        ):
            return False, raw_speed, raw_spread
        return True, raw_speed, raw_spread

    def _raw_motion_summary(self):
        if len(self.samples) < 2:
            return float('inf'), float('inf')
        t_first, p_first = self.samples[0]
        t_last, p_last = self.samples[-1]
        dt = max(1e-6, float(t_last - t_first))
        raw_speed = float(np.linalg.norm(p_last - p_first) / dt)
        pts = np.vstack([p for _, p in self.samples])
        center = np.mean(pts, axis=0)
        raw_spread = float(np.max(np.linalg.norm(pts - center, axis=1)))
        return raw_speed, raw_spread

    def _should_publish_stationary_start(self, vel_now):
        # Stationary detection uses raw recent samples because ballistic gravity
        # compensation creates a fake velocity for a ball resting on the paddle.
        stationary, _, _ = self._stationary_start_status()
        return stationary

    def _publish_target(self, source_header, ball_center_xyz, time_to_impact, vel_at_impact,
                        reason=''):
        target = StrikeTarget()
        target.header.frame_id = source_header.frame_id
        target.header.stamp = self.get_clock().now().to_msg()
        paddle_xyz = np.array(ball_center_xyz, dtype=np.float64)
        if self.target_paddle_under_ball:
            paddle_xyz -= (
                self.ball_to_paddle_offset + self.ball_to_paddle_offset_margin
            )
        target.impact_pose.position.x = float(paddle_xyz[0])
        target.impact_pose.position.y = float(paddle_xyz[1])
        target.impact_pose.position.z = float(paddle_xyz[2])
        # Paddle is mounted on the side of tool0; the wrist orientation that
        # leaves the paddle face pointing ~up in base_link is the home pose
        # captured below (measured via `tf2_echo base_link tool0`). Holding the
        # wrist at this orientation across the strike means the IK only has to
        # translate to the impact XY, not re-rotate the paddle.
        target.impact_pose.orientation.x = -0.007
        target.impact_pose.orientation.y = 0.699
        target.impact_pose.orientation.z = 0.0
        target.impact_pose.orientation.w = 0.715

        ttl = max(0.0, float(time_to_impact))
        target.time_to_impact = Duration(sec=int(ttl), nanosec=int((ttl % 1.0) * 1e9))
        target.ball_velocity_at_impact.x = float(vel_at_impact[0])
        target.ball_velocity_at_impact.y = float(vel_at_impact[1])
        target.ball_velocity_at_impact.z = float(vel_at_impact[2])
        self.target_pub.publish(target)
        self.get_logger().info(
            f'Published strike target ({reason}): '
            f'ball_center={np.array(ball_center_xyz, dtype=np.float64).tolist()} '
            f'paddle={paddle_xyz.tolist()} tti={ttl:.3f}s',
            throttle_duration_sec=0.5,
        )

    def _now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _fit_ballistic(self):
        """
        Fit z(t) = z0 + vz*t - 0.5*g*t^2 and x,y linear in t to the sample
        buffer. Return current state + predicted strike-height crossing.

        Returns
        -------
        (pos_now, vel_now, time_to_impact, impact_xyz, vel_at_impact) or None
        """
        ts = np.array([s[0] for s in self.samples], dtype=np.float64)
        pts = np.vstack([s[1] for s in self.samples]).astype(np.float64)

        t0 = ts[-1]
        dt = ts - t0

        # Guard against degenerate timestamps (all samples at same time).
        if np.ptp(dt) < 1e-6:
            return None

        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]

        # x(dt) = x0 + vx*dt
        vx, x0 = np.polyfit(dt, xs, 1)
        # y(dt) = y0 + vy*dt
        vy, y0 = np.polyfit(dt, ys, 1)

        # z(dt) = z0 + vz*dt - 0.5*g*dt^2
        # => z(dt) + 0.5*g*dt^2 = z0 + vz*dt
        z_linear = zs + 0.5 * GRAVITY * (dt ** 2)
        vz, z0 = np.polyfit(dt, z_linear, 1)

        pos_now = np.array([x0, y0, z0], dtype=np.float64)
        vel_now = np.array([vx, vy, vz], dtype=np.float64)

        # Solve for descending crossing of strike plane:
        # z0 + vz*t - 0.5*g*t^2 = strike_height, with t >= 0.
        a = -0.5 * GRAVITY
        b = vz
        c = z0 - self.strike_height
        roots = np.roots(np.array([a, b, c], dtype=np.float64))

        t_candidates = []
        for r in roots:
            if abs(r.imag) > 1e-7:
                continue
            t_hit = float(r.real)
            if t_hit < 0.0:
                continue
            vz_at_hit = vz - GRAVITY * t_hit
            # Prefer descending branch to avoid "upward" intersections.
            if vz_at_hit <= 0.0:
                t_candidates.append(t_hit)

        if not t_candidates:
            return pos_now, vel_now, None, None, None

        t_impact = min(t_candidates)
        impact_xyz = np.array(
            [x0 + vx * t_impact, y0 + vy * t_impact, self.strike_height],
            dtype=np.float64,
        )
        vel_impact = np.array(
            [vx, vy, vz - GRAVITY * t_impact],
            dtype=np.float64,
        )

        return pos_now, vel_now, t_impact, impact_xyz, vel_impact


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
