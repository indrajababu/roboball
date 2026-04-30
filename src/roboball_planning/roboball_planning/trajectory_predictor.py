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


class TrajectoryPredictor(Node):
    def __init__(self):
        super().__init__('trajectory_predictor')

        self.buffer_size = int(self.declare_parameter('buffer_size', 12).value)
        self.min_samples = int(self.declare_parameter('min_samples', 4).value)
        self.strike_height = float(self.declare_parameter('strike_height', 0.60).value)

        self.samples: "deque[tuple[float, np.ndarray]]" = deque(maxlen=self.buffer_size)

        self.ball_sub = self.create_subscription(
            PointStamped, '/ball_pose', self.ball_callback, 10
        )
        self.state_pub = self.create_publisher(BallState, '/ball_state', 10)
        self.target_pub = self.create_publisher(StrikeTarget, '/strike_target', 10)

        self.get_logger().info(
            f'Trajectory predictor up. buffer_size={self.buffer_size}, '
            f'min_samples={self.min_samples}, strike_height={self.strike_height} m'
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

        if t_impact is None:
            return

        target = StrikeTarget()
        target.header.frame_id = msg.header.frame_id
        target.header.stamp = self.get_clock().now().to_msg()
        target.impact_pose.position.x = float(impact_xyz[0])
        target.impact_pose.position.y = float(impact_xyz[1])
        target.impact_pose.position.z = float(impact_xyz[2])
        # Paddle is mounted on the side of tool0; the wrist orientation that
        # leaves the paddle face pointing ~up in base_link is the home pose
        # captured below (measured via `tf2_echo base_link tool0`). Holding the
        # wrist at this orientation across the strike means the IK only has to
        # translate to the impact XY, not re-rotate the paddle.
        target.impact_pose.orientation.x = -0.007
        target.impact_pose.orientation.y = 0.699
        target.impact_pose.orientation.z = 0.0
        target.impact_pose.orientation.w = 0.715

        ttl = max(0.0, t_impact)
        target.time_to_impact = Duration(sec=int(ttl), nanosec=int((ttl % 1.0) * 1e9))
        target.ball_velocity_at_impact.x = float(vel_impact[0])
        target.ball_velocity_at_impact.y = float(vel_impact[1])
        target.ball_velocity_at_impact.z = float(vel_impact[2])
        self.target_pub.publish(target)

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
