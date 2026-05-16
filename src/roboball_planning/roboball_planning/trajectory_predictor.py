"""
Trajectory predictor — Node 2 in the Roboball stack.

Subscribes to `/ball_pose` (PointStamped), keeps a rolling buffer of recent
samples, fits a ballistic model (x,y linear; z with gravity), and publishes:

  /ball_state (roboball_msgs/BallState) — filtered current state

The predictor is intentionally low-latency: publish as soon as a small
ballistic fit is possible, and let downstream planning/control account for
message age.
"""

from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PointStamped

from roboball_msgs.msg import BallState


GRAVITY = 9.81  # m/s^2, +z up in base_link


class TrajectoryPredictor(Node):
    def __init__(self):
        super().__init__('trajectory_predictor')

        self.buffer_size = int(self.declare_parameter('buffer_size', 12).value)
        self.min_samples = int(self.declare_parameter('min_samples', 4).value)

        self.samples: "deque[tuple[float, np.ndarray]]" = deque(maxlen=self.buffer_size)

        self.ball_sub = self.create_subscription(
            PointStamped, '/ball_pose', self.ball_callback, 10
        )
        self.state_pub = self.create_publisher(BallState, '/ball_state', 10)

        self.get_logger().info(
            f'Trajectory predictor up. buffer_size={self.buffer_size}, '
            f'min_samples={self.min_samples}'
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

        pos_now, vel_now = fit

        state = BallState()
        state.header = msg.header
        state.position.x, state.position.y, state.position.z = pos_now.tolist()
        state.velocity.x, state.velocity.y, state.velocity.z = vel_now.tolist()
        state.fit_valid = True
        self.state_pub.publish(state)

    def _fit_ballistic(self):
        """
        Fit z(t) = z0 + vz*t - 0.5*g*t^2 and x,y linear in t to the sample
        buffer. Return the filtered current position and velocity.

        Returns
        -------
        (pos_now, vel_now) or None
        """
        ts = np.array([s[0] for s in self.samples], dtype=np.float64)
        pts = np.vstack([s[1] for s in self.samples]).astype(np.float64)

        t0 = ts[-1]
        dt = ts - t0

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

        return pos_now, vel_now


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
