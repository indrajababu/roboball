"""
Trajectory predictor — Node 2 in the Roboball stack.

Subscribes to `/ball_pose` (PointStamped), keeps a rolling buffer of recent
samples, fits a ballistic model (x,y linear; z with gravity), and publishes:

  /ball_state    (roboball_msgs/BallState)   — filtered current state
  /strike_target (roboball_msgs/StrikeTarget) — where/when to hit it

This file is a **stub**. The buffer + ROS plumbing is wired up; the actual fit
is TODO — see `_fit_ballistic` below.
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

        self.buffer_size = int(self.declare_parameter('buffer_size', 20).value)
        self.min_samples = int(self.declare_parameter('min_samples', 8).value)
        self.strike_height = float(self.declare_parameter('strike_height', 0.5).value)

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
        target.header = msg.header
        target.impact_pose.position.x = float(impact_xyz[0])
        target.impact_pose.position.y = float(impact_xyz[1])
        target.impact_pose.position.z = float(impact_xyz[2])
        # Paddle pointing up (flip of lab5/7 "gripper down" convention):
        # [qx,qy,qz,qw] = [1,0,0,0] rotates 180° about X, so the tool z-axis
        # points up in base_link. Adjust once the paddle CAD is finalized.
        target.impact_pose.orientation.x = 1.0
        target.impact_pose.orientation.y = 0.0
        target.impact_pose.orientation.z = 0.0
        target.impact_pose.orientation.w = 0.0

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
        # TODO(step 6): implement the least-squares fit and the apex/height
        # crossing solve. Pseudocode:
        #   ts  = np.array([s[0] for s in self.samples])
        #   xs  = np.vstack([s[1] for s in self.samples])
        #   t0  = ts[-1]
        #   dt  = ts - t0
        #   # Fit x(t) = x0 + vx*dt  and  y(t) = y0 + vy*dt  via np.polyfit deg 1
        #   # Fit z(t) = z0 + vz*dt - 0.5*g*dt^2  by subtracting the gravity
        #   # term from zs and doing a deg-1 fit on the residual
        #   # Solve for dt* where z(dt*) == strike_height on the descending branch
        #   # pos_now  = [x0, y0, z0]
        #   # vel_now  = [vx, vy, vz]
        #   # impact_xyz = [x0 + vx*dt*, y0 + vy*dt*, strike_height]
        #   # vel_at_impact = [vx, vy, vz - g*dt*]
        return None


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPredictor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
