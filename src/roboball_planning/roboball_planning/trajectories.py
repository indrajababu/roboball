#!/usr/bin/env python3
"""
Trajectory classes for Lab 7
Defines Linear and Circular trajectories for the UR7e end effector

Author: EECS 106B Course Staff, Spring 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse


class Trajectory:
    """Base trajectory class"""

    def __init__(self, total_time):
        """
        Parameters
        ----------
        total_time : float
            Duration of the trajectory in seconds
        """

        self.total_time = total_time

    def target_pose(self, time):
        """
        Returns desired end-effector pose at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            7D vector [x, y, z, qx, qy, qz, qw] where:
            - (x, y, z) is position in meters
            - (qx, qy, qz, qw) is orientation as quaternion
            - Gripper pointing down corresponds to quaternion [0, 1, 0, 0]
        """

        raise NotImplementedError

    def target_velocity(self, time):
        """
        Returns desired end-effector velocity at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            6D twist [vx, vy, vz, wx, wy, wz] where:
            - (vx, vy, vz) is linear velocity in m/s
            - (wx, wy, wz) is angular velocity in rad/s
        """

        raise NotImplementedError

    def display_trajectory(self, num_waypoints=100, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.

        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animation : bool
            if True, saves a gif of the animated trajectory
        """

        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])

        fig = plt.figure(figsize=(12, 10))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        # Row 1: time series (position and translational velocity vs time)
        ax_ts_pos = fig.add_subplot(2, 2, 1)
        ax_ts_pos.plot(times, target_positions[:, 0], label='x')
        ax_ts_pos.plot(times, target_positions[:, 1], label='y')
        ax_ts_pos.plot(times, target_positions[:, 2], label='z')
        ax_ts_pos.set_ylabel('Position')
        ax_ts_pos.set_title('Target Position')
        ax_ts_pos.legend()
        ax_ts_pos.grid(True)

        ax_ts_vel = fig.add_subplot(2, 2, 2, sharex=ax_ts_pos)
        ax_ts_vel.plot(times, target_velocities[:, 0], label='vx')
        ax_ts_vel.plot(times, target_velocities[:, 1], label='vy')
        ax_ts_vel.plot(times, target_velocities[:, 2], label='vz')
        ax_ts_vel.set_xlabel('Time')
        ax_ts_vel.set_ylabel('Velocity')
        ax_ts_vel.set_title('Target Velocity')
        ax_ts_vel.legend()
        ax_ts_vel.grid(True)

        # Row 2: 3D evolution of position and body-frame velocity
        ax0 = fig.add_subplot(2, 2, 3, projection='3d')
        pos_padding = [[-0.1, 0.1],
                       [-0.1, 0.1],
                       [-0.1, 0.1]]
        ax0.set_xlim3d([min(target_positions[:, 0]) + pos_padding[0][0],
                        max(target_positions[:, 0]) + pos_padding[0][1]])
        ax0.set_ylim3d([min(target_positions[:, 1]) + pos_padding[1][0],
                        max(target_positions[:, 1]) + pos_padding[1][1]])
        ax0.set_zlim3d([min(target_positions[:, 2]) + pos_padding[2][0],
                        max(target_positions[:, 2]) + pos_padding[2][1]])
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        ax0.set_title(f'{trajectory_name} evolution of end-effector\'s position.')
        line0 = ax0.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], c=colormap, s=2)

        ax1 = fig.add_subplot(2, 2, 4, projection='3d')
        vel_padding = [[-0.1, 0.1],
                       [-0.1, 0.1],
                       [-0.1, 0.1]]
        ax1.set_xlim3d([min(target_velocities[:, 0]) + vel_padding[0][0],
                        max(target_velocities[:, 0]) + vel_padding[0][1]])
        ax1.set_ylim3d([min(target_velocities[:, 1]) + vel_padding[1][0],
                        max(target_velocities[:, 1]) + vel_padding[1][1]])
        ax1.set_zlim3d([min(target_velocities[:, 2]) + vel_padding[2][0],
                        max(target_velocities[:, 2]) + vel_padding[2][1]])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'{trajectory_name} evolution of end-effector\'s translational body-frame velocity.')
        line1 = ax1.scatter(target_velocities[:, 0], target_velocities[:, 1], target_velocities[:, 2], c=colormap, s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints,
                                                          fargs=([line0, line1],),
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))),
                                                          blit=False)
        fig.tight_layout()
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)


class LinearTrajectory(Trajectory):
    """
    Straight line trajectory from start to goal position.
    Uses trapezoidal velocity profile (constant acceleration, constant velocity, constant deceleration).
    """

    def __init__(self, start_position, goal_position, total_time):
        """
        Parameters
        ----------
        start_position : np.ndarray
            3D starting position [x, y, z] in meters
        goal_position : np.ndarray
            3D goal position [x, y, z] in meters
        total_time : float
            Total duration of trajectory in seconds
        """

        super().__init__(total_time)

        self.start_position = np.array(start_position)
        self.goal_position = np.array(goal_position)

        # Calculate trajectory parameters
        self.direction = self.goal_position - self.start_position
        self.distance = np.linalg.norm(self.direction)

        if self.distance > 0:
            self.unit_direction = self.direction / self.distance
        else:
            self.unit_direction = np.zeros(3)

        # Trapezoidal velocity profile parameters
        # Accelerate for first half, decelerate for second half
        self.t_half = total_time / 2.0
        self.v_max = 2.0 * self.distance / total_time  # Peak velocity
        self.acceleration = self.v_max / self.t_half

        # Gripper pointing down quaternion
        self.orientation = np.array([0.0, 1.0, 0.0, 0.0])  # [qx, qy, qz, qw]

    def target_pose(self, time):
        """
        Returns desired end-effector pose at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            7D vector [x, y, z, qx, qy, qz, qw] where:
            - (x, y, z) is position in meters
            - (qx, qy, qz, qw) is orientation as quaternion
            - Gripper pointing down corresponds to quaternion [0, 1, 0, 0]
        """

        # Clamp time
        t = np.clip(time, 0, self.total_time)

        if t <= self.t_half:
            # Accelerating phase: s = 0.5 * a * t^2 along the unit direction
            s = 0.5 * self.acceleration * t ** 2
            pos = self.start_position + s * self.unit_direction
        else:
            # Decelerating phase: mirror of acceleration using time-to-end
            # t_remaining counts backward from total_time, so s_remaining = 0.5*a*t_rem^2
            t_remaining = self.total_time - t
            s_remaining = 0.5 * self.acceleration * t_remaining ** 2
            pos = self.goal_position - s_remaining * self.unit_direction

        # Combine position and orientation
        return np.concatenate([pos, self.orientation])

    def target_velocity(self, time):
        """
        Returns desired end-effector velocity at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            6D twist [vx, vy, vz, wx, wy, wz] where:
            - (vx, vy, vz) is linear velocity in m/s
            - (wx, wy, wz) is angular velocity in rad/s
        """

        # Clamp time
        t = np.clip(time, 0, self.total_time)

        # Calculate speed based on trapezoidal profile
        if t <= self.t_half:
            # Accelerating: v = a * t
            speed = self.acceleration * t
        else:
            # Decelerating: mirror of acceleration using time-to-end
            speed = self.acceleration * (self.total_time - t)

        vel = speed * self.unit_direction

        # Combine position and orientation
        return np.concatenate([vel, np.zeros(3)])


class CircularTrajectory(Trajectory):
    """
    Circular trajectory around a center point in a horizontal plane.
    Uses angular trapezoidal velocity profile.
    """

    def __init__(self, center_position, radius, total_time):
        """
        Parameters
        ----------
        center_position : np.ndarray
            3D center position [x, y, z] in meters
        radius : float
            Radius of circle in meters
        total_time : float
            Total duration of trajectory in seconds
        """
        super().__init__(total_time)

        self.center_position = np.array(center_position)
        self.radius = radius

        # Angular trapezoidal profile (complete circle = 2π radians)
        self.total_angle = 2 * np.pi
        self.t_half = total_time / 2.0
        self.angular_v_max = 2.0 * self.total_angle / total_time
        self.angular_acceleration = self.angular_v_max / self.t_half

        # Gripper pointing down quaternion
        self.orientation = np.array([0.0, 1.0, 0.0, 0.0])

    def target_pose(self, time):
        """
        Returns desired end-effector pose at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            7D vector [x, y, z, qx, qy, qz, qw] where:
            - (x, y, z) is position in meters
            - (qx, qy, qz, qw) is orientation as quaternion
            - Gripper pointing down corresponds to quaternion [0, 1, 0, 0]
        """

        # Clamp time
        t = np.clip(time, 0, self.total_time)

        if t <= self.t_half:
            # Accelerating phase: theta = 0.5 * alpha * t^2
            theta = 0.5 * self.angular_acceleration * t ** 2
        else:
            # Decelerating phase: remaining angle from 2*pi shrinks as 0.5*alpha*t_rem^2
            t_remaining = self.total_time - t
            theta_remaining = 0.5 * self.angular_acceleration * t_remaining ** 2
            theta = self.total_angle - theta_remaining

        pos = self.center_position + self.radius * np.array([np.cos(theta), np.sin(theta), 0])

        # Combine position and orientation
        return np.concatenate([pos, self.orientation])

    def target_velocity(self, time):
        """
        Returns desired end-effector velocity at time t

        Parameters
        ----------
        time : float
            Time from start of trajectory

        Returns
        -------
        np.ndarray
            6D twist [vx, vy, vz, wx, wy, wz] where:
            - (vx, vy, vz) is linear velocity in m/s
            - (wx, wy, wz) is angular velocity in rad/s
        """

        # Clamp time
        t = np.clip(time, 0, self.total_time)

        if t <= self.t_half:
            # Accelerating: theta = 0.5*alpha*t^2, omega = alpha*t
            theta = 0.5 * self.angular_acceleration * t ** 2
            theta_dot = self.angular_acceleration * t
        else:
            # Decelerating: mirror of acceleration using time-to-end
            t_remaining = self.total_time - t
            theta_remaining = 0.5 * self.angular_acceleration * t_remaining ** 2
            theta = self.total_angle - theta_remaining
            theta_dot = self.angular_acceleration * t_remaining

        vel = self.radius * theta_dot * np.array([-np.sin(theta), np.cos(theta), 0])

        # Combine position and orientation
        return np.concatenate([vel, np.zeros(3)])


def define_trajectories(args):
    """Define each type of trajectory with the appropriate parameters."""

    trajectory = None
    if args.task == 'line':
        # Example linear trajectory
        start = np.array([0.3, 0.2, 0.3])
        goal = np.array([0.5, 0.4, 0.4])
        total_time = 5.0
        trajectory = LinearTrajectory(start, goal, total_time)
    elif args.task == 'circle':
        # Example circular trajectory
        center = np.array([0.4, 0.3, 0.3])
        radius = 0.1
        total_time = 8.0
        trajectory = CircularTrajectory(center, radius, total_time)

    return trajectory


if __name__ == '__main__':
    """
    Run this file to visualize plots of your paths. Note: the provided function
    only visualizes the end effector position, not its orientation. Use the
    animate function to visualize the full trajectory in a 3D plot.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, default='line', help='Options: line, circle.  Default: line')
    parser.add_argument('--animate', action='store_true', help='If you set this flag, the animated trajectory will be shown.')
    args = parser.parse_args()

    trajectory = define_trajectories(args)

    if trajectory:
        trajectory.display_trajectory(show_animation=args.animate)
