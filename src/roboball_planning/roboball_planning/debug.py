# debug.py

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from rclpy.time import Time
from tf2_ros import TransformException

from geometry_msgs.msg import Point, Pose, PoseArray
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def rgba(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


def quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    """
    Convert quaternion x,y,z,w to a 3x3 rotation matrix.
    """
    x, y, z, w = map(float, (x, y, z, w))

    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)

    s = 2.0 / n

    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s

    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_multiply(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    """
    Hamilton product.

    Both quaternions are [x, y, z, w].

    If q_tf represents target<-base and q_pose is base<-pose,
    then q_tf * q_pose gives target<-pose.
    """
    x1, y1, z1, w1 = map(float, q1)
    x2, y2, z2, w2 = map(float, q2)

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def normalize_quat(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


class StrikeDebugVisualizer:
    """
    Publishes RViz debug visuals for strike planning / IK waypoints.

    Expected inputs:
      - cart_points are expressed in base_frame
      - waypoint_quats are orientations expressed in base_frame
      - debug_marker_frame is the RViz frame you want to publish into,
        often "world" or "base_link"
    """

    def __init__(
        self,
        *,
        node,
        marker_pub,
        pose_pub,
        tf_buffer,
        base_frame: str = "base_link",
        debug_marker_frame: str = "base_link",
    ):
        self.node = node
        self.marker_pub = marker_pub
        self.pose_pub = pose_pub
        self.tf_buffer = tf_buffer
        self.base_frame = base_frame
        self.debug_marker_frame = debug_marker_frame

    def publish(
        self,
        cart_points: Sequence[Sequence[float]],
        waypoint_quats: Sequence[Sequence[float]],
        waypoint_success: Sequence[bool],
        failed_index: Optional[int],
    ) -> None:
        if self.marker_pub is None or self.pose_pub is None:
            return

        if not cart_points:
            return

        target_frame = self.debug_marker_frame

        pts_in_target = [
            np.asarray([float(p[0]), float(p[1]), float(p[2])], dtype=np.float64)
            for p in cart_points
        ]

        quats_in_target = [
            normalize_quat(q) for q in waypoint_quats
        ]

        if target_frame != self.base_frame:
            try:
                tf_t = self.tf_buffer.lookup_transform(
                    target_frame,
                    self.base_frame,
                    Time(),
                )

                q_tf_msg = tf_t.transform.rotation
                t_tf_msg = tf_t.transform.translation

                q_target_from_base = normalize_quat(
                    [q_tf_msg.x, q_tf_msg.y, q_tf_msg.z, q_tf_msg.w]
                )
                rot_target_from_base = quat_to_rot(*q_target_from_base)

                trans_target_from_base = np.array(
                    [t_tf_msg.x, t_tf_msg.y, t_tf_msg.z],
                    dtype=np.float64,
                )

                pts_in_target = [
                    rot_target_from_base @ p + trans_target_from_base
                    for p in pts_in_target
                ]

                quats_in_target = [
                    normalize_quat(quat_multiply(q_target_from_base, q_base))
                    for q_base in quats_in_target
                ]

            except TransformException:
                self.node.get_logger().warn(
                    f"Debug markers: TF lookup {self.base_frame}->{target_frame} "
                    f"failed; publishing in {self.base_frame} instead.",
                    throttle_duration_sec=5.0,
                )
                target_frame = self.base_frame

        stamp = self.node.get_clock().now().to_msg()

        pose_array = self._build_pose_array(
            pts_in_target,
            quats_in_target,
            target_frame,
            stamp,
        )
        self.pose_pub.publish(pose_array)

        markers = self._build_marker_array(
            pts_in_target,
            waypoint_success,
            failed_index,
            target_frame,
            stamp,
        )
        self.marker_pub.publish(markers)

    def _build_pose_array(
        self,
        pts: Sequence[np.ndarray],
        quats: Sequence[np.ndarray],
        frame_id: str,
        stamp,
    ) -> PoseArray:
        pose_array = PoseArray()
        pose_array.header.frame_id = frame_id
        pose_array.header.stamp = stamp

        for pt, quat in zip(pts, quats):
            p = Pose()

            p.position.x = float(pt[0])
            p.position.y = float(pt[1])
            p.position.z = float(pt[2])

            p.orientation.x = float(quat[0])
            p.orientation.y = float(quat[1])
            p.orientation.z = float(quat[2])
            p.orientation.w = float(quat[3])

            pose_array.poses.append(p)

        return pose_array

    def _build_marker_array(
        self,
        pts: Sequence[np.ndarray],
        waypoint_success: Sequence[bool],
        failed_index: Optional[int],
        frame_id: str,
        stamp,
    ) -> MarkerArray:
        markers = MarkerArray()

        clear = Marker()
        clear.header.frame_id = frame_id
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        path = Marker()
        path.header.frame_id = frame_id
        path.header.stamp = stamp
        path.ns = "strike_path"
        path.id = 1
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.scale.x = 0.008
        path.color = rgba(0.05, 0.75, 1.00, 1.00)

        for pt in pts:
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = float(pt[2])
            path.points.append(p)

        markers.markers.append(path)

        waypoints = Marker()
        waypoints.header.frame_id = frame_id
        waypoints.header.stamp = stamp
        waypoints.ns = "strike_waypoints"
        waypoints.id = 2
        waypoints.type = Marker.SPHERE_LIST
        waypoints.action = Marker.ADD
        waypoints.scale.x = 0.025
        waypoints.scale.y = 0.025
        waypoints.scale.z = 0.025

        for idx, pt in enumerate(pts):
            p = Point()
            p.x = float(pt[0])
            p.y = float(pt[1])
            p.z = float(pt[2])
            waypoints.points.append(p)

            color = rgba(0.20, 0.85, 0.20, 1.00)

            if failed_index is not None and idx == failed_index:
                color = rgba(0.95, 0.10, 0.10, 1.00)
            elif idx < len(waypoint_success) and not waypoint_success[idx]:
                color = rgba(0.95, 0.10, 0.10, 1.00)
            elif idx == len(pts) - 1:
                color = rgba(1.00, 0.70, 0.10, 1.00)

            waypoints.colors.append(color)

        markers.markers.append(waypoints)

        if failed_index is not None and 0 <= failed_index < len(pts):
            failed_pt = pts[failed_index]

            text = Marker()
            text.header.frame_id = frame_id
            text.header.stamp = stamp
            text.ns = "strike_failure"
            text.id = 3
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.scale.z = 0.05
            text.color = rgba(1.0, 0.2, 0.2, 1.0)

            text.pose.position.x = float(failed_pt[0])
            text.pose.position.y = float(failed_pt[1])
            text.pose.position.z = float(failed_pt[2] + 0.06)
            text.pose.orientation.w = 1.0

            text.text = f"IK FAIL wp {failed_index + 1}/{len(pts)}"

            markers.markers.append(text)

        return markers