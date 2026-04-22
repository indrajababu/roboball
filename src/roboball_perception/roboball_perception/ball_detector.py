"""
Ball detector — seeded from lab5/perception/process_pointcloud.py.

Pipeline:
  1. Subscribe to RealSense depth point cloud.
  2. Transform the cloud into `target_frame` (default `base_link`).
  3. Filter by a workspace bounding box AND by color (via an HSV mask computed
     on the synchronized RGB image — TODO).
  4. Publish the centroid of the surviving points as `/ball_pose` (PointStamped)
     and the filtered cloud as `/filtered_points` for RViz debug.

Starting state: the color-mask branch is stubbed out. The node currently
behaves exactly like lab5 (pure geometric filter) so you can verify the TF
calibration + workspace box work before wiring in HSV.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')

        self.target_frame = self.declare_parameter('target_frame', 'base_link').value
        self.source_frame = self.declare_parameter('source_frame', 'camera_depth_optical_frame').value

        # Workspace bounding box (in target_frame). Tune in RViz with
        # `ros2 param set /ball_detector <name> <value>`.
        self.min_x = float(self.declare_parameter('min_x', -0.80).value)
        self.max_x = float(self.declare_parameter('max_x',  0.80).value)
        self.min_y = float(self.declare_parameter('min_y', -0.80).value)
        self.max_y = float(self.declare_parameter('max_y',  0.80).value)
        self.min_z = float(self.declare_parameter('min_z',  0.00).value)
        self.max_z = float(self.declare_parameter('max_z',  2.00).value)

        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/camera/depth/color/points',
            self.pointcloud_callback,
            10,
        )

        self.ball_pose_pub = self.create_publisher(PointStamped, '/ball_pose', 1)
        self.filtered_points_pub = self.create_publisher(PointCloud2, '/filtered_points', 1)

        self.get_logger().info(
            f'Ball detector up. Target frame: {self.target_frame}. '
            f'Workspace box x=[{self.min_x},{self.max_x}], '
            f'y=[{self.min_y},{self.max_y}], z=[{self.min_z},{self.max_z}]'
        )

    def pointcloud_callback(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, Time())
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not transform {self.source_frame} to {self.target_frame}: {ex}',
                throttle_duration_sec=2.0,
            )
            return

        transformed_cloud = do_transform_cloud(msg, tf)

        raw_points = pc2.read_points(
            transformed_cloud,
            field_names=('x', 'y', 'z'),
            skip_nans=True,
        )
        points = np.column_stack(
            (raw_points['x'], raw_points['y'], raw_points['z'])
        ).astype(np.float32, copy=False)

        # Workspace box filter.
        mask = (
            (points[:, 0] >= self.min_x) & (points[:, 0] <= self.max_x)
            & (points[:, 1] >= self.min_y) & (points[:, 1] <= self.max_y)
            & (points[:, 2] >= self.min_z) & (points[:, 2] <= self.max_z)
        )

        # TODO(hsv): add a color mask here. Subscribe to
        # `/camera/camera/color/image_raw` + `camera_info`, convert to HSV,
        # threshold for the beach-ball color(s), then project matching pixels
        # into the depth frame and intersect with `mask`.
        filtered = points[mask]

        if filtered.size == 0:
            self.get_logger().warn(
                'No points inside workspace box — check TF and box params.',
                throttle_duration_sec=2.0,
            )
            return

        filtered_cloud = pc2.create_cloud_xyz32(
            transformed_cloud.header,
            filtered.tolist(),
        )
        self.filtered_points_pub.publish(filtered_cloud)

        centroid = np.mean(filtered, axis=0)

        ball_pose = PointStamped()
        ball_pose.header.stamp = self.get_clock().now().to_msg()
        ball_pose.header.frame_id = self.target_frame
        ball_pose.point.x = float(centroid[0])
        ball_pose.point.y = float(centroid[1])
        ball_pose.point.z = float(centroid[2])
        self.ball_pose_pub.publish(ball_pose)

    def _on_parameter_update(self, params):
        bounds = {
            'min_x': self.min_x, 'max_x': self.max_x,
            'min_y': self.min_y, 'max_y': self.max_y,
            'min_z': self.min_z, 'max_z': self.max_z,
        }
        for p in params:
            if p.name in bounds and p.type_ == Parameter.Type.DOUBLE:
                bounds[p.name] = float(p.value)

        if bounds['min_x'] > bounds['max_x'] \
                or bounds['min_y'] > bounds['max_y'] \
                or bounds['min_z'] > bounds['max_z']:
            return SetParametersResult(successful=False, reason='min_* must be <= max_*')

        self.min_x, self.max_x = bounds['min_x'], bounds['max_x']
        self.min_y, self.max_y = bounds['min_y'], bounds['max_y']
        self.min_z, self.max_z = bounds['min_z'], bounds['max_z']
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = BallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
