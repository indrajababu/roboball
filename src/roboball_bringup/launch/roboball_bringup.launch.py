"""
Top-level Roboball bring-up.

Launches, in order:
  - RealSense camera (rgb + depth + pointcloud)
  - MoveIt for UR7e (provides /compute_ik, /plan_kinematic_path)
  - ArUco + static TF broadcaster (camera -> ar_marker_N -> base_link)
  - Trajectory validator (safety layer in front of the UR trajectory controller)
  - Ball detector (publishes /ball_pose), default detector: HSV
  - Trajectory predictor (publishes /strike_target)
  - Strike planner (gated off by default — see start_strike_planner)

Seeded from lab5/planning/launch/lab5_bringup.launch.py.

Usage:
  ros2 launch roboball_bringup roboball_bringup.launch.py
Optional args:
  launch_rviz:=false                 — skip MoveIt's RViz
  ur_type:=ur7e                      — robot type forwarded to ur_moveit_config
  detector:=hsv|yolo                 — ball_detector backend (default hsv)
  strike_height:=<meters>            — strike-plane Z in base_link (default 0.60)
  start_strike_planner:=true|false   — auto-start the strike planner (default false).
                                       Off so go_home keeps using
                                       scaled_joint_trajectory_controller; start
                                       the planner manually once parked.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    ur_type = LaunchConfiguration('ur_type', default='ur7e')
    launch_rviz = LaunchConfiguration('launch_rviz', default='true')
    marker_number = LaunchConfiguration('marker_number', default='-1')
    detector = LaunchConfiguration('detector', default='hsv')
    strike_height = LaunchConfiguration('strike_height', default='0.60')
    start_strike_planner = LaunchConfiguration('start_strike_planner', default='false')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py',
            )
        ),
        launch_arguments={
            'serial_no': "'843112070166'",
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
            'rgb_camera.color_profile': '640x480x30',
            'depth_module.depth_profile': '640x480x30',
        }.items(),
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_moveit_config'),
                'launch',
                'ur_moveit.launch.py',
            )
        ),
        launch_arguments={
            'ur_type': ur_type,
            'launch_rviz': launch_rviz,
        }.items(),
    )

    static_tf_node = Node(
        package='roboball_bringup',
        executable='static_camera_tf',
        name='static_camera_tf',
        output='screen',
        parameters=[{
            'marker_number': marker_number,
        }],
    )

    aruco_node = Node(
        package='ros2_aruco',
        executable='aruco_node',
        name='aruco_node',
        output='screen',
        parameters=[{
            'marker_size': 0.15,
            'aruco_dictionary_id': 'DICT_5X5_250',
            'image_topic': '/camera/camera/color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'camera_frame': 'camera_color_optical_frame',
        }],
    )

    validator_node = Node(
        package='roboball_planning',
        executable='validate_trajectory',
        name='trajectory_validator',
        output='screen',
    )

    ball_detector_node = Node(
        package='roboball_perception',
        executable='ball_detector',
        name='ball_detector',
        output='screen',
        parameters=[{
            'detector': detector,
            'cloud_stride': 2,
        }],
    )

    predictor_node = Node(
        package='roboball_planning',
        executable='trajectory_predictor',
        name='trajectory_predictor',
        output='screen',
        parameters=[{
            'strike_height': ParameterValue(strike_height, value_type=float),
            'min_samples': 4,
            'buffer_size': 12,
        }],
    )

    strike_planner_node = Node(
        package='roboball_planning',
        executable='strike_planner',
        name='strike_planner',
        output='screen',
        parameters=[{
            'num_waypoints': 5,
            'ik_budget': 0.08,
            'ik_timeout': 0.15,
        }],
        condition=IfCondition(start_strike_planner),
    )

    shutdown_on_any_exit = RegisterEventHandler(
        OnProcessExit(
            on_exit=[EmitEvent(event=Shutdown(reason='A launched process exited'))]
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument('ur_type', default_value='ur7e'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        DeclareLaunchArgument('marker_number', default_value='-1'),
        DeclareLaunchArgument('detector', default_value='hsv'),
        DeclareLaunchArgument('strike_height', default_value='0.60'),
        DeclareLaunchArgument('start_strike_planner', default_value='false'),
        realsense_launch,
        moveit_launch,
        aruco_node,
        static_tf_node,
        validator_node,
        ball_detector_node,
        predictor_node,
        strike_planner_node,
        shutdown_on_any_exit,
    ])
