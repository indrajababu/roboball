from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([

        # Start Gazebo with your world
        ExecuteProcess(
            cmd=[
                'gz', 'sim',
                'src/simulation/worlds/world.sdf'
            ],
            output='screen'
        ),
    ])