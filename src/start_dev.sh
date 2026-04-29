#!/bin/bash

cd /ros2_ws
colcon build --symlink-install
source install/setup.bash

echo "Workspace ready!"
