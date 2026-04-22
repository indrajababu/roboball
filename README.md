# Roboball

UR7e juggles a beach ball with a paddle end-effector. EE106A Spring 2026 final project.

## Packages

- **roboball_msgs** — `BallState`, `StrikeTarget` custom messages.
- **roboball_perception** — `ball_detector` node: RealSense depth → `/ball_pose`.
- **roboball_planning** — IK wrapper, trajectory generators, PID/trajectory controllers, ballistic predictor, strike planner. Seeded from labs 3, 5, 7.
- **roboball_bringup** — calibration TF, go-to-home utility, top-level launch.

## Build

All ROS2 work happens inside the class `ros2` distrobox container.

```bash
distrobox enter ros2
cd ~/ros_workspaces/roboball
colcon build --symlink-install
source install/setup.bash
```

## Bring-up sequence (same pattern as lab7)

```bash
# Shell 1 — enable comms with the physical UR7e
ros2 run ur7e_utils enable_comms

# Shell 2 — launch the full stack
ros2 launch roboball_bringup roboball_bringup.launch.py
# Optional: launch_rviz:=false  ur_type:=ur7e
```

Once the stack is up, park the arm at the safe home before doing anything else:

```bash
ros2 run roboball_bringup go_home
```

## Debug / unit-testing

Visualize a trajectory without any ROS running (matplotlib only):

```bash
cd src/roboball_planning/roboball_planning
python3 trajectories.py --task line
python3 trajectories.py --task circle --animate
```

Individual nodes for isolated testing:

```bash
ros2 run roboball_bringup static_camera_tf
ros2 run roboball_perception ball_detector
ros2 run roboball_planning trajectory_predictor
ros2 run roboball_planning strike_planner
ros2 run roboball_planning validate_trajectory
ros2 run roboball_planning ik        # one-shot IK smoke test from lab5
```

## Status (Day 1 — scaffold only)

The skeleton is in. These are the **stubs with TODOs** in the order you'll complete them, tracking the 7-step plan:

| Step | File | What's TODO |
|------|------|-------------|
| 3 | `roboball_bringup/roboball_bringup/static_camera_tf.py` | Replace placeholder `G` with the measured `base_link → camera_color_optical_frame`. |
| 4 | `roboball_perception/roboball_perception/ball_detector.py` | Add HSV color mask on `/camera/camera/color/image_raw`. |
| 6 | `roboball_planning/roboball_planning/trajectory_predictor.py` | Implement `_fit_ballistic` — least-squares fit + height-crossing solve. |
| 7 | `roboball_planning/roboball_planning/strike_planner.py` | Swap action-based execution for PID velocity loop using `LinearTrajectory` + `PIDJointVelocityController`. |

Full plan: `~/.claude/plans/look-through-the-labs-swift-teacup.md`.
