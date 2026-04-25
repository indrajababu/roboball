# Testing `strike_planner.py`

How to verify the PID-velocity strike planner works, both off the robot
(pure Python → fake hardware) and on the physical UR7e.

The node under test is [src/roboball_planning/roboball_planning/strike_planner.py](src/roboball_planning/roboball_planning/strike_planner.py).
It activates `forward_velocity_controller` on startup and reverts to
`scaled_joint_trajectory_controller` on shutdown.

---

## Off the robot

### 1. Pure-Python checks (no ROS)

Sanity-check the math helpers and trajectory plotting:

```bash
cd src/roboball_planning/roboball_planning

# LinearTrajectory still plots a clean trapezoidal profile
python3 trajectories.py --task line --animate

# Finite-difference helper: synthetic linear-in-time joint trajectory
# should produce constant unit velocities
python3 -c "
import numpy as np
from strike_planner import _finite_diff
times = np.linspace(0, 1, 10)
positions = np.outer(times, np.ones(6))
print(_finite_diff(positions, times))   # expect rows of 1.0
"
```

### 2. Build check

In the class `ros2` distrobox container:

```bash
distrobox enter ros2
cd ~/ros_workspaces/roboball
colcon build --symlink-install --packages-select roboball_msgs roboball_planning
source install/setup.bash
```

Catches missing imports and message-generation issues.

### 3. Full sim with `use_fake_hardware`

UR's `ros2_control` stack supports a fake-hardware mode that fakes the
joint controller and `/joint_states` publisher. End-to-end pipeline runs
without a real arm. Three shells:

```bash
# Shell 1 — fake UR + ros2_control
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur7e robot_ip:=yyy.yyy.yyy.yyy \
    use_fake_hardware:=true \
    initial_joint_controller:=scaled_joint_trajectory_controller

# Shell 2 — MoveIt (provides /compute_ik)
ros2 launch ur_moveit_config ur_moveit.launch.py \
    ur_type:=ur7e use_fake_hardware:=true launch_rviz:=true

# Shell 3 — node under test
ros2 run roboball_planning strike_planner
```

Confirm the controller swap:

```bash
ros2 control list_controllers
# expected:
#   forward_velocity_controller         active
#   scaled_joint_trajectory_controller  inactive
```

If `forward_velocity_controller` is missing entirely, it isn't loaded by
the UR controllers YAML for fake hardware. Either add it there or skip
sim and use ursim / the real robot.

### 4. Hand-crafted strike target

With `strike_planner` running (sim or real), publish one target:

```bash
ros2 topic pub --once /strike_target roboball_msgs/StrikeTarget "{
  header: {frame_id: base_link},
  impact_pose: {
    position: {x: 0.4, y: 0.0, z: 0.6},
    orientation: {x: 1.0, y: 0.0, z: 0.0, w: 0.0}
  },
  time_to_impact: {sec: 2, nanosec: 0},
  ball_velocity_at_impact: {x: 0.0, y: 0.0, z: 0.0}
}"
```

Expected log lines from `strike_planner`:

```
Activated forward_velocity_controller, deactivated scaled_joint_trajectory_controller.
Strike armed: 10 waypoints over 2.00s from [...] to [0.4 0.0 0.6].
Strike complete (t=2.0Xs).
```

Inspect the live command stream and final EE position:

```bash
ros2 topic hz /forward_velocity_controller/commands     # ~10 Hz when busy, 0 otherwise
ros2 topic echo /forward_velocity_controller/commands --once
ros2 run tf2_ros tf2_echo base_link tool0               # should land near (0.4, 0.0, 0.6)
```

---

## On the physical UR7e

Same pipeline plus safety scaffolding. **Keep a finger on the e-stop
from step 4 onward.**

### 1. Pre-flight (no motion)

```bash
# Shell 1
distrobox enter ros2
ros2 run ur7e_utils enable_comms

# Shell 2
ros2 launch roboball_bringup roboball_bringup.launch.py
```

Verify `forward_velocity_controller` is loaded (active or inactive — the
strike planner will activate it):

```bash
ros2 control list_controllers
```

### 2. Park before activating velocity mode

```bash
ros2 run roboball_bringup go_home
```

`go_home` routes through [validate_trajectory.py](src/roboball_planning/roboball_planning/validate_trajectory.py)
so it's bound-checked. **Important:** the default
[roboball_bringup.launch.py](src/roboball_bringup/launch/roboball_bringup.launch.py)
starts `strike_planner` immediately, which switches to velocity control
on init. For the first robot test, comment out `strike_planner_node` in
the launch file, run `go_home`, then start the planner manually:

```bash
ros2 run roboball_planning strike_planner
```

### 3. Tiny, slow strike near home

Read the current paddle pose:

```bash
ros2 run tf2_ros tf2_echo base_link tool0
```

Publish a target with **small displacement (~5 cm)** and **long
`time_to_impact` (3–5 s)** so the arm crawls. If it lurches or moves the
wrong way, e-stop and check joint ordering and TF.

### 4. Scale up

Once the slow test is clean, in this order:

- [ ] Shrink `time_to_impact` toward 1 s
- [ ] Increase displacement to ~20 cm
- [ ] Watch `/joint_states` rate and PID command magnitudes
- [ ] Retune Kp/Ki/Kd in [strike_planner.py:55-57](src/roboball_planning/roboball_planning/strike_planner.py#L55-L57) if joints saturate or oscillate

### 5. Full pipeline (after README steps 4 & 6 are done)

With the predictor and ball detector working, drop the ball into the
workspace; `/strike_target` populates automatically; the paddle
intercepts.

### 6. Shutdown

Ctrl-C the strike planner. Confirm the controller reverted:

```bash
ros2 control list_controllers
# expected: scaled_joint_trajectory_controller   active
ros2 run roboball_bringup go_home   # should succeed
```

---

## Debug toolkit

```bash
ros2 topic hz   /forward_velocity_controller/commands   # ~10 Hz when busy
ros2 topic echo /forward_velocity_controller/commands
ros2 topic echo /strike_target
ros2 topic echo /joint_states --field position --once
ros2 control list_controllers
ros2 run tf2_tools view_frames                          # PDF of TF tree
ros2 node info /strike_planner
```

---

## Common failure modes

1. **Velocity controller not loaded.** `_switch_controller` warns but
   doesn't abort, so the node thinks it's running while commands go
   nowhere. Always confirm with `ros2 control list_controllers` after
   start.

2. **Joint name mismatch.** The IK service may return joints in a
   different order than `/joint_states`. The
   [`_reorder_positions`](src/roboball_planning/roboball_planning/strike_planner.py#L207-L209)
   and
   [`_current_joint_vector`](src/roboball_planning/roboball_planning/strike_planner.py#L212-L218)
   helpers handle this; if you see a `KeyError`, your robot exposes
   joint names that differ from the `JOINT_ORDER` constant.

3. **TF not yet populated.** First `/strike_target` arrives before TF
   tree is up → `TF lookup base_link->tool0 failed`. The node logs and
   skips; just retry.

4. **MoveIt IK build time.** 10 waypoints × ~50 ms each = ~0.5 s build
   per strike. The clock starts after the build completes, so flight
   time isn't lost, but the first PID tick may have a large initial
   position error. If logs show that, drop `NUM_WAYPOINTS` in
   [strike_planner.py](src/roboball_planning/roboball_planning/strike_planner.py#L43)
   from 10 to ~5.

5. **Strike planner crashed mid-strike.** The arm keeps its last
   commanded velocity (`forward_velocity_controller` does not auto-zero
   on disconnect). E-stop, then restart the planner so the
   shutdown-time controller revert + zero-velocity publish runs cleanly.
