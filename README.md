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

Here is the finalized `README.md` for your project. This document integrates the M4-specific fixes, the unified workspace structure, and the troubleshooting steps we discovered during your setup today.

```markdown
# Roboball Simulation & Development Environment

This environment is designed for developing ROS 2 Humble applications on an M4 Mac and transitioning them seamlessly to the UC Berkeley CS lab machines. It utilizes a Docker-based Ubuntu 22.04 environment with a web-accessible desktop (noVNC).

## 🛠 Prerequisites
* **Docker Desktop**: Installed and running.
* **M4 Apple Silicon**: High-performance optimization for ARM64 architecture.

---

## 🚀 1. First-Time Setup

### Step A: Folder Structure
To ensure code transfers seamlessly to the lab, maintain a unified ROS 2 workspace structure on your Mac:
```text
ROBOBALL/
├── Dockerfile           # M4 web-env configuration
├── .gitignore
└── src/                 # All ROS 2 packages live here (Unified)
    ├── roboball_bringup/    # Launch files
    ├── roboball_msgs/       # Custom message definitions
    ├── roboball_perception/ # Detection logic
    ├── roboball_planning/   # Control/Movement logic
    ├── simulation/          # Gazebo worlds and sim-specific configs
    └── ros2_aruco/          # Third-party vision tools
```

### Step B: Build the Image
From the root `ROBOBALL/` directory on your Mac, build the image:
```bash
docker build --platform linux/arm64 -t ros2_web_env .
```

---

## 🏃 2. Running the Environment

### Manual Start
Run the following to start the container and link your code. This uses a volume mount, so edits on your Mac reflect instantly in the simulator.
```bash
docker run -it --rm \
    --name roboball_sim \
    -p 8080:8080 \
    --volume="$PWD/src:/ros2_ws/src" \
    --shm-size=1g \
    ros2_web_env
```

### Accessing the Desktop
1. Open your browser to: `http://localhost:8080/vnc.html`
2. Click **Connect**.
3. You now have a full Ubuntu desktop. The terminal is the `$_` icon in the bottom dock.

---

## 🏗 3. Development Workflow (Inside VNC)

### The "Golden Build" Sequence
Always build in this order to ensure your custom messages in `roboball_msgs` are correctly linked to your planning and perception nodes:
```bash
cd /ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### Launching Simulation
To start Gazebo (Ignition Fortress) and the ROS-GZ Bridge simultaneously:
```bash
ros2 launch roboball_bringup sim.launch.py
```

### Manual Gazebo Launch (Troubleshooting)
If the launch file is not used, run Gazebo directly using the Ignition command:
```bash
ign gazebo ~/ros2_ws/src/simulation/worlds/world.sdf
```

---

## 🎹 Hotkey Cheat Sheet (Mac to VNC)

| Action | Key Combination |
| :--- | :--- |
| **Paste into Terminal** | `Ctrl + Shift + V` |
| **Copy from Terminal** | `Ctrl + Shift + C` |
| **Interrupt/Stop Node** | `Ctrl + C` |
| **Delete Line** | `Ctrl + U` |
| **Bridge Mac Clipboard** | Use the grey arrow on the left of the browser window |

---

## ⚡ 4. Productivity Shortcuts (Mac Setup)
Add this alias to your Mac's `~/.zshrc` to launch the environment with one command:
```bash
# Start the Roboball Simulation Environment
alias robo-start='docker run -it --rm --name roboball_sim -p 8080:8080 --volume="$PWD/src:/ros2_ws/src" --shm-size=1g ros2_web_env'
```
*After adding, run `source ~/.zshrc`.*
```