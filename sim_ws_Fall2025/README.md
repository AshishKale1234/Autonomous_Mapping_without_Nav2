# Autonomous Mapping of a Completely Unknown Environment  
## TurtleBot3 (Without Nav2) — ROS 2

This repository contains a **self-contained ROS 2 simulation workspace** that implements  
**autonomous exploration and mapping of a completely unknown indoor environment using TurtleBot3**,  
**without using the Nav2 navigation stack**.

The robot autonomously explores the environment, avoids obstacles, and builds an occupancy grid map
using onboard sensing and SLAM, without keyboard teleoperation, predefined waypoints, or Nav2.

---

## Supported ROS 2 Distributions
- **ROS 2 Humble** (Ubuntu 22.04)
- **ROS 2 Jazzy** (Ubuntu 24.04)

The same workflow applies to both distributions.

---

## System Requirements
- Ubuntu 22.04 (Humble) **or** Ubuntu 24.04 (Jazzy)
- ROS 2 Desktop installation
- Gazebo
- TurtleBot3 packages
- `slam_toolbox` (used by the provided launch file)

---

## Environment Setup (Required)
Set the TurtleBot3 model before building or running:

```bash
export TURTLEBOT3_MODEL=waffle
```

(Optional) Add permanently:
```bash
echo "export TURTLEBOT3_MODEL=waffle" >> ~/.bashrc
```

---

## Build Instructions
After cloning or downloading this repository:

```bash
cd sim_ws_Fall2025
colcon build --symlink-install
source install/local_setup.bash
```

⚠️ You must source the workspace in every new terminal before running.

---

## Running Task 1 — Autonomous Mapping
Launch the simulation, SLAM, and autonomous exploration node:

```bash
ros2 launch turtlebot3_gazebo mapper.launch.py
```

### Behavior
- The robot spawns in a **completely unknown environment**
- SLAM builds an occupancy grid map
- The robot autonomously explores the space
- No Nav2, no keyboard teleoperation, no fixed goals

Mapping continues until exploration coverage is maximized.

---

## Saving the Map (slam_toolbox)
While the system is running, you can save the current map using the SLAM Toolbox service:

```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "name:
  data: 'map_name'"
```

This will generate `map_name.pgm` and `map_name.yaml` (location depends on slam_toolbox configuration).
If you want to follow the course naming convention, use `map` or rename the outputs to:

```
map.pgm
map.yaml
```

---

## Output Map Files (Project Convention)
This workspace includes a `maps/` folder typically used for storing the final map:

```
sim_ws_Fall2025/src/turtlebot3_gazebo/maps/
 ├── map.pgm
 └── map.yaml
```

These files can later be reused for navigation tasks.

---

## Project Structure
Relevant implementation files:

```
sim_ws_Fall2025/
 └── src/
     └── turtlebot3_gazebo/
         ├── launch/
         │   └── mapper.launch.py
         ├── maps/
         │   ├── map.pgm
         │   └── map.yaml
         └── src/
             └── lab4/
                 ├── task1.py   # Autonomous mapping logic
                 ├── task2.py   # Navigation with static obstacles
                 └── task3.py   # Search and localization
```

---

## Key Features
- Fully autonomous frontier-based exploration
- Custom global planning and control
- Obstacle avoidance using onboard sensors
- No Nav2 usage
- Relative paths only (portable across systems)

---

## Notes
- No keyboard or joystick control is used
- No Nav2 stack is used at any stage
- All navigation logic is implemented manually
- Workspace is designed to run immediately after build

---

## Troubleshooting
If nodes fail to launch:

1. Source ROS 2:
```bash
source /opt/ros/<humble|jazzy>/setup.bash
```

2. Source the workspace:
```bash
source install/local_setup.bash
```

3. Confirm the TurtleBot model is set:
```bash
echo $TURTLEBOT3_MODEL
```

---

## Author
Ashish Kale  
Autonomous Systems — ROS 2
