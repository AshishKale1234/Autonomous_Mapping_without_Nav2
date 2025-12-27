# Autonomous Mapping without Nav2

- Fully autonomous exploration and mapping of a completely unknown indoor environment
- Implemented using TurtleBot3 and ROS 2 (Humble / Jazzy)
- No keyboard teleoperation or predefined waypoints
- No use of the Nav2 navigation stack
- Frontier-based exploration strategy for efficient coverage
- Custom global planning and reactive obstacle avoidance
- Online SLAM using `slam_toolbox`
- Generates and saves occupancy grid maps (`.pgm`, `.yaml`)
- Designed to run in Gazebo simulation
- Portable, self-contained ROS 2 workspace

Detailed build and run instructions are provided inside  
**`sim_ws_Fall2025/README.md`**
