# dVRK Trajectory Replay

## Overview
Replay JIGSAW kinematics trajectories on dVRK simulation using relative motion transformations.

## Requirements
- ROS 2 Humble
- dVRK ROS 2 packages
- Python 3.10
- PyKDL
- dvrk_python, crtk_python_client

## Quick Start

### 1. Launch Simulation
```bash
conda activate dvrk_env
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source ~/dvrk_ws/install/setup.bash
ros2 launch dvrk_model arm.launch.py arm:=PSM1 generation:=Classic simulated:=true
```

### 2. Run Replay
In a new terminal:
```bash
conda activate dvrk_env
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source ~/dvrk_ws/install/setup.bash
cd src/dvrk_replay
python replay_trajectory.py
```

## Configuration

Edit `replay_trajectory.py` to change:
- `ARM_NAME`: Target arm (default: 'PSM1')
- `SOURCE_ARM`: Source arm data from JIGSAW (default: 'PSM1')
- `FILEPATH`: Path to kinematics file
- `SCALE_TRANSLATION`: Motion scaling (default: 1.0)
- `SCALE_ROTATION`: Rotation scaling (default: 1.0)

## JIGSAW Data Format

76 columns per timestep:
- Columns 0-18: Master Left (MTML)
- Columns 19-37: Master Right (MTMR)
- Columns 38-56: Slave Left (PSM1)
- Columns 57-75: Slave Right (PSM2)

Each arm: position(3) + rotation(9) + velocity(3) + angular_vel(3) + gripper(1)

## Tested Gestures
- ✅ Needle Passing
- ✅ Knot Tying

## Troubleshooting

**"unable to get measured_cp"**
- Ensure tool is engaged in console GUI
- Verify: `ros2 topic echo /PSM1/measured_cp --once`

**Arm goes to FAULT**
- Reduce `SCALE_TRANSLATION` to 0.5 or 0.3
- Check console for IK errors

**ROS shutdown error**
- Known issue at completion (does not affect replay)

## Safety (Physical Robot)
⚠️ Before running on physical dVRK:
- Start with `SCALE_TRANSLATION = 0.3`
- Keep e-stop accessible
- Clear workspace
- Test short segments first

## Technical Details

Uses relative motion transformations:
```
T_relative = T_current^(-1) * T_next
T_target = T_measured * T_relative
```

This avoids coordinate frame mismatches and IK failures from absolute positioning.
