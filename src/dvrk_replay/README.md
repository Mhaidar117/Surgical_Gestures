# dVRK Trajectory Replay

## Overview
Replay and visualize JIGSAW kinematics trajectories on dVRK simulation using relative motion transformations.

## Files
- `replay_trajectory.py`: Main replay script for executing trajectories on dVRK
- `kinematics_parser.py`: Parser and visualization tools for JIGSAW kinematics data
- `__init__.py`: Package initialization

## Requirements
- ROS 2 Humble
- dVRK ROS 2 packages
- Python 3.10
- PyKDL, numpy, matplotlib
- dvrk_python, crtk_python_client

## Usage

### Visualize JIGSAW Kinematics (No Robot Required)
```bash
python kinematics_parser.py
```

This will:
- Load and parse JIGSAW kinematics file
- Print trajectory statistics (position ranges, gripper angles, etc.)
- Generate 3D trajectory plots for PSM1 and PSM2
- Show velocity profiles (linear and angular)

**Configuration:**
Edit the `filepath` variable in `kinematics_parser.py`:
```python
filepath = "/path/to/your/Needle_Passing_B001.txt"
```

**Visualization Features:**
- 3D trajectories color-coded by gripper angle
- Gripper angle over time
- Linear velocity (vx, vy, vz)
- Angular velocity (ωx, ωy, ωz)
- Statistics for both PSM1 and PSM2

### Replay on dVRK Simulation

#### 1. Launch Simulation
```bash
conda activate dvrk_env
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source ~/dvrk_ws/install/setup.bash
ros2 launch dvrk_model arm.launch.py arm:=PSM1 generation:=Classic simulated:=true
```

#### 2. Run Replay
In a new terminal:
```bash
conda activate dvrk_env
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source ~/dvrk_ws/install/setup.bash
cd src/dvrk_replay
python replay_trajectory.py
```

## Configuration

### Replay Parameters

Edit `replay_trajectory.py` to change:
```python
ARM_NAME = 'PSM1'              # Target arm
SOURCE_ARM = 'PSM1'            # Source arm data from JIGSAW
FILEPATH = "path/to/file.txt"  # Kinematics file path
SCALE_TRANSLATION = 1.0        # Motion scaling (0.5 = half speed)
SCALE_ROTATION = 1.0           # Rotation scaling
```

### Visualization Parameters

Edit `kinematics_parser.py` to visualize different arms:
```python
kin.plot_trajectories('PSM1')   # or 'PSM2', 'MTML', 'MTMR'
kin.plot_velocities('PSM1')
```

## JIGSAW Data Format

76 columns per timestep:

| Columns | Arm | Data |
|---------|-----|------|
| 0-18 | Master Left (MTML) | position(3) + rotation(9) + velocity(3) + angular_vel(3) + gripper(1) |
| 19-37 | Master Right (MTMR) | same format |
| 38-56 | Slave Left (PSM1) | same format |
| 57-75 | Slave Right (PSM2) | same format |

**Position (3):** x, y, z in meters  
**Rotation (9):** 3×3 rotation matrix (row-major, flattened)  
**Velocity (3):** vx, vy, vz in m/s  
**Angular velocity (3):** ωx, ωy, ωz in rad/s  
**Gripper (1):** angle in radians

## Tested Gestures
- ✅ Needle Passing
- ✅ Knot Tying
- ⚠️ Suturing (not tested)

## Workflow Example
```bash
# 1. First, visualize the trajectory
python kinematics_parser.py
# Review plots to understand the motion

# 2. Launch dVRK simulation
ros2 launch dvrk_model arm.launch.py arm:=PSM1 generation:=Classic simulated:=true

# 3. Execute replay
python replay_trajectory.py
```

## Troubleshooting

### Visualization Issues

**"FileNotFoundError"**
- Update `filepath` in `kinematics_parser.py`
- Ensure JIGSAW data is in correct location

**"No display found"**
- If running on headless server, use: `export MPLBACKEND=Agg`
- Or save plots instead of showing: `plt.savefig('trajectory.png')`

### Replay Issues

**"unable to get measured_cp"**
- Ensure tool is engaged in console GUI
- Verify: `ros2 topic echo /PSM1/measured_cp --once`
- Wait longer for subscribers to initialize

**Arm goes to FAULT**
- Reduce `SCALE_TRANSLATION` to 0.5 or 0.3
- Check console for IK errors
- Verify starting position is safe

**ROS shutdown error**
- Known issue at completion (does not affect replay)
- Does not impact trajectory execution quality

## Safety (Physical Robot)

⚠️ **Before running on physical dVRK:**
1. **Visualize first**: Always run `kinematics_parser.py` to review trajectory
2. **Start slow**: Set `SCALE_TRANSLATION = 0.3`
3. **E-stop ready**: Keep emergency stop accessible
4. **Clear workspace**: Ensure no obstacles
5. **Test short segments**: Use `--end-idx 100` for initial tests
6. **Monitor console**: Watch for tracking errors

## Technical Details

### Relative Motion Transformation

The replay uses relative transformations to avoid coordinate frame mismatches:
```
T_relative = T_current^(-1) * T_next
T_target = T_measured * T_relative
```

**Benefits:**
- No need for absolute position matching
- Avoids IK failures from unreachable positions
- Preserves motion shape regardless of starting position
- Generalizable across different robot setups

### Rotation Scaling

Rotations are scaled using axis-angle representation:
```python
angle_scaled = angle_original * scale_rotation
R_scaled = Rotation.Rot(axis, angle_scaled)
```

This preserves the rotation axis while scaling the magnitude.

## API Reference

### JIGSAWKinematics Class
```python
from kinematics_parser import JIGSAWKinematics

# Load data
kin = JIGSAWKinematics(filepath)

# Access parsed data
kin.psm1_pos        # PSM1 positions (N×3)
kin.psm1_rot        # PSM1 rotations (N×9, flattened)
kin.psm1_vel        # PSM1 velocities (N×3)
kin.psm1_gripper    # PSM1 gripper angles (N,)

# Get statistics
kin.get_statistics()

# Plot trajectories
kin.plot_trajectories('PSM1')
kin.plot_velocities('PSM1')

# Get rotation matrix
R = kin.get_rotation_matrix(kin.psm1_rot, idx=0)  # 3×3 matrix
```

## Future Work
- [ ] Fix ROS shutdown exception
- [ ] Two-arm synchronized replay
- [ ] Real-time trajectory visualization during replay
- [ ] Integration with predicted kinematics from ViT model
- [ ] Trajectory comparison metrics (human vs predicted)
