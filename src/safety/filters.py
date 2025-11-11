"""
Safety filters for trajectory validation.
"""
import torch
import numpy as np
from typing import Tuple, Optional, Dict


def check_workspace_bounds(
    position: torch.Tensor,
    bounds: Dict[str, Tuple[float, float]]
) -> bool:
    """
    Check if position is within workspace bounds.
    
    Args:
        position: Position of shape (3,) or (B, 3)
        bounds: Dictionary with 'x', 'y', 'z' keys mapping to (min, max) tuples
    
    Returns:
        True if within bounds
    """
    if len(position.shape) == 1:
        position = position.unsqueeze(0)
    
    x, y, z = position[:, 0], position[:, 1], position[:, 2]
    
    x_ok = (x >= bounds['x'][0]) & (x <= bounds['x'][1])
    y_ok = (y >= bounds['y'][0]) & (y <= bounds['y'][1])
    z_ok = (z >= bounds['z'][0]) & (z <= bounds['z'][1])
    
    return (x_ok & y_ok & z_ok).all().item()


def check_velocity_limits(
    trajectory: torch.Tensor,
    max_velocity: float = 0.1,
    dt: float = 1.0 / 30.0
) -> bool:
    """
    Check if velocities are within limits.
    
    Args:
        trajectory: Trajectory of shape (T, D) or (B, T, D)
        max_velocity: Maximum velocity per timestep
        dt: Time step
    
    Returns:
        True if all velocities are within limits
    """
    if len(trajectory.shape) == 2:
        trajectory = trajectory.unsqueeze(0)
    
    if trajectory.shape[1] < 2:
        return True
    
    velocities = (trajectory[:, 1:, :] - trajectory[:, :-1, :]) / dt
    max_vel = torch.abs(velocities).max().item()
    
    return max_vel <= max_velocity


def check_acceleration_limits(
    trajectory: torch.Tensor,
    max_acceleration: float = 0.5,
    dt: float = 1.0 / 30.0
) -> bool:
    """
    Check if accelerations are within limits.
    
    Args:
        trajectory: Trajectory of shape (T, D) or (B, T, D)
        max_acceleration: Maximum acceleration
        dt: Time step
    
    Returns:
        True if all accelerations are within limits
    """
    if len(trajectory.shape) == 2:
        trajectory = trajectory.unsqueeze(0)
    
    if trajectory.shape[1] < 3:
        return True
    
    velocities = (trajectory[:, 1:, :] - trajectory[:, :-1, :]) / dt
    accelerations = (velocities[:, 1:, :] - velocities[:, :-1, :]) / dt
    max_acc = torch.abs(accelerations).max().item()
    
    return max_acc <= max_acceleration


def validate_trajectory(
    trajectory: torch.Tensor,
    workspace_bounds: Optional[Dict] = None,
    max_velocity: float = 0.1,
    max_acceleration: float = 0.5
) -> Tuple[bool, str]:
    """
    Validate trajectory for safety.
    
    Args:
        trajectory: Trajectory of shape (T, D) where D >= 3 (position)
        workspace_bounds: Optional workspace bounds
        max_velocity: Maximum velocity
        max_acceleration: Maximum acceleration
    
    Returns:
        Tuple of (is_safe, error_message)
    """
    if trajectory.shape[0] == 0:
        return False, "Empty trajectory"
    
    # Check workspace bounds
    if workspace_bounds is not None:
        positions = trajectory[:, :3]
        if not check_workspace_bounds(positions, workspace_bounds):
            return False, "Position outside workspace bounds"
    
    # Check velocity limits
    if not check_velocity_limits(trajectory, max_velocity):
        return False, "Velocity exceeds maximum limit"
    
    # Check acceleration limits
    if not check_acceleration_limits(trajectory, max_acceleration):
        return False, "Acceleration exceeds maximum limit"
    
    return True, "Trajectory is safe"

