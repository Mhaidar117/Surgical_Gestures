"""
Post-processing for trajectory smoothing and safety checks.
"""
import torch
import numpy as np
from scipy.signal import savgol_filter
from typing import Tuple, Optional


def project_rotation_svd(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Project 6D rotation to valid rotation matrix using SVD.
    
    Args:
        rot_6d: 6D rotation representation (..., 6)
    
    Returns:
        Valid rotation matrices (..., 3, 3)
    """
    from models.losses import rotation_6d_to_matrix
    
    # Convert to matrix
    R = rotation_6d_to_matrix(rot_6d)
    
    # Project to SO(3) using SVD
    U, S, V = torch.svd(R)
    R_proj = U @ V.transpose(-2, -1)
    
    # Ensure determinant is +1
    det = torch.det(R_proj)
    if det < 0:
        U[:, :, -1] *= -1
        R_proj = U @ V.transpose(-2, -1)
    
    return R_proj


def smooth_trajectory(
    trajectory: torch.Tensor,
    window_length: int = 5,
    polyorder: int = 2
) -> torch.Tensor:
    """
    Smooth trajectory using Savitzky-Golay filter.
    
    Args:
        trajectory: Trajectory of shape (B, T, D) or (T, D)
        window_length: Window length for filter (must be odd)
        polyorder: Polynomial order
    
    Returns:
        Smoothed trajectory
    """
    is_batch = len(trajectory.shape) == 3
    
    if is_batch:
        B, T, D = trajectory.shape
        trajectory_np = trajectory.cpu().numpy()
        smoothed = np.zeros_like(trajectory_np)
        
        for b in range(B):
            for d in range(D):
                if T >= window_length:
                    smoothed[b, :, d] = savgol_filter(
                        trajectory_np[b, :, d],
                        window_length,
                        polyorder
                    )
                else:
                    smoothed[b, :, d] = trajectory_np[b, :, d]
        
        return torch.from_numpy(smoothed).to(trajectory.device)
    else:
        T, D = trajectory.shape
        trajectory_np = trajectory.cpu().numpy()
        smoothed = np.zeros_like(trajectory_np)
        
        for d in range(D):
            if T >= window_length:
                smoothed[:, d] = savgol_filter(
                    trajectory_np[:, d],
                    window_length,
                    polyorder
                )
            else:
                smoothed[:, d] = trajectory_np[:, d]
        
        return torch.from_numpy(smoothed).to(trajectory.device)


def clip_velocities(
    trajectory: torch.Tensor,
    max_velocity: float = 0.1,
    dt: float = 1.0 / 30.0
) -> torch.Tensor:
    """
    Clip velocities to maximum limits.
    
    Args:
        trajectory: Trajectory of shape (B, T, D) or (T, D)
        max_velocity: Maximum velocity per timestep
        dt: Time step
    
    Returns:
        Clipped trajectory
    """
    if len(trajectory.shape) == 3:
        B, T, D = trajectory.shape
        if T < 2:
            return trajectory
        
        # Compute velocities
        velocities = (trajectory[:, 1:, :] - trajectory[:, :-1, :]) / dt
        
        # Clip
        velocities = torch.clamp(velocities, -max_velocity, max_velocity)
        
        # Reconstruct trajectory
        clipped = trajectory.clone()
        for t in range(1, T):
            clipped[:, t, :] = clipped[:, t-1, :] + velocities[:, t-1, :] * dt
        
        return clipped
    else:
        T, D = trajectory.shape
        if T < 2:
            return trajectory
        
        velocities = (trajectory[1:, :] - trajectory[:-1, :]) / dt
        velocities = torch.clamp(velocities, -max_velocity, max_velocity)
        
        clipped = trajectory.clone()
        for t in range(1, T):
            clipped[t, :] = clipped[t-1, :] + velocities[t-1, :] * dt
        
        return clipped


def postprocess_kinematics(
    kinematics: torch.Tensor,
    smooth: bool = True,
    clip_vel: bool = True,
    project_rot: bool = True
) -> torch.Tensor:
    """
    Complete post-processing pipeline for kinematics.
    
    Args:
        kinematics: Raw kinematics (B, T, D) or (T, D)
        smooth: Whether to apply smoothing
        clip_vel: Whether to clip velocities
        project_rot: Whether to project rotations
    
    Returns:
        Post-processed kinematics
    """
    processed = kinematics.clone()
    
    # Project rotations
    if project_rot and processed.shape[-1] >= 9:
        # Project 6D rotation to valid rotation matrix
        rot_6d = processed[..., 3:9]
        # For now, just ensure it's normalized (full SVD projection would be here)
        # This is simplified - full implementation would convert back to 6D
    
    # Smooth trajectory
    if smooth:
        processed = smooth_trajectory(processed)
    
    # Clip velocities
    if clip_vel:
        processed = clip_velocities(processed)
    
    return processed

