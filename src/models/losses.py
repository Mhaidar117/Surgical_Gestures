"""
Loss functions for kinematics, gesture, skill, brain alignment, and control regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from scipy.stats import spearmanr


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    
    Args:
        rot_6d: Rotation in 6D format of shape (..., 6)
    
    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    
    # Gram-Schmidt orthogonalization
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Stack to form rotation matrix
    return torch.stack([b1, b2, b3], dim=-2)  # (..., 3, 3)


def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance between rotation matrices.
    
    Args:
        R1: First rotation matrix of shape (..., 3, 3)
        R2: Second rotation matrix of shape (..., 3, 3)
    
    Returns:
        Geodesic distance in radians
    """
    # R1^T * R2
    R_diff = torch.matmul(R1.transpose(-2, -1), R2)
    
    # Trace
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    
    # Clamp for numerical stability
    trace = torch.clamp(trace, -1.0, 3.0)
    
    # Geodesic distance: arccos((trace - 1) / 2)
    angle = torch.acos((trace - 1.0) / 2.0)
    
    return angle


def rotation_9d_to_matrix(rot_9d: torch.Tensor) -> torch.Tensor:
    """
    Convert 9D rotation (flattened 3x3 matrix, row-major) to 3x3 rotation matrix.

    Args:
        rot_9d: Rotation in 9D format of shape (..., 9)

    Returns:
        Rotation matrices of shape (..., 3, 3)
    """
    return rot_9d.view(*rot_9d.shape[:-1], 3, 3)


def kinematics_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
    kinematics_format: str = '19d'
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute kinematics loss with position, rotation, and gripper components.

    Args:
        pred: Predicted kinematics of shape (B, T, d_out)
        target: Target kinematics of same shape
        weights: Optional dictionary with weights for 'pos', 'rot', 'jaw', 'smooth'
        kinematics_format: '19d' for JIGSAWS raw format or '10d' for converted format
            - '19d': pos(3) + rot9D(9) + vel_trans(3) + vel_rot(3) + gripper(1)
            - '10d': pos(3) + rot6D(6) + gripper(1)

    Returns:
        Tuple of (total_loss, component_losses_dict)
    """
    if weights is None:
        # Only supervise gripper/jaw - set other weights to 0
        weights = {'pos': 0.0, 'rot': 0.0, 'jaw': 1.0, 'smooth': 0.0, 'vel': 0.0}

    # Position loss (SmoothL1) - make contiguous for MPS compatibility
    pos_pred = pred[..., :3].contiguous()
    pos_target = target[..., :3].contiguous()
    pos_loss = F.smooth_l1_loss(pos_pred, pos_target, reduction='mean')

    if kinematics_format == '19d':
        # JIGSAWS 19D format: pos(3) + rot9D(9) + vel_trans(3) + vel_rot(3) + gripper(1)
        # Rotation is 9D flattened matrix at indices 3:12
        rot_pred_9d = pred[..., 3:12].contiguous()
        rot_target_9d = target[..., 3:12].contiguous()

        rot_pred_mat = rotation_9d_to_matrix(rot_pred_9d)
        rot_target_mat = rotation_9d_to_matrix(rot_target_9d)

        rot_loss = geodesic_distance(rot_pred_mat, rot_target_mat).mean()

        # Velocity losses (translational and rotational)
        vel_trans_pred = pred[..., 12:15].contiguous()
        vel_trans_target = target[..., 12:15].contiguous()
        vel_trans_loss = F.mse_loss(vel_trans_pred, vel_trans_target, reduction='mean')

        vel_rot_pred = pred[..., 15:18].contiguous()
        vel_rot_target = target[..., 15:18].contiguous()
        vel_rot_loss = F.mse_loss(vel_rot_pred, vel_rot_target, reduction='mean')

        vel_loss = vel_trans_loss + vel_rot_loss

        # Gripper at dimension 18
        jaw_pred = pred[..., 18:19].contiguous()
        jaw_target = target[..., 18:19].contiguous()
        jaw_loss = F.mse_loss(jaw_pred, jaw_target, reduction='mean')

    else:
        # 10D format: pos(3) + rot6D(6) + gripper(1)
        rot_pred_6d = pred[..., 3:9].contiguous()
        rot_target_6d = target[..., 3:9].contiguous()

        rot_pred_mat = rotation_6d_to_matrix(rot_pred_6d)
        rot_target_mat = rotation_6d_to_matrix(rot_target_6d)

        rot_loss = geodesic_distance(rot_pred_mat, rot_target_mat).mean()

        vel_loss = torch.tensor(0.0, device=pred.device)

        # Gripper at dimension 9
        if pred.shape[-1] >= 10:
            jaw_pred = pred[..., 9:10].contiguous()
            jaw_target = target[..., 9:10].contiguous()
            jaw_loss = F.mse_loss(jaw_pred, jaw_target, reduction='mean')
        else:
            jaw_loss = torch.tensor(0.0, device=pred.device)

    # Convert to degrees for reporting
    rot_loss_deg = rot_loss * 180.0 / np.pi

    # Smoothness penalty (jerk)
    smooth_penalty = jerk_penalty(pred[..., :3])

    # Total loss
    total_loss = (
        weights['pos'] * pos_loss +
        weights['rot'] * rot_loss +
        weights['jaw'] * jaw_loss +
        weights['smooth'] * smooth_penalty +
        weights.get('vel', 0.5) * vel_loss
    )

    component_losses = {
        'pos': pos_loss,
        'rot': rot_loss,
        'rot_deg': rot_loss_deg,
        'jaw': jaw_loss,
        'vel': vel_loss,
        'smooth': smooth_penalty
    }

    return total_loss, component_losses


def jerk_penalty(trajectory: torch.Tensor) -> torch.Tensor:
    """
    Compute jerk penalty (third derivative) for smoothness.
    
    Args:
        trajectory: Trajectory of shape (B, T, D)
    
    Returns:
        Mean squared jerk
    """
    if trajectory.shape[1] < 3:
        return torch.tensor(0.0, device=trajectory.device)
    
    # First derivative (velocity)
    vel = trajectory[:, 1:, :] - trajectory[:, :-1, :]
    
    # Second derivative (acceleration)
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    
    # Third derivative (jerk)
    jerk = acc[:, 1:, :] - acc[:, :-1, :]
    
    # Mean squared jerk
    return (jerk ** 2).mean()


def velocity_penalty(trajectory: torch.Tensor) -> torch.Tensor:
    """
    Compute velocity penalty.
    
    Args:
        trajectory: Trajectory of shape (B, T, D)
    
    Returns:
        Mean squared velocity
    """
    if trajectory.shape[1] < 2:
        return torch.tensor(0.0, device=trajectory.device)
    
    vel = trajectory[:, 1:, :] - trajectory[:, :-1, :]
    return (vel ** 2).mean()


def acceleration_penalty(trajectory: torch.Tensor) -> torch.Tensor:
    """
    Compute acceleration penalty.
    
    Args:
        trajectory: Trajectory of shape (B, T, D)
    
    Returns:
        Mean squared acceleration
    """
    if trajectory.shape[1] < 3:
        return torch.tensor(0.0, device=trajectory.device)
    
    vel = trajectory[:, 1:, :] - trajectory[:, :-1, :]
    acc = vel[:, 1:, :] - vel[:, :-1, :]
    return (acc ** 2).mean()


def rsa_loss(
    model_rdm: torch.Tensor,
    eeg_rdm: torch.Tensor
) -> torch.Tensor:
    """
    Compute RSA loss: 1 - Spearman correlation between model and EEG RDMs.
    
    Args:
        model_rdm: Model RDM of shape (M, M)
        eeg_rdm: EEG RDM of shape (M, M)
    
    Returns:
        RSA loss (scalar)
    """
    # Flatten upper triangles
    model_flat = model_rdm.triu(diagonal=1).flatten()
    eeg_flat = eeg_rdm.triu(diagonal=1).flatten()
    
    # Remove zeros (diagonal elements)
    mask = (model_flat != 0) & (eeg_flat != 0)
    model_flat = model_flat[mask]
    eeg_flat = eeg_flat[mask]
    
    if len(model_flat) == 0:
        return torch.tensor(1.0, device=model_rdm.device)
    
    # Compute Spearman correlation
    model_np = model_flat.detach().cpu().numpy()
    eeg_np = eeg_flat.detach().cpu().numpy()
    
    corr, _ = spearmanr(model_np, eeg_np)
    corr = corr if not np.isnan(corr) else 0.0
    
    # Loss: 1 - correlation (we want high correlation = low loss)
    loss = 1.0 - corr
    
    return torch.tensor(loss, device=model_rdm.device, requires_grad=True)


def encoding_loss(
    model_features: torch.Tensor,
    eeg_patterns: torch.Tensor,
    ridge_alpha: float = 1.0
) -> torch.Tensor:
    """
    Compute encoding model loss: ridge regression error.
    
    Args:
        model_features: Model features of shape (M, D_model)
        eeg_patterns: EEG patterns of shape (M, D_eeg)
        ridge_alpha: Ridge regularization strength
    
    Returns:
        Encoding loss (scalar)
    """
    # Fit ridge regression: EEG = model_features @ W
    # W = (X^T X + alpha I)^(-1) X^T Y
    
    X = model_features.detach().cpu().numpy()
    Y = eeg_patterns.detach().cpu().numpy()
    
    # Ridge regression
    XTX = X.T @ X
    XTY = X.T @ Y
    I = np.eye(XTX.shape[0])
    
    try:
        W = np.linalg.solve(XTX + ridge_alpha * I, XTY)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        W = np.linalg.pinv(XTX + ridge_alpha * I) @ XTY
    
    # Predict EEG from model features
    pred_eeg = model_features @ torch.from_numpy(W).float().to(model_features.device)
    
    # MSE loss
    loss = F.mse_loss(pred_eeg, eeg_patterns, reduction='mean')
    
    return loss


def control_regularizer(
    kinematics: torch.Tensor,
    velocity_limits: Optional[torch.Tensor] = None,
    acceleration_limits: Optional[torch.Tensor] = None,
    joint_limits: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> torch.Tensor:
    """
    Compute control regularizer for physical plausibility.
    
    Args:
        kinematics: Predicted kinematics of shape (B, T, D)
        velocity_limits: Maximum velocity limits of shape (D,)
        acceleration_limits: Maximum acceleration limits of shape (D,)
        joint_limits: Tuple of (min_limits, max_limits) of shape (D,)
    
    Returns:
        Regularization penalty
    """
    penalty = torch.tensor(0.0, device=kinematics.device)
    
    # Velocity limits
    if velocity_limits is not None and kinematics.shape[1] > 1:
        vel = kinematics[:, 1:, :] - kinematics[:, :-1, :]
        vel_violations = torch.clamp(torch.abs(vel) - velocity_limits.unsqueeze(0).unsqueeze(0), min=0.0)
        penalty = penalty + vel_violations.sum()
    
    # Acceleration limits
    if acceleration_limits is not None and kinematics.shape[1] > 2:
        vel = kinematics[:, 1:, :] - kinematics[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        acc_violations = torch.clamp(torch.abs(acc) - acceleration_limits.unsqueeze(0).unsqueeze(0), min=0.0)
        penalty = penalty + acc_violations.sum()
    
    # Joint limits
    if joint_limits is not None:
        min_limits, max_limits = joint_limits
        below_min = torch.clamp(min_limits.unsqueeze(0).unsqueeze(0) - kinematics, min=0.0)
        above_max = torch.clamp(kinematics - max_limits.unsqueeze(0).unsqueeze(0), min=0.0)
        penalty = penalty + below_min.sum() + above_max.sum()
    
    return penalty


def compute_total_loss(
    pred_kinematics: torch.Tensor,
    target_kinematics: torch.Tensor,
    gesture_logits: torch.Tensor,
    gesture_labels: torch.Tensor,
    skill_logits: torch.Tensor,
    skill_labels: torch.Tensor,
    model_rdm: Optional[torch.Tensor] = None,
    eeg_rdm: Optional[torch.Tensor] = None,
    model_features: Optional[torch.Tensor] = None,
    eeg_patterns: Optional[torch.Tensor] = None,
    brain_mode: str = 'none',
    loss_weights: Optional[Dict[str, float]] = None,
    kinematics_format: str = '19d'
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute total loss with all components.

    Args:
        pred_kinematics: Predicted kinematics (B, T, D)
        target_kinematics: Target kinematics (B, T, D)
        gesture_logits: Gesture logits (B, num_gestures)
        gesture_labels: Gesture labels (B,)
        skill_logits: Skill logits (B, num_skills)
        skill_labels: Skill labels (B,)
        model_rdm: Model RDM for RSA (M, M)
        eeg_rdm: EEG RDM for RSA (M, M)
        model_features: Model features for encoding (M, D)
        eeg_patterns: EEG patterns for encoding (M, D_eeg)
        brain_mode: 'none', 'rsa', or 'encoding'
        loss_weights: Dictionary with weights for each loss component
        kinematics_format: '19d' for JIGSAWS raw format or '10d' for converted format

    Returns:
        Tuple of (total_loss, component_losses_dict)
    """
    if loss_weights is None:
        loss_weights = {
            'kin': 1.0,
            'gesture': 1.0,
            'skill': 0.5,
            'brain': 0.01,
            'control': 0.01
        }

    component_losses = {}

    # Kinematics loss
    kin_loss, kin_components = kinematics_loss(
        pred_kinematics, target_kinematics, kinematics_format=kinematics_format
    )
    component_losses['kin'] = kin_loss
    component_losses.update({f'kin_{k}': v for k, v in kin_components.items()})
    
    # Gesture loss (cross-entropy with label smoothing)
    gesture_loss = F.cross_entropy(gesture_logits, gesture_labels, label_smoothing=0.1)
    component_losses['gesture'] = gesture_loss
    
    # Skill loss (cross-entropy)
    skill_loss = F.cross_entropy(skill_logits, skill_labels)
    component_losses['skill'] = skill_loss
    
    # Brain alignment loss
    brain_loss = torch.tensor(0.0, device=pred_kinematics.device)
    if brain_mode == 'rsa' and model_rdm is not None and eeg_rdm is not None:
        brain_loss = rsa_loss(model_rdm, eeg_rdm)
        component_losses['brain_rsa'] = brain_loss
    elif brain_mode == 'encoding' and model_features is not None and eeg_patterns is not None:
        brain_loss = encoding_loss(model_features, eeg_patterns)
        component_losses['brain_encoding'] = brain_loss
    
    # Control regularizer
    control_reg = control_regularizer(pred_kinematics)
    component_losses['control'] = control_reg
    
    # Total loss
    total_loss = (
        loss_weights['kin'] * kin_loss +
        loss_weights['gesture'] * gesture_loss +
        loss_weights['skill'] * skill_loss +
        loss_weights['brain'] * brain_loss +
        loss_weights['control'] * control_reg
    )
    
    component_losses['total'] = total_loss
    
    return total_loss, component_losses

