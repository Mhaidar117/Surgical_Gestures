"""
Evaluation metrics for kinematics, gesture, skill, and brain alignment.
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import Dict, List, Tuple
from scipy.stats import spearmanr


def compute_kinematics_metrics(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute kinematics metrics: RMSE, end-effector error, rotation error.
    
    Args:
        pred: Predicted kinematics (B, T, D)
        target: Target kinematics (B, T, D)
    
    Returns:
        Dictionary of metrics
    """
    # Position RMSE
    pos_pred = pred[..., :3]
    pos_target = target[..., :3]
    pos_rmse = torch.sqrt(torch.mean((pos_pred - pos_target) ** 2)).item()
    
    # End-effector error (Euclidean distance)
    ee_error = torch.norm(pos_pred - pos_target, dim=-1).mean().item()
    
    # Rotation error (simplified - would use geodesic distance)
    if pred.shape[-1] >= 9:
        rot_pred = pred[..., 3:9]
        rot_target = target[..., 3:9]
        rot_rmse = torch.sqrt(torch.mean((rot_pred - rot_target) ** 2)).item()
    else:
        rot_rmse = 0.0
    
    return {
        'pos_rmse': pos_rmse,
        'ee_error': ee_error,
        'rot_rmse': rot_rmse
    }


def compute_gesture_metrics(
    pred_logits: torch.Tensor,
    target_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute gesture classification metrics.
    
    Args:
        pred_logits: Predicted logits (B, num_gestures)
        target_labels: Target labels (B,)
    
    Returns:
        Dictionary of metrics
    """
    pred_labels = torch.argmax(pred_logits, dim=-1).cpu().numpy()
    target_np = target_labels.cpu().numpy()
    
    # F1 scores
    f1_macro = f1_score(target_np, pred_labels, average='macro')
    f1_micro = f1_score(target_np, pred_labels, average='micro')
    
    # Accuracy
    accuracy = accuracy_score(target_np, pred_labels)
    
    return {
        'gesture_f1_macro': f1_macro,
        'gesture_f1_micro': f1_micro,
        'gesture_accuracy': accuracy
    }


def compute_skill_metrics(
    pred_logits: torch.Tensor,
    target_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute skill classification metrics.
    
    Args:
        pred_logits: Predicted logits (B, num_skills)
        target_labels: Target labels (B,)
    
    Returns:
        Dictionary of metrics
    """
    pred_labels = torch.argmax(pred_logits, dim=-1).cpu().numpy()
    target_np = target_labels.cpu().numpy()
    
    f1_macro = f1_score(target_np, pred_labels, average='macro')
    accuracy = accuracy_score(target_np, pred_labels)
    
    return {
        'skill_f1_macro': f1_macro,
        'skill_accuracy': accuracy
    }


def compute_rsa_metric(
    model_rdm: torch.Tensor,
    eeg_rdm: torch.Tensor
) -> float:
    """
    Compute RSA correlation between model and EEG RDMs.
    
    Args:
        model_rdm: Model RDM (M, M)
        eeg_rdm: EEG RDM (M, M)
    
    Returns:
        Spearman correlation
    """
    # Flatten upper triangles
    model_flat = model_rdm.triu(diagonal=1).flatten()
    eeg_flat = eeg_rdm.triu(diagonal=1).flatten()
    
    # Remove zeros
    mask = (model_flat != 0) & (eeg_flat != 0)
    model_flat = model_flat[mask].cpu().numpy()
    eeg_flat = eeg_flat[mask].cpu().numpy()
    
    if len(model_flat) == 0:
        return 0.0
    
    corr, _ = spearmanr(model_flat, eeg_flat)
    return corr if not np.isnan(corr) else 0.0

