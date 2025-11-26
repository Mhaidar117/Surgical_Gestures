"""
Evaluation metrics for kinematics, gesture, skill, and brain alignment.
"""
import torch
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score
)
from typing import Dict, List, Tuple, Optional
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


def compute_kinematics_metrics_detailed(
    pred: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """
    Compute detailed kinematics metrics with per-component breakdown.

    Args:
        pred: Predicted kinematics (B, T, D) where D >= 10
              Format: [pos(3), rot6d(6), jaw(1), ...]
        target: Target kinematics (B, T, D)

    Returns:
        Dictionary with detailed metrics:
        - position_rmse: Overall position RMSE
        - position_rmse_x/y/z: Per-axis RMSE
        - end_effector_error: Mean Euclidean distance
        - rotation_rmse: 6D rotation representation RMSE
        - rotation_geodesic_deg: Geodesic distance in degrees (if available)
        - gripper_mae: Gripper/jaw MAE
        - gripper_rmse: Gripper/jaw RMSE
    """
    metrics = {}

    # Position metrics
    pos_pred = pred[..., :3]
    pos_target = target[..., :3]

    # Overall position RMSE
    metrics['position_rmse'] = torch.sqrt(torch.mean((pos_pred - pos_target) ** 2)).item()

    # Per-axis RMSE
    for i, axis in enumerate(['x', 'y', 'z']):
        axis_error = (pos_pred[..., i] - pos_target[..., i]) ** 2
        metrics[f'position_rmse_{axis}'] = torch.sqrt(torch.mean(axis_error)).item()

    # End-effector error (Euclidean distance)
    ee_error = torch.norm(pos_pred - pos_target, dim=-1)
    metrics['end_effector_error'] = ee_error.mean().item()
    metrics['end_effector_error_std'] = ee_error.std().item()
    metrics['end_effector_error_max'] = ee_error.max().item()

    # Rotation metrics (6D representation)
    if pred.shape[-1] >= 9:
        rot_pred = pred[..., 3:9]
        rot_target = target[..., 3:9]

        # RMSE in 6D space
        rot_rmse = torch.sqrt(torch.mean((rot_pred - rot_target) ** 2)).item()
        metrics['rotation_rmse'] = rot_rmse

        # Geodesic distance (convert 6D to matrices)
        try:
            from models.losses import rotation_6d_to_matrix, geodesic_distance
            rot_pred_mat = rotation_6d_to_matrix(rot_pred)
            rot_target_mat = rotation_6d_to_matrix(rot_target)
            geodesic_dist = geodesic_distance(rot_pred_mat, rot_target_mat)
            metrics['rotation_geodesic_rad'] = geodesic_dist.mean().item()
            metrics['rotation_geodesic_deg'] = (geodesic_dist.mean() * 180.0 / np.pi).item()
        except ImportError:
            # If losses module not available, skip geodesic
            metrics['rotation_geodesic_deg'] = 0.0

    # Gripper/jaw metrics
    if pred.shape[-1] >= 10:
        jaw_pred = pred[..., 9]
        jaw_target = target[..., 9]

        # MAE
        jaw_mae = torch.mean(torch.abs(jaw_pred - jaw_target)).item()
        metrics['gripper_mae'] = jaw_mae

        # RMSE
        jaw_rmse = torch.sqrt(torch.mean((jaw_pred - jaw_target) ** 2)).item()
        metrics['gripper_rmse'] = jaw_rmse

    return metrics


def compute_gesture_metrics_detailed(
    pred_logits: torch.Tensor,
    target_labels: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute detailed gesture classification metrics.

    Args:
        pred_logits: Predicted logits (B, num_gestures)
        target_labels: Target labels (B,)
        class_names: Optional list of class names for reporting

    Returns:
        Dictionary with:
        - accuracy: Overall accuracy
        - f1_macro/micro/weighted: F1 scores
        - precision_per_class: Dict mapping class to precision
        - recall_per_class: Dict mapping class to recall
        - f1_per_class: Dict mapping class to F1
        - confusion_matrix: Numpy confusion matrix
        - classification_report: Sklearn classification report dict
    """
    pred_labels = torch.argmax(pred_logits, dim=-1).cpu().numpy()
    target_np = target_labels.cpu().numpy()

    # Get unique classes
    unique_classes = np.unique(np.concatenate([target_np, pred_labels]))
    n_classes = pred_logits.shape[-1]

    # Generate default class names if not provided
    if class_names is None:
        class_names = [f'G{i+1}' for i in range(n_classes)]

    metrics = {}

    # Overall metrics
    metrics['accuracy'] = accuracy_score(target_np, pred_labels)
    metrics['f1_macro'] = f1_score(target_np, pred_labels, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(target_np, pred_labels, average='micro', zero_division=0)
    metrics['f1_weighted'] = f1_score(target_np, pred_labels, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(target_np, pred_labels, average=None, zero_division=0, labels=range(n_classes))
    recall_per_class = recall_score(target_np, pred_labels, average=None, zero_division=0, labels=range(n_classes))
    f1_per_class = f1_score(target_np, pred_labels, average=None, zero_division=0, labels=range(n_classes))

    metrics['precision_per_class'] = {
        class_names[i]: float(precision_per_class[i]) for i in range(len(precision_per_class))
    }
    metrics['recall_per_class'] = {
        class_names[i]: float(recall_per_class[i]) for i in range(len(recall_per_class))
    }
    metrics['f1_per_class'] = {
        class_names[i]: float(f1_per_class[i]) for i in range(len(f1_per_class))
    }

    # Confusion matrix
    cm = confusion_matrix(target_np, pred_labels, labels=range(n_classes))
    metrics['confusion_matrix'] = cm

    # Classification report (as dict)
    report = classification_report(
        target_np, pred_labels,
        labels=range(n_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    metrics['classification_report'] = report

    return metrics


def compute_skill_metrics_detailed(
    pred_logits: torch.Tensor,
    target_labels: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute detailed skill classification metrics.

    Args:
        pred_logits: Predicted logits (B, num_skills)
        target_labels: Target labels (B,)
        class_names: Optional list of skill level names (default: ['Novice', 'Intermediate', 'Expert'])

    Returns:
        Dictionary with:
        - accuracy: Overall accuracy
        - f1_macro/weighted: F1 scores
        - precision_per_class: Dict mapping skill level to precision
        - recall_per_class: Dict mapping skill level to recall
        - f1_per_class: Dict mapping skill level to F1
        - confusion_matrix: Numpy confusion matrix
        - classification_report: Sklearn classification report dict
    """
    pred_labels = torch.argmax(pred_logits, dim=-1).cpu().numpy()
    target_np = target_labels.cpu().numpy()

    # Default skill level names
    if class_names is None:
        class_names = ['Novice', 'Intermediate', 'Expert']

    n_classes = pred_logits.shape[-1]

    metrics = {}

    # Overall metrics
    metrics['accuracy'] = accuracy_score(target_np, pred_labels)
    metrics['f1_macro'] = f1_score(target_np, pred_labels, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(target_np, pred_labels, average='weighted', zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(target_np, pred_labels, average=None, zero_division=0, labels=range(n_classes))
    recall_per_class = recall_score(target_np, pred_labels, average=None, zero_division=0, labels=range(n_classes))
    f1_per_class = f1_score(target_np, pred_labels, average=None, zero_division=0, labels=range(n_classes))

    metrics['precision_per_class'] = {
        class_names[i]: float(precision_per_class[i]) for i in range(len(precision_per_class))
    }
    metrics['recall_per_class'] = {
        class_names[i]: float(recall_per_class[i]) for i in range(len(recall_per_class))
    }
    metrics['f1_per_class'] = {
        class_names[i]: float(f1_per_class[i]) for i in range(len(f1_per_class))
    }

    # Confusion matrix
    cm = confusion_matrix(target_np, pred_labels, labels=range(n_classes))
    metrics['confusion_matrix'] = cm

    # Classification report (as dict)
    report = classification_report(
        target_np, pred_labels,
        labels=range(n_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    metrics['classification_report'] = report

    return metrics

