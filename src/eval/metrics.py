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


def compute_gesture_edit_distance(
    pred_sequences: List[List[int]],
    true_sequences: List[List[int]],
    collapse_runs: bool = True,
) -> Dict[str, float]:
    """Mean Levenshtein edit distance between predicted and true gesture sequences.

    Args:
        pred_sequences: list of predicted gesture-label sequences (one per trial).
        true_sequences: list of ground-truth gesture-label sequences (one per trial),
            matched 1-to-1 with ``pred_sequences``.
        collapse_runs: if True, collapse consecutive duplicates before distance
            (the usual JIGSAWS convention). E.g. ``[1,1,2,2,2,3]`` -> ``[1,2,3]``.

    Returns:
        dict with ``edit_distance_mean`` (mean over trials) and
        ``edit_distance_normalized`` (mean of edit_dist / max(len_pred, len_true)).
    """
    if len(pred_sequences) != len(true_sequences):
        raise ValueError(
            f'sequence count mismatch: pred={len(pred_sequences)} true={len(true_sequences)}'
        )

    def _collapse(seq: List[int]) -> List[int]:
        out: List[int] = []
        for g in seq:
            if not out or out[-1] != g:
                out.append(g)
        return out

    def _lev(a: List[int], b: List[int]) -> int:
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        # O(m*n) DP
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            cur = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = cur
        return prev[n]

    dists = []
    norms = []
    for p, t in zip(pred_sequences, true_sequences):
        if collapse_runs:
            p = _collapse(p)
            t = _collapse(t)
        d = _lev(p, t)
        dists.append(d)
        denom = max(len(p), len(t), 1)
        norms.append(d / denom)

    return {
        'edit_distance_mean': float(np.mean(dists)) if dists else 0.0,
        'edit_distance_normalized': float(np.mean(norms)) if norms else 0.0,
        'num_trials': len(pred_sequences),
    }


def compute_frame_weighted_gesture_accuracy(
    per_segment_pred: List[int],
    per_segment_true: List[int],
    per_segment_frames: List[int],
) -> float:
    """Per-segment accuracy weighted by segment frame length.

    A one-off wrong prediction on a 200-frame segment hurts more than a wrong
    prediction on a 10-frame segment. Complements the unweighted accuracy the
    existing ``gesture_accuracy`` metric reports.
    """
    if not per_segment_pred:
        return 0.0
    correct_frames = 0
    total_frames = 0
    for p, t, n in zip(per_segment_pred, per_segment_true, per_segment_frames):
        if n <= 0:
            continue
        total_frames += n
        if p == t:
            correct_frames += n
    return correct_frames / total_frames if total_frames > 0 else 0.0


def compute_gesture_frame_iou(
    per_segment_pred: List[int],
    per_segment_true: List[int],
    per_segment_frames: List[int],
    num_classes: int = 15,
) -> Dict[str, float]:
    """Per-class frame-level IoU over gesture segments.

    Treats a segment of length n as contributing n frames to its predicted and
    true classes. Per-class IoU = |pred_c ∩ true_c| / |pred_c ∪ true_c|.
    Returns mean IoU across classes that appear at least once.
    """
    if not per_segment_pred:
        return {'iou_mean': 0.0, 'iou_classes_seen': 0}

    inter = np.zeros(num_classes, dtype=np.float64)
    pred_total = np.zeros(num_classes, dtype=np.float64)
    true_total = np.zeros(num_classes, dtype=np.float64)
    for p, t, n in zip(per_segment_pred, per_segment_true, per_segment_frames):
        if n <= 0:
            continue
        pred_total[p] += n
        true_total[t] += n
        if p == t:
            inter[p] += n
    union = pred_total + true_total - inter
    mask = union > 0
    ious = np.zeros(num_classes, dtype=np.float64)
    ious[mask] = inter[mask] / union[mask]
    seen = (pred_total > 0) | (true_total > 0)
    return {
        'iou_mean': float(ious[seen].mean()) if seen.any() else 0.0,
        'iou_classes_seen': int(seen.sum()),
    }


def compute_ordinal_skill_metrics(
    pred_logits: torch.Tensor,
    target_labels: torch.Tensor,
) -> Dict[str, float]:
    """Skill metrics that respect the N < I < E ordering.

    - ``ord_mae``: mean absolute error between argmax prediction and true label.
    - ``ord_expected_mae``: MAE using softmax-expected value (continuous
      prediction = Σ_k k · p_k) instead of argmax. Softer credit for a confident
      prediction that leans the right direction.
    - ``ord_spearman``: rank correlation between expected value and true label.
    """
    probs = torch.softmax(pred_logits, dim=-1)
    k = torch.arange(probs.shape[-1], dtype=probs.dtype, device=probs.device)
    expected = (probs * k).sum(dim=-1).cpu().numpy()
    pred_hard = pred_logits.argmax(dim=-1).cpu().numpy()
    target = target_labels.cpu().numpy()

    ord_mae = float(np.mean(np.abs(pred_hard - target))) if len(target) else 0.0
    ord_exp_mae = float(np.mean(np.abs(expected - target))) if len(target) else 0.0
    if len(target) >= 2 and np.unique(target).size >= 2:
        rho, _ = spearmanr(expected, target)
        rho = float(rho) if not np.isnan(rho) else 0.0
    else:
        rho = 0.0
    return {
        'ord_mae': ord_mae,
        'ord_expected_mae': ord_exp_mae,
        'ord_spearman': rho,
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

