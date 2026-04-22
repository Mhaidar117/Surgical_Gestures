"""Tests for the new eval metrics: edit distance, IoU, frame-weighted accuracy,
ordinal skill metrics.

Run:
    python -m pytest tests/test_eval_metrics_extras.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))

from eval.metrics import (  # noqa: E402
    compute_gesture_edit_distance,
    compute_frame_weighted_gesture_accuracy,
    compute_gesture_frame_iou,
    compute_ordinal_skill_metrics,
)


# ---- edit distance ---------------------------------------------------------

def test_edit_distance_identical_sequences_is_zero():
    out = compute_gesture_edit_distance([[1, 2, 3]], [[1, 2, 3]])
    assert out['edit_distance_mean'] == 0
    assert out['edit_distance_normalized'] == 0
    assert out['num_trials'] == 1


def test_edit_distance_collapses_runs_by_default():
    # Pred '1 1 2 2 2 3' collapses to '1 2 3', matches true '1 2 3' -> distance 0
    out = compute_gesture_edit_distance([[1, 1, 2, 2, 2, 3]], [[1, 2, 3]])
    assert out['edit_distance_mean'] == 0


def test_edit_distance_no_collapse_flag():
    out = compute_gesture_edit_distance(
        [[1, 1, 2, 2, 2, 3]], [[1, 2, 3]], collapse_runs=False
    )
    # Lengths 6 vs 3 -> at minimum 3 edits
    assert out['edit_distance_mean'] >= 3


def test_edit_distance_one_substitution():
    out = compute_gesture_edit_distance([[1, 5, 3]], [[1, 2, 3]])
    assert out['edit_distance_mean'] == 1
    assert out['edit_distance_normalized'] == pytest.approx(1 / 3)


def test_edit_distance_rejects_mismatched_counts():
    with pytest.raises(ValueError):
        compute_gesture_edit_distance([[1, 2]], [[1, 2], [3, 4]])


# ---- frame-weighted accuracy + IoU -----------------------------------------

def test_frame_weighted_accuracy_weights_by_duration():
    # Two segments, one correct (50 frames), one wrong (10 frames).
    # Unweighted accuracy = 0.5; frame-weighted = 50/60 ≈ 0.833.
    acc = compute_frame_weighted_gesture_accuracy(
        per_segment_pred=[1, 2], per_segment_true=[1, 3],
        per_segment_frames=[50, 10],
    )
    assert acc == pytest.approx(50 / 60)


def test_frame_weighted_accuracy_empty():
    assert compute_frame_weighted_gesture_accuracy([], [], []) == 0.0


def test_gesture_frame_iou_perfect_match():
    iou = compute_gesture_frame_iou([1, 2, 3], [1, 2, 3], [10, 20, 30], num_classes=15)
    assert iou['iou_mean'] == pytest.approx(1.0)
    assert iou['iou_classes_seen'] == 3


def test_gesture_frame_iou_partial_overlap():
    # Two segments predicted as 1 (20 + 10 frames); truth: first is 1, second is 2.
    iou = compute_gesture_frame_iou(
        per_segment_pred=[1, 1], per_segment_true=[1, 2],
        per_segment_frames=[20, 10], num_classes=15,
    )
    # Class 1: inter=20, union=pred(30)+true(20)-inter(20)=30 -> iou=20/30=0.667
    # Class 2: inter=0,  union=pred(0)+true(10)-0=10 -> iou=0
    assert iou['iou_mean'] == pytest.approx((20 / 30 + 0) / 2, rel=1e-4)
    assert iou['iou_classes_seen'] == 2


# ---- ordinal skill ---------------------------------------------------------

def test_ordinal_skill_perfect_predictions():
    # 3 samples, classes 0 / 1 / 2, each predicted confidently at the right class.
    logits = torch.tensor([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
    ])
    labels = torch.tensor([0, 1, 2])
    out = compute_ordinal_skill_metrics(logits, labels)
    assert out['ord_mae'] == 0.0
    assert out['ord_expected_mae'] < 0.05  # near-perfect, softmax with 5.0 is sharp
    assert out['ord_spearman'] == pytest.approx(1.0)


def test_ordinal_skill_systematic_overshoot():
    # Predict always 2 when truth cycles through 0/1/2.
    logits = torch.tensor([
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 5.0],
        [0.0, 0.0, 5.0],
    ])
    labels = torch.tensor([0, 1, 2])
    out = compute_ordinal_skill_metrics(logits, labels)
    # argmax predictions: 2, 2, 2; errors: 2, 1, 0 -> MAE = 1.0
    assert out['ord_mae'] == pytest.approx(1.0)


def test_ordinal_skill_monotone_ranking():
    # Predictions with increasing expected value track truth -> rho ~ 1.
    logits = torch.tensor([
        [3.0, 0.0, 0.0],   # expected ~0
        [0.0, 3.0, 0.0],   # expected ~1
        [0.0, 0.0, 3.0],   # expected ~2
    ])
    labels = torch.tensor([0, 1, 2])
    out = compute_ordinal_skill_metrics(logits, labels)
    assert out['ord_spearman'] == pytest.approx(1.0)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
