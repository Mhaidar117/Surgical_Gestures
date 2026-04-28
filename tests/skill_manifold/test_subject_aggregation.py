"""Unit tests for src/skill_manifold/subject_aggregation.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from skill_manifold.subject_aggregation import (    # noqa: E402
    FLS_TASK_IDS, FLS_TASK_NAMES,
    aggregate_subject_mean, aggregate_subject_per_task_mean,
    assemble_subject_frame, carry_subject_metadata,
    composite_skill_per_subject, normalize_performance,
    per_task_skill_per_subject,
)


def _build_trial_frame(n_subjects: int = 4, tasks=(1, 2, 3),
                        n_trials_per: int = 3,
                        n_features: int = 5,
                        seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subj in range(1, n_subjects + 1):
        for task in tasks:
            for k in range(1, n_trials_per + 1):
                row = {
                    "trial_id": f"{subj}_{task}_{k}",
                    "subject_id": subj,
                    "task_id": task,
                    "try_num": k,
                    "age": 25 + subj,
                    "dominant_hand": "Right" if subj % 2 else "Left",
                    "dominant_eye": "Left" if subj % 3 else "Right",
                    "gender": "F" if subj % 2 else "M",
                    "performance": float(15 + subj * task + 0.1 * k),
                    "perf_min": 5.0,
                    "perf_max": 25.0,
                }
                for f in range(n_features):
                    row[f"feat_{f}"] = float(rng.standard_normal()
                                              + subj * 0.1 + task * 0.05)
                rows.append(row)
    return pd.DataFrame(rows)


def test_normalize_performance_to_unit_interval() -> None:
    df = pd.DataFrame({
        "performance": [5.0, 15.0, 25.0],
        "perf_min": [5.0, 5.0, 5.0],
        "perf_max": [25.0, 25.0, 25.0],
    })
    s = normalize_performance(df)
    np.testing.assert_allclose(s.to_numpy(), [0.0, 0.5, 1.0])


def test_normalize_performance_handles_zero_range() -> None:
    df = pd.DataFrame({
        "performance": [10.0],
        "perf_min": [10.0],
        "perf_max": [10.0],   # zero range -> NaN
    })
    s = normalize_performance(df)
    assert s.isna().all()


def test_composite_skill_per_subject_averages_normalized_score() -> None:
    df = _build_trial_frame(n_subjects=3, tasks=(1, 2), n_trials_per=2)
    out = composite_skill_per_subject(df)
    assert sorted(out.columns.tolist()) == ["composite_skill", "subject_id"]
    assert len(out) == 3
    assert (out["composite_skill"] >= 0).all() and (out["composite_skill"] <= 1).all()


def test_per_task_skill_columns_match_task_ids() -> None:
    df = _build_trial_frame(n_subjects=4, tasks=(1, 2, 3), n_trials_per=2)
    out = per_task_skill_per_subject(df)
    expected_cols = {"subject_id"} | {f"skill_task_{t}" for t in (1, 2, 3)}
    assert set(out.columns) == expected_cols
    assert len(out) == 4


def test_aggregate_subject_mean_collapses_correctly() -> None:
    df = _build_trial_frame(n_subjects=3, tasks=(1, 2), n_trials_per=2,
                             n_features=4, seed=1)
    feat_cols = [f"feat_{i}" for i in range(4)]
    agg = aggregate_subject_mean(df, feat_cols)
    assert len(agg) == 3
    # Hand-compute the mean for subject 1 and compare.
    sub1 = df[df["subject_id"] == 1][feat_cols].mean()
    np.testing.assert_allclose(
        agg.loc[agg["subject_id"] == 1, feat_cols].to_numpy().ravel(),
        sub1.to_numpy(),
        atol=1e-12,
    )


def test_aggregate_subject_per_task_mean_layout() -> None:
    df = _build_trial_frame(n_subjects=3, tasks=(1, 2, 3), n_trials_per=2,
                             n_features=2, seed=2)
    feat_cols = ["feat_0", "feat_1"]
    wide, new_cols = aggregate_subject_per_task_mean(df, feat_cols, task_ids=(1, 2, 3))
    assert len(wide) == 3
    # 3 tasks * 2 features = 6 derived feature columns.
    assert len(new_cols) == 6
    for tid in (1, 2, 3):
        tname = FLS_TASK_NAMES[tid]
        for f in feat_cols:
            assert f"{tname}__{f}" in wide.columns


def test_aggregate_subject_per_task_mean_handles_missing_task() -> None:
    """If a subject did not perform task 3, the task-3 columns are imputed
    with the column mean across other subjects so the dimensionality is
    preserved."""
    df = _build_trial_frame(n_subjects=3, tasks=(1, 2, 3), n_trials_per=2,
                             n_features=2, seed=3)
    # Drop subject 1's task-3 trials.
    df = df[~((df["subject_id"] == 1) & (df["task_id"] == 3))].copy()
    wide, new_cols = aggregate_subject_per_task_mean(df, ["feat_0", "feat_1"])
    assert len(wide) == 3
    for c in new_cols:
        assert wide[c].notna().all()


def test_carry_subject_metadata_collapses_constants() -> None:
    df = _build_trial_frame(n_subjects=2, tasks=(1, 2), n_trials_per=2)
    meta = carry_subject_metadata(df)
    assert sorted(meta["subject_id"].tolist()) == [1, 2]
    assert {"age", "dominant_hand", "dominant_eye", "gender"}.issubset(meta.columns)


def test_assemble_subject_frame_mean_mode_has_skill_and_features() -> None:
    df = _build_trial_frame(n_subjects=4, tasks=(1, 2, 3), n_trials_per=2,
                             n_features=3, seed=7)
    feat_cols = ["feat_0", "feat_1", "feat_2"]
    out, returned_cols = assemble_subject_frame(df, feat_cols, mode="mean")
    assert returned_cols == feat_cols
    assert len(out) == 4
    for c in feat_cols + ["composite_skill", "skill_task_1",
                          "skill_task_2", "skill_task_3"]:
        assert c in out.columns
