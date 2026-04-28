"""Unit tests for src/skill_manifold/features_fls_gaze.py.

Strategy
--------
Build a tiny FLS gaze CSV with the documented Tobii column layout (header
row + 20 data columns) and run it through `_summarize_one_trial`. The
assertion targets are:
  - the loader honours `has_header=True` (no skip would push column 19 into
    a string row and crash);
  - the resulting feature vector has shape (EYE_DIM,) and is finite;
  - the metadata columns survive the build_fls_gaze_feature_frame round-trip
    when an in-memory PerformanceScores frame is passed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from skill_manifold.features_eeg_eye import EYE_DIM            # noqa: E402
from skill_manifold.features_fls_gaze import (                  # noqa: E402
    _summarize_one_trial,
    build_fls_gaze_feature_frame,
    eye_feature_column_names,
)


HEADER_COLS: List[str] = [
    "Gaze point X", "Gaze point Y",
    "Gaze point 3D X", "Gaze point 3D Y", "Gaze point 3D Z",
    "Gaze direction left X", "Gaze direction left Y", "Gaze direction left Z",
    "Gaze direction right X", "Gaze direction right Y", "Gaze direction right Z",
    "Pupil position left X", "Pupil position left Y", "Pupil position left Z",
    "Pupil position right X", "Pupil position right Y", "Pupil position right Z",
    "Pupil diameter left", "Pupil diameter right",
    "Eye movement type index (1: Fixation; 2: Saccade; 0: Unknown)",
]


def _write_fake_gaze_csv(path: Path, n_samples: int = 600, seed: int = 0) -> None:
    """Write a synthetic FLS gaze CSV at `path` (with header)."""
    rng = np.random.default_rng(seed)
    rows = np.zeros((n_samples, 20), dtype=np.float64)
    # Gaze X, Y oscillating with noise.
    t = np.arange(n_samples)
    rows[:, 0] = 800 + 50 * np.sin(t / 40.0) + rng.standard_normal(n_samples)
    rows[:, 1] = 600 + 50 * np.cos(t / 40.0) + rng.standard_normal(n_samples)
    # Pupil L/R diameters around 3.5 mm with small noise; a few blinks.
    rows[:, 17] = 3.5 + 0.05 * rng.standard_normal(n_samples)
    rows[:, 18] = 3.5 + 0.05 * rng.standard_normal(n_samples)
    blink_idx = rng.choice(n_samples, size=10, replace=False)
    rows[blink_idx, 17] = 0.0; rows[blink_idx, 18] = 0.0
    # Movement type: mostly fixation (1), some saccade (2), occasional unknown (0).
    mt = rng.choice([1, 2, 0], size=n_samples, p=[0.7, 0.25, 0.05])
    rows[:, 19] = mt

    df = pd.DataFrame(rows, columns=HEADER_COLS)
    df.to_csv(path, index=False)


def test_summarize_one_trial_returns_18d_finite(tmp_path: Path) -> None:
    csv = tmp_path / "1_1_1.csv"
    _write_fake_gaze_csv(csv, n_samples=600, seed=42)
    vec = _summarize_one_trial(csv)
    assert vec.shape == (EYE_DIM,)
    assert np.all(np.isfinite(vec))


def test_eye_feature_column_names_match_eye_dim() -> None:
    cols = eye_feature_column_names()
    assert len(cols) == EYE_DIM


def test_build_fls_gaze_feature_frame_round_trip(tmp_path: Path) -> None:
    """Feed a small synthetic PerformanceScores frame + matching CSVs and
    confirm the build returns one row per trial with the right columns."""
    eye_dir = tmp_path / "EYE_FLS"
    eye_dir.mkdir()
    rows = []
    for subj in (1, 2):
        for task in (1, 2):
            tid = f"{subj}_{task}_1"
            _write_fake_gaze_csv(eye_dir / f"{tid}.csv",
                                  n_samples=500 + subj * task * 7,
                                  seed=subj * 100 + task)
            rows.append({
                "trial_id": tid,
                "subject_id": subj,
                "task_id": task,
                "try_num": 1,
                "age": 25 + subj,
                "dominant_hand": "Right",
                "dominant_eye": "Left",
                "gender": "F",
                "performance": float(10 + subj + task),
                "perf_min": 5.0,
                "perf_max": 25.0,
            })
    scores = pd.DataFrame(rows)
    df = build_fls_gaze_feature_frame(REPO, scores=scores, eye_dir=eye_dir)

    assert len(df) == len(rows), "every synthetic trial should land in the frame"
    feat_cols = eye_feature_column_names()
    for c in feat_cols:
        assert c in df.columns
        assert np.all(np.isfinite(df[c].to_numpy()))
    for meta in ("subject_id", "task_id", "performance", "perf_min", "perf_max",
                 "age", "dominant_hand", "dominant_eye", "gender"):
        assert meta in df.columns
