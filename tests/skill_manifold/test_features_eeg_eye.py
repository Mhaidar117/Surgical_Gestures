"""Smoke tests for skill_manifold.features_eeg_eye."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from skill_manifold.features_eeg_eye import (
    EEG_BASELINE_DIM, EEG_PC_DIM, EYE_DIM, _build_eye_vector,
    build_eeg_eye_feature_frame, feature_column_names,
    load_performance_scores, load_task_module_map, read_phase1_task_names,
)
from skill_manifold.io import CONFIG_DIR, performance_scores_csv


def test_load_performance_scores_strips_quotes(synthetic_data_root: Path):
    df = load_performance_scores(performance_scores_csv(synthetic_data_root))
    # trial_id is derived from Eye File Name with the trailing ' and .csv removed.
    assert "trial_id" in df.columns
    assert not df["trial_id"].astype(str).str.endswith("'").any()
    assert not df["trial_id"].astype(str).str.contains(".csv").any()
    # The synthetic fixture generates only Try==1 rows.
    assert (df["try_num"] == 1).all()
    # Expected columns after rename.
    assert set(["subject_id", "task_id", "age", "dominant_hand",
                "performance"]).issubset(df.columns)


def test_feature_column_dim_matches_reality():
    cols = feature_column_names()
    assert len(cols) == EEG_BASELINE_DIM + EEG_PC_DIM + EYE_DIM
    assert cols[0] == "eeg_base_0"
    assert "eye_blink_fraction" in cols


def test_build_eye_vector_handles_missing_fields():
    v = _build_eye_vector({})  # totally empty
    assert v.shape == (EYE_DIM,)
    assert np.isfinite(v).all()
    assert (v == 0).all()


def test_build_feature_frame_shape(synthetic_data_root: Path):
    mod_map = load_task_module_map(CONFIG_DIR / "skill_manifold_task_modules.yaml")
    df = build_eeg_eye_feature_frame(synthetic_data_root, mod_map, try_filter=1)
    cols = feature_column_names()
    assert len(df) > 0
    for c in cols:
        assert c in df.columns
    assert df[cols].isna().sum().sum() == 0
    # Metadata columns.
    for c in ["trial_id", "subject_id", "task_id", "task_module",
              "age", "dominant_hand", "performance"]:
        assert c in df.columns
    # Task_module values are drawn from the nine config labels, plus "unknown"
    # if a task_id is unmapped (shouldn't happen given the default config).
    valid_modules = {"pick_and_place", "pegboard", "match_board",
                     "ring_and_rail", "camera_scaling", "ring_walk",
                     "needle_suture", "energy", "dissection", "unknown"}
    assert set(df["task_module"].unique()).issubset(valid_modules)


def test_task_module_map_covers_27_ids():
    mod_map = load_task_module_map(CONFIG_DIR / "skill_manifold_task_modules.yaml")
    assert set(mod_map.keys()) == set(range(1, 28))
    assert len(set(mod_map.values())) >= 8  # at least 8 distinct modules


def test_read_phase1_task_names(synthetic_data_root: Path):
    names = read_phase1_task_names(synthetic_data_root)
    assert 1 in names and 27 in names
    assert names[1] == "Pick and place"
