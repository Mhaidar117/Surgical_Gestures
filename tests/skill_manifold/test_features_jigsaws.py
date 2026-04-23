"""Smoke tests for skill_manifold.features_jigsaws."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from skill_manifold.features_jigsaws import (
    ARM_FEATURE_NAMES, _parse_surgeon_letter, build_jigsaws_feature_frame,
    feature_column_names, gesture_histogram, parse_meta_file, zscore_features,
)
from skill_manifold.io import jigsaws_meta


def _gesture_pool():
    return ["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10",
            "G11", "G12", "G13", "G14", "G15"]


def test_surgeon_letter_parses_all_three_task_name_patterns():
    # Regression: earlier code did trial_id.split('_', 1)[1][0], which returned
    # 'T' for Knot_Tying trials (from "Tying") and 'P' for Needle_Passing
    # trials (from "Passing"). Every surgeon-letter parse must agree with the
    # <letter><digits> suffix.
    cases = {
        "Suturing_B001": "B",
        "Suturing_I005": "I",
        "Knot_Tying_B001": "B",
        "Knot_Tying_H003": "H",
        "Needle_Passing_B001": "B",
        "Needle_Passing_F002": "F",
    }
    for trial_id, expected in cases.items():
        assert _parse_surgeon_letter(trial_id) == expected, (
            f"{trial_id!r} -> {_parse_surgeon_letter(trial_id)}, expected {expected}"
        )
    # Something unparseable falls back to "?" rather than silently returning "U".
    assert _parse_surgeon_letter("Unknown_format") == "?"


def test_build_feature_frame_surgeons_span_full_alphabet(synthetic_data_root: Path):
    # All three tasks use surgeons B..I; the parser must recover that on each.
    import numpy as np  # noqa: F401
    gp = _gesture_pool()
    df = build_jigsaws_feature_frame(synthetic_data_root, gp)
    expected_surgeons = {"B", "C", "D", "E", "F", "G", "H", "I"}
    for task in ("Suturing", "Knot_Tying", "Needle_Passing"):
        surgeons = set(df.loc[df["task"] == task, "surgeon"])
        assert surgeons == expected_surgeons, (
            f"{task}: got {surgeons}, expected {expected_surgeons}"
        )


def test_parse_meta_file_basic(synthetic_data_root: Path):
    meta = jigsaws_meta(synthetic_data_root, "Suturing")
    trials = parse_meta_file(meta, "Suturing")
    assert len(trials) >= 8, "expect >= 8 synthetic Suturing trials"
    for t in trials:
        assert t.task == "Suturing"
        assert t.trial_id.startswith("Suturing_")
        assert t.skill in {"N", "I", "E"}
        assert 6 <= t.grs_total <= 30
        assert set(t.osats.keys()).issuperset({
            "respect_for_tissue", "time_and_motion", "overall_performance"
        })


def test_gesture_histogram_sums_to_at_most_one():
    rows = [(1, 30, "G1"), (31, 60, "G5"), (61, 90, "G11")]
    hist = gesture_histogram(rows, _gesture_pool(), n_frames=90)
    assert hist.shape == (len(_gesture_pool()),)
    # Three contiguous segments over 90 frames covers everything, so sum == 1.
    assert np.isclose(hist.sum(), 1.0, atol=1e-6)


def test_build_feature_frame_shape(synthetic_data_root: Path):
    gp = _gesture_pool()
    df = build_jigsaws_feature_frame(synthetic_data_root, gp)
    cols = feature_column_names(gp)
    # 14 gestures + 2*12 arm stats + 1 duration = 39.
    assert len(cols) == len(gp) + 2 * len(ARM_FEATURE_NAMES) + 1 == 39
    for c in cols:
        assert c in df.columns
    assert df[cols].isna().sum().sum() == 0, "no NaN in features"
    assert set(df["task"].unique()).issubset(
        {"Suturing", "Knot_Tying", "Needle_Passing"})
    # Metadata columns present.
    for col in ["task", "surgeon", "skill", "grs_total",
                "trial_index_within_surgeon_task"]:
        assert col in df.columns
    # Per (surgeon, task), trial_index should start at 0.
    assert (df.groupby(["task", "surgeon"])["trial_index_within_surgeon_task"]
              .min() == 0).all()


def test_zscore_features_normalizes(synthetic_data_root: Path):
    gp = _gesture_pool()
    df = build_jigsaws_feature_frame(synthetic_data_root, gp)
    cols = feature_column_names(gp)
    z = zscore_features(df, cols)
    for c in cols:
        v = z[c].to_numpy()
        if v.std() < 1e-12:
            continue
        assert abs(v.mean()) < 1e-8, f"{c} mean {v.mean()}"
        assert abs(v.std() - 1.0) < 1e-6, f"{c} std {v.std()}"
