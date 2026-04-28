"""Unit tests for helper functions inside pipeline/skill_manifold_gw_fls.py.

Loaded by spec since pipeline/ isn't a package.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

SPEC = importlib.util.spec_from_file_location(
    "skill_manifold_gw_fls",
    REPO / "pipeline" / "skill_manifold_gw_fls.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader is not None
SPEC.loader.exec_module(mod)


def _make_subject_frame(rows):
    df = pd.DataFrame(rows)
    return df


def test_drop_degenerate_subjects_removes_zero_eeg_row() -> None:
    """A subject with zero EEG features (and any gaze) should be dropped."""
    rows = [
        {"subject_id": 1, "g0": 1.0, "g1": -0.5, "e0": 0.3, "e1": 0.4},
        {"subject_id": 2, "g0": 0.2, "g1": 1.1, "e0": 0.0, "e1": 0.0},  # degenerate EEG
        {"subject_id": 3, "g0": -0.5, "g1": 0.1, "e0": 0.7, "e1": -0.2},
    ]
    df = _make_subject_frame(rows)
    out, dropped = mod.drop_degenerate_subjects(df, ["g0", "g1"], ["e0", "e1"])
    assert dropped == [2]
    assert out["subject_id"].tolist() == [1, 3]


def test_drop_degenerate_subjects_removes_zero_gaze_row() -> None:
    """A zero gaze row also triggers the drop."""
    rows = [
        {"subject_id": 5, "g0": 0.0, "g1": 0.0, "e0": 0.3, "e1": 0.4},
        {"subject_id": 6, "g0": 0.5, "g1": 0.2, "e0": 0.7, "e1": -0.2},
    ]
    df = _make_subject_frame(rows)
    out, dropped = mod.drop_degenerate_subjects(df, ["g0", "g1"], ["e0", "e1"])
    assert dropped == [5]
    assert out["subject_id"].tolist() == [6]


def test_drop_degenerate_subjects_preserves_normal_rows() -> None:
    """No drops when every subject has non-trivial features in both modalities."""
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(10, 15):
        rows.append({
            "subject_id": sid,
            "g0": float(rng.standard_normal()),
            "g1": float(rng.standard_normal()),
            "e0": float(rng.standard_normal()),
            "e1": float(rng.standard_normal()),
        })
    df = _make_subject_frame(rows)
    out, dropped = mod.drop_degenerate_subjects(df, ["g0", "g1"], ["e0", "e1"])
    assert dropped == []
    assert len(out) == len(df)


def test_drop_degenerate_subjects_threshold_is_configurable() -> None:
    """Tightening the threshold lets very small but non-zero rows through."""
    rows = [
        {"subject_id": 1, "g0": 1e-3, "g1": 1e-3, "e0": 1e-3, "e1": 1e-3},
        {"subject_id": 2, "g0": 1.0, "g1": 1.0, "e0": 1.0, "e1": 1.0},
    ]
    df = _make_subject_frame(rows)
    out, dropped = mod.drop_degenerate_subjects(
        df, ["g0", "g1"], ["e0", "e1"], norm_threshold=1e-6,
    )
    assert dropped == []   # 1e-3 norm > 1e-6 threshold
    out, dropped = mod.drop_degenerate_subjects(
        df, ["g0", "g1"], ["e0", "e1"], norm_threshold=1e-2,
    )
    assert dropped == [1]  # 1e-3 norm < 1e-2 threshold


# ---------- per_subject_zscore_features --------------------------------------

def test_per_subject_zscore_zero_mean_unit_std() -> None:
    """Each row of the output should have mean ~0 and std ~1 along the
    feature axis. This is the property that removes per-subject global
    gain from the EEG features."""
    rng = np.random.default_rng(0)
    rows = []
    for sid, gain in zip(range(1, 6), [0.0, 5.0, -3.0, 100.0, -50.0]):
        # Each subject has the same shape across features but a different
        # additive offset (the "gain"). z-scoring should equalize them.
        shape = rng.standard_normal(8)
        for f, val in enumerate(shape):
            rows.append({"subject_id": sid, **{f"f{j}": float(shape[j] + gain)
                                                 for j in range(8)}})
            break
    df = _make_subject_frame(rows)
    feat_cols = [f"f{j}" for j in range(8)]
    out = mod.per_subject_zscore_features(df, feat_cols)
    X = out[feat_cols].to_numpy()
    np.testing.assert_allclose(X.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(X.std(axis=1), 1.0, atol=1e-10)


def test_per_subject_zscore_invariant_to_subject_offset() -> None:
    """Two subjects who share the same shape but differ by an additive
    constant should produce identical z-scored rows."""
    cols = ["f0", "f1", "f2", "f3"]
    shape = [1.0, 2.0, 3.0, 4.0]
    df = _make_subject_frame([
        {"subject_id": 1, **{c: shape[i] + 10.0 for i, c in enumerate(cols)}},
        {"subject_id": 2, **{c: shape[i] - 5.0 for i, c in enumerate(cols)}},
    ])
    out = mod.per_subject_zscore_features(df, cols)
    np.testing.assert_allclose(
        out.iloc[0][cols].to_numpy(dtype=float),
        out.iloc[1][cols].to_numpy(dtype=float),
        atol=1e-10,
    )


def test_per_subject_zscore_handles_constant_row() -> None:
    """A constant row (zero std) should produce a zero output rather
    than NaN -- the downstream degeneracy filter catches it."""
    cols = ["f0", "f1", "f2"]
    df = _make_subject_frame([
        {"subject_id": 1, "f0": 5.0, "f1": 5.0, "f2": 5.0},
        {"subject_id": 2, "f0": 0.0, "f1": 1.0, "f2": 2.0},
    ])
    out = mod.per_subject_zscore_features(df, cols)
    np.testing.assert_allclose(out.iloc[0][cols].to_numpy(dtype=float),
                                [0.0, 0.0, 0.0])
    assert np.all(np.isfinite(out.iloc[1][cols].to_numpy(dtype=float)))


# ---------- per_task_3x3 excluded_subjects -----------------------------------

def _build_trial_resid_frame(n_subjects: int, n_features: int,
                              feature_prefix: str, seed: int) -> pd.DataFrame:
    """Synthetic trial-level frame with subject_id, task_id, try_num,
    perf_min, perf_max, performance, and `n_features` feature columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(1, n_subjects + 1):
        for task in (1, 2, 3):
            for k in range(1, 3):
                row = {
                    "subject_id": sid,
                    "task_id": task,
                    "try_num": k,
                    "performance": float(rng.uniform(8, 25)
                                          + (sid / n_subjects) * 5),
                    "perf_min": 5.0,
                    "perf_max": 25.0,
                }
                for j in range(n_features):
                    row[f"{feature_prefix}{j}"] = float(rng.standard_normal()
                                                          + sid * 0.1)
                rows.append(row)
    return pd.DataFrame(rows)


def test_per_task_3x3_excluded_subjects_filters_per_task_means() -> None:
    """A subject ID passed via excluded_subjects must not appear in any
    per-task subject set (the postmortem found that the headline was
    running at N=24 while per-task ran at N=25 -- this is the regression)."""
    n_features_g, n_features_e = 3, 3
    gaze = _build_trial_resid_frame(8, n_features_g, "g", seed=1)
    eeg  = _build_trial_resid_frame(8, n_features_e, "e", seed=2)
    feat_g = [f"g{j}" for j in range(n_features_g)]
    feat_e = [f"e{j}" for j in range(n_features_e)]

    # Drop subject 5.
    out = mod.per_task_3x3(
        gaze, eeg, feature_cols_gaze=feat_g, feature_cols_eeg=feat_e,
        n_perms=20, seed=0, excluded_subjects=[5], zscore_eeg=False,
    )
    # Each per-task result has n_subjects equal to the surviving set.
    assert out, "per-task result should not be empty"
    for tid, res in out.items():
        assert res["n_subjects"] <= 7, \
            f"task {tid}: expected <=7 surviving subjects, got {res['n_subjects']}"


# ---------- task in residualization design -----------------------------------

def test_residualize_trial_features_includes_task_when_flag_on() -> None:
    """With residualize_task=True, the residualized features should be
    closer to zero on a feature that's perfectly predicted by task_id
    than with residualize_task=False."""
    rng = np.random.default_rng(0)
    rows = []
    for sid in range(1, 6):
        for task in (1, 2, 3):
            for k in range(1, 3):
                # Feature equals task * 10 plus tiny noise: residualizing on
                # task should drive it to ~0; residualizing on demographics
                # alone leaves the task-mean structure in place.
                base = task * 10.0 + 0.01 * rng.standard_normal()
                rows.append({
                    "subject_id": sid, "task_id": task, "try_num": k,
                    "age": 25.0, "dominant_hand": "Right",
                    "dominant_eye": "Left", "gender": "F",
                    "feat0": base,
                })
    df = pd.DataFrame(rows)

    with_task = mod.residualize_trial_features(df, ["feat0"], residualize_task=True)
    without_task = mod.residualize_trial_features(df, ["feat0"], residualize_task=False)

    # With task in the design, residual std (of the z-scored residual) is
    # still 1 because residualize re-z-scores the residual by construction.
    # The relevant property is: the residual no longer correlates with
    # task_id when task_id is in the design.
    corr_with = np.corrcoef(with_task["feat0"], df["task_id"])[0, 1]
    corr_without = np.corrcoef(without_task["feat0"], df["task_id"])[0, 1]
    assert abs(corr_with) < 0.05, \
        f"residualize_task=True should kill the task correlation, got {corr_with:.3f}"
    assert abs(corr_without) > 0.5, \
        f"residualize_task=False should leave the task correlation, got {corr_without:.3f}"


# ---------- drop_one_subject_rdm_sensitivity --------------------------------

def test_drop_one_subject_rdm_sensitivity_shape() -> None:
    """Output should have one cell list per off-diagonal RDM cell, each
    of length N (number of subjects), plus a baseline cell value."""
    rng = np.random.default_rng(0)
    n = 9
    feats = rng.standard_normal((n, 6))
    tiers = np.array(["Low"] * 3 + ["Mid"] * 3 + ["High"] * 3)
    sids = np.arange(1, n + 1)
    out = mod.drop_one_subject_rdm_sensitivity(feats, tiers, sids)
    assert set(out["cells"].keys()) == {"Low_Mid", "Low_High", "Mid_High"}
    for cell, values in out["cells"].items():
        assert len(values) == n
        assert all(np.isfinite(v) for v in values)
        assert cell in out["baseline_cells"]


def test_drop_one_subject_rdm_sensitivity_detects_outlier() -> None:
    """A single subject far from their tier centroid should produce
    one obviously-different leave-one-out value for the cells touching
    that tier."""
    n = 9
    feats = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))   # all near same point
    tiers = np.array(["Low"] * 3 + ["Mid"] * 3 + ["High"] * 3)
    # Make one Mid subject an outlier on a different axis.
    feats[3] = np.array([0.0, 5.0, 0.0])
    sids = np.arange(1, n + 1)
    out = mod.drop_one_subject_rdm_sensitivity(feats, tiers, sids)
    low_mid = np.array(out["cells"]["Low_Mid"])
    # Dropping the outlier (index 3, subject 4) should give a much smaller
    # Low_Mid distance than dropping any non-outlier.
    drop_outlier = low_mid[3]
    drop_others = np.concatenate([low_mid[:3], low_mid[4:]])
    assert drop_outlier < drop_others.min() * 0.5, \
        "dropping the outlier should collapse Low_Mid distance"


# ---------- compare_fls_runs script ------------------------------------------

def test_compare_fls_runs_produces_table(tmp_path: Path) -> None:
    """End-to-end: feed two minimal results JSONs to the comparison
    script and confirm we get a markdown table with all the headline,
    companion, and per-task rows."""
    spec = importlib.util.spec_from_file_location(
        "compare_fls_runs", REPO / "pipeline" / "compare_fls_runs.py")
    cmp_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cmp_mod)

    base = {
        "n_subjects": 24,
        "tier_counts": {"Low": 8, "Mid": 8, "High": 8},
        "tertile_cutoffs": {"q33": 0.4, "q66": 0.6},
        "dropped_subjects": [23],
        "config": {"residualize_task": True, "eeg_zscore": True},
        "headline": {
            "gw_distance": 0.05, "z_score": -0.6, "p_value": 0.30,
            "argmax_assignment": {"Low": "Low", "Mid": "High", "High": "Mid"},
        },
        "companion": {
            "n_subjects": 24, "diag_mass_observed": 0.06,
            "z_score": 0.4, "diag_mass_p_value": 0.3,
            "diag_mass_z_score": 0.4, "argmax_match_rate": 0.08,
            "gw_distance": 0.07,
        },
        "per_task": {
            "1": {"task_name": "peg_transfer", "gw_distance": 0.9,
                  "z_score": 1.7, "p_value": 0.95,
                  "argmax_assignment": {"Low": "Mid"}},
        },
    }
    other = json.loads(json.dumps(base))
    other["headline"]["z_score"] = -1.1
    other["config"]["eeg_zscore"] = False

    p_left = tmp_path / "left.json"
    p_right = tmp_path / "right.json"
    p_left.write_text(json.dumps(base))
    p_right.write_text(json.dumps(other))

    md = cmp_mod.render_markdown(
        json.loads(p_left.read_text()),
        json.loads(p_right.read_text()),
        "with_z", "no_z",
    )
    # Sanity: every section present, both labels present, headline z values rendered.
    for tag in ("Configuration", "Headline", "Companion", "Per-task",
                "with_z", "no_z", "-0.600", "-1.100"):
        assert tag in md, f"comparison markdown missing tag: {tag!r}"
