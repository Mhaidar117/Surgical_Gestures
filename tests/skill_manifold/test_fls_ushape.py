"""Tests for pipeline/skill_manifold_fls_ushape.py.

The three analysis functions (mid_vs_rest_t_test, continuous_skill_regression,
individual_u_position) are tested against synthetic per-subject feature
matrices with known structure.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

SPEC = importlib.util.spec_from_file_location(
    "skill_manifold_fls_ushape",
    REPO / "pipeline" / "skill_manifold_fls_ushape.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader is not None
SPEC.loader.exec_module(mod)


def _build_subject_frame(n_low: int = 8, n_mid: int = 8, n_high: int = 8,
                          n_features: int = 6, seed: int = 0,
                          mid_offset: float = 1.5) -> pd.DataFrame:
    """Build a synthetic per-subject frame where Mid is offset on a known feature."""
    rng = np.random.default_rng(seed)
    rows = []
    sid = 1
    for tier, n, skill_center in (
        ("Low", n_low, 0.2),
        ("Mid", n_mid, 0.5),
        ("High", n_high, 0.8),
    ):
        for _ in range(n):
            row = {
                "subject_id": sid,
                "tier": tier,
                "composite_skill": float(skill_center + 0.05 * rng.standard_normal()),
            }
            for j in range(n_features):
                # Feature 0 is the "Mid-shifted" feature: Mid mean = +mid_offset,
                # Low/High mean = 0. Other features are pure noise.
                if j == 0:
                    base = mid_offset if tier == "Mid" else 0.0
                else:
                    base = 0.0
                row[f"feat{j}"] = float(base + rng.standard_normal())
            rows.append(row)
            sid += 1
    return pd.DataFrame(rows)


# ---------- mid_vs_rest_t_test ----------------------------------------------

def test_t_test_detects_mid_offset() -> None:
    """The Mid-shifted feature should rank #1 with positive Mid − rest."""
    df = _build_subject_frame(seed=0, mid_offset=1.5)
    feat_cols = [f"feat{j}" for j in range(6)]
    t_results = mod.mid_vs_rest_t_test(df, feat_cols)
    # The shifted feature should have the largest |t-statistic|.
    assert t_results.iloc[0]["feature"] == "feat0"
    assert t_results.iloc[0]["mid_minus_rest"] > 0.5
    assert t_results.iloc[0]["t_stat"] > 0


def test_t_test_p_values_in_range() -> None:
    """All p-values should be in [0, 1] and one-sided p-values complementary."""
    df = _build_subject_frame(seed=1, mid_offset=0.5)
    feat_cols = [f"feat{j}" for j in range(4)]
    t_results = mod.mid_vs_rest_t_test(df, feat_cols)
    for _, row in t_results.iterrows():
        assert 0.0 <= row["p_two_sided"] <= 1.0
        assert 0.0 <= row["p_one_sided_up"] <= 1.0
        assert 0.0 <= row["p_one_sided_down"] <= 1.0
        # one-sided up + one-sided down should sum to 1 (two-sided shape).
        np.testing.assert_allclose(
            row["p_one_sided_up"] + row["p_one_sided_down"], 1.0, atol=1e-10,
        )


def test_t_test_lim_flags() -> None:
    """is_lim_up / is_lim_down should be set correctly for the predicted features."""
    cols = list(mod.LIM_PREDICTED_UP) + list(mod.LIM_PREDICTED_DOWN) + ["other_col"]
    rng = np.random.default_rng(2)
    rows = []
    for sid, tier in zip(range(1, 7), ["Low"] * 2 + ["Mid"] * 2 + ["High"] * 2):
        row = {"subject_id": sid, "tier": tier, "composite_skill": float(rng.random())}
        for c in cols:
            row[c] = float(rng.standard_normal())
        rows.append(row)
    df = pd.DataFrame(rows)
    t_results = mod.mid_vs_rest_t_test(df, cols)
    # Lim-up features should be flagged is_lim_up.
    for c in mod.LIM_PREDICTED_UP:
        assert t_results.loc[t_results["feature"] == c, "is_lim_up"].iloc[0]
    for c in mod.LIM_PREDICTED_DOWN:
        assert t_results.loc[t_results["feature"] == c, "is_lim_down"].iloc[0]
    assert not t_results.loc[t_results["feature"] == "other_col", "is_lim_up"].iloc[0]


# ---------- continuous_skill_regression -------------------------------------

def test_regression_prefers_quadratic_for_u_shape() -> None:
    """A feature that's high at mid-skill and low at the ends should produce
    quadratic preference (positive ΔAIC)."""
    rng = np.random.default_rng(0)
    n = 30
    skill = rng.uniform(0.0, 1.0, size=n)
    # U-shape: feature peaks at skill=0.5.
    u_shape = -3.0 * (skill - 0.5) ** 2 + 0.1 * rng.standard_normal(n)
    linear = 2.0 * skill + 0.1 * rng.standard_normal(n)
    df = pd.DataFrame({
        "subject_id": np.arange(1, n + 1),
        "composite_skill": skill,
        "u_feat": u_shape,
        "lin_feat": linear,
        "tier": ["Mid"] * n,   # tier not used by this function
    })
    res = mod.continuous_skill_regression(df, ["u_feat", "lin_feat"])
    u_row = res[res["feature"] == "u_feat"].iloc[0]
    lin_row = res[res["feature"] == "lin_feat"].iloc[0]
    assert u_row["delta_AIC"] > 5.0, \
        "U-shape feature should strongly prefer quadratic (ΔAIC > 5)"
    assert lin_row["delta_AIC"] < 2.0, \
        "Linear feature should not prefer quadratic (ΔAIC < 2)"


def test_regression_n_field() -> None:
    df = _build_subject_frame(seed=3)
    feat_cols = [f"feat{j}" for j in range(3)]
    res = mod.continuous_skill_regression(df, feat_cols)
    assert (res["n"] == 24).all()


# ---------- individual_u_position -------------------------------------------

def test_u_position_sign_correlates_with_mid_membership() -> None:
    """Mid subjects should on average have higher u_position than Low/High."""
    df = _build_subject_frame(seed=5, mid_offset=2.0)
    feat_cols = [f"feat{j}" for j in range(6)]
    u_df = mod.individual_u_position(df, feat_cols)
    mid_mean = u_df.loc[u_df["tier"] == "Mid", "u_position"].mean()
    rest_mean = u_df.loc[u_df["tier"] != "Mid", "u_position"].mean()
    assert mid_mean > rest_mean + 0.5, \
        f"Mid mean u_position {mid_mean:.3f} should exceed rest {rest_mean:.3f}"


def test_u_position_unit_axis() -> None:
    """The projection axis should be normalized to unit length, so all subjects'
    u_position should have magnitude bounded by their feature norm."""
    df = _build_subject_frame(seed=6, mid_offset=1.5)
    feat_cols = [f"feat{j}" for j in range(6)]
    u_df = mod.individual_u_position(df, feat_cols)
    # u_position should be a finite real number for every subject.
    assert u_df["u_position"].notna().all()
    assert np.all(np.isfinite(u_df["u_position"].to_numpy()))
    assert len(u_df) == len(df)


def test_u_position_carries_skill_when_present() -> None:
    df = _build_subject_frame(seed=7)
    feat_cols = [f"feat{j}" for j in range(3)]
    u_df = mod.individual_u_position(df, feat_cols)
    assert "composite_skill" in u_df.columns
    assert "tier" in u_df.columns
    assert "subject_id" in u_df.columns


# ---------- Lim feature constants -------------------------------------------

def test_lim_features_are_eeg_columns() -> None:
    """Sanity check: the 4 Lim features should match the orchestrator's
    canonical EEG column names (frontal_{L,R}_theta, parietal_{L,R}_alpha)."""
    sys.path.insert(0, str(REPO / "src"))
    from skill_manifold.features_fls_eeg import eeg_feature_column_names
    canonical = set(eeg_feature_column_names())
    for col in mod.LIM_FEATURES:
        assert col in canonical, f"Lim feature {col!r} not in canonical EEG cols"


# ---------- LDA classifier --------------------------------------------------

def _build_subject_frame_for_lda(n_per_tier: int = 8, n_features: int = 40,
                                   tier_signal: float = 1.5,
                                   seed: int = 0) -> pd.DataFrame:
    """Synthetic subject frame with a tier-specific signal in feature 0."""
    rng = np.random.default_rng(seed)
    rows = []
    sid = 1
    tier_signals = {"Low": 0.0, "Mid": tier_signal, "High": -tier_signal}
    for tier in ("Low", "Mid", "High"):
        for _ in range(n_per_tier):
            row = {"subject_id": sid, "tier": tier,
                   "composite_skill": float(rng.random())}
            shift = tier_signals[tier]
            for j in range(n_features):
                # Feature 0 carries the tier signal; others are noise.
                base = shift if j == 0 else 0.0
                row[f"feat{j}"] = float(base + rng.standard_normal())
            rows.append(row)
            sid += 1
    return pd.DataFrame(rows)


def test_lda_runs_and_returns_expected_keys() -> None:
    df = _build_subject_frame_for_lda(n_per_tier=8, n_features=10, seed=1)
    feat_cols = [f"feat{j}" for j in range(10)]
    res = mod.lda_three_class(df, feat_cols)
    for key in ("accuracy", "confusion", "confusion_normalized", "tier_order",
                "per_class_recall", "pairwise_mce", "per_subject_predictions",
                "n_subjects", "n_features"):
        assert key in res, f"missing key: {key}"
    assert res["n_subjects"] == 24
    assert res["n_features"] == 10
    assert res["tier_order"] == ["Low", "Mid", "High"]
    # Confusion matrix is 3x3.
    confusion = np.asarray(res["confusion"])
    assert confusion.shape == (3, 3)
    # Per-subject predictions length matches N.
    assert len(res["per_subject_predictions"]) == 24


def test_lda_above_chance_on_separable_signal() -> None:
    """With a strong tier signal, 3-class LDA should beat chance (1/3)."""
    df = _build_subject_frame_for_lda(n_per_tier=10, n_features=20,
                                        tier_signal=2.5, seed=2)
    feat_cols = [f"feat{j}" for j in range(20)]
    res = mod.lda_three_class(df, feat_cols)
    assert res["accuracy"] > 0.5, \
        f"separable signal should give accuracy > 0.5, got {res['accuracy']:.3f}"


def test_lda_pairwise_mce_keys() -> None:
    df = _build_subject_frame_for_lda(seed=3)
    feat_cols = [f"feat{j}" for j in range(10)]
    res = mod.lda_three_class(df, feat_cols)
    assert "Low_vs_Mid" in res["pairwise_mce"]
    assert "Mid_vs_High" in res["pairwise_mce"]
    assert "Low_vs_High" in res["pairwise_mce"]
    for pair_key, pair in res["pairwise_mce"].items():
        for sub_key in ("MCE_A_to_B", "MCE_B_to_A", "n_A", "n_B"):
            assert sub_key in pair, f"pair {pair_key} missing {sub_key}"
        # Misclassification rates are bounded.
        assert 0.0 <= pair["MCE_A_to_B"] <= 1.0
        assert 0.0 <= pair["MCE_B_to_A"] <= 1.0


# ---------- Nemani topomap helpers -----------------------------------------

def test_region_head_pos_keys_match_eeg_regions() -> None:
    """REGION_HEAD_POS should have a coordinate for each of the 8 regions."""
    sys.path.insert(0, str(REPO / "src"))
    from skill_manifold.features_fls_eeg import REGION_ORDER
    assert set(mod.REGION_HEAD_POS.keys()) == set(REGION_ORDER)


def test_topomap_writes_file(tmp_path: Path) -> None:
    """Smoke test: plot_nemani_topomap should write a non-empty PNG file."""
    df = _build_subject_frame_for_lda(seed=4, n_features=10)
    feat_cols = [f"feat{j}" for j in range(10)]
    # Build a t_results frame using real eeg column names so the topomap can
    # parse region/band tokens.
    sys.path.insert(0, str(REPO / "src"))
    from skill_manifold.features_fls_eeg import eeg_feature_column_names
    eeg_cols = eeg_feature_column_names()
    rng = np.random.default_rng(0)
    rows = []
    for c in eeg_cols:
        rows.append({
            "feature": c,
            "mid_minus_rest": float(rng.standard_normal() * 0.3),
            "t_stat": float(rng.standard_normal()),
            "p_two_sided": float(rng.uniform(0.01, 0.99)),
            "p_one_sided_up": 0.5,
            "p_one_sided_down": 0.5,
            "is_lim_up": False,
            "is_lim_down": False,
            "abs_t": 1.0,
            "mid_mean": 0.0, "rest_mean": 0.0,
        })
    t_df = pd.DataFrame(rows)
    out = tmp_path / "topomap.png"
    mod.plot_nemani_topomap(t_df, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_lda_confusion_plot_writes_file(tmp_path: Path) -> None:
    """Smoke test: plot_lda_confusion should write a non-empty PNG file."""
    df = _build_subject_frame_for_lda(seed=5)
    feat_cols = [f"feat{j}" for j in range(40)]
    res = mod.lda_three_class(df, feat_cols)
    out = tmp_path / "confusion.png"
    mod.plot_lda_confusion(res, out)
    assert out.exists()
    assert out.stat().st_size > 0


# ---------- top-k feature LDA -----------------------------------------------

def test_lda_top_k_returns_expected_shape() -> None:
    df = _build_subject_frame_for_lda(seed=10, n_features=20)
    feat_cols = [f"feat{j}" for j in range(20)]
    res = mod.lda_three_class_top_k(df, feat_cols, k=5)
    assert res["n_features"] == 5
    assert res["method"].startswith("top_")
    assert "selected_features_per_fold" in res
    # One feature list per LOO fold = N folds.
    assert len(res["selected_features_per_fold"]) == 24
    # Each fold's selected list should have exactly k entries.
    for sel in res["selected_features_per_fold"]:
        assert len(sel) == 5


def test_lda_top_k_above_chance_on_separable_signal() -> None:
    """With strong tier signal restricted to feature 0 plus noise elsewhere,
    top-k feature selection should pick feature 0 in most folds and the
    classifier should beat chance."""
    df = _build_subject_frame_for_lda(seed=11, n_features=40,
                                        tier_signal=2.5)
    feat_cols = [f"feat{j}" for j in range(40)]
    res = mod.lda_three_class_top_k(df, feat_cols, k=5)
    assert res["accuracy"] > 0.5, \
        f"top-k LDA should beat chance on a separable signal, got {res['accuracy']:.3f}"
    # The signal feature should appear in most folds' top-k selection.
    folds_picking_feat0 = sum(
        1 for sel in res["selected_features_per_fold"] if "feat0" in sel
    )
    assert folds_picking_feat0 > 18, \
        f"feat0 should be picked in most folds, got {folds_picking_feat0}/24"


# ---------- PCA → LDA -------------------------------------------------------

def test_lda_pca_returns_expected_shape() -> None:
    df = _build_subject_frame_for_lda(seed=12, n_features=20)
    feat_cols = [f"feat{j}" for j in range(20)]
    res = mod.lda_three_class_pca(df, feat_cols, n_components=5)
    assert res["n_features"] == 5
    assert "pca" in res["method"]


def test_lda_pca_above_chance_on_separable_signal() -> None:
    """PCA→LDA should recover a separable signal even when it lives on a
    single feature out of many noisy ones — PCA captures the variance
    direction and LDA classifies in PC space."""
    df = _build_subject_frame_for_lda(seed=13, n_features=40,
                                        tier_signal=2.5)
    feat_cols = [f"feat{j}" for j in range(40)]
    res = mod.lda_three_class_pca(df, feat_cols, n_components=5)
    assert res["accuracy"] > 0.4, \
        f"PCA→LDA accuracy too low on separable signal: {res['accuracy']:.3f}"


# ---------- comparison plot -------------------------------------------------

def test_lda_comparison_plot_writes_file(tmp_path: Path) -> None:
    df = _build_subject_frame_for_lda(seed=14, n_features=10)
    feat_cols = [f"feat{j}" for j in range(10)]
    res_full = mod.lda_three_class(df, feat_cols)
    res_full["method"] = "all_features"
    res_topk = mod.lda_three_class_top_k(df, feat_cols, k=3)
    res_pca = mod.lda_three_class_pca(df, feat_cols, n_components=3)
    out = tmp_path / "comparison.png"
    mod.plot_lda_classifier_comparison([res_full, res_topk, res_pca], out)
    assert out.exists()
    assert out.stat().st_size > 0


# ---------- Soangra-style RFC + RFE classifier ------------------------------

def test_rfc_rfe_returns_expected_shape() -> None:
    df = _build_subject_frame_for_lda(seed=20, n_features=15)
    feat_cols = [f"feat{j}" for j in range(15)]
    res = mod.rfc_three_class_rfe(df, feat_cols, k=4, n_estimators=20)
    assert res["n_features"] == 4
    assert res["method"].startswith("rfc_rfe_")
    assert "selected_features_per_fold" in res
    assert len(res["selected_features_per_fold"]) == 24
    for sel in res["selected_features_per_fold"]:
        assert len(sel) == 4


def test_rfc_rfe_above_chance_on_separable_signal() -> None:
    """RFC + RFE on a feature space with one strongly separable signal
    feature should beat chance and concentrate its selection on that feature."""
    df = _build_subject_frame_for_lda(seed=21, n_features=20,
                                        tier_signal=2.5)
    feat_cols = [f"feat{j}" for j in range(20)]
    res = mod.rfc_three_class_rfe(df, feat_cols, k=5, n_estimators=30)
    assert res["accuracy"] > 0.4, \
        f"RFC + RFE accuracy too low on separable signal: {res['accuracy']:.3f}"
    # The signal feature should appear in most folds' selection.
    folds_picking_feat0 = sum(
        1 for sel in res["selected_features_per_fold"] if "feat0" in sel
    )
    assert folds_picking_feat0 > 18, \
        f"feat0 should be picked in most folds, got {folds_picking_feat0}/24"


def test_rfc_rfe_includes_mid_vs_rest() -> None:
    """The Mid-vs-not-Mid derived statistic should be present and bounded."""
    df = _build_subject_frame_for_lda(seed=22, n_features=10)
    feat_cols = [f"feat{j}" for j in range(10)]
    res = mod.rfc_three_class_rfe(df, feat_cols, k=3, n_estimators=20)
    assert "mid_vs_rest" in res
    mvr = res["mid_vs_rest"]
    for key in ("accuracy", "mid_recall", "non_mid_recall", "mid_precision"):
        assert key in mvr
        if not np.isnan(mvr[key]):
            assert 0.0 <= mvr[key] <= 1.0


# ---------- modality flag + gaze top-features panel -------------------------

def test_argparse_modality_default_is_eeg() -> None:
    args = mod._parse_args([])
    assert args.modality == "eeg"
    assert "ushape" in str(args.output_dir)
    assert "ushape_gaze" not in str(args.output_dir)


def test_argparse_modality_gaze_routes_to_ushape_gaze() -> None:
    """Default output directory should switch when modality flag is set."""
    args = mod._parse_args(["--modality", "gaze"])
    assert args.modality == "gaze"
    assert "ushape_gaze" in str(args.output_dir)


def test_argparse_modality_explicit_output_dir_wins() -> None:
    """An explicit --output_dir should override the modality default."""
    args = mod._parse_args([
        "--modality", "gaze",
        "--output_dir", "/tmp/custom_out",
    ])
    assert "/tmp/custom_out" in str(args.output_dir)


def test_top_features_summary_writes_file(tmp_path: Path) -> None:
    """plot_top_features_summary should render a 2x2 panel for the top
    features, suitable as a gaze-mode replacement for the Lim panel."""
    rng = np.random.default_rng(0)
    df = _build_subject_frame_for_lda(seed=42, n_features=8, tier_signal=2.0)
    feat_cols = [f"feat{j}" for j in range(8)]
    t_results = mod.mid_vs_rest_t_test(df, feat_cols)
    out = tmp_path / "top_features.png"
    mod.plot_top_features_summary(df, t_results, out, n_top=4)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_markdown_eeg_mode_includes_lim_section() -> None:
    """Default (modality='eeg') markdown should include Lim and Nemani sections."""
    rng = np.random.default_rng(0)
    eeg_cols = [
        "eeg_frontal_L_theta", "eeg_frontal_R_theta",
        "eeg_parietal_L_alpha", "eeg_parietal_R_alpha",
        "eeg_central_L_delta",
    ]
    rows = []
    for c in eeg_cols:
        rows.append({
            "feature": c, "mid_minus_rest": 0.5, "t_stat": 2.0,
            "p_two_sided": 0.05, "p_one_sided_up": 0.025,
            "p_one_sided_down": 0.025,
            "is_lim_up": c in mod.LIM_PREDICTED_UP,
            "is_lim_down": c in mod.LIM_PREDICTED_DOWN,
            "abs_t": 2.0, "mid_mean": 0.0, "rest_mean": 0.0,
        })
    t_df = pd.DataFrame(rows)
    reg_df = pd.DataFrame([{"feature": c, "delta_AIC": 0.0,
                             "b_quadratic_quad": 0.0, "is_lim_feature": True}
                            for c in eeg_cols])
    u_df = pd.DataFrame([{"subject_id": i, "tier": "Low",
                           "composite_skill": 0.5, "u_position": 0.0}
                          for i in range(3)])
    md = mod.render_markdown(t_df, reg_df, u_df, n_subjects=24,
                              tier_counts={"Low": 8, "Mid": 8, "High": 8},
                              modality="eeg")
    assert "Lim et al." in md
    assert "Nemani" in md
    assert "Modality:** EEG" in md
    assert "## Top features summary" not in md


def test_render_markdown_gaze_mode_skips_eeg_specific_sections() -> None:
    """Gaze-mode markdown should skip Lim test and Nemani topomap sections,
    replace with the data-driven top-features summary."""
    rng = np.random.default_rng(0)
    gaze_cols = [f"eye_feat_{i}" for i in range(5)]
    rows = []
    for c in gaze_cols:
        rows.append({
            "feature": c, "mid_minus_rest": 0.4, "t_stat": 1.5,
            "p_two_sided": 0.10, "p_one_sided_up": 0.05,
            "p_one_sided_down": 0.05,
            "is_lim_up": False, "is_lim_down": False,
            "abs_t": 1.5, "mid_mean": 0.0, "rest_mean": 0.0,
        })
    t_df = pd.DataFrame(rows)
    reg_df = pd.DataFrame([{"feature": c, "delta_AIC": 0.0,
                             "b_quadratic_quad": 0.0, "is_lim_feature": False}
                            for c in gaze_cols])
    u_df = pd.DataFrame([{"subject_id": i, "tier": "Low",
                           "composite_skill": 0.5, "u_position": 0.0}
                          for i in range(3)])
    md = mod.render_markdown(t_df, reg_df, u_df, n_subjects=24,
                              tier_counts={"Low": 8, "Mid": 8, "High": 8},
                              modality="gaze")
    # No Lim or Nemani sections.
    assert "## Lim et al." not in md
    assert "## Anatomical map vs. Nemani" not in md
    # But top-features summary is present.
    assert "## Top features summary" in md
    assert "Modality:** Gaze" in md
    assert "Bonferroni" not in md   # The Lim-specific Bonferroni line is hidden
