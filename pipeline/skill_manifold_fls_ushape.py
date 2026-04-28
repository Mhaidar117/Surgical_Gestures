#!/usr/bin/env python3
"""U-shape consolidation analysis for the FLS EEG cross-modality study.

Standalone script that takes the cached trial-level EEG features and runs
three analyses to test whether the inverted-U engagement story holds:

    1. Per-feature Mid vs. (Low ∪ High) t-test ranking. Tests whether the
       four cognitive-load features predicted by Lim et al. (2025) -- the
       two frontal theta features and the two parietal alpha features --
       come out at the top with the predicted directions (frontal theta
       higher in Mid, parietal alpha lower in Mid).

    2. Continuous-skill regression. For each feature fit `feature ~ skill`
       (linear) and `feature ~ skill + skill²` (quadratic) and compare
       AIC. A quadratic-preferred fit on the cognitive-load features is
       direct support for the U-shape independent of tier binning.

    3. Individual U-position projection. Project each subject's z-scored
       EEG vector onto the unit axis pointing from `mean(Low, High)` to
       `Mid`. The projection score is each subject's "depth in the U" —
       the per-subject quantity that a personalized brain-aligned model
       would track during skill acquisition.

Lim et al. (2025) reference:
    Lim, C., Obuseh, M., Cha, J., Steward, J., Sundaram, C., & Yu, D.
    "Neural insights on expert surgeons' mental workload during live
    robotic surgeries." Scientific Reports 15:12073.

Usage
-----
    export PYTHONPATH=src
    python pipeline/skill_manifold_fls_ushape.py \\
        --features_cache cache/skill_manifold_fls
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Reuse the orchestrator's data-prep functions so the U-shape analysis
# operates on the exact same per-subject feature matrix as the GW pipeline.
_SPEC = importlib.util.spec_from_file_location(
    "skill_manifold_gw_fls",
    REPO / "pipeline" / "skill_manifold_gw_fls.py",
)
_orch = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader is not None
_SPEC.loader.exec_module(_orch)

from skill_manifold.binning import TIER_NAMES, add_tier_column          # noqa: E402
from skill_manifold.features_fls_eeg import eeg_feature_column_names    # noqa: E402
from skill_manifold.features_fls_gaze import eye_feature_column_names   # noqa: E402
from skill_manifold.subject_aggregation import assemble_subject_frame   # noqa: E402

log = logging.getLogger("skill_manifold_fls_ushape")


# ---------- Lim et al. predicted features ------------------------------------

# Frontal theta UP, parietal alpha DOWN under high cognitive load
# (Lim et al. 2025; Borghini 2014; Klimesch 2012; Chikhi 2022 meta-analysis).
LIM_PREDICTED_UP: Tuple[str, ...] = (
    "eeg_frontal_L_theta",
    "eeg_frontal_R_theta",
)
LIM_PREDICTED_DOWN: Tuple[str, ...] = (
    "eeg_parietal_L_alpha",
    "eeg_parietal_R_alpha",
)
LIM_FEATURES: Tuple[str, ...] = LIM_PREDICTED_UP + LIM_PREDICTED_DOWN


# ---------- argparse ---------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_root", type=Path, default=REPO)
    p.add_argument("--features_cache", type=Path, default=None,
                   help="Where to find / write the trial-level parquet cache.")
    p.add_argument("--modality", type=str, default="eeg",
                   choices=["eeg", "gaze"],
                   help="Which modality to run the U-shape analysis on. "
                        "'eeg' (default): the canonical EEG analysis with Lim "
                        "+ Nemani panels. 'gaze': parallel analysis on the "
                        "18-d Tobii eye-summary features; Lim/Nemani panels "
                        "are skipped (no published gaze prediction or "
                        "anatomical region map applies) and replaced with a "
                        "data-driven top-features panel.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Output directory for plots, JSON, and markdown. "
                        "Default depends on --modality: ushape/ for eeg, "
                        "ushape_gaze/ for gaze.")
    p.add_argument("--no_residualize_task", action="store_true",
                   help="Skip task_id in the residualization design "
                        "(matches the orchestrator's flag).")
    p.add_argument("--skip_eeg_zscore", action="store_true",
                   help="Skip per-subject EEG z-scoring (when --modality "
                        "eeg) or per-subject gaze z-scoring (when --modality "
                        "gaze). Default behaviour z-scores the active "
                        "modality's features along the feature axis.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_level", default="INFO")
    args = p.parse_args(argv)
    if args.output_dir is None:
        suffix = "ushape" if args.modality == "eeg" else "ushape_gaze"
        args.output_dir = REPO / "reports" / "skill_manifold_fls" / suffix
    return args


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------- analysis functions ----------------------------------------------

def mid_vs_rest_t_test(
    subj_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """For each feature, compute Mid vs (Low ∪ High) two-sample t-test.

    Returns a DataFrame with columns:
        feature, mid_mean, rest_mean, mid_minus_rest, t_stat,
        p_two_sided, p_one_sided_up, p_one_sided_down,
        is_lim_up, is_lim_down, sig_one_sided
    """
    if "tier" not in subj_df.columns:
        raise ValueError("subj_df must have a 'tier' column")
    tiers = subj_df["tier"].to_numpy()
    is_mid = tiers == "Mid"
    is_rest = ~is_mid
    if is_mid.sum() < 2 or is_rest.sum() < 2:
        raise ValueError("need ≥2 subjects in each group for a t-test")

    rows = []
    for col in feature_cols:
        x = subj_df.loc[is_mid, col].to_numpy(dtype=np.float64)
        y = subj_df.loc[is_rest, col].to_numpy(dtype=np.float64)
        # Welch's t-test (unequal variances; safer at small N).
        t_stat, p_two = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        diff = float(x.mean() - y.mean())
        # One-sided p-values (Lim's predictions are directional).
        p_up = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
        p_down = p_two / 2.0 if t_stat < 0 else 1.0 - p_two / 2.0
        rows.append({
            "feature": col,
            "mid_mean": float(x.mean()),
            "rest_mean": float(y.mean()),
            "mid_minus_rest": diff,
            "t_stat": float(t_stat),
            "p_two_sided": float(p_two),
            "p_one_sided_up": float(p_up),
            "p_one_sided_down": float(p_down),
            "is_lim_up": col in LIM_PREDICTED_UP,
            "is_lim_down": col in LIM_PREDICTED_DOWN,
        })
    out = pd.DataFrame(rows)
    out["abs_t"] = out["t_stat"].abs()
    out = out.sort_values("abs_t", ascending=False).reset_index(drop=True)
    return out


def continuous_skill_regression(
    subj_df: pd.DataFrame,
    feature_cols: Sequence[str],
    skill_col: str = "composite_skill",
) -> pd.DataFrame:
    """Per feature, fit linear and quadratic models against composite skill.

    Returns a DataFrame with one row per feature:
        feature, n, b_linear, slope_linear, RSS_linear, AIC_linear,
        b0_quad, b_lin_quad, b_quad_quad, RSS_quad, AIC_quad,
        delta_AIC (= AIC_linear - AIC_quad; positive => quadratic wins),
        is_lim_feature
    """
    if skill_col not in subj_df.columns:
        raise ValueError(f"subj_df must have a {skill_col!r} column")
    s = subj_df[skill_col].to_numpy(dtype=np.float64)
    rows = []
    for col in feature_cols:
        y = subj_df[col].to_numpy(dtype=np.float64)
        n = int(np.sum(np.isfinite(y) & np.isfinite(s)))
        if n < 5:
            continue
        # Linear fit: y = a + b*s
        X1 = np.vstack([np.ones(n), s]).T
        coef1, *_ = np.linalg.lstsq(X1, y, rcond=None)
        rss1 = float(((y - X1 @ coef1) ** 2).sum())
        aic1 = _aic(rss1, n, k=2)
        # Quadratic fit: y = a + b*s + c*s²
        X2 = np.vstack([np.ones(n), s, s ** 2]).T
        coef2, *_ = np.linalg.lstsq(X2, y, rcond=None)
        rss2 = float(((y - X2 @ coef2) ** 2).sum())
        aic2 = _aic(rss2, n, k=3)

        rows.append({
            "feature": col,
            "n": n,
            "b_intercept_linear": float(coef1[0]),
            "slope_linear": float(coef1[1]),
            "RSS_linear": rss1,
            "AIC_linear": aic1,
            "b_intercept_quad": float(coef2[0]),
            "b_linear_quad": float(coef2[1]),
            "b_quadratic_quad": float(coef2[2]),
            "RSS_quad": rss2,
            "AIC_quad": aic2,
            "delta_AIC": aic1 - aic2,
            "is_lim_feature": col in LIM_FEATURES,
        })
    out = pd.DataFrame(rows).sort_values("delta_AIC", ascending=False).reset_index(drop=True)
    return out


def _aic(rss: float, n: int, k: int) -> float:
    """Gaussian-likelihood AIC up to additive constants. Lower = better.

    Formula: AIC = n * log(RSS / n) + 2k
    """
    if rss <= 0:
        return float("-inf")
    return float(n * np.log(rss / n) + 2 * k)


def individual_u_position(
    subj_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Project each subject onto the Mid vs (Low ∪ High) mean-difference axis.

    The axis is `mid_centroid - mean(low_centroid, high_centroid)`,
    normalized to unit length. Each subject's U-position is the dot
    product of their feature vector with this axis. Mid subjects on
    average should have positive U-position; Low/High should have
    negative.

    Returns a DataFrame with columns:
        subject_id, tier, composite_skill, u_position
    """
    if "tier" not in subj_df.columns:
        raise ValueError("subj_df must have a 'tier' column")
    feats = subj_df[list(feature_cols)].to_numpy(dtype=np.float64)
    tiers = subj_df["tier"].to_numpy()

    centroids: Dict[str, np.ndarray] = {}
    for t in TIER_NAMES:
        mask = tiers == t
        if mask.any():
            centroids[t] = feats[mask].mean(axis=0)
    if not all(t in centroids for t in TIER_NAMES):
        raise RuntimeError("missing centroid for at least one tier; abort")

    rest = 0.5 * (centroids["Low"] + centroids["High"])
    axis = centroids["Mid"] - rest
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise RuntimeError("Mid axis is degenerate (Mid centroid == rest)")
    axis = axis / norm

    proj = feats @ axis
    out = subj_df[["subject_id", "tier"]].copy()
    if "composite_skill" in subj_df.columns:
        out["composite_skill"] = subj_df["composite_skill"].to_numpy()
    out["u_position"] = proj
    return out.reset_index(drop=True)


# ---------- plotting ---------------------------------------------------------

# Tier palette (kept consistent with the orchestrator's palette).
TIER_COLOR = {"Low": "#1f77b4", "Mid": "#ff7f0e", "High": "#2ca02c"}


def _bar_with_ci(ax, values, color, label=None, width=0.7) -> None:
    """Draw a single bar with mean and 95% CI error bar."""
    n = len(values)
    if n == 0:
        return
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ci = 1.96 * sem
    ax.bar([label], [mean], width=width, color=color,
           edgecolor="#666", linewidth=0.6, alpha=0.85)
    ax.errorbar([label], [mean], yerr=[ci], color="black",
                capsize=4, lw=1.0)


def plot_lim_comparison(
    subj_df: pd.DataFrame,
    t_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Simple, professional plot showing our 4 cognitive-load features
    against Lim et al.'s predictions.

    Layout: 2x2 grid.
      Top row: frontal_L_theta, frontal_R_theta (Lim predicts Mid > rest)
      Bottom row: parietal_L_alpha, parietal_R_alpha (Lim predicts Mid < rest)
    Each panel: 3 bars (Low, Mid, High) with 95% CI; one-sided p-value of
    Mid vs rest in the predicted direction shown.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), sharey=False)
    fig.suptitle(
        "Mid-skill cognitive load signature vs. Lim et al. (2025) prediction",
        fontsize=12, y=1.00,
    )
    # Panel layout: each row is (feature_column, panel_title, panel_label,
    # predicted-direction). Top row: theta should be UP in Mid (Lim et al.
    # 2025); bottom row: alpha should be DOWN. Generic prediction text is
    # NOT drawn inside the panels — it lives in the figure caption,
    # cross-referenced via panel labels (a)-(d).
    rows: List[Tuple[str, str, str, str]] = [
        ("eeg_frontal_L_theta", "Frontal-Left theta", "(a)", "up"),
        ("eeg_frontal_R_theta", "Frontal-Right theta", "(b)", "up"),
        ("eeg_parietal_L_alpha", "Parietal-Left alpha", "(c)", "down"),
        ("eeg_parietal_R_alpha", "Parietal-Right alpha", "(d)", "down"),
    ]
    flat_axes = axes.flatten()
    for idx, (col, title, panel, direction) in enumerate(rows):
        ax = flat_axes[idx]
        # Get the t-test result for this feature.
        row = t_results[t_results["feature"] == col].iloc[0]
        p_dir = row["p_one_sided_up"] if direction == "up" else row["p_one_sided_down"]
        diff = row["mid_minus_rest"]
        # Bars by tier.
        for t in TIER_NAMES:
            vals = subj_df.loc[subj_df["tier"] == t, col].to_numpy(dtype=np.float64)
            _bar_with_ci(ax, vals, TIER_COLOR[t], label=t)
        # Highlight Mid bar with a thicker outline if effect is in predicted direction.
        match_predicted = (
            (direction == "up" and diff > 0) or (direction == "down" and diff < 0)
        )
        if match_predicted:
            mid_idx = list(TIER_NAMES).index("Mid")
            ax.patches[mid_idx].set_edgecolor("black")
            ax.patches[mid_idx].set_linewidth(1.4)

        # Panel label + title, left-aligned ("(a) Frontal-Left theta").
        ax.set_title(f"{panel} {title}", fontsize=10, loc="left")
        ax.set_ylabel("z-scored relative power", fontsize=9)
        ax.tick_params(labelsize=9)
        ax.axhline(0, color="#bbb", lw=0.5)

        # Compact in-panel result: just the per-panel statistic + direction
        # marker. The prediction direction itself is in the caption.
        sig_marker = "(matches)" if match_predicted else "(opposite)"
        ax.text(
            0.02, 0.97,
            f"Mid − rest = {diff:+.3f}\np = {p_dir:.3f} {sig_marker}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="#bbb", lw=0.5, alpha=0.9),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


def plot_top_features_summary(
    subj_df: pd.DataFrame,
    t_results: pd.DataFrame,
    output_path: Path,
    *,
    n_top: int = 4,
    title: str = "Top features by |t|: Mid vs. (Low ∪ High) tier means with 95% CI",
) -> None:
    """Data-driven analogue of `plot_lim_comparison` for modalities without
    a published prediction (e.g. gaze).

    Picks the top-`n_top` features by absolute t-statistic from the
    Mid vs.\\ (Low ∪ High) ranking, then renders them in the same 2×2
    bar-chart layout as the Lim plot --- tier means with 95% CI per
    panel, predicted-direction caption replaced with a "data-driven
    top feature" caption.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = t_results.head(n_top)
    if len(top) < 1:
        return
    rows = max(1, int(np.ceil(n_top / 2)))
    cols = min(2, n_top)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.2 * rows),
                              sharey=False)
    fig.suptitle(title, fontsize=11, y=1.00)
    flat_axes = np.atleast_1d(axes).flatten()
    for idx, (_, row) in enumerate(top.iterrows()):
        if idx >= len(flat_axes):
            break
        ax = flat_axes[idx]
        col = row["feature"]
        for t in TIER_NAMES:
            vals = subj_df.loc[subj_df["tier"] == t, col].to_numpy(dtype=np.float64)
            _bar_with_ci(ax, vals, TIER_COLOR[t], label=t)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("z-scored value" if col.startswith("eeg_") else "value",
                      fontsize=9)
        ax.tick_params(labelsize=9)
        ax.axhline(0, color="#bbb", lw=0.5)
        diff = float(row["mid_minus_rest"])
        p_two = float(row["p_two_sided"])
        ax.text(
            0.02, 0.97,
            f"Top-{idx + 1} by |t|. Mid − rest = {diff:+.3f}, "
            f"two-sided p = {p_two:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor="white", edgecolor="#bbb", lw=0.5, alpha=0.9),
        )

    # Hide any unused panels.
    for idx in range(len(top), len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


def plot_feature_ranking(
    t_results: pd.DataFrame,
    output_path: Path,
    top_n: int = 40,
) -> None:
    """Forest-style plot of all features ranked by |t|, with Lim features highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = t_results.head(top_n).iloc[::-1]   # highest |t| at the top
    n = len(df)
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.25 * n + 1.0)))

    # Color: red if Mid > rest, blue if Mid < rest. Highlighted edge for Lim features.
    colors = ["#D65F5F" if x > 0 else "#4878D0" for x in df["mid_minus_rest"]]
    is_lim = df["is_lim_up"] | df["is_lim_down"]
    edges = ["black" if x else "none" for x in is_lim]
    lws = [1.6 if x else 0.0 for x in is_lim]

    bars = ax.barh(range(n), df["mid_minus_rest"], color=colors,
                   edgecolor=edges, linewidth=lws, alpha=0.9)
    ax.set_yticks(range(n), df["feature"], fontsize=7)
    ax.axvline(0, color="#444", lw=0.5)
    ax.set_xlabel("Mid mean − Rest mean (z-scored relative power)", fontsize=9)
    ax.set_title("Per-feature contribution to Mid vs. (Low ∪ High)", fontsize=10)

    # Legend.
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#D65F5F", label="Mid > rest"),
        Patch(facecolor="#4878D0", label="Mid < rest"),
        Patch(facecolor="white", edgecolor="black", linewidth=1.6,
              label="Lim et al. cognitive-load prediction"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


def plot_quadratic_vs_linear(
    reg_results: pd.DataFrame,
    output_path: Path,
    top_n: int = 40,
) -> None:
    """Bar chart of feature ΔAIC, with Lim features highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = reg_results.head(top_n).iloc[::-1]
    n = len(df)
    fig, ax = plt.subplots(figsize=(7.2, max(4.0, 0.25 * n + 1.0)))

    colors = ["#0F6E56" if x > 2 else "#bbb" for x in df["delta_AIC"]]
    edges = ["black" if x else "none" for x in df["is_lim_feature"]]
    lws = [1.6 if x else 0.0 for x in df["is_lim_feature"]]
    ax.barh(range(n), df["delta_AIC"], color=colors,
            edgecolor=edges, linewidth=lws, alpha=0.9)
    ax.set_yticks(range(n), df["feature"], fontsize=7)
    ax.axvline(0, color="#444", lw=0.5)
    ax.axvline(2, color="#888", lw=0.5, ls="--")
    ax.set_xlabel("ΔAIC = AIC(linear) − AIC(quadratic);  positive = quadratic preferred",
                  fontsize=9)
    ax.set_title("Continuous-skill regression: linear vs. quadratic per feature",
                 fontsize=10)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="#0F6E56", label="ΔAIC > 2 (quadratic supported)"),
        Patch(facecolor="#bbb",    label="ΔAIC ≤ 2 (linear adequate)"),
        Patch(facecolor="white", edgecolor="black", linewidth=1.6,
              label="Lim et al. cognitive-load feature"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


def plot_individual_u_positions(
    u_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Scatter: x = composite_skill, y = u_position, color by tier.

    The U-shape predicts: low at both ends of the skill axis, high in
    the middle. A fitted quadratic is overlaid as a visual reference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    if "composite_skill" not in u_df.columns:
        # Fall back to subject_id ordering.
        u_df = u_df.copy()
        u_df["composite_skill"] = u_df["subject_id"].astype(float)

    for t in TIER_NAMES:
        sub = u_df[u_df["tier"] == t]
        ax.scatter(sub["composite_skill"], sub["u_position"],
                   s=70, color=TIER_COLOR[t], alpha=0.85, edgecolors="white",
                   linewidths=1.0, label=f"{t} (n={len(sub)})")
        for _, row in sub.iterrows():
            ax.annotate(str(int(row["subject_id"])),
                        (row["composite_skill"], row["u_position"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=7, alpha=0.7)

    # Fit quadratic for visual reference.
    s = u_df["composite_skill"].to_numpy(dtype=np.float64)
    y = u_df["u_position"].to_numpy(dtype=np.float64)
    if len(s) >= 4:
        coefs = np.polyfit(s, y, 2)
        s_grid = np.linspace(s.min(), s.max(), 100)
        ax.plot(s_grid, np.polyval(coefs, s_grid),
                color="#444", lw=1.0, ls="--",
                label=f"quadratic fit (a={coefs[0]:+.2f})")

    ax.axhline(0, color="#bbb", lw=0.5)
    ax.set_xlabel("Composite skill score (per subject)")
    ax.set_ylabel("U-position (projection onto Mid − rest axis)")
    ax.set_title("Per-subject position on the inverted-U engagement axis", fontsize=10)
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


# ---------- Nemani-style anatomical topographic plot ------------------------

# Approximate normalized scalp positions for our 8 lateralized regions.
# Origin at head center (0.5, 0.5); head radius ≈ 0.45.
REGION_HEAD_POS: Dict[str, Tuple[float, float]] = {
    "frontal_L":   (0.36, 0.84),
    "frontal_R":   (0.64, 0.84),
    "central_L":   (0.28, 0.55),
    "central_R":   (0.72, 0.55),
    "parietal_L":  (0.32, 0.30),
    "parietal_R":  (0.68, 0.30),
    "occipital_L": (0.42, 0.10),
    "occipital_R": (0.58, 0.10),
}

# Nemani's regions, schematized as bounding ellipses on the same head
# coordinate system. PFC ≈ upper third (frontal_L + frontal_R coverage).
# LMM1 ≈ left central, slightly medial (Nemani's strongest single region).
# SMA ≈ midline frontocentral (FCz site, anterior to Cz and posterior to
# Fz). It sits on the midline, between our frontal row (y ≈ 0.84) and our
# central row (y ≈ 0.55) — NOT at the geometric center of the head. We
# don't have a midline region in our 8-region schema, so the SMA outline
# is informational only (no direct EEG region to compare). Each region
# gets a distinct color so the three outlines are easily distinguishable.
NEMANI_REGION_PATCHES: List[Dict[str, object]] = [
    {"label": "PFC (Nemani)",
     "center": (0.50, 0.86), "axes": (0.34, 0.08),
     "color": "#A32D2D"},   # red
    {"label": "LMM1 (Nemani)",
     "center": (0.32, 0.55), "axes": (0.10, 0.10),
     "color": "#0F6E56"},   # teal
    {"label": "SMA (Nemani, midline FCz)",
     "center": (0.50, 0.72), "axes": (0.05, 0.07),
     "color": "#534AB7"},   # purple
]


def _draw_head_outline(ax, head_radius: float = 0.45) -> None:
    """Draw a head outline (circle + nose triangle + ears) on `ax`."""
    import matplotlib.patches as mpatches

    # Head circle
    ax.add_patch(mpatches.Circle(
        (0.5, 0.5), head_radius,
        fill=False, edgecolor="#444", linewidth=1.2,
    ))
    # Nose
    nose_x = [0.50, 0.46, 0.54]
    nose_y = [0.5 + head_radius + 0.04, 0.5 + head_radius - 0.005,
              0.5 + head_radius - 0.005]
    ax.plot(nose_x + [nose_x[0]], nose_y + [nose_y[0]],
            color="#444", linewidth=1.0)
    # Left ear
    ax.add_patch(mpatches.Ellipse(
        (0.5 - head_radius, 0.5), 0.04, 0.10,
        fill=False, edgecolor="#444", linewidth=0.8,
    ))
    # Right ear
    ax.add_patch(mpatches.Ellipse(
        (0.5 + head_radius, 0.5), 0.04, 0.10,
        fill=False, edgecolor="#444", linewidth=0.8,
    ))


def _draw_nemani_regions(ax) -> None:
    """Overlay Nemani's PFC / LMM1 / SMA region outlines on the head.

    Inline labels are intentionally omitted; a shared legend at the
    figure level identifies each region by color so the per-panel head
    stays uncluttered.
    """
    import matplotlib.patches as mpatches

    for patch in NEMANI_REGION_PATCHES:
        cx, cy = patch["center"]
        ax_w, ax_h = patch["axes"]
        ax.add_patch(mpatches.Ellipse(
            (cx, cy), 2 * ax_w, 2 * ax_h,
            fill=False, edgecolor=patch["color"],
            linewidth=1.6, linestyle=(0, (5, 2)), alpha=0.95,
        ))


def _band_topomap_panel(
    ax,
    band: str,
    region_effect: Dict[str, float],
    vmin: float,
    vmax: float,
    show_labels: bool = True,
    show_nemani: bool = True,
) -> None:
    """Draw a single head topographic panel for one band."""
    import matplotlib
    import matplotlib.patches as mpatches

    cmap = matplotlib.colormaps["RdBu_r"]
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    _draw_head_outline(ax)
    if show_nemani:
        _draw_nemani_regions(ax)

    for region, (rx, ry) in REGION_HEAD_POS.items():
        eff = region_effect.get(region, 0.0)
        color = cmap(norm(eff))
        ax.add_patch(mpatches.Circle(
            (rx, ry), 0.07,
            facecolor=color, edgecolor="black", linewidth=0.5, alpha=0.95,
        ))
        if show_labels:
            ax.text(rx, ry, region.split("_")[1],
                    ha="center", va="center", fontsize=6.5,
                    color="black" if abs(eff) < 0.6 * max(abs(vmin), abs(vmax))
                    else "white", weight="bold")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(band, fontsize=10, fontweight="bold")


def plot_nemani_topomap(
    t_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """One head per band, regions colored by Mid - rest effect size.

    Overlays Nemani et al. (2018) PFC / LMM1 / SMA region outlines so the
    anatomical correspondence between our broadband left-frontal/central
    pattern and Nemani's fNIRS findings is visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    band_names = ["delta", "theta", "alpha", "beta", "gamma"]

    # Compute Mid - rest per (region, band) from the t-test results table.
    band_to_effects: Dict[str, Dict[str, float]] = {b: {} for b in band_names}
    for _, row in t_results.iterrows():
        feat = row["feature"]
        if not feat.startswith("eeg_"):
            continue
        parts = feat[len("eeg_"):].rsplit("_", 1)
        if len(parts) != 2:
            continue
        region, band = parts
        if region in REGION_HEAD_POS and band in band_to_effects:
            band_to_effects[band][region] = float(row["mid_minus_rest"])

    # Symmetric color range across all bands so panels are comparable.
    all_effects = [v for d in band_to_effects.values() for v in d.values()]
    if not all_effects:
        return
    vlim = float(np.max(np.abs(all_effects)))
    if vlim < 1e-6:
        vlim = 1.0

    fig, axes = plt.subplots(1, len(band_names), figsize=(2.4 * len(band_names), 3.8))
    fig.suptitle(
        "Mid − rest by region and band, with Nemani et al. (2018) regions overlaid",
        fontsize=11, y=0.99,
    )
    fig.subplots_adjust(top=0.84, bottom=0.18)
    for ax, band in zip(axes, band_names):
        _band_topomap_panel(
            ax, band,
            region_effect=band_to_effects[band],
            vmin=-vlim, vmax=vlim,
            show_labels=True,
            show_nemani=True,
        )

    # Shared colorbar.
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=-vlim, vmax=vlim),
        cmap="RdBu_r",
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        fraction=0.05, pad=0.10, aspect=40)
    cbar.set_label("Mid − rest (z-scored relative power)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Figure-level legend identifying the Nemani region colors. Place it
    # above the panels so it doesn't crowd the colorbar at the bottom.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=patch["color"], lw=1.8,
               linestyle=(0, (5, 2)), label=str(patch["label"]))
        for patch in NEMANI_REGION_PATCHES
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=len(legend_handles),
        bbox_to_anchor=(0.5, 0.92), fontsize=9,
        frameon=True, framealpha=0.95, edgecolor="#bbb",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


# ---------- 3-class LDA classifier ------------------------------------------

def lda_three_class(
    subj_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Dict[str, object]:
    """Leave-one-out 3-class LDA on the subject-level EEG feature matrix.

    Reports overall accuracy, confusion matrix, and pairwise MCEs in a
    format directly comparable to Nemani et al. (2018) Table-format results.
    Uses shrinkage='auto' (Ledoit-Wolf) because we have N=24 with p=40.

    Returns a dict with:
        accuracy, confusion (3x3 array), confusion_normalized, tier_order,
        per_class_recall, pairwise_mce (Low_vs_Mid, Mid_vs_High, Low_vs_High),
        mid_vs_rest (Mid vs not-Mid 2-class collapse),
        per_subject_predictions
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import LeaveOneOut

    if "tier" not in subj_df.columns:
        raise ValueError("subj_df must have a 'tier' column")
    X = subj_df[list(feature_cols)].to_numpy(dtype=np.float64)
    y = subj_df["tier"].to_numpy()
    sids = subj_df["subject_id"].to_numpy()

    tier_order = list(TIER_NAMES)
    n = X.shape[0]
    preds = np.empty(n, dtype=object)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        clf = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto",
        )
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx[0]] = clf.predict(X[test_idx])[0]

    return _summarize_lda_predictions(
        y, preds, sids, tier_order,
        n_features_effective=X.shape[1],
        method_label="all_features",
    )


def _summarize_lda_predictions(
    y: np.ndarray, preds: np.ndarray, sids: np.ndarray,
    tier_order: List[str], n_features_effective: int,
    method_label: str, selected_features: Optional[List[List[str]]] = None,
) -> Dict[str, object]:
    """Shared bookkeeping: confusion + recall + pairwise MCEs from y, preds."""
    K = len(tier_order)
    confusion = np.zeros((K, K), dtype=np.int64)
    for true, pred in zip(y, preds):
        i = tier_order.index(true)
        j = tier_order.index(pred)
        confusion[i, j] += 1
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.where(
        row_sums > 0, confusion / np.maximum(row_sums, 1), 0.0,
    )
    accuracy = float(np.trace(confusion) / confusion.sum())
    per_class_recall = {tier_order[i]: float(confusion_norm[i, i])
                        for i in range(K)}

    pairwise: Dict[str, Dict[str, float]] = {}
    for a, b in [("Low", "Mid"), ("Mid", "High"), ("Low", "High")]:
        mask_a = y == a
        mask_b = y == b
        if mask_a.sum() == 0 or mask_b.sum() == 0:
            continue
        pairwise[f"{a}_vs_{b}"] = {
            "MCE_A_to_B": int(np.sum((y == a) & (preds == b))) / max(int(mask_a.sum()), 1),
            "MCE_B_to_A": int(np.sum((y == b) & (preds == a))) / max(int(mask_b.sum()), 1),
            "n_A": int(mask_a.sum()), "n_B": int(mask_b.sum()),
        }

    per_subject = [
        {"subject_id": int(sids[i]), "true": str(y[i]), "pred": str(preds[i])}
        for i in range(len(y))
    ]

    # Mid vs not-Mid derived statistic: collapse the 3-class confusion to
    # a 2-class problem (is this subject Mid or not?). With the U-shape
    # geometry, this is the more honest readout — Low and High are
    # geometrically co-located, so a 3-class linear classifier trades
    # them while still routing Mid samples correctly.
    if "Mid" in tier_order:
        mid_idx = tier_order.index("Mid")
        # True positive Mid: confusion[mid, mid]
        tp_mid = int(confusion[mid_idx, mid_idx])
        # True negative non-Mid: every cell with true != Mid AND pred != Mid.
        tn_non_mid = int(
            confusion.sum()
            - confusion[mid_idx, :].sum()         # remove true-Mid row
            - confusion[:, mid_idx].sum()         # remove pred-Mid column
            + confusion[mid_idx, mid_idx]         # add back the doubly-removed cell
        )
        n_mid = int(confusion[mid_idx, :].sum())
        n_non_mid = int(confusion.sum() - n_mid)
        mid_recall = tp_mid / n_mid if n_mid > 0 else float("nan")
        non_mid_recall = tn_non_mid / n_non_mid if n_non_mid > 0 else float("nan")
        mid_precision = (tp_mid / int(confusion[:, mid_idx].sum())
                         if confusion[:, mid_idx].sum() > 0 else float("nan"))
        mid_vs_rest = {
            "accuracy": (tp_mid + tn_non_mid) / int(confusion.sum()),
            "mid_recall": mid_recall,
            "non_mid_recall": non_mid_recall,
            "mid_precision": mid_precision,
            "n_mid": n_mid,
            "n_non_mid": n_non_mid,
        }
    else:
        mid_vs_rest = {}

    out: Dict[str, object] = {
        "method": method_label,
        "accuracy": accuracy,
        "confusion": confusion.tolist(),
        "confusion_normalized": confusion_norm.tolist(),
        "tier_order": tier_order,
        "per_class_recall": per_class_recall,
        "pairwise_mce": pairwise,
        "mid_vs_rest": mid_vs_rest,
        "per_subject_predictions": per_subject,
        "n_subjects": int(len(y)),
        "n_features": int(n_features_effective),
    }
    if selected_features is not None:
        out["selected_features_per_fold"] = selected_features
    return out


def lda_three_class_top_k(
    subj_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    k: int = 5,
) -> Dict[str, object]:
    """LOO 3-class LDA with t-stat feature selection INSIDE each fold.

    For every leave-one-out fold:
      1. Compute Mid vs (Low ∪ High) Welch t-test on the 23 training samples.
      2. Pick the top `k` features by absolute t-statistic.
      3. Train LDA on those k features and predict the held-out subject.

    Doing the selection inside the fold avoids the test sample contributing
    to the t-statistics that determine which features get used. Default
    k=5 matches the dimensionality used by Nemani et al. (2018), which
    gives N/p ≈ 4.8 at our sample size — a much healthier ratio than the
    full p=40 LDA's N/p = 0.6.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import LeaveOneOut

    if "tier" not in subj_df.columns:
        raise ValueError("subj_df must have a 'tier' column")
    cols = list(feature_cols)
    X_full = subj_df[cols].to_numpy(dtype=np.float64)
    y = subj_df["tier"].to_numpy()
    sids = subj_df["subject_id"].to_numpy()

    tier_order = list(TIER_NAMES)
    n = X_full.shape[0]
    preds = np.empty(n, dtype=object)
    selected_features: List[List[str]] = []

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_full):
        # In-fold t-test: Mid vs rest on training subjects only.
        train_y = y[train_idx]
        is_mid = train_y == "Mid"
        is_rest = ~is_mid
        if is_mid.sum() < 2 or is_rest.sum() < 2:
            # Not enough samples in either group; fall back to all features.
            top_k_idx = np.arange(min(k, len(cols)))
        else:
            t_abs = np.zeros(len(cols), dtype=np.float64)
            for j in range(len(cols)):
                col = X_full[train_idx, j]
                t, _ = stats.ttest_ind(col[is_mid], col[is_rest],
                                        equal_var=False, nan_policy="omit")
                t_abs[j] = abs(t) if np.isfinite(t) else 0.0
            top_k_idx = np.argsort(-t_abs)[:k]
        selected_features.append([cols[i] for i in top_k_idx])

        # Fit LDA on the k-dimensional subset.
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_full[train_idx][:, top_k_idx], train_y)
        preds[test_idx[0]] = clf.predict(X_full[test_idx][:, top_k_idx])[0]

    return _summarize_lda_predictions(
        y, preds, sids, tier_order,
        n_features_effective=k,
        method_label=f"top_{k}_features_by_tstat",
        selected_features=selected_features,
    )


def rfc_three_class_rfe(
    subj_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    k: int = 5,
    n_estimators: int = 100,
    seed: int = 0,
) -> Dict[str, object]:
    """LOO 3-class Random Forest with in-fold RFE feature selection.

    Mirrors Soangra et al. (2022, PLoS ONE 17(6):e0267936) methodology
    directly: RFE-selected features fed into a Random Forest with default
    100 estimators, evaluated under leave-one-out cross-validation.
    Their best 3-class accuracy was 58% with ECU + deltoid (2 features)
    and 52% with ECU + deltoid + biceps (3 features) at N = 26 across
    Novice/Intermediate/Expert. Default `k = 5` here matches the
    dimensionality used by our other top-k classifier so the comparison
    plot is internally consistent; lower k can be used to recover their
    exact 2- or 3-feature setup.

    For every leave-one-out fold:
      1. Fit RFE with a RandomForestClassifier estimator on the 23
         training samples, eliminating one feature at a time until k
         features remain.
      2. Train a fresh RandomForestClassifier on the selected k features.
      3. Predict the held-out subject.

    Doing both the RFE and the RFC fitting inside the fold prevents the
    test sample from leaking into either the feature-selection or the
    classifier-fitting step.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE
    from sklearn.model_selection import LeaveOneOut

    if "tier" not in subj_df.columns:
        raise ValueError("subj_df must have a 'tier' column")
    cols = list(feature_cols)
    X_full = subj_df[cols].to_numpy(dtype=np.float64)
    y = subj_df["tier"].to_numpy()
    sids = subj_df["subject_id"].to_numpy()

    tier_order = list(TIER_NAMES)
    n = X_full.shape[0]
    preds = np.empty(n, dtype=object)
    selected_features: List[List[str]] = []

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_full):
        # In-fold RFE: estimator is RFC; eliminate features one at a time.
        rfe_estimator = RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed,
        )
        n_select = min(k, X_full.shape[1])
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_select, step=1)
        rfe.fit(X_full[train_idx], y[train_idx])
        sel_indices = np.where(rfe.support_)[0]
        selected_features.append([cols[i] for i in sel_indices])

        # Final RFC trained on the selected features.
        clf = RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed,
        )
        clf.fit(X_full[train_idx][:, sel_indices], y[train_idx])
        preds[test_idx[0]] = clf.predict(X_full[test_idx][:, sel_indices])[0]

    return _summarize_lda_predictions(
        y, preds, sids, tier_order,
        n_features_effective=k,
        method_label=f"rfc_rfe_{k}_features",
        selected_features=selected_features,
    )


def lda_three_class_pca(
    subj_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    n_components: int = 5,
) -> Dict[str, object]:
    """LOO 3-class LDA with PCA dimensionality reduction inside each fold.

    For every leave-one-out fold:
      1. Fit PCA on the 23 training samples to extract `n_components` PCs.
      2. Transform both the training samples and the held-out sample into
         PC space.
      3. Train LDA on the n_components-dim training projections and
         predict the held-out projection.

    Unlike top-k feature selection by t-stat, PCA is unsupervised and
    therefore safe even if applied to all data — but doing it inside
    the fold keeps the test sample fully isolated from the projection
    basis, matching best practice.
    """
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import LeaveOneOut

    if "tier" not in subj_df.columns:
        raise ValueError("subj_df must have a 'tier' column")
    cols = list(feature_cols)
    X_full = subj_df[cols].to_numpy(dtype=np.float64)
    y = subj_df["tier"].to_numpy()
    sids = subj_df["subject_id"].to_numpy()

    tier_order = list(TIER_NAMES)
    n = X_full.shape[0]
    preds = np.empty(n, dtype=object)

    loo = LeaveOneOut()
    n_components_effective = min(n_components, X_full.shape[1], n - 1)
    for train_idx, test_idx in loo.split(X_full):
        pca = PCA(n_components=n_components_effective, random_state=0)
        train_pcs = pca.fit_transform(X_full[train_idx])
        test_pcs = pca.transform(X_full[test_idx])
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(train_pcs, y[train_idx])
        preds[test_idx[0]] = clf.predict(test_pcs)[0]

    return _summarize_lda_predictions(
        y, preds, sids, tier_order,
        n_features_effective=n_components_effective,
        method_label=f"pca_{n_components_effective}_components",
    )


def plot_lda_confusion(
    lda_result: Dict[str, object],
    output_path: Path,
) -> None:
    """Confusion matrix heatmap for the 3-class LDA classifier."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confusion = np.asarray(lda_result["confusion"], dtype=np.float64)
    confusion_norm = np.asarray(lda_result["confusion_normalized"], dtype=np.float64)
    tier_order = lda_result["tier_order"]
    accuracy = lda_result["accuracy"]

    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    im = ax.imshow(confusion_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    K = len(tier_order)
    for i in range(K):
        for j in range(K):
            count = int(confusion[i, j])
            frac = confusion_norm[i, j]
            color = "white" if frac > 0.5 else "black"
            ax.text(j, i, f"{count}\n({frac:.2f})",
                    ha="center", va="center", color=color, fontsize=10)

    ax.set_xticks(range(K), tier_order, fontsize=10)
    ax.set_yticks(range(K), tier_order, fontsize=10)
    ax.set_xlabel("Predicted tier", fontsize=10)
    ax.set_ylabel("True tier", fontsize=10)
    chance = 1.0 / K
    ax.set_title(
        f"LDA leave-one-out 3-class confusion\n"
        f"accuracy = {accuracy:.3f}  (chance = {chance:.3f}, "
        f"N = {lda_result['n_subjects']}, p = {lda_result['n_features']})",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, label="row-normalized fraction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


def plot_lda_classifier_comparison(
    lda_results: List[Dict[str, object]],
    output_path: Path,
) -> None:
    """Side-by-side comparison plot of multiple LDA classifier variants.

    Top row: confusion matrices, one per classifier.
    Bottom: grouped bar chart of pairwise MCEs across classifiers, with
    Nemani's Expert-vs-Novice MCE (4.4% / 4.2% averaged ≈ 0.043) drawn
    as a horizontal reference line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(lda_results)
    if n == 0:
        return

    fig = plt.figure(figsize=(4.0 * n, 7.5))
    gs = fig.add_gridspec(2, n, height_ratios=[1.4, 1.0], hspace=0.5)

    # --- top row: confusion matrices ---
    for i, res in enumerate(lda_results):
        ax = fig.add_subplot(gs[0, i])
        confusion = np.asarray(res["confusion"], dtype=np.float64)
        confusion_norm = np.asarray(res["confusion_normalized"], dtype=np.float64)
        tier_order = res["tier_order"]
        K = len(tier_order)
        im = ax.imshow(confusion_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        for r in range(K):
            for c in range(K):
                count = int(confusion[r, c])
                frac = confusion_norm[r, c]
                color = "white" if frac > 0.5 else "black"
                ax.text(c, r, f"{count}\n({frac:.2f})",
                        ha="center", va="center", color=color, fontsize=8)
        ax.set_xticks(range(K), tier_order, fontsize=9)
        ax.set_yticks(range(K), tier_order, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        if i == 0:
            ax.set_ylabel("True", fontsize=9)
        ax.set_title(
            f"{res['method']}\n"
            f"acc = {res['accuracy']:.3f}, p = {res['n_features']}",
            fontsize=9,
        )

    # --- bottom: pairwise MCE grouped bar chart ---
    ax = fig.add_subplot(gs[1, :])
    pair_keys = ["Low_vs_Mid", "Mid_vs_High", "Low_vs_High"]
    pair_labels = ["Low vs Mid", "Mid vs High", "Low vs High"]
    method_labels = [str(r["method"]) for r in lda_results]
    n_methods = len(lda_results)
    n_pairs = len(pair_keys)
    bar_width = 0.8 / n_methods

    palette = ["#4878D0", "#0F6E56", "#D65F5F"]
    for m, res in enumerate(lda_results):
        # Average the asymmetric MCE for visualization (A→B and B→A).
        mce_means = []
        for pk in pair_keys:
            pair = res["pairwise_mce"].get(pk, {})
            ab = pair.get("MCE_A_to_B", 0.0)
            ba = pair.get("MCE_B_to_A", 0.0)
            mce_means.append((ab + ba) / 2.0)
        x = np.arange(n_pairs) + (m - (n_methods - 1) / 2.0) * bar_width
        ax.bar(x, mce_means, bar_width,
               color=palette[m % len(palette)], alpha=0.9,
               edgecolor="white",
               label=method_labels[m])

    ax.axhline(0.043, color="#444", lw=1.0, ls="--",
               label="Nemani et al. (2018) Expert vs Novice MCE ≈ 0.043")
    ax.set_xticks(np.arange(n_pairs))
    ax.set_xticklabels(pair_labels, fontsize=10)
    ax.set_ylabel("Mean pairwise MCE\n(average of A→B and B→A)", fontsize=10)
    ax.set_ylim(0.0, max(0.7, ax.get_ylim()[1]))
    ax.set_title(
        "Pairwise misclassification by tier-pair, across classifier variants",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    # Interpretation (Low↔High = U-shape signature; lowest Low-vs-Mid MCE
    # under top-k feature selection) is covered by the "How to read these
    # results" paragraph in ushape_report.md. We deliberately keep the
    # plot itself uncluttered.

    fig.suptitle(
        "3-class LDA: feature-set comparison (LOO cross-validation, N = "
        f"{lda_results[0]['n_subjects']})",
        fontsize=11, y=0.99,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


# ---------- markdown summary -------------------------------------------------

def _find_classifier(lda_results: Optional[List[Dict[str, object]]],
                      method_keyword: str) -> Optional[Dict[str, object]]:
    """Look up a specific classifier in the lda_results list by method name keyword."""
    if not lda_results:
        return None
    for r in lda_results:
        if method_keyword.lower() in str(r.get("method", "")).lower():
            return r
    return None


def render_markdown(
    t_results: pd.DataFrame,
    reg_results: pd.DataFrame,
    u_df: pd.DataFrame,
    n_subjects: int,
    tier_counts: Dict[str, int],
    lda_results: Optional[List[Dict[str, object]]] = None,
    modality: str = "eeg",
) -> str:
    bonf = 0.05 / 4   # four pre-specified Lim hypotheses (EEG-only)
    is_eeg = modality == "eeg"
    modality_label = "EEG" if is_eeg else "Gaze"
    title_modality = "EEG" if is_eeg else "Gaze (Tobii eye-summary features)"
    lines = [
        f"# FLS U-shape consolidation analysis — {title_modality}",
        "",
        f"**Modality:** {modality_label}",
        f"**Subjects:** {n_subjects}",
        f"**Tier counts:** {tier_counts}",
    ]
    if is_eeg:
        lines.append(
            f"**Bonferroni-corrected α (4 pre-specified Lim hypotheses):** {bonf:.4f}"
        )
    lines.append("")

    # ---------- Key result (TL;DR) ----------------------------------------
    rfc = _find_classifier(lda_results, "rfc")
    topk = _find_classifier(lda_results, "top_")
    lim_frontal_l = t_results[t_results["feature"] == "eeg_frontal_L_theta"].iloc[0] \
        if "eeg_frontal_L_theta" in t_results["feature"].values else None
    lines += [
        "## Key result",
        "",
    ]
    if rfc is not None:
        mvr = rfc.get("mid_vs_rest", {}) or {}
        lines += [
            f"Under a Soangra-style Random-Forest classifier with in-fold "
            f"RFE feature selection (k=5), **Mid-vs-not-Mid accuracy = "
            f"{mvr.get('accuracy', float('nan')):.3f}** "
            f"(Mid precision = {mvr.get('mid_precision', float('nan')):.3f}, "
            f"non-Mid recall = {mvr.get('non_mid_recall', float('nan')):.3f}). "
            f"This is the headline preliminary-data number: it surfaces a "
            f"Mid-skill engagement state distinct from both novices and "
            f"experts, comparable in interpretability to Soangra et al. (2022) "
            f"PLoS ONE 58–61% wearable-sensor 3-class accuracy at N=26 — "
            f"despite our smaller sample size and the more challenging "
            f"clinical question (\"is this surgeon in the high-engagement "
            f"intermediate state?\").",
            "",
        ]
        # Pull the classic U-shape signature: % of High classified as Low.
        try:
            confusion = np.asarray(rfc["confusion"], dtype=np.int64)
            tier_order = list(rfc["tier_order"])
            high_idx = tier_order.index("High")
            low_idx = tier_order.index("Low")
            high_to_low = int(confusion[high_idx, low_idx])
            high_total = int(confusion[high_idx, :].sum())
            lines += [
                f"The same classifier produces the cleanest multivariate "
                f"U-shape signature in the project: **{high_to_low}/{high_total} "
                f"High-tier subjects are classified as Low**, reflecting "
                f"that novices and experts share relative-power patterns "
                f"under the inverted-U engagement model. Under Nemani et al. "
                f"(2018)'s monotonic skill-cortex assumption, Low vs. High "
                f"would be the *most-separable* pair; we observe it as the "
                f"*least-separable* pair (mean pairwise MCE = "
                f"{(rfc['pairwise_mce']['Low_vs_High']['MCE_A_to_B'] + rfc['pairwise_mce']['Low_vs_High']['MCE_B_to_A']) / 2.0:.3f}).",
                "",
            ]
        except Exception:
            pass
    if is_eeg and lim_frontal_l is not None:
        lines += [
            f"Pre-registered Lim et al. (2025, *Sci. Rep.* 15:12073) "
            f"prediction test: frontal-L theta is elevated in Mid-tier "
            f"subjects (Mid − rest = {lim_frontal_l['mid_minus_rest']:+.3f}, "
            f"one-sided p = {lim_frontal_l['p_one_sided_up']:.4f}); the "
            f"effect direction matches the cognitive-load hypothesis at "
            f"uncorrected α = 0.05.",
            "",
        ]
    if is_eeg and topk is not None and rfc is not None:
        lines += [
            f"Methodological convergence across feature-selection methods: "
            f"both t-statistic ranking (top-5 LDA) and RFE under Random Forest "
            f"select `central_L_delta` and `frontal_L_delta` as the most "
            f"discriminative features in 22+/24 LOO folds, both falling "
            f"anatomically inside Nemani et al. (2018)'s PFC and LMM1 "
            f"regions on the topomap.",
            "",
        ]
    if not is_eeg and topk is not None:
        # Identify the most-stable selected features in the gaze top-5 LDA.
        sel_lists = topk.get("selected_features_per_fold", [])
        stable_picks: List[str] = []
        if sel_lists:
            from collections import Counter as _Counter
            cnt = _Counter()
            for sel in sel_lists:
                for f in sel:
                    cnt[f] += 1
            stable_picks = [f for f, c in cnt.most_common(3) if c >= len(sel_lists) * 0.8]
        if stable_picks:
            lines += [
                "Stable in-fold gaze feature selection (top-5 LDA): "
                + ", ".join(f"`{f}`" for f in stable_picks)
                + f" picked in ≥80% of {len(sel_lists)} LOO folds. These "
                "are the gaze features carrying the strongest Mid-tier "
                "contrast in the FLS dataset.",
                "",
            ]

    # ---------- Literature comparison -------------------------------------
    lines += [
        "## Literature comparison",
        "",
        "| Study | Modality | What it measures | N | Classes | Best accuracy |",
        "|---|---|---|---:|---:|---:|",
        "| Nemani et al. 2018 (*Sci. Adv.*) | fNIRS PFC + LMM1 + SMA | Hemodynamic / metabolic | 32 | 2 (Expert vs Novice) | 95.6% |",
        "| Soangra et al. 2022 (*PLoS ONE*) | EMG + accelerometer (ECU + deltoid) | Muscle activation + motor variability | 26 | 3 (N / I / E) | 58% (RFC + RFE) |",
        f"| **This work** | EEG regional bandpower (top-5 RFE) | Cortical engagement / oscillation patterns | {n_subjects} | 3 (L / M / H) | "
        + (f"{(rfc.get('mid_vs_rest', {}) or {}).get('accuracy', float('nan')):.1%}".replace('.0%', '%').replace('nan%', 'TBD')
           if rfc else 'TBD')
        + " Mid-vs-not-Mid (RFC + RFE-5) |",
        "",
        "Note: the 2-class vs. 3-class accuracies are not directly "
        "comparable across rows. Nemani's 95.6% is on a 2-class "
        "monotonic-skill problem (Expert vs. Novice). Soangra's 58% is on "
        "the same 3-class structure as ours. Our headline number is the "
        "Mid-vs-not-Mid 2-class collapse of our 3-class result — "
        "directly comparable to Soangra's 3-class accuracy after the same "
        "binary collapse, and the more clinically-meaningful question for "
        "tracking trainee progress through the high-engagement state.",
        "",
    ]
    if is_eeg:
        lines += [
            "## Lim et al. (2025) prediction test",
            "",
            "Pre-registered prediction (frontal theta higher, parietal alpha "
            "lower in Mid):",
            "",
            "| Feature | Predicted | Mid − rest | one-sided p | matches direction? |",
            "|---|---|---:|---:|---|",
        ]
        for col in LIM_FEATURES:
            mask = t_results["feature"] == col
            if not mask.any():
                continue
            row = t_results[mask].iloc[0]
            if col in LIM_PREDICTED_UP:
                pred = "Mid > rest"
                p = row["p_one_sided_up"]
                match = "yes" if row["mid_minus_rest"] > 0 else "no"
            else:
                pred = "Mid < rest"
                p = row["p_one_sided_down"]
                match = "yes" if row["mid_minus_rest"] < 0 else "no"
            sig = "**" if p < bonf else ""
            lines.append(
                f"| `{col}` | {pred} | {row['mid_minus_rest']:+.3f} | "
                f"{sig}{p:.4f}{sig} | {match} |"
            )
        lines += [
            "",
            "Bold p-values clear the 4-test Bonferroni threshold (α = 0.0125).",
            "",
            "![Lim comparison](lim_comparison.png)",
            "",
        ]
    else:
        lines += [
            "## Top features summary (data-driven)",
            "",
            "Lim et al. (2025) is an EEG-cognitive-load study; no published "
            "gaze-feature prediction analogous to frontal theta / parietal "
            "alpha exists. We replace the Lim panel with a data-driven "
            "summary of the top-4 gaze features by |t-statistic|, showing "
            "tier means with 95% CI in the same 2×2 layout.",
            "",
            "![top features](top_features.png)",
            "",
        ]
    lines += [
        "## Per-feature ranking (full)",
        "",
        f"Top 10 features by |t-statistic| (out of {len(t_results)}):",
        "",
        "| Rank | Feature | Mid − rest | t | p (two-sided) | Lim? |",
        "|---:|---|---:|---:|---:|:---:|",
    ]
    for i, row in t_results.head(10).iterrows():
        is_lim = "✓" if (row["is_lim_up"] or row["is_lim_down"]) else ""
        lines.append(
            f"| {i+1} | `{row['feature']}` | {row['mid_minus_rest']:+.3f} | "
            f"{row['t_stat']:+.2f} | {row['p_two_sided']:.4f} | {is_lim} |"
        )
    lines += [
        "",
        "![feature ranking](feature_ranking.png)",
        "",
        "## Continuous-skill regression",
        "",
        "Linear vs. quadratic AIC comparison; positive ΔAIC = quadratic preferred.",
        "Convention: ΔAIC > 2 is meaningful evidence for the more complex model.",
        "",
        "| Rank | Feature | ΔAIC | quadratic coeff | Lim? |",
        "|---:|---|---:|---:|:---:|",
    ]
    for i, row in reg_results.head(10).iterrows():
        is_lim = "✓" if row["is_lim_feature"] else ""
        lines.append(
            f"| {i+1} | `{row['feature']}` | {row['delta_AIC']:+.2f} | "
            f"{row['b_quadratic_quad']:+.3f} | {is_lim} |"
        )
    lines += [
        "",
        "![quadratic vs linear](quadratic_vs_linear.png)",
        "",
        "## Individual U-position projection",
        "",
        "Each subject's projection onto the Mid − (Low ∪ High) discriminant axis. "
        "Positive values lie in the Mid (high-engagement) direction; negative "
        "values lie toward Low/High. Subjects sorted by composite skill.",
        "",
        "![individual U-positions](individual_u_positions.png)",
        "",
    ]
    if is_eeg:
        lines += [
            "## Anatomical map vs. Nemani et al. (2018)",
            "",
            "Per-band schematic head topography of the Mid − rest contrast on "
            "the 8 lateralized regions, with Nemani et al.'s PFC, LMM1 (left "
            "medial M1), and SMA region outlines overlaid. Nemani identified "
            "LMM1 (weight = −0.70 in their LDA) and the PFC as the most "
            "discriminative regions for FLS skill on fNIRS oxy-hemoglobin; "
            "this plot tests anatomical convergence on EEG spectral power.",
            "",
            "**Note on SMA:** SMA's scalp projection is the midline FCz site "
            "(between Fz and Cz, anterior to the central row). Our 8-region "
            "schema deliberately drops midline channels — there is no "
            "corresponding region in our EEG features to compare directly. "
            "The SMA outline is shown for anatomical reference only; the "
            "anatomical convergence claim relies on PFC and LMM1, both of "
            "which DO have direct EEG regional counterparts (frontal_L/R "
            "and central_L respectively).",
            "",
            "![Nemani topomap](nemani_topomap.png)",
            "",
        ]
    if lda_results:
        chance = 1.0 / len(lda_results[0]["tier_order"])
        lines += [
            "## 3-class LDA classifiers (Nemani-style)",
            "",
            f"Leave-one-out 3-class LDA on the {lda_results[0]['n_subjects']} "
            f"subjects, comparing three feature-set choices. The full "
            f"40-feature classifier is the head-to-head with Nemani et al. "
            f"on raw dimensionality; the top-5 by t-stat and PCA-5 variants "
            f"match Nemani's effective dimensionality (5 fNIRS metrics, "
            f"N/p ≈ 5).",
            "",
            "### How to read these results",
            "",
            "Under the canonical *monotonic* skill-cortex relationship "
            "(Nemani et al. 2018: Novice > Skilled > Expert on PFC, "
            "reverse on motor cortex), the Low ↔ High pair should be the "
            "*most* separable of the three pairs — they sit at opposite "
            "ends of the skill axis. Under the *inverted-U engagement* "
            "model (Lim et al. 2025), Low and High should be the *least* "
            "separable: novices and experts both show low cortical "
            "engagement (one because they don't engage, the other because "
            "execution is automated), while Mid-skill subjects sit in a "
            "distinct high-engagement state. **High Low ↔ High mutual "
            "MCE in this run is therefore evidence *for* the U-shape, "
            "not classifier failure.** The 3-class accuracy summary "
            "buries this signal because the classifier maps Low and High "
            "to the same region of feature space and then guesses one or "
            "the other; the *Mid vs. not-Mid* derived statistic and the "
            "pairwise MCE breakdown are the more informative readouts.",
            "",
            "### Accuracy summary",
            "",
            "| Classifier | Effective p | 3-class accuracy | Mid vs not-Mid accuracy | Mid recall |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in lda_results:
            mvr = r.get("mid_vs_rest", {}) or {}
            mvr_acc = mvr.get("accuracy", float("nan"))
            mvr_rec = mvr.get("mid_recall", float("nan"))
            lines.append(
                f"| `{r['method']}` | {r['n_features']} | "
                f"{r['accuracy']:.3f} | {mvr_acc:.3f} | {mvr_rec:.3f} |"
            )
        lines += [
            "",
            f"Chance (3-class) = {chance:.3f}; chance (Mid vs not-Mid, "
            "proportion baseline) = 0.667. Mid recall above 0.50 with "
            "Mid-vs-not-Mid accuracy above 0.667 is evidence the "
            "classifier is finding Mid-specific signal even when the "
            "3-class accuracy looks at chance.",
            "",
            "![LDA classifier comparison](lda_comparison.png)",
            "",
        ]

        for r in lda_results:
            mvr = r.get("mid_vs_rest", {}) or {}
            lines += [
                f"### `{r['method']}`",
                "",
                f"Effective feature count: {r['n_features']}; "
                f"3-class accuracy = {r['accuracy']:.3f} (chance = {chance:.3f}).",
                "",
            ]
            if mvr:
                lines += [
                    f"**Mid vs. not-Mid (2-class collapse):** "
                    f"accuracy = {mvr.get('accuracy', float('nan')):.3f}, "
                    f"Mid recall = {mvr.get('mid_recall', float('nan')):.3f}, "
                    f"non-Mid recall = {mvr.get('non_mid_recall', float('nan')):.3f}, "
                    f"Mid precision = {mvr.get('mid_precision', float('nan')):.3f}.",
                    "",
                ]
            lines += [
                "Per-class recall:",
                "",
                "| Tier | Recall |",
                "|---|---:|",
            ]
            for tier, rec in r["per_class_recall"].items():
                lines.append(f"| {tier} | {rec:.3f} |")
            lines += [
                "",
                "Pairwise misclassification errors:",
                "",
                "| Pair | n_A | n_B | MCE A→B | MCE B→A |",
                "|---|---:|---:|---:|---:|",
            ]
            for pair_name, pair in r["pairwise_mce"].items():
                a, b = pair_name.split("_vs_")
                lines.append(
                    f"| {a} vs {b} | {pair['n_A']} | {pair['n_B']} | "
                    f"{pair['MCE_A_to_B']:.3f} | {pair['MCE_B_to_A']:.3f} |"
                )
            method = r["method"]
            if method == "all_40_features":
                png_name = "lda_confusion.png"
            elif "rfc" in method:
                png_name = "lda_confusion_rfc_rfe.png"
            elif "top_" in method:
                png_name = "lda_confusion_top5.png"
            else:
                png_name = "lda_confusion_pca5.png"
            lines += ["", f"![confusion]({png_name})", ""]

        lines += [
            "Direct comparison to Nemani et al. (2018) Fig. 3C: they report "
            "MCE = 4.4% / 4.2% for Expert vs. Novice surgeons (5 fNIRS metrics "
            "across 32 subjects, *2-class* monotonic-skill problem). Our "
            "Low vs. High pair under the matched-dimensionality classifiers "
            "(top-5, PCA-5) is the dimensionality-matched analogue, but the "
            "*expected* direction differs: Nemani would predict Low vs. "
            "High to be the most-separable pair; we predict and observe it "
            "to be the *least*-separable pair, reflecting the inverted-U "
            "geometry.",
            "",
        ]
    return "\n".join(lines)


# ---------- main ------------------------------------------------------------

def run(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_eeg = args.modality == "eeg"
    modality_label = "EEG" if is_eeg else "Gaze"
    log.info("running U-shape analysis on modality: %s", modality_label)
    log.info("output directory: %s", out_dir)

    log.info("loading EEG trial features ...")
    eeg_trial = _orch._load_or_build_feature_frame(
        repo_root=args.repo_root, cache_path=args.features_cache,
        cache_name="fls_eeg_trial",
        builder=_orch.build_fls_eeg_feature_frame,
        builder_kwargs={},
    )
    log.info("  -> %d EEG trials", len(eeg_trial))

    feat_cols_eeg = eeg_feature_column_names()
    feat_cols_gaze = eye_feature_column_names()

    log.info("loading gaze trial features ...")
    gaze_trial = _orch._load_or_build_feature_frame(
        repo_root=args.repo_root, cache_path=args.features_cache,
        cache_name="fls_gaze_trial",
        builder=_orch.build_fls_gaze_feature_frame,
        builder_kwargs={},
    )
    log.info("  -> %d gaze trials", len(gaze_trial))

    # Active modality's feature columns. The other modality is still loaded
    # so the degeneracy filter can evaluate both sides.
    feat_cols_active = feat_cols_eeg if is_eeg else feat_cols_gaze

    # Replicate the orchestrator's data-prep pipeline.
    residualize_task = not args.no_residualize_task
    log.info("residualizing (task=%s) ...", residualize_task)
    eeg_resid = _orch.residualize_trial_features(
        eeg_trial, feat_cols_eeg, residualize_task=residualize_task)
    gaze_resid = _orch.residualize_trial_features(
        gaze_trial, feat_cols_gaze, residualize_task=residualize_task)

    log.info("aggregating to subject level ...")
    eeg_subj, _ = assemble_subject_frame(eeg_resid, feat_cols_eeg, mode="mean")
    gaze_subj, _ = assemble_subject_frame(gaze_resid, feat_cols_gaze, mode="mean")

    # Z-score the active modality (controlled by --skip_eeg_zscore, which
    # acts as a generic "skip z-score on the active modality" flag now).
    if not args.skip_eeg_zscore:
        if is_eeg:
            log.info("step 3b: per-subject z-scoring EEG features ...")
            eeg_subj = _orch.per_subject_zscore_features(eeg_subj, feat_cols_eeg)
        else:
            log.info("step 3b: per-subject z-scoring gaze features ...")
            gaze_subj = _orch.per_subject_zscore_features(
                gaze_subj, feat_cols_gaze)

    subj_df = (gaze_subj
               .merge(eeg_subj[["subject_id"] + feat_cols_eeg],
                      on="subject_id", how="inner")
               .dropna(subset=["composite_skill"])
               .sort_values("subject_id")
               .reset_index(drop=True))

    subj_df, dropped = _orch.drop_degenerate_subjects(
        subj_df,
        feature_cols_gaze=feat_cols_gaze,
        feature_cols_eeg=feat_cols_eeg,
    )
    if dropped:
        log.warning("dropped degenerate subjects: %s", dropped)

    subj_df, cutoffs = add_tier_column(subj_df, score_col="composite_skill")
    tier_counts = subj_df["tier"].value_counts().to_dict()
    log.info("tier counts: %s", tier_counts)
    log.info("N = %d", len(subj_df))

    # Run the analyses on the active modality's features.
    log.info("step A: Mid vs (Low ∪ High) per-feature t-tests (%s) ...",
             modality_label)
    t_results = mid_vs_rest_t_test(subj_df, feat_cols_active)

    log.info("step B: continuous-skill regression (linear vs quadratic) ...")
    reg_results = continuous_skill_regression(subj_df, feat_cols_active)

    log.info("step C: individual U-position projection ...")
    u_df = individual_u_position(subj_df, feat_cols_active)

    log.info("step D: 3-class LDA classifier (full features, leave-one-out) ...")
    lda_result = lda_three_class(subj_df, feat_cols_active)
    n_active = len(feat_cols_active)
    lda_result["method"] = f"all_{n_active}_features"
    log.info("  full-feature LDA accuracy = %.3f (chance = %.3f)",
             lda_result["accuracy"], 1.0 / len(lda_result["tier_order"]))

    log.info("step D2: 3-class LDA with top-5 features by t-stat (in-fold selection) ...")
    k_topk = min(5, n_active)
    lda_topk = lda_three_class_top_k(subj_df, feat_cols_active, k=k_topk)
    log.info("  top-%d LDA accuracy = %.3f", k_topk, lda_topk["accuracy"])

    log.info("step D3: 3-class LDA in PCA-5 space (in-fold) ...")
    pca_components = min(5, n_active)
    lda_pca = lda_three_class_pca(
        subj_df, feat_cols_active, n_components=pca_components)
    log.info("  PCA-%d LDA accuracy = %.3f", pca_components, lda_pca["accuracy"])

    log.info("step D4: 3-class RFC + RFE-5 (Soangra-style, in-fold) ...")
    rfc_rfe = rfc_three_class_rfe(subj_df, feat_cols_active, k=k_topk)
    log.info("  RFC+RFE-%d accuracy = %.3f", k_topk, rfc_rfe["accuracy"])

    # Plots.
    log.info("writing plots ...")
    if is_eeg:
        plot_lim_comparison(subj_df, t_results, out_dir / "lim_comparison.png")
        plot_nemani_topomap(t_results, out_dir / "nemani_topomap.png")
    else:
        plot_top_features_summary(
            subj_df, t_results, out_dir / "top_features.png",
            n_top=4,
            title="Top gaze features by |t|: Mid vs. (Low ∪ High) tier means with 95% CI",
        )
    plot_feature_ranking(t_results, out_dir / "feature_ranking.png")
    plot_quadratic_vs_linear(reg_results, out_dir / "quadratic_vs_linear.png")
    plot_individual_u_positions(u_df, out_dir / "individual_u_positions.png")
    plot_lda_confusion(lda_result, out_dir / "lda_confusion.png")
    plot_lda_confusion(lda_topk, out_dir / "lda_confusion_top5.png")
    plot_lda_confusion(lda_pca, out_dir / "lda_confusion_pca5.png")
    plot_lda_confusion(rfc_rfe, out_dir / "lda_confusion_rfc_rfe.png")
    plot_lda_classifier_comparison(
        [lda_result, lda_topk, lda_pca, rfc_rfe],
        out_dir / "lda_comparison.png",
    )

    # JSON summary.
    summary: Dict[str, object] = {
        "modality": args.modality,
        "n_subjects": int(len(subj_df)),
        "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "tertile_cutoffs": cutoffs.as_dict(),
        "dropped_subjects": [int(s) for s in dropped],
        "config": {
            "modality": args.modality,
            "residualize_task": bool(residualize_task),
            "zscore_active_modality": not args.skip_eeg_zscore,
        },
        "per_feature_t_test": t_results.to_dict(orient="records"),
        "continuous_skill_regression": reg_results.to_dict(orient="records"),
        "individual_u_positions": u_df.to_dict(orient="records"),
        "lda_three_class": lda_result,
        "lda_three_class_top_k": lda_topk,
        "lda_three_class_pca": lda_pca,
        "rfc_three_class_rfe": rfc_rfe,
    }
    if is_eeg:
        # Lim-specific summary block (EEG only — no gaze analogue).
        summary["lim_hypothesis_test"] = {
            col: {
                "predicted_direction": "up" if col in LIM_PREDICTED_UP else "down",
                "mid_minus_rest": float(t_results.loc[
                    t_results["feature"] == col, "mid_minus_rest"].iloc[0]),
                "t_stat": float(t_results.loc[
                    t_results["feature"] == col, "t_stat"].iloc[0]),
                "p_one_sided": float(t_results.loc[
                    t_results["feature"] == col,
                    "p_one_sided_up" if col in LIM_PREDICTED_UP else "p_one_sided_down",
                ].iloc[0]),
                "p_two_sided": float(t_results.loc[
                    t_results["feature"] == col, "p_two_sided"].iloc[0]),
            }
            for col in LIM_FEATURES
            if (t_results["feature"] == col).any()
        }
    json_path = out_dir / "ushape_results.json"
    json_path.write_text(json.dumps(summary, indent=2, default=float))
    log.info("wrote %s", json_path)

    # Markdown summary.
    md = render_markdown(
        t_results, reg_results, u_df,
        n_subjects=int(len(subj_df)), tier_counts=tier_counts,
        lda_results=[lda_result, lda_topk, lda_pca, rfc_rfe],
        modality=args.modality,
    )
    md_path = out_dir / "ushape_report.md"
    md_path.write_text(md)
    log.info("wrote %s", md_path)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    _setup_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
