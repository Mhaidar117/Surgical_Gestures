#!/usr/bin/env python3
"""Comparison B orchestrator -- GW skill manifold between JIGSAWS and EEG/Eye.

Reads src/configs/skill_manifold.yaml + src/configs/skill_manifold_task_modules.yaml,
then runs steps 1-10 of the Comparison B specification and writes:

  reports/skill_manifold/
    plots/rdm_jigsaws.png, plots/rdm_eeg_eye.png
    plots/coupling_headline.png, plots/null_histogram.png
    plots/mds_jigsaws.png, plots/mds_eeg_eye.png
    plots/osats_axis_alignment.png, plots/coupling_trial_level.png
    report_comparison_B.md
    results_comparison_B.json

See docs/agent_prompt_gw_skill_manifold.md for the full specification.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from skill_manifold.io import (  # noqa: E402
    CONFIG_DIR, JIGSAWS_OSATS_COLUMNS, data_root, ensure_output_dirs, load_config,
)
from skill_manifold.features_jigsaws import (  # noqa: E402
    build_jigsaws_feature_frame, feature_column_names as jigsaws_feat_cols,
    zscore_features,
)
from skill_manifold.features_eeg_eye import (  # noqa: E402
    build_eeg_eye_feature_frame, feature_column_names as eeg_eye_feat_cols,
    load_task_module_map, mimic_modality_columns,
)
from skill_manifold.residualize import residualize  # noqa: E402
from skill_manifold.binning import TIER_NAMES, add_tier_column  # noqa: E402
from skill_manifold.rdms import (  # noqa: E402
    centroid_rdm, is_valid_rdm, pairwise_cosine_rdm,
)
from skill_manifold.gw import (  # noqa: E402
    entropic_gromov_wasserstein, gromov_wasserstein_centroids,
    permutation_null_centroid,
)
from skill_manifold.trial_null import (  # noqa: E402
    BONFERRONI_Z_THREE_CELLS, coupling_matrix_diagnostics,
    eeg_baseline_pc_correlation, epsilon_sensitivity,
    jigsaws_modality_split_analysis, modality_split_analysis,
    pooled_eeg_random_split_null, stratified_bootstrap_verdict,
    subsample_robustness, subsample_robustness_stratified,
    trial_level_block_null, trial_level_block_null_all_trials,
)
from skill_manifold.binning import (  # noqa: E402
    assign_fixed_tier, assign_jigsaws_skill_tier,
)
from skill_manifold.features_jigsaws import jigsaws_modality_columns  # noqa: E402

log = logging.getLogger("skill_manifold_gw")


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------- plotting helpers (kept self-contained; import matplotlib lazily) ----------

def _heatmap(M: np.ndarray, title: str, labels: Sequence[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < M.max() * 0.6 else "black",
                    fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _null_hist(null: np.ndarray, observed: float, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=40, color="#777", edgecolor="white")
    ax.axvline(observed, color="red", linestyle="--", label=f"observed = {observed:.4f}")
    ax.set_xlabel("GW distance under tier-shuffle null")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _mds_plot(feats: np.ndarray, tiers: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS

    # Using MDS on the cosine-distance matrix.
    D = pairwise_cosine_rdm(feats)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
              normalized_stress="auto")
    xy = mds.fit_transform(D)
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"Low": "#d73027", "Mid": "#fee090", "High": "#1a9850"}
    for t in ("Low", "Mid", "High"):
        mask = tiers == t
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1], s=22, alpha=0.7,
                   color=colors[t], label=f"{t} (n={int(mask.sum())})",
                   edgecolors="none")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _osats_bar(rows: List[dict], null_cis: List[tuple[float, float]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = [r["axis"] for r in rows]
    gw = [r["gw_distance"] for r in rows]
    lo = [c[0] for c in null_cis]
    hi = [c[1] for c in null_cis]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(axes))
    ax.bar(x, gw, color="#4575b4", label="observed GW")
    ax.fill_between(x, lo, hi, color="#aaaaaa", alpha=0.4, step="mid",
                    label="null 95% CI")
    ax.set_xticks(x, axes, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("GW distance")
    ax.set_title("OSATS axis-alignment vs Mimic composite")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _renormalize_uniform(T: np.ndarray, max_iter: int = 5000,
                         atol: float = 1e-9) -> np.ndarray:
    """Sinkhorn-style projection onto doubly-stochastic uniform marginals.

    Iterate row-then-column rescaling until both marginals are within `atol`
    of 1/M and 1/N respectively. POT's entropic GW leaves drift of 1e-5 on
    small N; we tighten it to machine precision so the downstream block-mass
    marginals are exactly uniform by construction.
    """
    T = np.asarray(T, dtype=np.float64)
    T = np.clip(T, 0.0, None)
    total = float(T.sum())
    if total <= 1e-12:
        return T
    T = T / total
    M, N = T.shape
    target_row = 1.0 / M
    target_col = 1.0 / N
    for _ in range(max_iter):
        rs = T.sum(axis=1, keepdims=True)
        T = np.where(rs > 1e-15, T * (target_row / rs), T)
        cs = T.sum(axis=0, keepdims=True)
        T = np.where(cs > 1e-15, T * (target_col / cs), T)
        max_drift = max(
            float(np.max(np.abs(T.sum(axis=1) - target_row))),
            float(np.max(np.abs(T.sum(axis=0) - target_col))),
        )
        if max_drift < atol:
            break
    return T


def _trial_block_null_plot(result: dict, path: Path, *,
                           show_expected: bool = False) -> None:
    """Two-panel plot: block-mass heatmap annotated with per-cell z, and the
    diag_mass null histogram. When `show_expected`, each cell also shows the
    expected mass under the null (which is uniform 1/9 with balanced tiers
    but `(n_j_a/NJ)*(n_e_b/NE)` with unbalanced tiers), and the histogram
    gets a vertical line at the expected trace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    B = np.asarray(result["block_mass"])
    tiers = list(result["tier_names"])
    per_z = np.asarray(result["per_cell_z"])
    null = np.asarray(result["null"])
    obs = float(result["observed"])
    p = float(result["p_value"])
    z = float(result["z_score"])

    expected_B = result.get("expected_block_mass")
    expected_trace = result.get("expected_trace_under_null")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    im = axes[0].imshow(B, cmap="magma", vmin=0, vmax=max(B.max(), 1.0 / 9 * 1.8))
    axes[0].set_xticks(range(3), tiers)
    axes[0].set_yticks(range(3), tiers)
    axes[0].set_xlabel("EEG/Eye tier"); axes[0].set_ylabel("JIGSAWS tier")
    for i in range(3):
        for j in range(3):
            label = f"{B[i, j]:.3f}"
            if show_expected and expected_B is not None:
                label += f"\nexp {float(np.asarray(expected_B)[i, j]):.3f}"
            if i == j:
                label += f"\nz={per_z[i]:+.2f}"
            axes[0].text(j, i, label, ha="center", va="center",
                         color="white" if B[i, j] < B.max() * 0.55 else "black",
                         fontsize=8)
    axes[0].set_title(f"Block-mass B   diag_mass = {obs:.3f}")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    axes[1].hist(null, bins=40, color="#888", edgecolor="white")
    axes[1].axvline(obs, color="red", linestyle="--",
                    label=f"observed = {obs:.3f}")
    exp_trace_val = (float(expected_trace) if expected_trace is not None
                     else 1.0 / 3.0)
    axes[1].axvline(exp_trace_val, color="blue", linestyle=":",
                    label=f"expected = {exp_trace_val:.3f}")
    axes[1].set_xlabel("diag_mass under tier-shuffle null")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Null (p={p:.3f}, z={z:+.2f})")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _serialize_stratified(result: dict) -> dict:
    """Render a stratified-bootstrap result dict into JSON-safe native types."""
    return {
        "n_bootstraps": int(result["n_bootstraps"]),
        "n_degenerate": int(result["n_degenerate"]),
        "tier_names": list(result["tier_names"]),
        "tier_counts_j": np.asarray(result["tier_counts_j"]).tolist(),
        "tier_counts_e": np.asarray(result["tier_counts_e"]).tolist(),
        "diag_mass_observed": np.asarray(result["diag_mass_observed"]).tolist(),
        "trace_z_scores": np.asarray(result["trace_z_scores"]).tolist(),
        "p_values": np.asarray(result["p_values"]).tolist(),
        "per_cell_z": np.asarray(result["per_cell_z"]).tolist(),
        "summary": {k: dict(v) for k, v in result["summary"].items()},
        "frac_significant_trace": float(result["frac_significant_trace"]),
        "frac_per_cell_bonf": np.asarray(result["frac_per_cell_bonf"]).tolist(),
        "frac_per_cell_positive": np.asarray(result["frac_per_cell_positive"]).tolist(),
    }


def _eeg_corr_plot(corr: dict, path: Path) -> None:
    """Histogram of per-trial Pearson r between mean baseline and mean pc EEG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = np.asarray(corr["per_trial_pearson_r"])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(r, bins=30, color="#4575b4", edgecolor="white")
    ax.axvline(float(corr["median"]), color="red", linestyle="--",
               label=f"median = {corr['median']:+.2f}")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("per-trial Pearson r  (baseline mean vs pc mean)")
    ax.set_ylabel("trials")
    ax.set_title(
        f"EEG baseline/pc correlation (median |r| = {corr['median_abs']:.2f}, "
        f"frac |r| > 0.5 = {corr['frac_abs_gt_0p5']:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _eeg_pooled_plot(pooled: dict, path: Path) -> None:
    """Histogram of the signed trace-z delta under random 64+64 EEG splits,
    with the observed baseline-minus-pc delta marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    null = np.asarray(pooled["delta_distribution"])
    obs = float(pooled["observed_delta_baseline_minus_pc"])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(null, bins=20, color="#888", edgecolor="white")
    ax.axvline(obs, color="red", linestyle="--",
               label=f"observed baseline - pc = {obs:+.2f}")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("trace_z(half A) - trace_z(half B)   (random 64+64 EEG splits)")
    ax.set_ylabel("random splits")
    ax.set_title(
        f"Pooled-128 EEG null (n = {pooled['n_random_splits']}, "
        f"two-sided p = {pooled['p_two_sided']:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def _serialize_modality_split(mod_res: dict, *, jigsaws_side: bool = False) -> dict:
    """Render `modality_split_analysis(...)` into JSON-safe native types. Set
    `jigsaws_side=True` when serializing a JIGSAWS-split result (keys use
    `jigsaws_feature_dim` instead of `mimic_feature_dim`)."""
    out: dict = {"_meta": dict(mod_res.get("_meta", {}))}
    dim_key = "jigsaws_feature_dim" if jigsaws_side else "mimic_feature_dim"
    for mod, block in mod_res.items():
        if mod == "_meta":
            continue
        m_out: dict = {dim_key: int(block[dim_key])}
        for binning in ("fixed", "tertile"):
            sub = block[binning]
            pr = sub["primary"]; bs = sub["bootstrap"]
            m_out[binning] = {
                "primary": {
                    "observed_diag_mass": float(pr["observed"]),
                    "expected_trace_under_null": float(pr["expected_trace_under_null"]),
                    "trace_p_value": float(pr["p_value"]),
                    "trace_z_score": float(pr["z_score"]),
                    "per_cell_z": np.asarray(pr["per_cell_z"]).tolist(),
                    "per_cell_observed": np.asarray(pr["per_cell_observed"]).tolist(),
                    "per_cell_expected": np.asarray(pr["per_cell_expected"]).tolist(),
                    "coupling_shape": list(pr["coupling_shape"]),
                    "row_sum_drift": float(pr["row_sum_drift"]),
                    "col_sum_drift": float(pr["col_sum_drift"]),
                    "tier_counts_j": np.asarray(pr["tier_counts_j"]).tolist(),
                    "tier_counts_e": np.asarray(pr["tier_counts_e"]).tolist(),
                },
                "bootstrap": _serialize_stratified(bs),
                "verdict": dict(sub["verdict"]),
            }
        out[mod] = m_out
    return out


def _modality_split_plot(mod_res: dict, path: Path, *,
                         title: Optional[str] = None) -> None:
    """2x3 (or 2xK) grid: row 0 = fixed-cutoff bootstrap, row 1 = tertile
    bootstrap. Each column is one modality. Three strips per panel
    (Low/Mid/High). Shared y-axis for magnitude comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mods = [m for m in mod_res.keys() if m != "_meta"]
    tier_names = list(mod_res["_meta"]["tier_names"])
    n_cols = len(mods)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.7 * n_cols, 7.5), sharey=True)
    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    rng = np.random.default_rng(0)

    def _panel(ax, pcz, verdict, label):
        for k, name in enumerate(tier_names):
            xs = 0.15 + k * 0.35 + 0.06 * rng.normal(size=pcz.shape[0])
            ax.scatter(xs, pcz[:, k], s=22, alpha=0.75,
                       color="#4575b4", edgecolors="none")
            med = float(np.median(pcz[:, k]))
            p05 = float(np.quantile(pcz[:, k], 0.05))
            p95 = float(np.quantile(pcz[:, k], 0.95))
            ax.hlines(med, 0.15 + k * 0.35 - 0.07, 0.15 + k * 0.35 + 0.07,
                      color="black", linewidth=2)
            ax.fill_betweenx([p05, p95],
                             0.15 + k * 0.35 - 0.09, 0.15 + k * 0.35 + 0.09,
                             color="#4575b4", alpha=0.18)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.axhline(BONFERRONI_Z_THREE_CELLS, color="red", linestyle=":",
                   linewidth=0.8)
        ax.axhline(-BONFERRONI_Z_THREE_CELLS, color="red", linestyle=":",
                   linewidth=0.8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0.15 + k * 0.35 for k in range(len(tier_names))],
                      [f"{n}" for n in tier_names], fontsize=9)
        v = (verdict or {}).get("verdict", "?")
        ax.set_title(f"{label}\nverdict: {v}", fontsize=10)

    for j, mod in enumerate(mods):
        fb = mod_res[mod]["fixed"]["bootstrap"]
        tb = mod_res[mod]["tertile"]["bootstrap"]
        _panel(axes[0, j], fb["per_cell_z"],
               mod_res[mod]["fixed"]["verdict"], f"{mod}  (fixed-cutoff)")
        _panel(axes[1, j], tb["per_cell_z"],
               mod_res[mod]["tertile"]["verdict"], f"{mod}  (tertile)")
    axes[0, 0].set_ylabel("per-cell z\n(fixed-cutoff)")
    axes[1, 0].set_ylabel("per-cell z\n(tertile)")
    fig.suptitle(
        title or (f"Step 10 — Mimic-side modality split "
                  f"(stratified bootstrap, B = {mod_res['_meta']['n_bootstraps']})"),
        y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _stratified_bootstrap_plot(result: dict, verdict: dict, path: Path,
                               *, title: str) -> None:
    """Three side-by-side strip plots of per-cell z with Bonferroni reference,
    plus a top-line verdict stamp."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pcz = np.asarray(result["per_cell_z"])
    tier_names = list(result["tier_names"])
    frac_bonf = np.asarray(result["frac_per_cell_bonf"])
    frac_pos = np.asarray(result["frac_per_cell_positive"])
    K = pcz.shape[1]

    fig, axes = plt.subplots(1, K, figsize=(3.5 * K, 4.4), sharey=True)
    if K == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    for k in range(K):
        ax = axes[k]
        xs = 0.4 + 0.2 * rng.normal(size=pcz.shape[0])
        ax.scatter(xs, pcz[:, k], s=26, alpha=0.75,
                   color="#4575b4", edgecolors="none")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.axhline(BONFERRONI_Z_THREE_CELLS, color="red", linestyle=":",
                   linewidth=0.8, label=f"±{BONFERRONI_Z_THREE_CELLS:.2f} Bonf.")
        ax.axhline(-BONFERRONI_Z_THREE_CELLS, color="red", linestyle=":",
                   linewidth=0.8)
        med = float(np.median(pcz[:, k]))
        p05 = float(np.quantile(pcz[:, k], 0.05))
        p95 = float(np.quantile(pcz[:, k], 0.95))
        ax.hlines(med, 0.1, 0.9, color="black", linewidth=2)
        ax.fill_betweenx([p05, p95], 0.1, 0.9, color="#4575b4", alpha=0.18,
                         label="5-95% band")
        ax.set_xlim(0, 1); ax.set_xticks([])
        ax.set_title(
            f"{tier_names[k]}<->{tier_names[k]}\n"
            f"median {med:+.2f}   "
            f"{int(frac_pos[k] * 100):d}% > 0   "
            f"{int(frac_bonf[k] * 100):d}% > Bonf.")
        if k == 0:
            ax.set_ylabel("per-cell z-score")
            ax.legend(fontsize=8, loc="upper left")
    v = verdict.get("verdict", "unknown") if verdict else "unknown"
    fig.suptitle(f"{title}   |   verdict: {v}", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _per_cell_bootstrap_plot(sr, path: Path) -> None:
    """Three side-by-side strip plots, one per diagonal cell, showing the
    z-score distribution across the B bootstraps with horizontal reference
    lines at z = 0 and z = ±BONFERRONI_Z_THREE_CELLS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = sr.per_cell_z.shape[1]
    fig, axes = plt.subplots(1, K, figsize=(3.5 * K, 4.4), sharey=True)
    if K == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    for k in range(K):
        ax = axes[k]
        xs = 0.4 + 0.2 * rng.normal(size=sr.n_runs)
        zs = sr.per_cell_z[:, k]
        ax.scatter(xs, zs, s=22, alpha=0.7, color="#4575b4", edgecolors="none")
        ax.axhline(0, color="black", linewidth=0.7)
        ax.axhline(BONFERRONI_Z_THREE_CELLS, color="red", linestyle=":",
                   linewidth=0.8, label=f"±{BONFERRONI_Z_THREE_CELLS:.2f} Bonf.")
        ax.axhline(-BONFERRONI_Z_THREE_CELLS, color="red", linestyle=":",
                   linewidth=0.8)
        med = sr.per_cell_median[k]
        p05, p95 = sr.per_cell_p05[k], sr.per_cell_p95[k]
        ax.hlines(med, 0.1, 0.9, color="black", linewidth=2)
        ax.fill_betweenx([p05, p95], 0.1, 0.9, color="#4575b4", alpha=0.18,
                         label="5-95% band")
        frac = sr.frac_significant_per_cell_bonf[k]
        ax.set_xlim(0, 1); ax.set_xticks([])
        ax.set_title(
            f"{sr.tier_names[k]}<->{sr.tier_names[k]}\n"
            f"median {med:+.2f}  ({int(frac * sr.n_runs)}/{sr.n_runs} > Bonf.)")
        if k == 0:
            ax.set_ylabel("per-cell z-score")
            ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"Bootstrap per-cell diagonal z-scores (B = {sr.n_runs})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _epsilon_sensitivity_plot(rows, primary: dict, path: Path) -> None:
    """Two panels: diag_mass vs epsilon (y-axis clamped to [0.28, 0.42] so
    sub-1% wiggles don't dramatize), and per-cell z vs epsilon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_xs = [r["epsilon"] for r in rows]
    diag = [r["diag_mass"] for r in rows]
    per_cell = np.asarray([r["per_cell_z"] for r in rows])     # (n_eps, 3)
    tier_names = primary.get("tier_names", list(TIER_NAMES))
    expected_trace = float(primary.get("expected_trace_under_null", 1.0 / 3.0))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(eps_xs, diag, marker="o", color="#4575b4", label="observed")
    axes[0].axhline(expected_trace, color="blue", linestyle=":",
                    label=f"expected = {expected_trace:.3f}")
    axes[0].set_xscale("log"); axes[0].set_xlabel("entropic-GW epsilon")
    axes[0].set_ylabel("diag_mass")
    axes[0].set_ylim(0.28, 0.42)
    axes[0].set_title("diag_mass vs epsilon (all-trials coupling)")
    axes[0].legend(fontsize=8)

    colors = {"Low": "#d73027", "Mid": "#fee090", "High": "#1a9850"}
    for k, name in enumerate(tier_names):
        axes[1].plot(eps_xs, per_cell[:, k], marker="o",
                     color=colors.get(name, "black"),
                     label=f"{name}<->{name}")
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].axhline(BONFERRONI_Z_THREE_CELLS, color="red", linestyle=":",
                    linewidth=0.8, label=f"±{BONFERRONI_Z_THREE_CELLS:.2f} Bonf.")
    axes[1].axhline(-BONFERRONI_Z_THREE_CELLS, color="red", linestyle=":",
                    linewidth=0.8)
    axes[1].set_xscale("log"); axes[1].set_xlabel("entropic-GW epsilon")
    axes[1].set_ylabel("per-cell z-score")
    axes[1].set_title("per-cell diagonal z vs epsilon")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _robustness_plot(sr, path: Path) -> None:
    """Two-panel plot: per-subsample diag_mass distribution and p-value distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(sr.observed, bins=15, color="#4575b4", edgecolor="white")
    axes[0].axvline(sr.observed_median, color="red", linestyle="--",
                    label=f"median = {sr.observed_median:.3f}")
    axes[0].axvline(1.0 / 3.0, color="blue", linestyle=":",
                    label="expected = 0.333")
    axes[0].set_xlabel("diag_mass")
    axes[0].set_ylabel("subsamples")
    axes[0].set_title(f"{sr.n_runs} independent 300-trial subsamples")
    axes[0].legend(fontsize=8)

    axes[1].hist(sr.p_values, bins=np.linspace(0, 1, 21),
                 color="#1a9850", edgecolor="white")
    axes[1].axvline(0.05, color="red", linestyle="--", label="alpha = 0.05")
    axes[1].set_xlabel("per-subsample p-value")
    axes[1].set_ylabel("subsamples")
    axes[1].set_title(f"p < 0.05 in {int(sr.frac_p_lt_05 * sr.n_runs)}/{sr.n_runs} runs")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _trial_coupling_plot(T: np.ndarray, n_per_tier: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(T, cmap="magma", aspect="auto")
    # Draw tier boundaries.
    for k in (1, 2):
        ax.axhline(k * n_per_tier - 0.5, color="white", linewidth=0.5)
        ax.axvline(k * n_per_tier - 0.5, color="white", linewidth=0.5)
    ax.set_xlabel("EEG/Eye trials (Low | Mid | High)")
    ax.set_ylabel("JIGSAWS trials (Low | Mid | High)")
    ax.set_title("Entropic-GW trial-level coupling (300x300)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- pipeline ----------

def _ensure_assertions(rdm_j: np.ndarray, rdm_e: np.ndarray, coupling: np.ndarray,
                       null: Dict[str, float], results: dict,
                       trial_null_result: Optional[dict] = None) -> List[dict]:
    """Run the verification checklist and record PASS/FAIL records."""
    checks: List[dict] = []
    checks.append({"name": "rdm_j_valid",
                   "pass": bool(is_valid_rdm(rdm_j))})
    checks.append({"name": "rdm_e_valid",
                   "pass": bool(is_valid_rdm(rdm_e))})
    row_ok = np.allclose(coupling.sum(axis=1), 1.0 / coupling.shape[0], atol=1e-6)
    col_ok = np.allclose(coupling.sum(axis=0), 1.0 / coupling.shape[1], atol=1e-6)
    checks.append({"name": "coupling_marginals_uniform",
                   "pass": bool(row_ok and col_ok)})
    checks.append({"name": "null_has_1000_valid_samples",
                   "pass": bool(null["n_permutations"] >= 1000)})

    if trial_null_result:
        # All-trials PRIMARY checks (tier-proportional block marginals).
        primary = trial_null_result
        B = np.asarray(primary["block_mass"])
        nj = np.asarray(primary.get("tier_counts_j", []))
        ne = np.asarray(primary.get("tier_counts_e", []))
        NJ = int(nj.sum()) if nj.size else 0
        NE = int(ne.sum()) if ne.size else 0
        shape_ok = (NJ > 0 and NE > 0 and
                    tuple(primary.get("coupling_shape", ())) == (NJ, NE))
        checks.append({"name": "trial_null_primary_coupling_shape",
                       "pass": bool(shape_ok)})

        b_sum_ok = abs(float(B.sum()) - 1.0) < 1e-9
        if NJ > 0 and NE > 0:
            row_targets = nj / float(NJ)
            col_targets = ne / float(NE)
        else:
            row_targets = np.full(B.shape[0], 1.0 / B.shape[0])
            col_targets = np.full(B.shape[1], 1.0 / B.shape[1])
        b_row_ok = np.allclose(B.sum(axis=1), row_targets, atol=1e-6)
        b_col_ok = np.allclose(B.sum(axis=0), col_targets, atol=1e-6)
        checks.append({"name": "block_mass_marginals_match_tier_counts",
                       "pass": bool(b_row_ok and b_col_ok and b_sum_ok)})

        # Pre-renormalization drift on the NJ x NE coupling.
        row_drift = float(primary.get("row_sum_drift", 1.0))
        col_drift = float(primary.get("col_sum_drift", 1.0))
        checks.append({
            "name": "trial_null_primary_coupling_drift_tight",
            "pass": bool(max(row_drift, col_drift) < 1e-9),
        })

        # Null mean should be near the tier-count-weighted expected trace
        # (which equals 1/3 only when tiers are perfectly balanced).
        K = int(primary["n_permutations"])
        sd = float(primary["null_std"])
        mu = float(primary["null_mean"])
        expected = float(primary.get("expected_trace_under_null", 1.0 / 3.0))
        tol = max(3 * sd / np.sqrt(max(1, K)), 1e-6)
        checks.append({
            "name": "trial_null_mean_near_expected_trace",
            "pass": bool(abs(mu - expected) < tol),
        })

    # Stratified-bootstrap checks (Step 9g).
    fb = results.get("fixed_cutoff_bootstrap") or {}
    if fb:
        fixed_src = results.get("fixed_cutoff_sensitivity") or {}
        exp_j = tuple(fixed_src.get("tier_counts_j", ())) or None
        exp_e = tuple(fixed_src.get("tier_counts_e", ())) or None
        got_j = tuple(fb.get("tier_counts_j", ()))
        got_e = tuple(fb.get("tier_counts_e", ()))
        checks.append({
            "name": "fixed_cutoff_bootstrap_tier_counts_j_preserved",
            "pass": bool(exp_j is not None and got_j == exp_j),
        })
        checks.append({
            "name": "fixed_cutoff_bootstrap_tier_counts_e_preserved",
            "pass": bool(exp_e is not None and got_e == exp_e),
        })
        pcz = np.asarray(fb.get("per_cell_z", []), dtype=np.float64)
        shape_ok = (pcz.ndim == 2
                    and pcz.shape == (int(fb.get("n_bootstraps", 0)),
                                       len(fb.get("tier_names", []))))
        checks.append({
            "name": "fixed_cutoff_bootstrap_per_cell_z_shape_clean",
            "pass": bool(shape_ok and np.isfinite(pcz).all()),
        })
        checks.append({
            "name": "fixed_cutoff_bootstrap_no_degenerate",
            "pass": bool(int(fb.get("n_degenerate", 0)) == 0),
        })

    tb = results.get("tertile_bootstrap_stratified") or {}
    if tb:
        checks.append({
            "name": "tertile_bootstrap_stratified_no_degenerate",
            "pass": bool(int(tb.get("n_degenerate", 0)) == 0),
        })

    # Step 10 modality-split checks.
    ms = results.get("modality_split") or {}
    if ms and "_meta" in ms:
        mods = [m for m in ms.keys() if m != "_meta"]
        exp_sizes = {"eeg_baseline": 64, "eeg_predictive_coding": 64, "eye": 18}
        dims_ok = (set(mods) == set(exp_sizes.keys())
                   and all(ms[m]["mimic_feature_dim"] == exp_sizes[m]
                           for m in mods))
        checks.append({"name": "modality_split_dims_64_64_18", "pass": bool(dims_ok)})

        # All three modalities should preserve the same tier counts under each
        # binning (they share the tier assignments; only features differ).
        def _first(mod: str, binning: str, key: str):
            return ms[mod][binning]["bootstrap"][key]

        counts_match = True
        no_degen = True
        shape_ok = True
        for binning in ("fixed", "tertile"):
            ref_j = _first(mods[0], binning, "tier_counts_j")
            ref_e = _first(mods[0], binning, "tier_counts_e")
            for m in mods:
                counts_match = counts_match and (
                    _first(m, binning, "tier_counts_j") == ref_j
                    and _first(m, binning, "tier_counts_e") == ref_e
                )
                no_degen = no_degen and (
                    int(ms[m][binning]["bootstrap"].get("n_degenerate", 0)) == 0)
                pcz = np.asarray(ms[m][binning]["bootstrap"]["per_cell_z"])
                shape_ok = shape_ok and (
                    pcz.ndim == 2 and pcz.shape[1] == 3
                    and np.isfinite(pcz).all())
        checks.append({"name": "modality_split_tier_counts_consistent",
                       "pass": bool(counts_match)})
        checks.append({"name": "modality_split_no_degenerate",
                       "pass": bool(no_degen)})
        checks.append({"name": "modality_split_per_cell_z_shape_clean",
                       "pass": bool(shape_ok)})

    # Step 11 JIGSAWS-side split checks.
    jms = results.get("jigsaws_modality_split") or {}
    if jms and "_meta" in jms:
        mods = [m for m in jms.keys() if m != "_meta"]
        checks.append({
            "name": "jigsaws_modality_split_present",
            "pass": bool(set(mods) == {"gestures", "kinematics"}),
        })
        no_degen_j = all(
            int(jms[m][b]["bootstrap"].get("n_degenerate", 0)) == 0
            for m in mods for b in ("fixed", "tertile")
        )
        checks.append({
            "name": "jigsaws_modality_split_no_degenerate",
            "pass": bool(no_degen_j),
        })

    # Comparison A checks.
    ca = results.get("comparison_a") or {}
    if ca and "primary" in ca:
        pr = ca["primary"]
        shape_ok = tuple(pr["coupling_shape"]) == (
            int(sum(ca["tier_counts_j"])), int(sum(ca["tier_counts_e"])))
        checks.append({
            "name": "comparison_a_coupling_shape",
            "pass": bool(shape_ok),
        })
        bs = ca.get("bootstrap") or {}
        checks.append({
            "name": "comparison_a_bootstrap_no_degenerate",
            "pass": bool(int(bs.get("n_degenerate", 0)) == 0),
        })
        # Tier counts preserved between primary and bootstrap.
        preserved = (
            tuple(bs.get("tier_counts_j", ())) == tuple(ca["tier_counts_j"])
            and tuple(bs.get("tier_counts_e", ())) == tuple(ca["tier_counts_e"])
        )
        checks.append({
            "name": "comparison_a_bootstrap_tier_counts_preserved",
            "pass": bool(preserved),
        })

    # P1/P3/P4 diagnostics presence.
    if results.get("eeg_baseline_pc_correlation"):
        checks.append({"name": "eeg_baseline_pc_correlation_populated",
                       "pass": True})
    if results.get("eeg_pooled_random_split_null"):
        checks.append({"name": "eeg_pooled_random_split_null_populated",
                       "pass": True})
    if results.get("eye_coupling_diagnostic"):
        checks.append({"name": "eye_coupling_diagnostic_populated",
                       "pass": True})

    results["checks"] = checks
    return checks


def run(config: dict,
        task_modules_yaml: Path,
        data_root_path: Path,
        *,
        n_perms_override: int | None = None,
        subsample_override: int | None = None,
        smoke_fraction: float | None = None) -> dict:
    """Execute steps 1-10 and return the results dict that gets JSON-dumped."""
    reports_dir, plots_dir = ensure_output_dirs()
    log.info("data root: %s", data_root_path)
    log.info("reports:   %s", reports_dir)

    n_perms = int(n_perms_override or config["n_perms"])
    subsample_n = int(subsample_override or config["subsample_per_tier"])
    seed = int(config["seed"])
    modality_split_n_perms = int(
        config.get("modality_split_n_perms", n_perms))
    modality_split_n_bootstraps = int(
        config.get("modality_split_n_bootstraps",
                   config.get("trial_null_n_subsamples", 30)))

    # ===== Step 1: JIGSAWS features =====
    log.info("Step 1: building JIGSAWS feature frame ...")
    gesture_pool = list(config["gestures_pool"])
    jf = build_jigsaws_feature_frame(
        data_root_path, gesture_pool,
        gripper_open_eps=float(config["gripper_open_eps"]),
        economy_clip=float(config["economy_clip"]),
    )
    j_feat_cols = jigsaws_feat_cols(gesture_pool)
    log.info("  JIGSAWS trials: %d, feature dim: %d", len(jf), len(j_feat_cols))
    assert len(jf) > 0, "JIGSAWS feature frame is empty; check data root"
    jf = zscore_features(jf, j_feat_cols)

    if smoke_fraction is not None:
        jf = jf.sample(frac=smoke_fraction, random_state=seed).reset_index(drop=True)
        log.info("  smoke subsample -> JIGSAWS rows: %d", len(jf))

    # ===== Step 2: EEG/Eye features =====
    log.info("Step 2: building EEG/Eye feature frame ...")
    task_mod_map = load_task_module_map(task_modules_yaml)
    ef = build_eeg_eye_feature_frame(
        data_root_path, task_mod_map, try_filter=int(config["try_filter"]),
    )
    e_feat_cols = eeg_eye_feat_cols()
    log.info("  EEG/Eye Try==1 trials with cache hits: %d, feature dim: %d",
             len(ef), len(e_feat_cols))
    assert len(ef) > 0, "EEG/Eye feature frame is empty; check caches"
    # Z-score.
    for c in e_feat_cols:
        v = ef[c].to_numpy(dtype=np.float64)
        sd = v.std()
        ef[c] = ((v - v.mean()) / sd) if sd > 1e-12 else 0.0

    if smoke_fraction is not None:
        ef = ef.sample(frac=smoke_fraction, random_state=seed).reset_index(drop=True)
        log.info("  smoke subsample -> EEG/Eye rows: %d", len(ef))

    # ===== Step 3: residualization =====
    log.info("Step 3: residualizing nuisances ...")
    jr = residualize(jf, j_feat_cols,
                     categorical=["task", "surgeon"],
                     ordinal=["trial_index_within_surgeon_task"])
    er = residualize(ef, e_feat_cols,
                     categorical=["task_module", "subject_id", "dominant_hand"],
                     ordinal=["age"])

    max_post_r2_j = float(jr.post_fit_r2.max())
    max_post_r2_e = float(er.post_fit_r2.max())
    log.info("  JIGSAWS max post-fit R^2 = %.3e (should be ~0)", max_post_r2_j)
    log.info("  EEG/Eye max post-fit R^2 = %.3e (should be ~0)", max_post_r2_e)

    jr_frame = jr.residuals
    er_frame = er.residuals

    # ===== Step 4: tertile binning =====
    log.info("Step 4: tertile binning ...")
    jr_frame, j_cutoffs = add_tier_column(jr_frame, "grs_total", tier_col="tier")
    er_frame, e_cutoffs = add_tier_column(er_frame, "performance", tier_col="tier")

    # ===== Step 5: centroid RDMs =====
    log.info("Step 5: centroid RDMs ...")
    j_feat_mat = jr_frame[j_feat_cols].to_numpy(dtype=np.float64)
    e_feat_mat = er_frame[e_feat_cols].to_numpy(dtype=np.float64)
    rdm_j = centroid_rdm(j_feat_mat, jr_frame["tier"].to_numpy(), TIER_NAMES)
    rdm_e = centroid_rdm(e_feat_mat, er_frame["tier"].to_numpy(), TIER_NAMES)
    _heatmap(rdm_j, "JIGSAWS centroid RDM (Low/Mid/High)", TIER_NAMES,
             plots_dir / "rdm_jigsaws.png")
    _heatmap(rdm_e, "EEG/Eye centroid RDM (Low/Mid/High)", TIER_NAMES,
             plots_dir / "rdm_eeg_eye.png")

    # ===== Step 6: Gromov-Wasserstein =====
    log.info("Step 6: Gromov-Wasserstein ...")
    gw = gromov_wasserstein_centroids(rdm_j, rdm_e, TIER_NAMES)
    _heatmap(gw.coupling, "Headline GW coupling", TIER_NAMES,
             plots_dir / "coupling_headline.png")

    # ===== Step 7: permutation null =====
    log.info("Step 7: permutation null (%d perms)...", n_perms)
    null = permutation_null_centroid(
        j_feat_mat, jr_frame["tier"].to_numpy(),
        e_feat_mat, er_frame["tier"].to_numpy(),
        TIER_NAMES, n_perms=n_perms, seed=seed,
    )
    _null_hist(null["null"], null["observed"],
               plots_dir / "null_histogram.png",
               f"Tier-shuffle null (n={null['n_permutations']})")

    # ===== Step 8: OSATS axis breakdown =====
    log.info("Step 8: OSATS axis breakdown ...")
    osats_results: List[dict] = []
    null_cis: List[tuple[float, float]] = []
    for axis in JIGSAWS_OSATS_COLUMNS:
        colname = f"osats_{axis}"
        if colname not in jr_frame.columns:
            continue
        sub, _cut = add_tier_column(jr_frame, colname, tier_col=f"tier_{axis}")
        rdm_axis = centroid_rdm(
            sub[j_feat_cols].to_numpy(dtype=np.float64),
            sub[f"tier_{axis}"].to_numpy(), TIER_NAMES,
        )
        gw_axis = gromov_wasserstein_centroids(rdm_axis, rdm_e, TIER_NAMES)
        n_axis = permutation_null_centroid(
            sub[j_feat_cols].to_numpy(dtype=np.float64),
            sub[f"tier_{axis}"].to_numpy(),
            e_feat_mat, er_frame["tier"].to_numpy(),
            TIER_NAMES, n_perms=max(200, n_perms // 5), seed=seed,
        )
        osats_results.append({
            "axis": axis,
            "gw_distance": float(gw_axis.distance),
            "p_value": float(n_axis["p_value"]),
            "z_score": float(n_axis["z_score"]),
            "null_mean": float(n_axis["null_mean"]),
            "null_std": float(n_axis["null_std"]),
        })
        lo = float(np.quantile(n_axis["null"], 0.025))
        hi = float(np.quantile(n_axis["null"], 0.975))
        null_cis.append((lo, hi))
    _osats_bar(osats_results, null_cis, plots_dir / "osats_axis_alignment.png")

    # ===== Step 9 PRIMARY: all-trials block-null (unbalanced, no upsampling) =====
    log.info("Step 9a: all-trials primary block-null (NJ=%d x NE=%d)...",
             len(jr_frame), len(er_frame))

    def _egw_with_eps(Cj_i, Ce_i, eps):
        r = entropic_gromov_wasserstein(Cj_i, Ce_i, epsilon=float(eps))
        # Renormalize to doubly-stochastic marginals only if drift is non-trivial.
        T = np.asarray(r.coupling, dtype=np.float64)
        if max(np.max(np.abs(T.sum(axis=1) - 1.0 / T.shape[0])),
               np.max(np.abs(T.sum(axis=0) - 1.0 / T.shape[1]))) > 1e-9:
            T = _renormalize_uniform(T)
        return float(r.distance), T

    primary = trial_level_block_null_all_trials(
        features_j=jr_frame[j_feat_cols].to_numpy(dtype=np.float64),
        tiers_j=jr_frame["tier"].to_numpy(),
        features_e=er_frame[e_feat_cols].to_numpy(dtype=np.float64),
        tiers_e=er_frame["tier"].to_numpy(),
        tier_names=TIER_NAMES,
        epsilon=float(config["gw_epsilon"]),
        n_perms=n_perms,
        seed=seed,
        entropic_gw_fn=_egw_with_eps,
        pairwise_rdm_fn=pairwise_cosine_rdm,
    )
    _trial_block_null_plot(primary, plots_dir / "trial_level_block_null.png",
                           show_expected=True)

    # ===== Step 9e SENSITIVITY: balanced 100-per-tier with replacement =====
    log.info("Step 9e: balanced-subsample sensitivity (n=%d per tier)...", subsample_n)
    rng = np.random.default_rng(seed)

    def _balanced_sample(df: pd.DataFrame, feat_cols: Sequence[str], n: int) -> tuple[np.ndarray, np.ndarray]:
        parts_feat: List[np.ndarray] = []
        parts_tier: List[np.ndarray] = []
        for t in TIER_NAMES:
            sub = df[df["tier"] == t]
            if len(sub) == 0:
                raise RuntimeError(f"no rows in tier {t}")
            if len(sub) >= n:
                idx = rng.choice(len(sub), size=n, replace=False)
            else:
                idx = rng.choice(len(sub), size=n, replace=True)
            picked = sub.iloc[idx]
            parts_feat.append(picked[feat_cols].to_numpy(dtype=np.float64))
            parts_tier.append(np.asarray(picked["tier"]))
        return np.vstack(parts_feat), np.concatenate(parts_tier)

    fj, tj = _balanced_sample(jr_frame, j_feat_cols, subsample_n)
    fe, te = _balanced_sample(er_frame, e_feat_cols, subsample_n)
    Cj = pairwise_cosine_rdm(fj); Ce = pairwise_cosine_rdm(fe)
    trial_null_result: dict = {}
    trial_coupling_marginal_drift = float("nan")
    entropic_gw_distance = float("nan")
    try:
        egw = entropic_gromov_wasserstein(Cj, Ce, epsilon=float(config["gw_epsilon"]))
        _trial_coupling_plot(egw.coupling, subsample_n,
                             plots_dir / "coupling_trial_level.png")
        entropic_gw_distance = float(egw.distance)
        trial_coupling_marginal_drift = max(
            float(np.max(np.abs(egw.coupling.sum(axis=0) - 1.0 / egw.coupling.shape[1]))),
            float(np.max(np.abs(egw.coupling.sum(axis=1) - 1.0 / egw.coupling.shape[0]))),
        )
        T_used = _renormalize_uniform(egw.coupling)
        trial_null_result = trial_level_block_null(
            T_used, tj, te, TIER_NAMES, n_perms=n_perms, seed=seed,
        )
    except Exception as e:
        log.warning("balanced subsample block null failed: %s", e)

    # ----- Step 9c: bootstrap robustness with per-cell z distributions -----
    robustness_records: List[dict] = []
    robustness_summary: dict = {}
    n_subsamples = int(config.get("trial_null_n_subsamples", 30))
    if n_subsamples > 0 and not smoke_fraction:
        log.info("Step 9c: bootstrap robustness (B=%d, per-cell z)...", n_subsamples)

        def _egw_dist_coupling(Cj_i: np.ndarray, Ce_i: np.ndarray):
            r = entropic_gromov_wasserstein(Cj_i, Ce_i,
                                            epsilon=float(config["gw_epsilon"]))
            return r.distance, _renormalize_uniform(r.coupling)

        sr = subsample_robustness(
            feat_j=jr_frame[j_feat_cols].to_numpy(dtype=np.float64),
            tiers_j_full=jr_frame["tier"].to_numpy(),
            feat_e=er_frame[e_feat_cols].to_numpy(dtype=np.float64),
            tiers_e_full=er_frame["tier"].to_numpy(),
            tier_names=TIER_NAMES,
            entropic_gw_fn=_egw_dist_coupling,
            pairwise_rdm_fn=pairwise_cosine_rdm,
            n_per_tier=subsample_n,
            base_seed=seed,
            n_subsamples=n_subsamples,
            n_perms=max(200, n_perms // 5),
        )
        robustness_records = [
            {"run": b, "diag_mass": float(sr.observed[b]),
             "p_value": float(sr.p_values[b]),
             "z_score": float(sr.z_scores[b]),
             "gw_distance": float(sr.gw_distance[b]),
             "per_cell_z": sr.per_cell_z[b].tolist()}
            for b in range(sr.n_runs)
        ]
        robustness_summary = {
            "n_runs": sr.n_runs,
            "observed_median": sr.observed_median,
            "observed_p05": sr.observed_p05,
            "observed_p95": sr.observed_p95,
            "frac_p_lt_05": sr.frac_p_lt_05,
            "per_cell_median": sr.per_cell_median.tolist(),
            "per_cell_p05": sr.per_cell_p05.tolist(),
            "per_cell_p95": sr.per_cell_p95.tolist(),
            "frac_significant_per_cell_bonf":
                sr.frac_significant_per_cell_bonf.tolist(),
            "bonferroni_z_threshold": BONFERRONI_Z_THREE_CELLS,
            "tier_names": list(sr.tier_names),
        }
        _robustness_plot(sr, plots_dir / "trial_level_robustness.png")
        _per_cell_bootstrap_plot(sr, plots_dir / "trial_level_per_cell_bootstrap.png")

    # ----- Step 9d: epsilon sensitivity on the ALL-TRIALS coupling -----
    epsilon_rows: List[dict] = []
    try:
        Cj_all = pairwise_cosine_rdm(jr_frame[j_feat_cols].to_numpy(dtype=np.float64))
        Ce_all = pairwise_cosine_rdm(er_frame[e_feat_cols].to_numpy(dtype=np.float64))
        eps_results = epsilon_sensitivity(
            Cj=Cj_all, Ce=Ce_all,
            tiers_j=jr_frame["tier"].to_numpy(),
            tiers_e=er_frame["tier"].to_numpy(),
            tier_names=TIER_NAMES,
            entropic_gw_fn_eps=_egw_with_eps,
            epsilons=(0.005, 0.01, 0.02, 0.05),
            n_perms=max(200, n_perms // 5),
            seed=seed,
        )
        for row in eps_results:
            epsilon_rows.append({
                "epsilon": row["epsilon"],
                "diag_mass": row["diag_mass"],
                "gw_distance": row["gw_distance"],
                "per_cell_z": np.asarray(row["per_cell_z"]).tolist(),
                "per_cell_observed": np.asarray(row["per_cell_observed"]).tolist(),
                "row_sum_drift": row["row_sum_drift"],
                "col_sum_drift": row["col_sum_drift"],
            })
        _epsilon_sensitivity_plot(
            epsilon_rows, primary,
            plots_dir / "trial_level_epsilon_sensitivity.png")
    except Exception as e:
        log.warning("epsilon sensitivity failed: %s", e)

    # ----- Step 9f: fixed-cutoff sensitivity (optional Change 4) -----
    fixed_cutoff_result: dict = {}
    try:
        log.info("Step 9f: fixed-cutoff sensitivity (JIGSAWS <16/16-22/>22, "
                 "Mimic <70/70-85/>85)...")
        jt_fixed = assign_fixed_tier(jr_frame["grs_total"].to_numpy(), 16, 22)
        et_fixed = assign_fixed_tier(er_frame["performance"].to_numpy(), 70, 85)
        # Expose fixed-cutoff labels on the frames so downstream cells (and
        # Step 9g's stratified bootstrap) can pick them up by column name.
        jr_frame = jr_frame.assign(tier_fixed=jt_fixed)
        er_frame = er_frame.assign(tier_fixed=et_fixed)

        if all((jt_fixed == t).any() for t in TIER_NAMES) and \
           all((et_fixed == t).any() for t in TIER_NAMES):
            fixed = trial_level_block_null_all_trials(
                features_j=jr_frame[j_feat_cols].to_numpy(dtype=np.float64),
                tiers_j=jt_fixed,
                features_e=er_frame[e_feat_cols].to_numpy(dtype=np.float64),
                tiers_e=et_fixed,
                tier_names=TIER_NAMES,
                epsilon=float(config["gw_epsilon"]),
                n_perms=n_perms, seed=seed,
                entropic_gw_fn=_egw_with_eps,
                pairwise_rdm_fn=pairwise_cosine_rdm,
            )
            fixed_cutoff_result = {
                "jigsaws_cutoffs": {"low": 16, "high": 22},
                "eeg_eye_cutoffs": {"low": 70, "high": 85},
                "tier_counts_j": fixed["tier_counts_j"].tolist(),
                "tier_counts_e": fixed["tier_counts_e"].tolist(),
                "diag_mass_observed": float(fixed["observed"]),
                "diag_mass_expected": float(fixed["expected_trace_under_null"]),
                "p_value": float(fixed["p_value"]),
                "z_score": float(fixed["z_score"]),
                "per_cell_z": np.asarray(fixed["per_cell_z"]).tolist(),
                "per_cell_observed": np.asarray(fixed["per_cell_observed"]).tolist(),
                "per_cell_expected": np.asarray(fixed["per_cell_expected"]).tolist(),
                "block_mass": np.asarray(fixed["block_mass"]).tolist(),
            }
        else:
            log.warning("fixed-cutoff binning left at least one tier empty; "
                        "skipping Step 9f.")
            fixed_cutoff_result = {"skipped_reason": "empty_tier"}
    except Exception as e:
        log.warning("fixed-cutoff sensitivity failed: %s", e)
        fixed_cutoff_result = {"skipped_reason": str(e)}

    # ----- Step 9g: stratified bootstrap on fixed-cutoff AND tertile bins -----
    fixed_bootstrap_serialized: dict = {}
    fixed_bootstrap_verdict: dict = {}
    tertile_bootstrap_serialized: dict = {}
    tertile_bootstrap_verdict: dict = {}
    n_bootstraps = int(config.get("trial_null_n_subsamples", 30))
    if n_bootstraps > 0 and not smoke_fraction and "tier_fixed" in jr_frame.columns:
        feat_j_arr = jr_frame[j_feat_cols].to_numpy(dtype=np.float64)
        feat_e_arr = er_frame[e_feat_cols].to_numpy(dtype=np.float64)

        log.info("Step 9g: fixed-cutoff stratified bootstrap (B=%d)...",
                 n_bootstraps)
        try:
            fb = subsample_robustness_stratified(
                features_j=feat_j_arr,
                tiers_j=jr_frame["tier_fixed"].to_numpy(),
                features_e=feat_e_arr,
                tiers_e=er_frame["tier_fixed"].to_numpy(),
                tier_names=TIER_NAMES,
                epsilon=float(config["gw_epsilon"]),
                n_perms=max(200, n_perms // 5),
                n_bootstraps=n_bootstraps, seed=seed,
                entropic_gw_fn=_egw_with_eps,
                pairwise_rdm_fn=pairwise_cosine_rdm,
            )
            fixed_bootstrap_serialized = _serialize_stratified(fb)
            fixed_bootstrap_verdict = stratified_bootstrap_verdict(fb, TIER_NAMES)
            _stratified_bootstrap_plot(
                fb, fixed_bootstrap_verdict,
                plots_dir / "fixed_cutoff_bootstrap.png",
                title=f"Fixed-cutoff stratified bootstrap (B = {fb['n_bootstraps']})")
        except Exception as e:
            log.warning("fixed-cutoff stratified bootstrap failed: %s", e)

        log.info("Step 9g': tertile stratified bootstrap (apples-to-apples)...")
        try:
            tb = subsample_robustness_stratified(
                features_j=feat_j_arr, tiers_j=jr_frame["tier"].to_numpy(),
                features_e=feat_e_arr, tiers_e=er_frame["tier"].to_numpy(),
                tier_names=TIER_NAMES,
                epsilon=float(config["gw_epsilon"]),
                n_perms=max(200, n_perms // 5),
                n_bootstraps=n_bootstraps, seed=seed,
                entropic_gw_fn=_egw_with_eps,
                pairwise_rdm_fn=pairwise_cosine_rdm,
            )
            tertile_bootstrap_serialized = _serialize_stratified(tb)
            tertile_bootstrap_verdict = stratified_bootstrap_verdict(tb, TIER_NAMES)
            _stratified_bootstrap_plot(
                tb, tertile_bootstrap_verdict,
                plots_dir / "tertile_bootstrap_stratified.png",
                title=f"Tertile stratified bootstrap (B = {tb['n_bootstraps']})")
        except Exception as e:
            log.warning("tertile stratified bootstrap failed: %s", e)

    # ===== Step 10: Mimic-side modality split =====
    modality_serialized: dict = {}
    mod_cols_split = mimic_modality_columns()
    if ("tier_fixed" in jr_frame.columns and "tier_fixed" in er_frame.columns
            and not smoke_fraction):
        log.info("Step 10: Mimic-side modality split "
                 "(baseline / predictive-coding / eye) "
                 "(B=%d, n_perms=%d)...",
                 modality_split_n_bootstraps, modality_split_n_perms)
        try:
            mod_res = modality_split_analysis(
                features_j=jr_frame[j_feat_cols].to_numpy(dtype=np.float64),
                tiers_j_tertile=jr_frame["tier"].to_numpy(),
                tiers_j_fixed=jr_frame["tier_fixed"].to_numpy(),
                features_e_full=er_frame[e_feat_cols].to_numpy(dtype=np.float64),
                e_cols_full=e_feat_cols,
                modality_columns=mod_cols_split,
                tiers_e_tertile=er_frame["tier"].to_numpy(),
                tiers_e_fixed=er_frame["tier_fixed"].to_numpy(),
                tier_names=TIER_NAMES,
                epsilon=float(config["gw_epsilon"]),
                n_perms=modality_split_n_perms,
                n_bootstraps=modality_split_n_bootstraps,
                seed=seed,
                entropic_gw_fn=_egw_with_eps,
                pairwise_rdm_fn=pairwise_cosine_rdm,
            )
            modality_serialized = _serialize_modality_split(mod_res)
            _modality_split_plot(
                mod_res, plots_dir / "modality_split.png")
        except Exception as e:
            log.warning("modality split failed: %s", e)

    # ----- P1: baseline/pc EEG correlation diagnostic -----
    eeg_corr: dict = {}
    try:
        log.info("Step 10 diagnostics: baseline/pc EEG correlation...")
        corr = eeg_baseline_pc_correlation(
            features_e_full=er_frame[e_feat_cols].to_numpy(dtype=np.float64),
            baseline_cols=mod_cols_split["eeg_baseline"],
            pc_cols=mod_cols_split["eeg_predictive_coding"],
            e_cols_full=e_feat_cols,
        )
        eeg_corr = {k: v for k, v in corr.items() if k != "per_trial_pearson_r"}
        eeg_corr["per_trial_pearson_r"] = corr["per_trial_pearson_r"].tolist()
        _eeg_corr_plot(corr, plots_dir / "eeg_baseline_pc_correlation.png")
    except Exception as e:
        log.warning("EEG correlation diagnostic failed: %s", e)

    # ----- P3: pooled-128 EEG random-split negative control -----
    eeg_pooled_null: dict = {}
    if modality_serialized and "tier_fixed" in jr_frame.columns and not smoke_fraction:
        try:
            log.info("Step 10 diagnostics: pooled-128 EEG random-split null...")
            pooled = pooled_eeg_random_split_null(
                features_j=jr_frame[j_feat_cols].to_numpy(dtype=np.float64),
                tiers_j=jr_frame["tier_fixed"].to_numpy(),
                features_e_full=er_frame[e_feat_cols].to_numpy(dtype=np.float64),
                e_cols_full=e_feat_cols,
                eeg_combined_cols=(mod_cols_split["eeg_baseline"]
                                   + mod_cols_split["eeg_predictive_coding"]),
                half_size=len(mod_cols_split["eeg_baseline"]),
                tiers_e=er_frame["tier_fixed"].to_numpy(),
                tier_names=TIER_NAMES,
                epsilon=float(config["gw_epsilon"]),
                n_perms=200, n_random_splits=20, seed=seed,
                entropic_gw_fn=_egw_with_eps,
                pairwise_rdm_fn=pairwise_cosine_rdm,
            )
            # Observed baseline-vs-pc z delta under fixed-cutoff binning.
            fb_zs = {m: modality_serialized[m]["fixed"]["primary"]["trace_z_score"]
                     for m in ("eeg_baseline", "eeg_predictive_coding")}
            observed_delta = float(
                fb_zs["eeg_baseline"] - fb_zs["eeg_predictive_coding"])
            null_delta = np.asarray(pooled["delta_distribution"])
            # Two-sided p = (1 + #{|null| >= |obs|}) / (1 + n).
            n = null_delta.size
            p_two_sided = (1.0 + float((np.abs(null_delta)
                                        >= abs(observed_delta)).sum())) / (1.0 + n)
            eeg_pooled_null = {
                "n_random_splits": int(pooled["n_random_splits"]),
                "half_size": int(pooled["half_size"]),
                "delta_distribution": null_delta.tolist(),
                "delta_median": float(pooled["delta_median"]),
                "delta_p05": float(pooled["delta_p05"]),
                "delta_p95": float(pooled["delta_p95"]),
                "delta_abs_median": float(pooled["delta_abs_median"]),
                "delta_abs_p95": float(pooled["delta_abs_p95"]),
                "observed_delta_baseline_minus_pc": observed_delta,
                "p_two_sided": float(p_two_sided),
            }
            _eeg_pooled_plot(eeg_pooled_null,
                             plots_dir / "eeg_pooled_random_split.png")
        except Exception as e:
            log.warning("pooled-128 EEG null failed: %s", e)

    # ----- P4: eye-only coupling-matrix diagnostic -----
    eye_coupling_diag: dict = {}
    if modality_serialized:
        try:
            log.info("Step 10 diagnostics: eye-only coupling matrix check...")
            eye_cols = mod_cols_split["eye"]
            col_to_idx = {c: i for i, c in enumerate(e_feat_cols)}
            eye_idx = np.array([col_to_idx[c] for c in eye_cols], dtype=np.int64)
            eye_feat = er_frame[e_feat_cols].to_numpy(dtype=np.float64)[:, eye_idx]
            eye_primary = mod_res["eye"]["fixed"]["primary"]
            # Reconstruct the eye coupling cheaply (fixed-cutoff primary).
            # The primary dict does not persist the full coupling; we just run
            # the GW one more time to inspect the coupling directly.
            Cj_eye = pairwise_cosine_rdm(
                jr_frame[j_feat_cols].to_numpy(dtype=np.float64))
            Ce_eye = pairwise_cosine_rdm(eye_feat)
            _dist, T_eye = _egw_with_eps(Cj_eye, Ce_eye,
                                         float(config["gw_epsilon"]))
            eye_coupling_diag = coupling_matrix_diagnostics(T_eye, eye_feat)
        except Exception as e:
            log.warning("eye coupling diagnostic failed: %s", e)

    # ===== Step 11: JIGSAWS-side modality split (P2) =====
    jigsaws_modality_serialized: dict = {}
    if ("tier_fixed" in jr_frame.columns and not smoke_fraction):
        log.info("Step 11: JIGSAWS-side modality split "
                 "(gestures / kinematics) (B=%d, n_perms=%d)...",
                 modality_split_n_bootstraps, modality_split_n_perms)
        try:
            j_mod_cols = jigsaws_modality_columns(list(config["gestures_pool"]))
            jmod_res = jigsaws_modality_split_analysis(
                features_j_full=jr_frame[j_feat_cols].to_numpy(dtype=np.float64),
                j_cols_full=j_feat_cols,
                modality_columns=j_mod_cols,
                tiers_j_tertile=jr_frame["tier"].to_numpy(),
                tiers_j_fixed=jr_frame["tier_fixed"].to_numpy(),
                features_e=er_frame[e_feat_cols].to_numpy(dtype=np.float64),
                tiers_e_tertile=er_frame["tier"].to_numpy(),
                tiers_e_fixed=er_frame["tier_fixed"].to_numpy(),
                tier_names=TIER_NAMES,
                epsilon=float(config["gw_epsilon"]),
                n_perms=modality_split_n_perms,
                n_bootstraps=modality_split_n_bootstraps,
                seed=seed,
                entropic_gw_fn=_egw_with_eps,
                pairwise_rdm_fn=pairwise_cosine_rdm,
            )
            jigsaws_modality_serialized = _serialize_modality_split(
                jmod_res, jigsaws_side=True)
            _modality_split_plot(
                jmod_res, plots_dir / "jigsaws_modality_split.png",
                title=f"Step 11 — JIGSAWS-side modality split "
                       f"(stratified bootstrap, B = {jmod_res['_meta']['n_bootstraps']})")
        except Exception as e:
            log.warning("JIGSAWS-side modality split failed: %s", e)

    # ===== COMPARISON A: practice manifold =====
    # NEW Comparison A residualization drops the tier-defining nuisance so the
    # skill / practice signal survives. JIGSAWS side drops `surgeon` because
    # the N/I/E label is constant per surgeon; Mimic side drops `subject_id`
    # because experience_trials is constant per subject.
    comparison_a_results: dict = {}
    if not smoke_fraction:
        log.info("Comparison A: residualization (task+trial_index on J; "
                 "task_module+dominant_hand+age on Mimic)...")
        try:
            from skill_manifold.residualize import residualize as _residualize
            jr_a = _residualize(
                jf, j_feat_cols,
                categorical=["task"],
                ordinal=["trial_index_within_surgeon_task"],
            )
            er_a = _residualize(
                ef, e_feat_cols,
                categorical=["task_module", "dominant_hand"],
                ordinal=["age"],
            )
            jf_a = jr_a.residuals.copy()
            ef_a = er_a.residuals.copy()

            # JIGSAWS tiers: categorical E/I/N -> Low/Mid/High.
            jf_a["tier"] = assign_jigsaws_skill_tier(jf_a["skill"].to_numpy())
            # Mimic tiers: tertile of per-subject experience_trials (practice depth).
            ef_a, exp_cutoffs = add_tier_column(
                ef_a, "experience_trials", tier_col="tier")

            log.info("Comparison A: all-trials block-null (JIGSAWS E/I/N x "
                     "Mimic experience-tertile)...")
            comp_a_primary = trial_level_block_null_all_trials(
                features_j=jf_a[j_feat_cols].to_numpy(dtype=np.float64),
                tiers_j=jf_a["tier"].to_numpy(),
                features_e=ef_a[e_feat_cols].to_numpy(dtype=np.float64),
                tiers_e=ef_a["tier"].to_numpy(),
                tier_names=TIER_NAMES,
                epsilon=float(config["gw_epsilon"]),
                n_perms=n_perms, seed=seed,
                entropic_gw_fn=_egw_with_eps,
                pairwise_rdm_fn=pairwise_cosine_rdm,
            )
            log.info("Comparison A: stratified bootstrap (B=%d)...",
                     int(config.get("trial_null_n_subsamples", 30)))
            comp_a_bootstrap = subsample_robustness_stratified(
                features_j=jf_a[j_feat_cols].to_numpy(dtype=np.float64),
                tiers_j=jf_a["tier"].to_numpy(),
                features_e=ef_a[e_feat_cols].to_numpy(dtype=np.float64),
                tiers_e=ef_a["tier"].to_numpy(),
                tier_names=TIER_NAMES,
                epsilon=float(config["gw_epsilon"]),
                n_perms=max(200, n_perms // 5),
                n_bootstraps=int(config.get("trial_null_n_subsamples", 30)),
                seed=seed,
                entropic_gw_fn=_egw_with_eps,
                pairwise_rdm_fn=pairwise_cosine_rdm,
            )
            comp_a_verdict = stratified_bootstrap_verdict(
                comp_a_bootstrap, TIER_NAMES)

            comparison_a_results = {
                "description": (
                    "Comparison A — practice manifold. JIGSAWS tiered by "
                    "self-reported E/I/N (N->Low, I->Mid, E->High); Mimic "
                    "tiered by tertile of per-subject non-first-try "
                    "experience_trials count (practice-depth proxy)."),
                "tier_counts_j": comp_a_primary["tier_counts_j"].tolist(),
                "tier_counts_e": comp_a_primary["tier_counts_e"].tolist(),
                "mimic_experience_cutoffs": exp_cutoffs.as_dict(),
                "primary": {
                    "observed_diag_mass": float(comp_a_primary["observed"]),
                    "expected_trace_under_null":
                        float(comp_a_primary["expected_trace_under_null"]),
                    "null_mean": float(comp_a_primary["null_mean"]),
                    "null_std": float(comp_a_primary["null_std"]),
                    "trace_p_value": float(comp_a_primary["p_value"]),
                    "trace_z_score": float(comp_a_primary["z_score"]),
                    "per_cell_z": np.asarray(
                        comp_a_primary["per_cell_z"]).tolist(),
                    "per_cell_observed": np.asarray(
                        comp_a_primary["per_cell_observed"]).tolist(),
                    "per_cell_expected": np.asarray(
                        comp_a_primary["per_cell_expected"]).tolist(),
                    "block_mass": np.asarray(
                        comp_a_primary["block_mass"]).tolist(),
                    "expected_block_mass": np.asarray(
                        comp_a_primary["expected_block_mass"]).tolist(),
                    "coupling_shape": list(comp_a_primary["coupling_shape"]),
                    "row_sum_drift": float(comp_a_primary["row_sum_drift"]),
                    "col_sum_drift": float(comp_a_primary["col_sum_drift"]),
                    "n_permutations": int(comp_a_primary["n_permutations"]),
                    "gw_distance": float(comp_a_primary["gw_distance"]),
                    "epsilon": float(comp_a_primary["epsilon"]),
                    "tier_names": list(comp_a_primary["tier_names"]),
                },
                "bootstrap": _serialize_stratified(comp_a_bootstrap),
                "verdict": dict(comp_a_verdict),
                "residualization_diagnostics": {
                    "jigsaws_max_r2_feature": float(jr_a.r2_per_feature.max()),
                    "jigsaws_max_post_fit_r2":
                        float(jr_a.post_fit_r2.max()),
                    "mimic_max_r2_feature": float(er_a.r2_per_feature.max()),
                    "mimic_max_post_fit_r2":
                        float(er_a.post_fit_r2.max()),
                    "jigsaws_nuisances": ["task", "trial_index_within_surgeon_task"],
                    "mimic_nuisances":   ["task_module", "dominant_hand", "age"],
                },
            }
            _trial_block_null_plot(
                comp_a_primary,
                plots_dir / "comparison_a_block_null.png",
                show_expected=True)
            _stratified_bootstrap_plot(
                comp_a_bootstrap, comp_a_verdict,
                plots_dir / "comparison_a_bootstrap.png",
                title=f"Comparison A stratified bootstrap (B = {comp_a_bootstrap['n_bootstraps']})")
        except Exception as e:
            log.warning("Comparison A failed: %s", e)
            comparison_a_results = {"skipped_reason": str(e)}

    # ===== Step 12 (ex-Step 11): MDS plots =====
    log.info("Step 12: MDS plots ...")
    _mds_plot(j_feat_mat, jr_frame["tier"].to_numpy(),
              plots_dir / "mds_jigsaws.png", "JIGSAWS residualized features (MDS)")
    _mds_plot(e_feat_mat, er_frame["tier"].to_numpy(),
              plots_dir / "mds_eeg_eye.png", "EEG/Eye residualized features (MDS)")

    # ---- aggregate result dict ----
    results = {
        "config_used": {"n_perms": n_perms, "gw_epsilon": float(config["gw_epsilon"]),
                         "subsample_per_tier": subsample_n, "seed": seed,
                         "smoke_fraction": smoke_fraction,
                         "modality_split_n_perms": modality_split_n_perms,
                         "modality_split_n_bootstraps": modality_split_n_bootstraps},
        "coverage": {
            "jigsaws_trials": int(len(jr_frame)),
            "eeg_eye_trials": int(len(er_frame)),
        },
        "tertile_cutoffs": {
            "jigsaws_grs_total": j_cutoffs.as_dict(),
            "eeg_eye_performance": e_cutoffs.as_dict(),
        },
        "rdm_jigsaws": rdm_j.tolist(),
        "rdm_eeg_eye": rdm_e.tolist(),
        "headline_gw": {
            "distance": float(gw.distance),
            "coupling": gw.coupling.tolist(),
            "argmax_assignment": gw.argmax_assignment,
        },
        "null": {
            "observed": float(null["observed"]),
            "p_value": float(null["p_value"]),
            "z_score": float(null["z_score"]),
            "null_mean": float(null["null_mean"]),
            "null_std": float(null["null_std"]),
            "n_permutations": int(null["n_permutations"]),
            "n_degenerate": int(null["n_degenerate"]),
        },
        "osats_axis_alignment": osats_results,
        "entropic_gw_trial_level": {
            "distance": entropic_gw_distance,
            "n_per_tier": subsample_n,
            "epsilon": float(config["gw_epsilon"]),
            "coupling_marginal_drift": trial_coupling_marginal_drift,
        },
        # PRIMARY trial-level: all-trials, unbalanced, uses every trial once.
        "trial_level_null_primary": {
            "observed_diag_mass": float(primary["observed"]),
            "expected_trace_under_null": float(primary["expected_trace_under_null"]),
            "null_mean": float(primary["null_mean"]),
            "null_std": float(primary["null_std"]),
            "p_value": float(primary["p_value"]),
            "z_score": float(primary["z_score"]),
            "n_permutations": int(primary["n_permutations"]),
            "coupling_shape": list(primary["coupling_shape"]),
            "row_sum_drift": float(primary["row_sum_drift"]),
            "col_sum_drift": float(primary["col_sum_drift"]),
            "gw_distance": float(primary["gw_distance"]),
            "epsilon": float(primary["epsilon"]),
            "tier_names": list(primary["tier_names"]),
            "tier_counts_j": np.asarray(primary["tier_counts_j"]).tolist(),
            "tier_counts_e": np.asarray(primary["tier_counts_e"]).tolist(),
            "block_mass": np.asarray(primary["block_mass"]).tolist(),
            "expected_block_mass": np.asarray(primary["expected_block_mass"]).tolist(),
            "per_cell_z": np.asarray(primary["per_cell_z"]).tolist(),
            "per_cell_observed": np.asarray(primary["per_cell_observed"]).tolist(),
            "per_cell_expected": np.asarray(primary["per_cell_expected"]).tolist(),
        },
        # SENSITIVITY: balanced 100-per-tier (upsampled on the JIGSAWS side).
        "trial_level_null_balanced": (
            {
                "observed_diag_mass": float(trial_null_result["observed"]),
                "null_mean": float(trial_null_result["null_mean"]),
                "null_std": float(trial_null_result["null_std"]),
                "p_value": float(trial_null_result["p_value"]),
                "z_score": float(trial_null_result["z_score"]),
                "n_permutations": int(trial_null_result["n_permutations"]),
                "tier_names": list(trial_null_result["tier_names"]),
                "block_mass": np.asarray(trial_null_result["block_mass"]).tolist(),
                "per_cell_z": np.asarray(trial_null_result["per_cell_z"]).tolist(),
                "per_cell_observed": np.asarray(
                    trial_null_result["per_cell_observed"]).tolist(),
            }
            if trial_null_result else None
        ),
        # Backwards-compat alias: older consumers read `trial_level_null`.
        "trial_level_null": {
            "observed_diag_mass": float(primary["observed"]),
            "expected_trace_under_null": float(primary["expected_trace_under_null"]),
            "null_mean": float(primary["null_mean"]),
            "null_std": float(primary["null_std"]),
            "p_value": float(primary["p_value"]),
            "z_score": float(primary["z_score"]),
            "n_permutations": int(primary["n_permutations"]),
            "tier_names": list(primary["tier_names"]),
            "block_mass": np.asarray(primary["block_mass"]).tolist(),
            "per_cell_z": np.asarray(primary["per_cell_z"]).tolist(),
            "per_cell_observed": np.asarray(primary["per_cell_observed"]).tolist(),
        },
        "trial_level_robustness": {
            "summary": robustness_summary,
            "per_run": robustness_records,
        },
        "epsilon_sensitivity": epsilon_rows,
        "fixed_cutoff_sensitivity": fixed_cutoff_result,
        "fixed_cutoff_bootstrap": fixed_bootstrap_serialized,
        "fixed_cutoff_bootstrap_verdict": fixed_bootstrap_verdict,
        "tertile_bootstrap_stratified": tertile_bootstrap_serialized,
        "tertile_bootstrap_stratified_verdict": tertile_bootstrap_verdict,
        "modality_split": modality_serialized,
        "eeg_baseline_pc_correlation": eeg_corr,
        "eeg_pooled_random_split_null": eeg_pooled_null,
        "eye_coupling_diagnostic": eye_coupling_diag,
        "jigsaws_modality_split": jigsaws_modality_serialized,
        "comparison_a": comparison_a_results,
        "residualization_diagnostics": {
            "max_post_fit_r2_jigsaws": max_post_r2_j,
            "max_post_fit_r2_eeg_eye": max_post_r2_e,
        },
    }
    _ensure_assertions(rdm_j, rdm_e, gw.coupling, null, results,
                       trial_null_result=primary)

    (reports_dir / "results_comparison_B.json").write_text(
        json.dumps(results, indent=2))
    _write_markdown_report(results, reports_dir / "report_comparison_B.md")
    _print_self_check(results)
    return results


def _write_markdown_report(results: dict, path: Path) -> None:
    cov = results["coverage"]
    hgw = results["headline_gw"]
    nl = results["null"]
    lines = []
    lines.append("# Comparison B — GW Skill Manifold\n")
    lines.append(f"**JIGSAWS trials:** {cov['jigsaws_trials']}  "
                 f"&nbsp;&nbsp; **EEG/Eye trials:** {cov['eeg_eye_trials']}")
    lines.append("")
    lines.append("## Headline result")
    lines.append("")
    lines.append(f"- GW distance: **{hgw['distance']:.6f}**")
    lines.append(f"- Permutation p-value: **{nl['p_value']:.4f}**  "
                 f"(z = {nl['z_score']:.2f} over {nl['n_permutations']} perms)")
    lines.append(f"- Argmax tier-to-tier assignment: `{hgw['argmax_assignment']}`")
    lines.append("")
    lines.append("![rdm_jigsaws](plots/rdm_jigsaws.png)")
    lines.append("![rdm_eeg_eye](plots/rdm_eeg_eye.png)")
    lines.append("![coupling_headline](plots/coupling_headline.png)")
    lines.append("![null](plots/null_histogram.png)")
    lines.append("")
    lines.append("## OSATS axis-alignment")
    lines.append("")
    lines.append("| axis | GW | p | z |")
    lines.append("|---|---|---|---|")
    for r in results["osats_axis_alignment"]:
        lines.append(
            f"| {r['axis']} | {r['gw_distance']:.4f} | {r['p_value']:.4f} | {r['z_score']:.2f} |"
        )
    lines.append("")
    lines.append("![osats](plots/osats_axis_alignment.png)")
    lines.append("")
    lines.append("## Trial-level entropic GW")
    egw = results["entropic_gw_trial_level"]
    lines.append(f"- distance: **{egw['distance']}**, n per tier = {egw['n_per_tier']}, epsilon = {egw['epsilon']}")
    drift = egw.get("coupling_marginal_drift")
    if drift is not None and np.isfinite(drift):
        lines.append(f"- coupling marginal drift from uniform: {drift:.2e}")
    lines.append("![trial_coupling](plots/coupling_trial_level.png)")
    lines.append("")

    tn_primary = results.get("trial_level_null_primary")
    if tn_primary:
        names = tn_primary.get("tier_names", ["Low", "Mid", "High"])
        per_z = tn_primary.get("per_cell_z", [])
        cs = tn_primary.get("coupling_shape", [])
        lines.append("## Trial-level block-diagonality null (all trials, primary)")
        lines.append("")
        lines.append(
            f"On the full {cs[0]} x {cs[1]} coupling (entropic GW, epsilon = "
            f"{tn_primary['epsilon']}), observed `diag_mass = "
            f"{tn_primary['observed_diag_mass']:.3f}` vs expected "
            f"`{tn_primary['expected_trace_under_null']:.3f}` under the "
            f"tier-shuffle null (p = **{tn_primary['p_value']:.4f}**, z = "
            f"**{tn_primary['z_score']:+.2f}**, "
            f"{tn_primary['n_permutations']} permutations).")
        lines.append("")
        if len(per_z) == len(names):
            zs = ", ".join(f"{n}<->{n} {z:+.2f}" for n, z in zip(names, per_z))
            lines.append(f"Per-cell diagonal z-scores: {zs}. "
                         f"Bonferroni-corrected threshold for 3 diagonal cells "
                         f"two-sided is |z| > {BONFERRONI_Z_THREE_CELLS:.2f}.")
        lines.append("")
        lines.append(f"Pre-renorm coupling drift: row "
                     f"{tn_primary['row_sum_drift']:.2e}, col "
                     f"{tn_primary['col_sum_drift']:.2e}.")
        lines.append("")
        lines.append("![trial_block_null](plots/trial_level_block_null.png)")
        lines.append("")

    rb = results.get("trial_level_robustness", {}).get("summary", {})
    if rb:
        names = rb.get("tier_names", ["Low", "Mid", "High"])
        lines.append("### Bootstrap robustness (B = {}, balanced 100-per-tier with replacement)".format(rb["n_runs"]))
        lines.append("")
        med = rb.get("per_cell_median", [])
        p05 = rb.get("per_cell_p05", [])
        p95 = rb.get("per_cell_p95", [])
        fbonf = rb.get("frac_significant_per_cell_bonf", [])
        for i, n in enumerate(names):
            if i < len(med):
                lines.append(
                    f"- {n}<->{n}: median {med[i]:+.2f} "
                    f"(5-95%: {p05[i]:+.2f} to {p95[i]:+.2f}); "
                    f"{int(fbonf[i] * rb['n_runs'])}/{rb['n_runs']} exceed "
                    f"|z| > {BONFERRONI_Z_THREE_CELLS:.2f}")
        lines.append("")
        lines.append(
            f"Median trace `diag_mass` = {rb['observed_median']:.3f} "
            f"(5-95%: {rb['observed_p05']:.3f}-{rb['observed_p95']:.3f}); "
            f"trace p < 0.05 in "
            f"{int(rb['frac_p_lt_05'] * rb['n_runs'])}/{rb['n_runs']} runs.")
        lines.append("")
        lines.append("![robustness](plots/trial_level_robustness.png)")
        lines.append("![per_cell_bootstrap](plots/trial_level_per_cell_bootstrap.png)")
        lines.append("")

    eps_rows = results.get("epsilon_sensitivity") or []
    if eps_rows:
        names = (tn_primary or {}).get("tier_names", ["Low", "Mid", "High"])
        lines.append("### Epsilon sensitivity (all-trials coupling)")
        lines.append("")
        header_cells = [f"z {n}<->{n}" for n in names]
        lines.append("| epsilon | diag_mass | GW distance | " +
                     " | ".join(header_cells) + " | row-sum drift |")
        lines.append("|---:|---:|---:|" + "---:|" * len(names) + "---:|")
        for r in eps_rows:
            per_z_vals = np.asarray(r["per_cell_z"]).tolist()
            per_cells = " | ".join(f"{z:+.2f}" for z in per_z_vals)
            lines.append(
                f"| {r['epsilon']:.3f} | {r['diag_mass']:.3f} | "
                f"{r['gw_distance']:.4f} | {per_cells} | "
                f"{r['row_sum_drift']:.2e} |"
            )
        lines.append("")
        lines.append("![epsilon_sensitivity](plots/trial_level_epsilon_sensitivity.png)")
        lines.append("")

    # Balanced subsample sensitivity (demoted from primary).
    tn_bal = results.get("trial_level_null_balanced")
    if tn_bal:
        names = tn_bal.get("tier_names", ["Low", "Mid", "High"])
        per_z = tn_bal.get("per_cell_z", [])
        zs = ", ".join(f"{n}<->{n} {z:+.2f}" for n, z in zip(names, per_z))
        lines.append("### Balanced-subsample sensitivity (n = 100 per tier, with replacement)")
        lines.append("")
        lines.append(
            f"Observed `diag_mass = {tn_bal['observed_diag_mass']:.3f}` "
            f"(null mean {tn_bal['null_mean']:.3f} +/- {tn_bal['null_std']:.3f}, "
            f"p = {tn_bal['p_value']:.4f}, z = {tn_bal['z_score']:+.2f}). "
            f"Per-cell diagonal z-scores: {zs}.")
        lines.append("")

    # Fixed-cutoff sensitivity (optional).
    fc = results.get("fixed_cutoff_sensitivity") or {}
    if fc and "diag_mass_observed" in fc and tn_primary:
        tier_names = tn_primary.get("tier_names", ["Low", "Mid", "High"])
        try:
            mid_idx = tier_names.index("Mid")
        except ValueError:
            mid_idx = 1
        tertile_mid_z = float(np.asarray(tn_primary["per_cell_z"])[mid_idx])
        fixed_mid_z = float(np.asarray(fc["per_cell_z"])[mid_idx])
        surv_label = ("Survives" if fixed_mid_z > 2.0
                      else ("Collapses" if abs(fixed_mid_z) < 1.0
                            else "Shifts"))
        lines.append("### Fixed-cutoff sensitivity (hard cutoffs)")
        lines.append("")
        lines.append(
            f"JIGSAWS (<{fc['jigsaws_cutoffs']['low']}/{fc['jigsaws_cutoffs']['low']}-"
            f"{fc['jigsaws_cutoffs']['high']}/>{fc['jigsaws_cutoffs']['high']}), "
            f"Mimic (<{fc['eeg_eye_cutoffs']['low']}/{fc['eeg_eye_cutoffs']['low']}-"
            f"{fc['eeg_eye_cutoffs']['high']}/>{fc['eeg_eye_cutoffs']['high']}). "
            f"Tier counts J {fc['tier_counts_j']}, E {fc['tier_counts_e']}.")
        lines.append("")
        lines.append(
            f"tertile Mid<->Mid z = **{tertile_mid_z:+.2f}**, "
            f"fixed-cutoff Mid<->Mid z = **{fixed_mid_z:+.2f}**. "
            f"**{surv_label}** the binning change.")
        lines.append("")

    # --- Stratified bootstrap sections (Step 9g) ---
    def _fmt_bootstrap(label: str, b: dict, v: dict) -> None:
        if not b:
            return
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            f"Tier counts preserved: J = {b['tier_counts_j']}, "
            f"E = {b['tier_counts_e']}.")
        lines.append(f"n_bootstraps = {b['n_bootstraps']}, "
                     f"n_degenerate = {b['n_degenerate']}.")
        lines.append(
            f"Fraction with trace p < 0.05: "
            f"**{b['frac_significant_trace']:.2f}**.")
        lines.append("")
        lines.append("| cell | median z | 5% | 95% | frac(|z|>2.394) | frac(z>0) |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        names = b.get("tier_names", ["Low", "Mid", "High"])
        for k, name in enumerate(names):
            s = b["summary"][f"per_cell_z_{name}"]
            lines.append(
                f"| {name}<->{name} | {s['median']:+.2f} | {s['p05']:+.2f} | "
                f"{s['p95']:+.2f} | {b['frac_per_cell_bonf'][k]:.2f} | "
                f"{b['frac_per_cell_positive'][k]:.2f} |"
            )
        lines.append("")
        if v:
            lines.append(
                f"**Verdict: {v['verdict']}** "
                f"(Low median {v['low_median']:+.2f}, Mid median {v['mid_median']:+.2f}; "
                f"frac(z>0) Low {v['frac_positive_low']:.2f}, Mid {v['frac_positive_mid']:.2f}).")
            lines.append("")

    fb = results.get("fixed_cutoff_bootstrap") or {}
    fb_verdict = results.get("fixed_cutoff_bootstrap_verdict") or {}
    if fb:
        _fmt_bootstrap("Fixed-cutoff bootstrap robustness (Step 9g, stratified)",
                       fb, fb_verdict)
        lines.append("![fixed_cutoff_bootstrap](plots/fixed_cutoff_bootstrap.png)")
        lines.append("")

    tb = results.get("tertile_bootstrap_stratified") or {}
    tb_verdict = results.get("tertile_bootstrap_stratified_verdict") or {}
    if tb:
        _fmt_bootstrap("Tertile bootstrap robustness (stratified, apples-to-apples)",
                       tb, tb_verdict)
        lines.append("![tertile_bootstrap_stratified]"
                     "(plots/tertile_bootstrap_stratified.png)")
        lines.append("")

    # --- Step 10: Mimic-side modality split ---
    ms = results.get("modality_split") or {}
    if ms and "_meta" in ms:
        meta = ms["_meta"]
        tier_names = list(meta["tier_names"])
        mods = [m for m in ms.keys() if m != "_meta"]
        lines.append("## Step 10 — Mimic-side modality split")
        lines.append("")
        lines.append(
            "JIGSAWS side = full combined 39-dim manifold for every comparison. "
            "Mimic side restricted to one modality at a time (dim: "
            + ", ".join(f"{m} = {ms[m]['mimic_feature_dim']}" for m in mods) + ").")
        lines.append("")

        for binning_label, binning_key in (("Fixed-cutoff binning (primary)", "fixed"),
                                            ("Tertile binning (sensitivity)", "tertile")):
            lines.append(f"### {binning_label}")
            lines.append("")
            lines.append("**Primary trial-level (observed):**")
            lines.append("")
            lines.append(
                "| modality | trace_z | trace_p | "
                + " | ".join(f"z {n}<->{n}" for n in tier_names) + " |")
            lines.append("|---|---:|---:|" + "---:|" * len(tier_names))
            for m in mods:
                pr = ms[m][binning_key]["primary"]
                zs = " | ".join(f"{z:+.2f}" for z in pr["per_cell_z"])
                lines.append(
                    f"| {m} | {pr['trace_z_score']:+.2f} | "
                    f"{pr['trace_p_value']:.4f} | {zs} |"
                )
            # Combined reference (Step 9a for tertile / 9f for fixed).
            comb_key = ("trial_level_null_primary" if binning_key == "tertile"
                        else "fixed_cutoff_sensitivity")
            comb = results.get(comb_key) or {}
            if comb:
                if comb_key == "trial_level_null_primary":
                    zs = " | ".join(f"{z:+.2f}" for z in comb["per_cell_z"])
                    tz = comb["z_score"]; tp = comb["p_value"]
                else:
                    zs = " | ".join(f"{z:+.2f}" for z in comb["per_cell_z"])
                    tz = comb["z_score"]; tp = comb["p_value"]
                lines.append(
                    f"| **combined (Step 9 ref)** | {tz:+.2f} | {tp:.4f} | {zs} |")
            lines.append("")

            lines.append("**Bootstrap robustness (B = {}):**".format(meta["n_bootstraps"]))
            lines.append("")
            lines.append(
                "| modality | cell | median | 5% | 95% | frac(\\|z\\|>2.39) | frac(z>0) | verdict |")
            lines.append("|---|---|---:|---:|---:|---:|---:|:--:|")
            for m in mods:
                bs = ms[m][binning_key]["bootstrap"]
                v = ms[m][binning_key]["verdict"]["verdict"]
                for k, n in enumerate(tier_names):
                    s = bs["summary"][f"per_cell_z_{n}"]
                    lines.append(
                        f"| {m} | {n}<->{n} | {s['median']:+.2f} | "
                        f"{s['p05']:+.2f} | {s['p95']:+.2f} | "
                        f"{bs['frac_per_cell_bonf'][k]:.2f} | "
                        f"{bs['frac_per_cell_positive'][k]:.2f} | "
                        f"{v if k == 0 else ''} |"
                    )
            lines.append("")
        lines.append("![modality_split](plots/modality_split.png)")
        lines.append("")

    # --- P1: EEG baseline/pc correlation ---
    ec = results.get("eeg_baseline_pc_correlation") or {}
    if ec:
        lines.append("### Step 10 — P1 diagnostic: EEG baseline/pc correlation")
        lines.append("")
        lines.append(
            f"Per-trial Pearson r between mean-pooled baseline and pc EEG: "
            f"median = **{ec['median']:+.2f}** (|r| median = {ec['median_abs']:.2f}, "
            f"5-95 % = {ec['p05']:+.2f} to {ec['p95']:+.2f}); |r| > 0.5 on "
            f"**{ec['frac_abs_gt_0p5']:.2f}** of {ec['n_trials']} trials.")
        lines.append(
            "**Caveat.** Baseline and pc encoders consume the same Phase 1 "
            "EEG windows and are therefore not independent evidence streams; "
            "per-modality verdicts should be read with the dependence in mind.")
        lines.append("")
        lines.append("![eeg_base_pc_corr](plots/eeg_baseline_pc_correlation.png)")
        lines.append("")

    # --- P3: pooled-128 EEG random-split null ---
    pool = results.get("eeg_pooled_random_split_null") or {}
    if pool:
        lines.append("### Step 10 — P3 diagnostic: Pooled-128 EEG random-split null")
        lines.append("")
        lines.append(
            f"Observed baseline-minus-pc trace_z delta = "
            f"**{pool['observed_delta_baseline_minus_pc']:+.2f}**. "
            f"Random 64+64 splits of the combined 128 EEG dims "
            f"({pool['n_random_splits']} draws): median delta "
            f"{pool['delta_median']:+.2f}, 5-95 % "
            f"[{pool['delta_p05']:+.2f}, {pool['delta_p95']:+.2f}]. "
            f"Two-sided p = **{pool['p_two_sided']:.3f}**.")
        lines.append("")
        lines.append("![eeg_pooled](plots/eeg_pooled_random_split.png)")
        lines.append("")

    # --- P4: eye coupling-matrix diagnostic ---
    eye_d = results.get("eye_coupling_diagnostic") or {}
    if eye_d:
        lines.append("### Step 10 — P4 diagnostic: Eye-only coupling matrix")
        lines.append("")
        lines.append(
            f"Eye-only cosine-distance matrix rank = **{eye_d['distance_rank']}**; "
            f"near-duplicate rows (cos-distance < 1e-4) = "
            f"{eye_d['near_duplicate_pairs']}/{eye_d['total_pairs']} "
            f"({eye_d['near_duplicate_fraction']:.2%}). "
            f"Pre-renorm coupling drift: row {eye_d['coupling_row_drift']:.2e}, "
            f"col {eye_d['coupling_col_drift']:.2e}. "
            f"({eye_d['n_trials']} trials, {eye_d['n_features']} features)")
        lines.append("")

    # --- Step 11: JIGSAWS-side modality split (P2) ---
    jms = results.get("jigsaws_modality_split") or {}
    if jms and "_meta" in jms:
        meta = jms["_meta"]
        tier_names = list(meta["tier_names"])
        mods = [m for m in jms.keys() if m != "_meta"]
        lines.append("## Step 11 — JIGSAWS-side modality split (P2)")
        lines.append("")
        lines.append(
            "Mimic side = full combined 146-dim manifold for every comparison. "
            "JIGSAWS side restricted to one modality at a time (dim: "
            + ", ".join(f"{m} = {jms[m]['jigsaws_feature_dim']}" for m in mods)
            + "). Duration (1-d) is omitted because cosine distance on 1-d "
            + "features is degenerate.")
        lines.append("")

        for binning_label, binning_key in (("Fixed-cutoff binning", "fixed"),
                                            ("Tertile binning",     "tertile")):
            lines.append(f"### {binning_label}")
            lines.append("")
            lines.append("**Primary trial-level (observed):**")
            lines.append("")
            lines.append("| modality | trace_z | trace_p | "
                         + " | ".join(f"z {n}<->{n}" for n in tier_names) + " |")
            lines.append("|---|---:|---:|" + "---:|" * len(tier_names))
            for m in mods:
                pr = jms[m][binning_key]["primary"]
                zs = " | ".join(f"{z:+.2f}" for z in pr["per_cell_z"])
                lines.append(
                    f"| {m} | {pr['trace_z_score']:+.2f} | "
                    f"{pr['trace_p_value']:.4f} | {zs} |")
            lines.append("")
            lines.append("**Bootstrap (B = {}):**".format(meta["n_bootstraps"]))
            lines.append("")
            lines.append("| modality | cell | median | 5% | 95% | "
                         "frac(\\|z\\|>2.39) | frac(z>0) | verdict |")
            lines.append("|---|---|---:|---:|---:|---:|---:|:--:|")
            for m in mods:
                bs = jms[m][binning_key]["bootstrap"]
                v = jms[m][binning_key]["verdict"]["verdict"]
                for k, n in enumerate(tier_names):
                    s = bs["summary"][f"per_cell_z_{n}"]
                    lines.append(
                        f"| {m} | {n}<->{n} | {s['median']:+.2f} | "
                        f"{s['p05']:+.2f} | {s['p95']:+.2f} | "
                        f"{bs['frac_per_cell_bonf'][k]:.2f} | "
                        f"{bs['frac_per_cell_positive'][k]:.2f} | "
                        f"{v if k == 0 else ''} |")
            lines.append("")
        lines.append("![jigsaws_modality_split](plots/jigsaws_modality_split.png)")
        lines.append("")

    # --- Comparison A: practice manifold ---
    ca = results.get("comparison_a") or {}
    if ca and "primary" in ca:
        pr = ca["primary"]; bs = ca["bootstrap"]; v = ca["verdict"]
        lines.append("## Comparison A — Practice manifold")
        lines.append("")
        lines.append(ca["description"])
        lines.append("")
        lines.append(
            f"Tier counts JIGSAWS (N/I/E -> Low/Mid/High) = "
            f"{dict(zip(pr['tier_names'], ca['tier_counts_j']))}, "
            f"Mimic (experience-tertile) = "
            f"{dict(zip(pr['tier_names'], ca['tier_counts_e']))}. "
            f"Mimic experience cutoffs (non-first-try count): "
            f"q33 = {ca['mimic_experience_cutoffs']['q33']:g}, "
            f"q66 = {ca['mimic_experience_cutoffs']['q66']:g}.")
        lines.append("")
        lines.append(
            f"**Primary (all-trials):** diag_mass = **{pr['observed_diag_mass']:.3f}** "
            f"vs expected `{pr['expected_trace_under_null']:.3f}`, "
            f"trace p = **{pr['trace_p_value']:.4f}**, "
            f"trace z = **{pr['trace_z_score']:+.2f}**. "
            f"Per-cell z-scores: "
            + ", ".join(f"{n}<->{n} {z:+.2f}" for n, z in zip(
                pr['tier_names'], pr['per_cell_z'])) + ".")
        lines.append("")
        lines.append(
            f"**Stratified bootstrap (B = {bs['n_bootstraps']}):** ")
        for k, name in enumerate(pr['tier_names']):
            s = bs['summary'][f'per_cell_z_{name}']
            lines.append(
                f"- {name}<->{name}: median {s['median']:+.2f} "
                f"(5-95% {s['p05']:+.2f}..{s['p95']:+.2f}); "
                f"|z|>2.39 in {bs['frac_per_cell_bonf'][k]:.2f}; "
                f"z>0 in {bs['frac_per_cell_positive'][k]:.2f}")
        lines.append("")
        lines.append(
            f"**Verdict: {v['verdict']}** "
            f"(Low med {v['low_median']:+.2f}, Mid med {v['mid_median']:+.2f}; "
            f"frac(z>0) L/M {v['frac_positive_low']:.2f}/{v['frac_positive_mid']:.2f}).")
        lines.append("")
        lines.append("![comparison_a_block](plots/comparison_a_block_null.png)")
        lines.append("![comparison_a_bootstrap](plots/comparison_a_bootstrap.png)")
        lines.append("")

    lines.append("## Verification checklist")
    for c in results.get("checks", []):
        lines.append(f"- [{('x' if c['pass'] else ' ')}] {c['name']}")
    path.write_text("\n".join(lines))


def _print_self_check(results: dict) -> None:
    log.info("================== Self-check ==================")
    for c in results["checks"]:
        status = "PASS" if c["pass"] else "FAIL"
        log.info("  [%s] %s", status, c["name"])
    log.info("================================================")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=str, default=None,
                   help="Parent of Gestures/, Eye/, cache/. Defaults to dataset_paths.resolve_dataset_root().")
    p.add_argument("--config", type=Path,
                   default=CONFIG_DIR / "skill_manifold.yaml")
    p.add_argument("--task_modules", type=Path,
                   default=CONFIG_DIR / "skill_manifold_task_modules.yaml")
    p.add_argument("--n_perms", type=int, default=None)
    p.add_argument("--subsample_per_tier", type=int, default=None)
    p.add_argument("--smoke", action="store_true",
                   help="Run on a 10% subsample of each dataset for a fast smoke pass.")
    p.add_argument("--log_level", type=str, default="INFO")
    args = p.parse_args()

    _setup_logging(args.log_level)
    cfg = load_config(args.config.name if args.config.parent == CONFIG_DIR
                      else args.config)
    # When --config is a full path, load by reading that path directly.
    if args.config.is_absolute():
        import yaml
        cfg = yaml.safe_load(args.config.read_text())

    root = data_root(args.data_root)
    run(cfg, task_modules_yaml=args.task_modules,
        data_root_path=root,
        n_perms_override=args.n_perms,
        subsample_override=args.subsample_per_tier,
        smoke_fraction=0.1 if args.smoke else None)


if __name__ == "__main__":
    main()
