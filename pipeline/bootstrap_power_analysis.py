"""Bootstrap power analysis for the cross-dataset GW pipeline.

Question this script answers
---------------------------
At our actual sample size (N <= 33 per side), how often does the
stratified bootstrap produce a Bonferroni-significant Mid-Mid z-score
when the *true* effect size is varied?

This addresses the reviewer concern that our 50% bootstrap-survival
rate (median Mid-Mid z = +0.88 across 30 resamples) is evidence the
preliminary cross-dataset data is fragile / underpowered. By
generating synthetic data with a *known* tier-aligned Mid-elevation
of magnitude delta sigma units and running the same bootstrap
machinery, we can:

  * Calibration check (delta = 0):   fraction-significant should be <= 5%.
  * Sanity check    (delta = 2 sigma): fraction-significant should be ~ 1.
  * Power curve at our N (~33):       locate observed effect on the curve.
  * Scaling argument (N = 100, 200):  show that survival rises with N.

What this script does NOT test
------------------------------
The synthetic generator skips the residualization stage of the real
pipeline. It exercises only the GW + bootstrap stage. So this analysis
defends the bootstrap stability claim, NOT a claim about residualization
inflating effect sizes. We flag this honestly in the writeup.

Usage
-----
    # Smoke run (3 deltas, 2 N values, 5 sims each, ~5 min):
    PYTHONPATH=src:. python pipeline/bootstrap_power_analysis.py --smoke

    # Full grid (5 deltas, 3 N values, 30 sims each, ~2 hr):
    PYTHONPATH=src:. python pipeline/bootstrap_power_analysis.py

Outputs
-------
    reports/skill_manifold/bootstrap_power.json
    reports/skill_manifold/plots/bootstrap_power_curve.png
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from skill_manifold.gw import entropic_gromov_wasserstein
from skill_manifold.rdms import pairwise_cosine_rdm
from skill_manifold.trial_null import (
    BONFERRONI_Z_THREE_CELLS,
    subsample_robustness_stratified,
)


log = logging.getLogger(__name__)
TIER_NAMES = ("Low", "Mid", "High")


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def make_synthetic_side(
    n_trials: int,
    n_dims: int,
    delta: float,
    rng: np.random.Generator,
    *,
    tier_axis: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one side's synthetic trial features and tier labels.

    Tier counts are split as evenly as possible across Low/Mid/High; any
    remainder goes to Mid (matches our tertile-binning convention when N
    is not divisible by 3).

    Mid-tier trials are sampled from N(delta * e, I) where `e` is a unit
    vector (the tier-discriminating axis). Low and High are sampled from
    N(0, I) -- both occupy the same region of feature space, encoding the
    inverted-U geometry where Low and High collapse together while Mid
    sticks out.

    Returns (features [n_trials x n_dims], tier_labels [n_trials]).
    """
    # Tier-discriminating axis. If supplied (so J and E share it), reuse;
    # otherwise pick a random unit vector. Sharing the axis between J and
    # E sides is what allows the GW pipeline to find a tier-aligned
    # coupling -- without that, neither side knows which tier-axis the
    # other is using.
    if tier_axis is None:
        tier_axis = rng.standard_normal(n_dims)
        tier_axis = tier_axis / np.linalg.norm(tier_axis)
    elif tier_axis.shape[0] != n_dims:
        raise ValueError(
            f"tier_axis has dim {tier_axis.shape[0]}, expected {n_dims}")

    # Tier counts. Split evenly; remainder goes to Mid.
    base, rem = divmod(n_trials, 3)
    counts = {"Low": base, "Mid": base + rem, "High": base}

    tiers: List[str] = []
    for t in TIER_NAMES:
        tiers.extend([t] * counts[t])
    tiers_arr = np.asarray(tiers, dtype=object)

    feats = rng.standard_normal((n_trials, n_dims))
    mid_mask = tiers_arr == "Mid"
    feats[mid_mask] += delta * tier_axis[None, :]
    return feats, tiers_arr


def make_synthetic_pair(
    n_per_side: int,
    delta: float,
    rng: np.random.Generator,
    *,
    n_dims_j: int = 39,
    n_dims_e: int = 146,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired synthetic data for J-side and E-side.

    The two sides have different feature dimensionalities (matching the
    real pipeline: ~39d for JIGSAWS, ~146d for the EEG/Eye combined).
    Each side picks its OWN random tier axis -- they can't be the same
    vector because the dims differ. GW alignment depends on both sides
    having a tier-discriminating direction in their own feature space,
    not on those directions being identical.
    """
    # Independent tier axes per side (different dims anyway).
    fj, tj = make_synthetic_side(n_per_side, n_dims_j, delta, rng)
    fe, te = make_synthetic_side(n_per_side, n_dims_e, delta, rng)
    return fj, tj, fe, te


# ---------------------------------------------------------------------------
# Bootstrap survival: one simulation
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    """Result for one (delta, N) cell of the power grid."""
    delta: float
    n_per_side: int
    n_simulations: int
    n_bootstraps_per_sim: int
    fraction_significant_per_sim: List[float]   # fraction of bootstraps with Mid-Mid z > Bonferroni
    median_mid_mid_z_per_sim: List[float]
    mean_fraction_significant: float
    sem_fraction_significant: float
    mean_median_z: float


def _egw_dist_coupling(c1, c2, eps):
    """Wrapper to match the (dist, coupling) signature expected by trial_null."""
    res = entropic_gromov_wasserstein(c1, c2, epsilon=float(eps))
    return res.distance, res.coupling


def run_one_simulation(
    *,
    delta: float,
    n_per_side: int,
    n_dims_j: int,
    n_dims_e: int,
    epsilon: float,
    n_perms_inner: int,
    n_bootstraps: int,
    seed: int,
) -> tuple[float, float]:
    """One simulation: generate synthetic data, run bootstrap, return
    (fraction-significant, median Mid-Mid z) for this realization."""
    rng = np.random.default_rng(seed)
    fj, tj, fe, te = make_synthetic_pair(
        n_per_side, delta, rng, n_dims_j=n_dims_j, n_dims_e=n_dims_e)

    boot = subsample_robustness_stratified(
        features_j=fj, tiers_j=tj,
        features_e=fe, tiers_e=te,
        tier_names=list(TIER_NAMES),
        epsilon=epsilon,
        n_perms=n_perms_inner,
        n_bootstraps=n_bootstraps,
        seed=seed + 1,
        entropic_gw_fn=_egw_dist_coupling,
        pairwise_rdm_fn=pairwise_cosine_rdm,
    )
    # per_cell_z: shape (n_bootstraps, K=3). Mid is index 1.
    per_cell_z = np.asarray(boot["per_cell_z"], dtype=np.float64)
    mid_z = per_cell_z[:, 1]
    fraction_sig = float((mid_z > BONFERRONI_Z_THREE_CELLS).mean())
    median_z = float(np.median(mid_z))
    return fraction_sig, median_z


def run_cell(
    *,
    delta: float,
    n_per_side: int,
    n_simulations: int,
    n_bootstraps: int,
    n_perms_inner: int,
    epsilon: float,
    n_dims_j: int,
    n_dims_e: int,
    base_seed: int,
) -> CellResult:
    """Run all simulations for one (delta, N) cell."""
    fracs: List[float] = []
    medians: List[float] = []
    for s in range(n_simulations):
        seed = base_seed + 100 * s
        f, m = run_one_simulation(
            delta=delta, n_per_side=n_per_side,
            n_dims_j=n_dims_j, n_dims_e=n_dims_e,
            epsilon=epsilon,
            n_perms_inner=n_perms_inner,
            n_bootstraps=n_bootstraps,
            seed=seed,
        )
        fracs.append(f)
        medians.append(m)
        log.info("    sim %d/%d: frac=%.2f, median_z=%+.2f",
                 s + 1, n_simulations, f, m)
    arr_f = np.asarray(fracs)
    arr_m = np.asarray(medians)
    sem = float(arr_f.std(ddof=1) / np.sqrt(len(arr_f))) if len(arr_f) > 1 else 0.0
    return CellResult(
        delta=delta,
        n_per_side=n_per_side,
        n_simulations=n_simulations,
        n_bootstraps_per_sim=n_bootstraps,
        fraction_significant_per_sim=fracs,
        median_mid_mid_z_per_sim=medians,
        mean_fraction_significant=float(arr_f.mean()),
        sem_fraction_significant=sem,
        mean_median_z=float(arr_m.mean()),
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_power_curve(
    cells: Sequence[CellResult],
    output_path: Path,
    *,
    observed_n: int = 33,
    observed_survival: float = 0.50,
    observed_effect_size: Optional[float] = None,
) -> None:
    """Plot fraction-significant vs. delta, one line per N value.

    `observed_*` are our actual real-data summary numbers, drawn as
    reference lines so the reader can locate where we sit on the curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by N.
    by_n: dict[int, List[CellResult]] = {}
    for c in cells:
        by_n.setdefault(c.n_per_side, []).append(c)
    for n in by_n:
        by_n[n] = sorted(by_n[n], key=lambda c: c.delta)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    palette = {33: "#d62728", 100: "#1f77b4", 200: "#2ca02c"}
    for n, cs in sorted(by_n.items()):
        deltas = [c.delta for c in cs]
        fracs = [c.mean_fraction_significant for c in cs]
        sems = [c.sem_fraction_significant for c in cs]
        color = palette.get(n, "#666")
        ax.errorbar(deltas, fracs, yerr=sems, color=color, marker="o",
                    capsize=3, lw=1.5, label=f"N = {n}")

    # Reference lines.
    ax.axhline(0.05, color="#888", ls=":", lw=0.8)
    ax.text(ax.get_xlim()[1], 0.05, " 5% (Type I)",
            fontsize=8, color="#666", va="bottom", ha="right")
    ax.axhline(observed_survival, color="#aaa", ls="--", lw=0.8)
    ax.text(ax.get_xlim()[1], observed_survival,
            f" Observed survival ({observed_survival:.0%})",
            fontsize=8, color="#666", va="bottom", ha="right")
    if observed_effect_size is not None:
        ax.axvline(observed_effect_size, color="#aaa", ls=":", lw=0.8)
        ax.text(observed_effect_size, 1.02, f"Observed Δ ≈ {observed_effect_size:.2f}σ",
                fontsize=8, color="#666", va="bottom", ha="center")

    ax.set_xlabel("True Mid-tier effect size Δ (σ units)", fontsize=10)
    ax.set_ylabel("Bootstrap survival fraction\n(Mid-Mid z > Bonferroni)", fontsize=10)
    ax.set_title("Bootstrap power vs. effect size at small surgical-skill N", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", output_path)


# ---------------------------------------------------------------------------
# CLI / main driver
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny grid for sanity-check (3 deltas x 2 N x 5 sims)")
    p.add_argument("--cell", nargs=3, metavar=("DELTA", "N", "N_SIMS"),
                   help="Run a single (delta, N, n_sims) cell only and write"
                        " a partial JSON. Useful for chaining bash calls when"
                        " each must finish under a timeout.")
    p.add_argument("--n_bootstraps", type=int, default=15,
                   help="Bootstrap resamples per simulation (default 15;"
                        " full pipeline uses 30).")
    p.add_argument("--aggregate", action="store_true",
                   help="Read all partial JSONs from"
                        " <output_dir>/bootstrap_power_partials/ and emit"
                        " the consolidated bootstrap_power.json + power-curve"
                        " plot.")
    p.add_argument("--output_dir", type=Path,
                   default=Path("reports/skill_manifold"))
    p.add_argument("--epsilon", type=float, default=0.01,
                   help="Entropic GW regularization (matches real pipeline)")
    p.add_argument("--n_perms_inner", type=int, default=200,
                   help="Permutations per bootstrap resample (lower than"
                        " the real pipeline's 1000 to keep runtime bounded;"
                        " 200 is enough for stable z-scores).")
    p.add_argument("--n_dims", type=int, default=5,
                   help="Synthetic feature dimensionality per side. Default"
                        " 5 mirrors the effective rank of the residualized,"
                        " z-scored real features (PC1 ~ 51%%, top few PCs"
                        " carry most variance). Higher values dilute"
                        " single-axis tier signal.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")

    # Aggregate previously-saved partial JSONs into the final outputs.
    if args.aggregate:
        partial_dir = args.output_dir / "bootstrap_power_partials"
        if not partial_dir.exists():
            raise SystemExit(f"no partials directory at {partial_dir}")
        cells: List[CellResult] = []
        for path in sorted(partial_dir.glob("cell_n*_d*.json")):
            d = json.loads(path.read_text())
            cells.append(CellResult(
                delta=d["delta"], n_per_side=d["n_per_side"],
                n_simulations=d["n_simulations"],
                n_bootstraps_per_sim=d["n_bootstraps_per_sim"],
                fraction_significant_per_sim=d["fraction_significant_per_sim"],
                median_mid_mid_z_per_sim=d["median_mid_mid_z_per_sim"],
                mean_fraction_significant=d["mean_fraction_significant"],
                sem_fraction_significant=d["sem_fraction_significant"],
                mean_median_z=d["mean_median_z"],
            ))
        log.info("aggregating %d partial cells", len(cells))
        deltas = sorted({c.delta for c in cells})
        n_values = sorted({c.n_per_side for c in cells})
        json_path = args.output_dir / "bootstrap_power.json"
        json_path.write_text(json.dumps({
            "deltas": deltas,
            "n_values": n_values,
            "n_dims": args.n_dims,
            "epsilon": args.epsilon,
            "bonferroni_z_threshold": float(BONFERRONI_Z_THREE_CELLS),
            "cells": [
                {
                    "delta": c.delta, "n_per_side": c.n_per_side,
                    "mean_fraction_significant": c.mean_fraction_significant,
                    "sem_fraction_significant": c.sem_fraction_significant,
                    "mean_median_z": c.mean_median_z,
                    "n_simulations": c.n_simulations,
                    "n_bootstraps_per_sim": c.n_bootstraps_per_sim,
                }
                for c in cells
            ],
        }, indent=2))
        log.info("wrote %s", json_path)
        plot_power_curve(cells, args.output_dir / "plots" / "bootstrap_power_curve.png")
        return

    # Single-cell mode for chained bash calls.
    if args.cell is not None:
        delta = float(args.cell[0])
        n = int(args.cell[1])
        n_sims = int(args.cell[2])
        log.info("[N=%d, delta=%.2f] running %d simulations (single-cell mode)",
                 n, delta, n_sims)
        cell = run_cell(
            delta=delta, n_per_side=n,
            n_simulations=n_sims,
            n_bootstraps=args.n_bootstraps,
            n_perms_inner=args.n_perms_inner,
            epsilon=args.epsilon,
            n_dims_j=args.n_dims, n_dims_e=args.n_dims,
            base_seed=20260427 + 100_000 * n + 1_000 * int(round(delta * 100)),
        )
        partial_dir = args.output_dir / "bootstrap_power_partials"
        partial_dir.mkdir(parents=True, exist_ok=True)
        out = partial_dir / f"cell_n{n}_d{int(round(delta * 100)):03d}.json"
        out.write_text(json.dumps({
            "delta": cell.delta,
            "n_per_side": cell.n_per_side,
            "n_simulations": cell.n_simulations,
            "n_bootstraps_per_sim": cell.n_bootstraps_per_sim,
            "mean_fraction_significant": cell.mean_fraction_significant,
            "sem_fraction_significant": cell.sem_fraction_significant,
            "mean_median_z": cell.mean_median_z,
            "fraction_significant_per_sim": cell.fraction_significant_per_sim,
            "median_mid_mid_z_per_sim": cell.median_mid_mid_z_per_sim,
        }, indent=2))
        log.info("[N=%d, delta=%.2f] mean survival = %.3f (SEM=%.3f), median z = %+.2f",
                 n, delta, cell.mean_fraction_significant,
                 cell.sem_fraction_significant, cell.mean_median_z)
        log.info("wrote %s", out)
        return

    if args.smoke:
        deltas = [0.0, 1.0, 2.0]
        n_values = [33, 100]
        n_simulations = 5
        n_bootstraps = args.n_bootstraps
        log.info("SMOKE grid: %d deltas x %d N x %d sims x %d boot",
                 len(deltas), len(n_values), n_simulations, n_bootstraps)
    else:
        deltas = [0.0, 0.5, 1.0, 1.5, 2.0]
        n_values = [33, 100, 200]
        n_simulations = 30
        n_bootstraps = args.n_bootstraps
        log.info("FULL grid: %d deltas x %d N x %d sims x %d boot",
                 len(deltas), len(n_values), n_simulations, n_bootstraps)

    cells: List[CellResult] = []
    base_seed = 20260427  # arbitrary fixed seed for reproducibility
    for ni, n in enumerate(n_values):
        for di, d in enumerate(deltas):
            log.info("[N=%d, delta=%.2f] running %d simulations ...",
                     n, d, n_simulations)
            cell = run_cell(
                delta=d, n_per_side=n,
                n_simulations=n_simulations,
                n_bootstraps=n_bootstraps,
                n_perms_inner=args.n_perms_inner,
                epsilon=args.epsilon,
                n_dims_j=args.n_dims, n_dims_e=args.n_dims,
                base_seed=base_seed + 10_000 * ni + 1_000 * di,
            )
            log.info("[N=%d, delta=%.2f] mean survival = %.3f (SEM=%.3f), median z = %+.2f",
                     n, d, cell.mean_fraction_significant,
                     cell.sem_fraction_significant, cell.mean_median_z)
            cells.append(cell)

    # Persist results.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    json_path = args.output_dir / f"bootstrap_power{suffix}.json"
    payload = {
        "deltas": deltas,
        "n_values": n_values,
        "n_simulations": n_simulations,
        "n_bootstraps_per_sim": n_bootstraps,
        "n_perms_inner": args.n_perms_inner,
        "epsilon": args.epsilon,
        "bonferroni_z_threshold": float(BONFERRONI_Z_THREE_CELLS),
        "cells": [
            {
                "delta": c.delta,
                "n_per_side": c.n_per_side,
                "mean_fraction_significant": c.mean_fraction_significant,
                "sem_fraction_significant": c.sem_fraction_significant,
                "mean_median_z": c.mean_median_z,
                "fraction_significant_per_sim": c.fraction_significant_per_sim,
                "median_mid_mid_z_per_sim": c.median_mid_mid_z_per_sim,
            }
            for c in cells
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2))
    log.info("wrote %s", json_path)

    plot_path = args.output_dir / "plots" / f"bootstrap_power_curve{suffix}.png"
    plot_power_curve(
        cells, plot_path,
        observed_effect_size=None,  # leave the x-axis ref off until full run
    )


if __name__ == "__main__":
    main()
