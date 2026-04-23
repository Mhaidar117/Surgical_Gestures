"""Gromov-Wasserstein distance computation and permutation nulls.

Uses POT (https://pythonot.github.io/) to compare two RDMs of possibly
different sizes. GW only requires the intra-set dissimilarities, which is
the right tool when the two datasets have disjoint feature spaces and no
shared task taxonomy -- exactly our setting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import ot  # POT

from skill_manifold.rdms import centroid_rdm
from skill_manifold.trial_null import (  # re-export for discoverability
    BONFERRONI_Z_THREE_CELLS,
    compute_block_mass,
    diag_mass,
    epsilon_sensitivity,
    modality_split_analysis,
    stratified_bootstrap_verdict,
    subsample_robustness,
    subsample_robustness_stratified,
    trial_level_block_null,
    trial_level_block_null_all_trials,
)

__all__ = [
    "BONFERRONI_Z_THREE_CELLS",
    "GWResult",
    "compute_block_mass",
    "diag_mass",
    "entropic_gromov_wasserstein",
    "epsilon_sensitivity",
    "gromov_wasserstein_centroids",
    "modality_split_analysis",
    "permutation_null_centroid",
    "stratified_bootstrap_verdict",
    "subsample_robustness",
    "subsample_robustness_stratified",
    "trial_level_block_null",
    "trial_level_block_null_all_trials",
]

log = logging.getLogger(__name__)


@dataclass
class GWResult:
    distance: float
    coupling: np.ndarray         # (K, K) transport plan
    argmax_assignment: Dict[str, str] = field(default_factory=dict)  # row tier -> col tier


def _uniform(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.float64) / float(n)


def gromov_wasserstein_centroids(rdm_a: np.ndarray,
                                 rdm_b: np.ndarray,
                                 tier_order: Sequence[str]) -> GWResult:
    """GW distance between two same-sized tier-centroid RDMs.

    The coupling is reported with rows labelled by `rdm_a`'s tiers and cols
    by `rdm_b`'s tiers (both assumed ordered the same way, i.e. tier_order).
    """
    p = _uniform(rdm_a.shape[0])
    q = _uniform(rdm_b.shape[0])
    dist, log_dict = ot.gromov.gromov_wasserstein2(
        rdm_a, rdm_b, p, q, loss_fun="square_loss", log=True,
    )
    T = log_dict["T"].astype(np.float64)

    # Argmax tier-to-tier assignment from the coupling's rows.
    assignment: Dict[str, str] = {}
    for i, t_row in enumerate(tier_order):
        j = int(np.argmax(T[i]))
        assignment[t_row] = tier_order[j]
    return GWResult(distance=float(dist), coupling=T, argmax_assignment=assignment)


def entropic_gromov_wasserstein(c1: np.ndarray,
                                c2: np.ndarray,
                                *,
                                epsilon: float = 0.01) -> GWResult:
    """Entropic GW for larger cost matrices. Returns distance and coupling.

    We call the log-enabled variant so we can recover the transport plan.
    """
    p = _uniform(c1.shape[0])
    q = _uniform(c2.shape[0])
    T, log_dict = ot.gromov.entropic_gromov_wasserstein(
        c1, c2, p, q, loss_fun="square_loss",
        epsilon=epsilon, log=True,
    )
    # POT returns the coupling; the distance (as a scalar) is in log_dict.
    # Different POT versions name it differently.
    dist = log_dict.get("gw_dist") or log_dict.get("gw_loss") or log_dict.get("loss")
    if dist is None:
        # Fall back to the explicit 2-form call.
        dist = ot.gromov.entropic_gromov_wasserstein2(
            c1, c2, p, q, loss_fun="square_loss", epsilon=epsilon,
        )
    return GWResult(distance=float(dist), coupling=np.asarray(T, dtype=np.float64))


def _shuffle_tiers(tiers: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random permutation of tier labels preserving per-tier counts.

    This is equivalent to rng.permutation(tiers) because numpy's permutation
    is a relabeling that preserves counts by construction.
    """
    return rng.permutation(tiers)


def permutation_null_centroid(
    features_a: np.ndarray, tiers_a: Sequence[str],
    features_b: np.ndarray, tiers_b: Sequence[str],
    tier_order: Sequence[str],
    *,
    n_perms: int = 1000,
    seed: int = 1337,
) -> Dict[str, float | np.ndarray]:
    """Permutation-null distribution of GW distance under independent tier shuffles.

    For each resample we shuffle tier labels on each side independently,
    recompute the 3x3 centroid RDMs, and take the GW distance. Returns a
    dict with the null distribution array, p-value, z-score, and the counts
    of degenerate (collapsed) resamples encountered.
    """
    rng = np.random.default_rng(seed)
    ta = np.asarray(tiers_a); tb = np.asarray(tiers_b)
    null: List[float] = []
    degenerate = 0

    # Observed value (under the *actual* tier assignments) for reference.
    rdm_a_obs = centroid_rdm(features_a, ta, tier_order)
    rdm_b_obs = centroid_rdm(features_b, tb, tier_order)
    observed = gromov_wasserstein_centroids(rdm_a_obs, rdm_b_obs, tier_order).distance

    while len(null) < n_perms:
        sa = _shuffle_tiers(ta, rng)
        sb = _shuffle_tiers(tb, rng)
        # Degenerate if all samples on either side fell into a single tier.
        if len(np.unique(sa)) < len(tier_order) or len(np.unique(sb)) < len(tier_order):
            degenerate += 1
            continue
        rdm_a = centroid_rdm(features_a, sa, tier_order)
        rdm_b = centroid_rdm(features_b, sb, tier_order)
        try:
            d = gromov_wasserstein_centroids(rdm_a, rdm_b, tier_order).distance
        except Exception as e:
            log.warning("GW failure in permutation: %s", e)
            degenerate += 1
            continue
        null.append(float(d))

    null_arr = np.asarray(null, dtype=np.float64)
    p = (1.0 + float((null_arr <= observed).sum())) / (1.0 + len(null_arr))
    mu = float(null_arr.mean())
    sd = float(null_arr.std(ddof=1)) if len(null_arr) > 1 else float("nan")
    z = (observed - mu) / sd if sd > 1e-12 else float("nan")

    return {
        "observed": float(observed),
        "null": null_arr,
        "p_value": float(p),
        "z_score": float(z),
        "null_mean": mu,
        "null_std": sd,
        "n_permutations": int(len(null_arr)),
        "n_degenerate": int(degenerate),
    }
