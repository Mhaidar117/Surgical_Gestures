"""Trial-level permutation null for the entropic-GW coupling.

The 3x3 centroid GW test from Step 6 has a proper null via
`skill_manifold.gw.permutation_null_centroid`. The trial-level analysis from
Step 9, however, has no null by default. Adding a naive "shuffle tier labels
and recompute GW" loop would be wrong: GW is invariant to row/column
permutations of its input distance matrices, so shuffling trial order on
either side leaves `Cj`, `Ce`, and the coupling `T` unchanged.

The correct null operates on the **interaction** between the fixed observed
coupling and the trial-level tier labels. Concretely, we aggregate `T`
(shape N x N, row/col sums = 1/N) into a 3x3 block-mass matrix `B` by tier,
then ask whether the diagonal mass `trace(B)` is unusually large under
independent shuffles of the tier labels on each side.

Under a null in which tier labels carry no information, expected block
marginals are uniform 1/9 and E[trace(B)] = 1/3. Observed `diag_mass >> 1/3`
means the coupling respects tier boundaries.

See docs/agent_prompt_trial_level_null.md for the full specification.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# Bonferroni-corrected two-sided z threshold for K = 3 diagonal cells, alpha = 0.05.
# z_{1 - alpha/(2*3)} = Phi^{-1}(1 - 0.05/6) ~= 2.3940.
BONFERRONI_Z_THREE_CELLS = 2.3940


# --------- core statistic -----------------------------------------------------

def compute_block_mass(coupling: np.ndarray,
                       tiers_j: np.ndarray,
                       tiers_e: np.ndarray,
                       tier_names: Sequence[str]) -> np.ndarray:
    """Aggregate a trial-level coupling into a K x K block-mass matrix.

    `B[a, b] = sum over (i, j) with tiers_j[i]==a and tiers_e[j]==b of T[i, j]`.
    Returns a (K, K) matrix that sums to `coupling.sum()` and, when `coupling`
    has uniform row/col marginals 1/N, has uniform 1/K block marginals.
    """
    T = np.asarray(coupling, dtype=np.float64)
    tj = np.asarray(tiers_j); te = np.asarray(tiers_e)
    if T.ndim != 2 or T.shape[0] != tj.shape[0] or T.shape[1] != te.shape[0]:
        raise ValueError(
            f"shape mismatch: T={T.shape}, tiers_j={tj.shape}, tiers_e={te.shape}")
    K = len(tier_names)
    B = np.zeros((K, K), dtype=np.float64)
    # One pass per (a, b) cell using boolean indexing.
    for a, t_a in enumerate(tier_names):
        mask_a = (tj == t_a)
        if not mask_a.any():
            continue
        T_a_rows = T[mask_a]
        for b, t_b in enumerate(tier_names):
            mask_b = (te == t_b)
            if not mask_b.any():
                continue
            B[a, b] = float(T_a_rows[:, mask_b].sum())
    return B


def diag_mass(block_mass: np.ndarray) -> float:
    """Return `trace(B)` -- total mass on the tier diagonal."""
    return float(np.trace(np.asarray(block_mass, dtype=np.float64)))


def _sinkhorn_renormalize(T: np.ndarray, max_iter: int = 5000,
                          atol: float = 1e-12) -> np.ndarray:
    """Project a non-negative matrix onto exact doubly-stochastic uniform
    marginals by alternating row/column rescaling. POT's entropic GW leaves
    mild drift; we tighten it so `compute_block_mass` gives exact uniform
    block marginals under uniform tier counts (and exact tier-proportional
    marginals under unbalanced tier counts).
    """
    T = np.clip(np.asarray(T, dtype=np.float64), 0.0, None)
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
        drift = max(
            float(np.max(np.abs(T.sum(axis=1) - target_row))),
            float(np.max(np.abs(T.sum(axis=0) - target_col))),
        )
        if drift < atol:
            break
    return T


def _tier_counts(tiers: np.ndarray, tier_names: Sequence[str]) -> np.ndarray:
    tiers = np.asarray(tiers)
    return np.array([int((tiers == t).sum()) for t in tier_names], dtype=np.int64)


# --------- permutation null ---------------------------------------------------

def trial_level_block_null(coupling: np.ndarray,
                           tiers_j: np.ndarray,
                           tiers_e: np.ndarray,
                           tier_names: Sequence[str],
                           *,
                           n_perms: int = 1000,
                           seed: int = 1337) -> Dict[str, object]:
    """Permutation null for the block-diagonal mass of a trial-level coupling.

    The observed coupling is fixed; only the tier labels are shuffled on each
    side independently. Returns a dict with observed, null samples, p-value
    (upper tail), z-score, per-cell diagonal z-scores, and the observed B.
    """
    T = np.asarray(coupling, dtype=np.float64)
    tj = np.asarray(tiers_j); te = np.asarray(tiers_e)
    K = len(tier_names)

    B_obs = compute_block_mass(T, tj, te, tier_names)
    observed = diag_mass(B_obs)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perms, dtype=np.float64)
    diag_null = np.empty((n_perms, K), dtype=np.float64)
    for k in range(n_perms):
        sj = rng.permutation(tj)
        se = rng.permutation(te)
        B_k = compute_block_mass(T, sj, se, tier_names)
        null[k] = np.trace(B_k)
        diag_null[k] = np.diag(B_k)

    # Upper-tail p: tighter alignment = larger observed.
    p = (1.0 + float((null >= observed).sum())) / (1.0 + n_perms)
    mu = float(null.mean())
    sd = float(null.std(ddof=1)) if n_perms > 1 else float("nan")
    z = (observed - mu) / sd if sd > 1e-12 else float("nan")

    # Per-cell diagonal z-scores.
    obs_diag = np.diag(B_obs)
    cell_mu = diag_null.mean(axis=0)
    cell_sd = diag_null.std(axis=0, ddof=1) if n_perms > 1 else np.full(K, np.nan)
    per_cell_z = np.where(cell_sd > 1e-12, (obs_diag - cell_mu) / cell_sd, np.nan)

    return {
        "observed": float(observed),
        "null": null,
        "null_mean": mu,
        "null_std": sd,
        "p_value": float(p),
        "z_score": float(z),
        "n_permutations": int(n_perms),
        "per_cell_z": per_cell_z.astype(np.float64),
        "per_cell_observed": obs_diag.astype(np.float64),
        "per_cell_null_mean": cell_mu.astype(np.float64),
        "per_cell_null_std": cell_sd.astype(np.float64),
        "block_mass": B_obs,
        "tier_names": list(tier_names),
    }


# --------- all-trials primary analysis ---------------------------------------

def trial_level_block_null_all_trials(
    *,
    features_j: np.ndarray,
    tiers_j: np.ndarray,
    features_e: np.ndarray,
    tiers_e: np.ndarray,
    tier_names: Sequence[str],
    epsilon: float,
    n_perms: int = 1000,
    seed: int = 1337,
    entropic_gw_fn: Optional[
        Callable[[np.ndarray, np.ndarray, float], tuple[float, np.ndarray]]] = None,
    pairwise_rdm_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, object]:
    """Primary trial-level block-null on the full unbalanced dataset.

    Computes a single entropic-GW coupling on the NJ x NE problem (with
    uniform marginals 1/NJ, 1/NE), aggregates to a 3x3 block-mass matrix,
    and runs the tier-shuffle permutation null on that fixed coupling.

    Because tier counts are unbalanced (e.g. NJ=34/35/34, NE=170/170/170),
    the null does NOT send every cell to 1/9. Expected values are
    `(n_j_a / NJ) * (n_e_b / NE)` per cell, and expected trace is the sum of
    those over a == b. The function returns both observed and expected so
    downstream reporting can draw both lines on the null histogram.

    POT and SciPy are injected via `entropic_gw_fn` and `pairwise_rdm_fn` so
    this module has no circular import with `gw`. If either is None we fall
    back to the project's default wrappers (lazy imported to avoid cycles).
    """
    if pairwise_rdm_fn is None:
        from skill_manifold.rdms import pairwise_cosine_rdm as _pcr
        pairwise_rdm_fn = _pcr
    if entropic_gw_fn is None:
        def _default_gw(cj, ce, eps):
            from skill_manifold.gw import entropic_gromov_wasserstein as _egw
            r = _egw(cj, ce, epsilon=float(eps))
            return float(r.distance), np.asarray(r.coupling, dtype=np.float64)
        entropic_gw_fn = _default_gw

    features_j = np.asarray(features_j, dtype=np.float64)
    features_e = np.asarray(features_e, dtype=np.float64)
    tj = np.asarray(tiers_j); te = np.asarray(tiers_e)
    NJ = features_j.shape[0]; NE = features_e.shape[0]
    if tj.shape[0] != NJ or te.shape[0] != NE:
        raise ValueError(
            f"tier/feature length mismatch: NJ={NJ}, len(tj)={tj.shape[0]}, "
            f"NE={NE}, len(te)={te.shape[0]}")

    Cj = pairwise_rdm_fn(features_j)
    Ce = pairwise_rdm_fn(features_e)
    gw_distance, coupling = entropic_gw_fn(Cj, Ce, float(epsilon))
    coupling = np.asarray(coupling, dtype=np.float64)

    row_sum_drift = float(np.max(np.abs(coupling.sum(axis=1) - 1.0 / NJ)))
    col_sum_drift = float(np.max(np.abs(coupling.sum(axis=0) - 1.0 / NE)))
    T_used = coupling
    if max(row_sum_drift, col_sum_drift) > 1e-9:
        T_used = _sinkhorn_renormalize(coupling)

    # Observed block mass.
    B_obs = compute_block_mass(T_used, tj, te, tier_names)
    observed = float(np.trace(B_obs))

    # Expected cell values under independent tier-shuffle null: with uniform
    # coupling marginals, E[B[a,b]] = (n_j_a / NJ) * (n_e_b / NE).
    nj = _tier_counts(tj, tier_names)
    ne = _tier_counts(te, tier_names)
    expected_cell = np.outer(nj / float(NJ), ne / float(NE))   # (K, K)
    expected_trace = float(np.trace(expected_cell))

    rng = np.random.default_rng(seed)
    K = len(tier_names)
    null = np.empty(n_perms, dtype=np.float64)
    diag_null = np.empty((n_perms, K), dtype=np.float64)
    for k in range(n_perms):
        sj = rng.permutation(tj)
        se = rng.permutation(te)
        B_k = compute_block_mass(T_used, sj, se, tier_names)
        null[k] = np.trace(B_k)
        diag_null[k] = np.diag(B_k)

    mu = float(null.mean())
    sd = float(null.std(ddof=1)) if n_perms > 1 else float("nan")
    p = (1.0 + float((null >= observed).sum())) / (1.0 + n_perms)
    z = (observed - mu) / sd if sd > 1e-12 else float("nan")

    # Per-cell diagonal z-scores.
    obs_diag = np.diag(B_obs)
    cell_mu = diag_null.mean(axis=0)
    cell_sd = diag_null.std(axis=0, ddof=1) if n_perms > 1 else np.full(K, np.nan)
    per_cell_z = np.where(cell_sd > 1e-12, (obs_diag - cell_mu) / cell_sd, np.nan)

    return {
        "observed": observed,
        "null": null,
        "null_mean": mu,
        "null_std": sd,
        "p_value": float(p),
        "z_score": float(z),
        "expected_trace_under_null": expected_trace,
        "per_cell_z": per_cell_z.astype(np.float64),
        "per_cell_observed": obs_diag.astype(np.float64),
        "per_cell_expected": np.diag(expected_cell).astype(np.float64),
        "per_cell_null_mean": cell_mu.astype(np.float64),
        "per_cell_null_std": cell_sd.astype(np.float64),
        "block_mass": B_obs,
        "expected_block_mass": expected_cell,
        "tier_counts_j": nj,
        "tier_counts_e": ne,
        "tier_names": list(tier_names),
        "n_permutations": int(n_perms),
        "coupling_shape": (int(NJ), int(NE)),
        "row_sum_drift": row_sum_drift,
        "col_sum_drift": col_sum_drift,
        "gw_distance": float(gw_distance),
        "epsilon": float(epsilon),
    }


# --------- subsample robustness ----------------------------------------------

@dataclass
class SubsampleRobustness:
    """Aggregated outputs of the B=30 subsample-robustness loop."""
    observed: np.ndarray           # shape (B,) diag_mass per subsample
    p_values: np.ndarray           # shape (B,) trial-level null p per subsample
    z_scores: np.ndarray           # shape (B,) trial-level null z per subsample
    gw_distance: np.ndarray        # shape (B,) entropic-GW distance per subsample
    per_cell_z: np.ndarray         # shape (B, K) per-cell diagonal z per subsample
    tier_names: List[str]
    n_runs: int
    frac_p_lt_05: float
    observed_median: float
    observed_p05: float
    observed_p95: float
    # Per-cell summaries (length K).
    per_cell_median: np.ndarray
    per_cell_p05: np.ndarray
    per_cell_p95: np.ndarray
    frac_significant_per_cell_bonf: np.ndarray  # fraction of |z| > 2.394 per cell


def subsample_robustness(
    *,
    feat_j: np.ndarray,
    tiers_j_full: np.ndarray,
    feat_e: np.ndarray,
    tiers_e_full: np.ndarray,
    tier_names: Sequence[str],
    entropic_gw_fn: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]],
    pairwise_rdm_fn: Callable[[np.ndarray], np.ndarray],
    n_per_tier: int,
    base_seed: int,
    n_subsamples: int = 30,
    n_perms: int = 1000,
) -> SubsampleRobustness:
    """Repeat Step 9 + trial-level null on `n_subsamples` independent draws.

    Each subsample:
      1. Draw `n_per_tier` trials per tier from each side (balanced; with
         replacement if a tier has fewer rows than `n_per_tier`).
      2. Build cosine-distance RDMs `Cj`, `Ce`.
      3. Run entropic GW to get the coupling `T`.
      4. Run the trial-level block-mass null on `T` and collect diag_mass, p, z.

    The work inside each subsample is one entropic-GW call (seconds) plus
    n_perms block-mass aggregations (milliseconds). `entropic_gw_fn` and
    `pairwise_rdm_fn` are injected so this module does not import POT or
    SciPy directly.
    """
    rng = np.random.default_rng(base_seed)
    K = len(tier_names)
    obs = np.empty(n_subsamples, dtype=np.float64)
    ps = np.empty(n_subsamples, dtype=np.float64)
    zs = np.empty(n_subsamples, dtype=np.float64)
    gw = np.empty(n_subsamples, dtype=np.float64)
    per_cell_z = np.empty((n_subsamples, K), dtype=np.float64)

    for b in range(n_subsamples):
        seed_b = int(rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(seed_b)

        def _balanced(feats, tiers):
            parts_f, parts_t = [], []
            for t in tier_names:
                idxs = np.flatnonzero(tiers == t)
                if idxs.size == 0:
                    raise RuntimeError(f"tier {t!r} has no trials")
                choose = sub_rng.choice(idxs, size=n_per_tier,
                                        replace=idxs.size < n_per_tier)
                parts_f.append(feats[choose])
                parts_t.append(tiers[choose])
            return np.vstack(parts_f), np.concatenate(parts_t)

        fj, tj = _balanced(feat_j, tiers_j_full)
        fe, te = _balanced(feat_e, tiers_e_full)
        Cj = pairwise_rdm_fn(fj)
        Ce = pairwise_rdm_fn(fe)
        dist, T = entropic_gw_fn(Cj, Ce)
        result = trial_level_block_null(T, tj, te, tier_names,
                                        n_perms=n_perms, seed=seed_b)
        obs[b] = result["observed"]
        ps[b] = result["p_value"]
        zs[b] = result["z_score"]
        gw[b] = float(dist)
        per_cell_z[b] = np.asarray(result["per_cell_z"], dtype=np.float64)

    return SubsampleRobustness(
        observed=obs, p_values=ps, z_scores=zs, gw_distance=gw,
        per_cell_z=per_cell_z, tier_names=list(tier_names),
        n_runs=n_subsamples,
        frac_p_lt_05=float((ps < 0.05).mean()),
        observed_median=float(np.median(obs)),
        observed_p05=float(np.quantile(obs, 0.05)),
        observed_p95=float(np.quantile(obs, 0.95)),
        per_cell_median=np.median(per_cell_z, axis=0),
        per_cell_p05=np.quantile(per_cell_z, 0.05, axis=0),
        per_cell_p95=np.quantile(per_cell_z, 0.95, axis=0),
        frac_significant_per_cell_bonf=np.mean(
            np.abs(per_cell_z) > BONFERRONI_Z_THREE_CELLS, axis=0),
    )


# --------- stratified bootstrap (preserves observed tier counts) ------------

def subsample_robustness_stratified(
    *,
    features_j: np.ndarray,
    tiers_j: np.ndarray,
    features_e: np.ndarray,
    tiers_e: np.ndarray,
    tier_names: Sequence[str],
    epsilon: float,
    n_perms: int = 1000,
    n_bootstraps: int = 30,
    seed: int = 1337,
    entropic_gw_fn: Optional[
        Callable[[np.ndarray, np.ndarray, float], tuple[float, np.ndarray]]] = None,
    pairwise_rdm_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, object]:
    """Stratified bootstrap that preserves observed tier counts on each side.

    For each resample:
      1. Within each JIGSAWS tier `a`, draw `n_j_a` trials with replacement
         (preserving the observed n_j_a). Do the same on the Mimic side.
      2. Call `trial_level_block_null_all_trials` on the resampled features
         and tier labels (NJ x NE, same shape as the observed coupling).
      3. Collect trace p / z and per-cell z-scores.

    This is the right bootstrap when the fixed-cutoff binning is imbalanced
    (e.g. JIGSAWS High = 14 trials): drawing a balanced 100-per-tier would
    upsample each High trial ~7x and artificially inflate precision. The
    stratified approach matches the single-shot coupling shape exactly and
    only asks "how sensitive is the answer to *which* trials we observed
    within each tier?".
    """
    tj = np.asarray(tiers_j); te = np.asarray(tiers_e)
    feat_j = np.asarray(features_j, dtype=np.float64)
    feat_e = np.asarray(features_e, dtype=np.float64)
    K = len(tier_names)
    tier_counts_j = _tier_counts(tj, tier_names)
    tier_counts_e = _tier_counts(te, tier_names)

    # Precompute per-tier index pools once.
    pools_j = {t: np.flatnonzero(tj == t) for t in tier_names}
    pools_e = {t: np.flatnonzero(te == t) for t in tier_names}
    for t, pool in pools_j.items():
        if pool.size == 0 and int(tier_counts_j[tier_names.index(t)]) > 0:
            raise RuntimeError(f"JIGSAWS tier {t!r} has no trials to resample")
    for t, pool in pools_e.items():
        if pool.size == 0 and int(tier_counts_e[tier_names.index(t)]) > 0:
            raise RuntimeError(f"Mimic tier {t!r} has no trials to resample")

    rng = np.random.default_rng(seed)
    obs = np.empty(n_bootstraps, dtype=np.float64)
    ps = np.empty(n_bootstraps, dtype=np.float64)
    zs = np.empty(n_bootstraps, dtype=np.float64)
    per_cell_z = np.empty((n_bootstraps, K), dtype=np.float64)
    counts_j_per = np.empty((n_bootstraps, K), dtype=np.int64)
    counts_e_per = np.empty((n_bootstraps, K), dtype=np.int64)
    n_degenerate = 0

    b = 0
    while b < n_bootstraps:
        seed_b = int(rng.integers(0, 2**31 - 1))
        sub_rng = np.random.default_rng(seed_b)
        idx_j_parts = []; idx_e_parts = []
        tiers_j_parts = []; tiers_e_parts = []
        for k, t in enumerate(tier_names):
            n_a = int(tier_counts_j[k])
            if n_a > 0:
                pick = sub_rng.choice(pools_j[t], size=n_a, replace=True)
                idx_j_parts.append(pick)
                tiers_j_parts.append(np.full(n_a, t, dtype=object))
            n_b = int(tier_counts_e[k])
            if n_b > 0:
                pick = sub_rng.choice(pools_e[t], size=n_b, replace=True)
                idx_e_parts.append(pick)
                tiers_e_parts.append(np.full(n_b, t, dtype=object))

        idx_j = np.concatenate(idx_j_parts)
        idx_e = np.concatenate(idx_e_parts)
        tiers_j_bs = np.concatenate(tiers_j_parts)
        tiers_e_bs = np.concatenate(tiers_e_parts)

        try:
            result = trial_level_block_null_all_trials(
                features_j=feat_j[idx_j], tiers_j=tiers_j_bs,
                features_e=feat_e[idx_e], tiers_e=tiers_e_bs,
                tier_names=tier_names, epsilon=float(epsilon),
                n_perms=n_perms, seed=seed_b,
                entropic_gw_fn=entropic_gw_fn, pairwise_rdm_fn=pairwise_rdm_fn,
            )
        except Exception as e:
            log.warning("stratified bootstrap iter %d failed: %s", b, e)
            n_degenerate += 1
            continue

        obs[b] = float(result["observed"])
        ps[b] = float(result["p_value"])
        zs[b] = float(result["z_score"])
        per_cell_z[b] = np.asarray(result["per_cell_z"], dtype=np.float64)
        counts_j_per[b] = _tier_counts(tiers_j_bs, tier_names)
        counts_e_per[b] = _tier_counts(tiers_e_bs, tier_names)
        b += 1

    summary = {
        "p_values":       _median_p05_p95(ps),
        "trace_z_scores": _median_p05_p95(zs),
        "diag_mass":      _median_p05_p95(obs),
        **{
            f"per_cell_z_{tier_names[k]}": _median_p05_p95(per_cell_z[:, k])
            for k in range(K)
        },
    }
    return {
        "p_values": ps,
        "trace_z_scores": zs,
        "per_cell_z": per_cell_z,
        "diag_mass_observed": obs,
        "tier_counts_j": tier_counts_j,
        "tier_counts_e": tier_counts_e,
        "per_bootstrap_counts_j": counts_j_per,
        "per_bootstrap_counts_e": counts_e_per,
        "summary": summary,
        "frac_significant_trace": float((ps < 0.05).mean()),
        "frac_per_cell_bonf": np.mean(
            np.abs(per_cell_z) > BONFERRONI_Z_THREE_CELLS, axis=0).astype(float),
        "frac_per_cell_positive": np.mean(per_cell_z > 0, axis=0).astype(float),
        "n_bootstraps": int(n_bootstraps),
        "n_degenerate": int(n_degenerate),
        "tier_names": list(tier_names),
    }


def _median_p05_p95(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p05":    float(np.quantile(arr, 0.05)),
        "p95":    float(np.quantile(arr, 0.95)),
    }


def stratified_bootstrap_verdict(result: Dict[str, object],
                                 tier_names: Sequence[str]) -> Dict[str, object]:
    """Apply the GO / CAUTION / NO-GO rule from the spec.

    GO      : Mid<->Mid AND Low<->Low both have median z > +2
              AND frac(z > 0) > 0.90 on both
              AND frac(|z| > 2.394) > 0.50 on both.
    CAUTION : median z > +2 on Low OR Mid BUT frac(z > 0) < 0.90
              (robust magnitude, unstable sign).
    NO-GO   : neither Low nor Mid clears median z > +2
              (or the band is centered near zero).
    """
    names = list(tier_names)
    try:
        i_low = names.index("Low"); i_mid = names.index("Mid")
    except ValueError:
        return {"verdict": "unknown", "reason": "tier names missing Low/Mid"}
    pcz = np.asarray(result["per_cell_z"], dtype=np.float64)   # (B, K)
    frac_pos = np.asarray(result["frac_per_cell_positive"], dtype=np.float64)
    frac_bonf = np.asarray(result["frac_per_cell_bonf"], dtype=np.float64)
    med = np.median(pcz, axis=0)

    magnitudes_ok = med[i_low] > 2.0 and med[i_mid] > 2.0
    signs_ok = frac_pos[i_low] > 0.90 and frac_pos[i_mid] > 0.90
    bonf_ok = frac_bonf[i_low] > 0.50 and frac_bonf[i_mid] > 0.50
    any_median_gt2 = (med[i_low] > 2.0) or (med[i_mid] > 2.0)

    if magnitudes_ok and signs_ok and bonf_ok:
        v = "GO"
    elif any_median_gt2:
        v = "CAUTION"
    else:
        v = "NO-GO"
    return {
        "verdict": v,
        "low_median": float(med[i_low]),
        "mid_median": float(med[i_mid]),
        "frac_positive_low": float(frac_pos[i_low]),
        "frac_positive_mid": float(frac_pos[i_mid]),
        "frac_bonf_low": float(frac_bonf[i_low]),
        "frac_bonf_mid": float(frac_bonf[i_mid]),
    }


# --------- modality split (Mimic-side) --------------------------------------

def modality_split_analysis(
    *,
    features_j: np.ndarray,
    tiers_j_tertile: np.ndarray,
    tiers_j_fixed: np.ndarray,
    features_e_full: np.ndarray,
    e_cols_full: Sequence[str],
    modality_columns: Dict[str, Sequence[str]],
    tiers_e_tertile: np.ndarray,
    tiers_e_fixed: np.ndarray,
    tier_names: Sequence[str],
    epsilon: float,
    n_perms: int,
    n_bootstraps: int,
    seed: int,
    entropic_gw_fn: Optional[
        Callable[[np.ndarray, np.ndarray, float], tuple[float, np.ndarray]]] = None,
    pairwise_rdm_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, object]:
    """Run the trial-level all-trials null AND the stratified bootstrap for
    each Mimic-side modality subset under both tertile and fixed-cutoff
    binning. JIGSAWS side stays as the full combined manifold throughout.

    Returns a dict keyed by modality name with {'fixed': {...}, 'tertile':
    {...}, 'mimic_feature_dim': int} per modality plus '_meta'. Each inner
    'fixed' and 'tertile' carries the primary analysis, the bootstrap, and
    the go/no-go verdict computed by `stratified_bootstrap_verdict`.
    """
    col_to_idx = {c: i for i, c in enumerate(e_cols_full)}
    mod_order = list(modality_columns.keys())

    def _seed_for(mod_idx: int, bin_idx: int) -> int:
        return int(seed) + (mod_idx + 1) * 10_000 + (bin_idx + 1) * 100

    def _run_one(features_e_sub: np.ndarray,
                 tj: np.ndarray, te: np.ndarray,
                 seed_primary: int, seed_bootstrap: int) -> Dict[str, object]:
        primary = trial_level_block_null_all_trials(
            features_j=features_j, tiers_j=tj,
            features_e=features_e_sub, tiers_e=te,
            tier_names=tier_names, epsilon=float(epsilon),
            n_perms=int(n_perms), seed=seed_primary,
            entropic_gw_fn=entropic_gw_fn, pairwise_rdm_fn=pairwise_rdm_fn,
        )
        bootstrap = subsample_robustness_stratified(
            features_j=features_j, tiers_j=tj,
            features_e=features_e_sub, tiers_e=te,
            tier_names=tier_names, epsilon=float(epsilon),
            n_perms=max(200, int(n_perms) // 5),
            n_bootstraps=int(n_bootstraps), seed=seed_bootstrap,
            entropic_gw_fn=entropic_gw_fn, pairwise_rdm_fn=pairwise_rdm_fn,
        )
        verdict = stratified_bootstrap_verdict(bootstrap, tier_names)
        return {"primary": primary, "bootstrap": bootstrap, "verdict": verdict}

    out: Dict[str, object] = {}
    for m_idx, mod in enumerate(mod_order):
        cols = list(modality_columns[mod])
        # Feature sub-matrix = column subset of the already-residualized
        # Mimic matrix. We do NOT re-residualize per modality; a 64-column
        # subset of a residualized 146-column matrix is itself residualized
        # against the same nuisances (residualization is per-column).
        missing = [c for c in cols if c not in col_to_idx]
        if missing:
            raise ValueError(
                f"modality {mod!r} references unknown columns: {missing[:3]}...")
        idx = np.array([col_to_idx[c] for c in cols], dtype=np.int64)
        features_e_sub = features_e_full[:, idx]

        fixed = _run_one(
            features_e_sub, tiers_j_fixed, tiers_e_fixed,
            seed_primary=_seed_for(m_idx, 0),
            seed_bootstrap=_seed_for(m_idx, 0) + 1,
        )
        tertile = _run_one(
            features_e_sub, tiers_j_tertile, tiers_e_tertile,
            seed_primary=_seed_for(m_idx, 1),
            seed_bootstrap=_seed_for(m_idx, 1) + 1,
        )
        out[mod] = {
            "fixed": fixed,
            "tertile": tertile,
            "mimic_feature_dim": int(len(cols)),
        }

    out["_meta"] = {
        "jigsaws_feature_dim": int(features_j.shape[1]),
        "n_bootstraps": int(n_bootstraps),
        "epsilon": float(epsilon),
        "tier_names": list(tier_names),
        "modalities": mod_order,
    }
    return out


def jigsaws_modality_split_analysis(
    *,
    features_j_full: np.ndarray,
    j_cols_full: Sequence[str],
    modality_columns: Dict[str, Sequence[str]],
    tiers_j_tertile: np.ndarray,
    tiers_j_fixed: np.ndarray,
    features_e: np.ndarray,
    tiers_e_tertile: np.ndarray,
    tiers_e_fixed: np.ndarray,
    tier_names: Sequence[str],
    epsilon: float,
    n_perms: int,
    n_bootstraps: int,
    seed: int,
    entropic_gw_fn: Optional[
        Callable[[np.ndarray, np.ndarray, float], tuple[float, np.ndarray]]] = None,
    pairwise_rdm_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, object]:
    """Mirror of `modality_split_analysis` with the JIGSAWS side as the
    ablation target. The Mimic side stays as a fixed full combined manifold;
    each JIGSAWS-side modality subset (gestures, kinematics, ...) is run
    against it under both tertile and fixed-cutoff binning.

    Returns the same schema as `modality_split_analysis` but with the roles
    swapped conceptually -- per-modality primary/bootstrap/verdict for the
    JIGSAWS feature subsets.
    """
    col_to_idx = {c: i for i, c in enumerate(j_cols_full)}
    mod_order = list(modality_columns.keys())

    def _seed_for(mod_idx: int, bin_idx: int) -> int:
        # Offset by 100_000 so seeds do not collide with the Mimic-side call.
        return int(seed) + 100_000 + (mod_idx + 1) * 10_000 + (bin_idx + 1) * 100

    def _run_one(features_j_sub: np.ndarray,
                 tj: np.ndarray, te: np.ndarray,
                 seed_primary: int, seed_bootstrap: int) -> Dict[str, object]:
        primary = trial_level_block_null_all_trials(
            features_j=features_j_sub, tiers_j=tj,
            features_e=features_e, tiers_e=te,
            tier_names=tier_names, epsilon=float(epsilon),
            n_perms=int(n_perms), seed=seed_primary,
            entropic_gw_fn=entropic_gw_fn, pairwise_rdm_fn=pairwise_rdm_fn,
        )
        bootstrap = subsample_robustness_stratified(
            features_j=features_j_sub, tiers_j=tj,
            features_e=features_e, tiers_e=te,
            tier_names=tier_names, epsilon=float(epsilon),
            n_perms=max(200, int(n_perms) // 5),
            n_bootstraps=int(n_bootstraps), seed=seed_bootstrap,
            entropic_gw_fn=entropic_gw_fn, pairwise_rdm_fn=pairwise_rdm_fn,
        )
        verdict = stratified_bootstrap_verdict(bootstrap, tier_names)
        return {"primary": primary, "bootstrap": bootstrap, "verdict": verdict}

    out: Dict[str, object] = {}
    for m_idx, mod in enumerate(mod_order):
        cols = list(modality_columns[mod])
        missing = [c for c in cols if c not in col_to_idx]
        if missing:
            raise ValueError(
                f"modality {mod!r} references unknown columns: {missing[:3]}...")
        idx = np.array([col_to_idx[c] for c in cols], dtype=np.int64)
        features_j_sub = features_j_full[:, idx]

        fixed = _run_one(
            features_j_sub, tiers_j_fixed, tiers_e_fixed,
            seed_primary=_seed_for(m_idx, 0),
            seed_bootstrap=_seed_for(m_idx, 0) + 1,
        )
        tertile = _run_one(
            features_j_sub, tiers_j_tertile, tiers_e_tertile,
            seed_primary=_seed_for(m_idx, 1),
            seed_bootstrap=_seed_for(m_idx, 1) + 1,
        )
        out[mod] = {
            "fixed": fixed,
            "tertile": tertile,
            "jigsaws_feature_dim": int(len(cols)),
        }

    out["_meta"] = {
        "mimic_feature_dim": int(features_e.shape[1]),
        "n_bootstraps": int(n_bootstraps),
        "epsilon": float(epsilon),
        "tier_names": list(tier_names),
        "modalities": mod_order,
    }
    return out


# --------- P1: EEG baseline/pc correlation diagnostic -----------------------

def eeg_baseline_pc_correlation(
    features_e_full: np.ndarray,
    baseline_cols: Sequence[str],
    pc_cols: Sequence[str],
    e_cols_full: Sequence[str],
) -> Dict[str, object]:
    """Per-trial Pearson correlation between the EEG baseline and predictive-
    coding encoder means, plus the mean/std/abs-median of that distribution.

    The two EEG encoders consume the same Phase 1 windows so they are NOT
    independent evidence streams. This diagnostic surfaces the magnitude of
    that dependence so downstream per-modality verdicts can be read with the
    right caveat.
    """
    col_to_idx = {c: i for i, c in enumerate(e_cols_full)}
    b_idx = np.array([col_to_idx[c] for c in baseline_cols], dtype=np.int64)
    p_idx = np.array([col_to_idx[c] for c in pc_cols], dtype=np.int64)
    B = features_e_full[:, b_idx]
    P = features_e_full[:, p_idx]
    # Per-trial Pearson correlation = cov(B_i, P_i) / (std(B_i) * std(P_i)).
    # Features are pre-z-scored across trials, not across EEG dims; compute
    # the per-trial statistic directly.
    Bm = B - B.mean(axis=1, keepdims=True)
    Pm = P - P.mean(axis=1, keepdims=True)
    num = (Bm * Pm).sum(axis=1)
    den = np.sqrt((Bm ** 2).sum(axis=1) * (Pm ** 2).sum(axis=1))
    rho = np.where(den > 1e-12, num / den, 0.0)
    return {
        "per_trial_pearson_r": rho.astype(np.float64),
        "mean": float(rho.mean()),
        "median": float(np.median(rho)),
        "median_abs": float(np.median(np.abs(rho))),
        "std": float(rho.std()),
        "p05": float(np.quantile(rho, 0.05)),
        "p95": float(np.quantile(rho, 0.95)),
        "frac_abs_gt_0p5": float((np.abs(rho) > 0.5).mean()),
        "n_trials": int(rho.size),
    }


# --------- P3: pooled-128 random-split null ---------------------------------

def pooled_eeg_random_split_null(
    *,
    features_j: np.ndarray,
    tiers_j: np.ndarray,
    features_e_full: np.ndarray,
    e_cols_full: Sequence[str],
    eeg_combined_cols: Sequence[str],      # all 128 baseline+pc columns
    half_size: int,                         # 64 (how many cols per random half)
    tiers_e: np.ndarray,
    tier_names: Sequence[str],
    epsilon: float,
    n_perms: int,
    n_random_splits: int,
    seed: int,
    entropic_gw_fn: Optional[
        Callable[[np.ndarray, np.ndarray, float], tuple[float, np.ndarray]]] = None,
    pairwise_rdm_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, object]:
    """Negative control for the baseline-vs-pc modality contrast.

    If the Step 10 difference between baseline trace_z and pc trace_z is
    genuinely "predictive-coding carries more signal than baseline," that
    signed difference should be *larger* than most random 64+64 splits of
    the combined 128 EEG dims. If the observed difference falls inside the
    random-split distribution, the split is statistically uninformative --
    any 64-column draw gives roughly the same per-cell z.

    Uses `trial_level_block_null_all_trials` under the hood with a fresh
    seed per random split. The coupling null is small (n_perms capped at
    200 for speed) since this block is a sensitivity check, not a primary
    significance test.
    """
    col_to_idx = {c: i for i, c in enumerate(e_cols_full)}
    combined_idx = np.array([col_to_idx[c] for c in eeg_combined_cols],
                            dtype=np.int64)
    n_combined = combined_idx.size
    if half_size * 2 > n_combined:
        raise ValueError(
            f"half_size*2 ({half_size*2}) exceeds combined EEG dims ({n_combined})")

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_random_splits, dtype=np.float64)
    z_a = np.empty(n_random_splits, dtype=np.float64)
    z_b = np.empty(n_random_splits, dtype=np.float64)

    for k in range(n_random_splits):
        perm = rng.permutation(combined_idx)
        half_a = perm[:half_size]
        half_b = perm[half_size:2 * half_size]

        primary_a = trial_level_block_null_all_trials(
            features_j=features_j, tiers_j=tiers_j,
            features_e=features_e_full[:, half_a], tiers_e=tiers_e,
            tier_names=tier_names, epsilon=float(epsilon),
            n_perms=min(int(n_perms), 200),
            seed=int(rng.integers(0, 2**31 - 1)),
            entropic_gw_fn=entropic_gw_fn, pairwise_rdm_fn=pairwise_rdm_fn,
        )
        primary_b = trial_level_block_null_all_trials(
            features_j=features_j, tiers_j=tiers_j,
            features_e=features_e_full[:, half_b], tiers_e=tiers_e,
            tier_names=tier_names, epsilon=float(epsilon),
            n_perms=min(int(n_perms), 200),
            seed=int(rng.integers(0, 2**31 - 1)),
            entropic_gw_fn=entropic_gw_fn, pairwise_rdm_fn=pairwise_rdm_fn,
        )
        z_a[k] = float(primary_a["z_score"])
        z_b[k] = float(primary_b["z_score"])
        deltas[k] = z_a[k] - z_b[k]

    return {
        "n_random_splits": int(n_random_splits),
        "half_size": int(half_size),
        "z_a": z_a, "z_b": z_b,
        "delta_distribution": deltas,   # signed z_a - z_b across random draws
        "delta_abs_median": float(np.median(np.abs(deltas))),
        "delta_abs_p95": float(np.quantile(np.abs(deltas), 0.95)),
        "delta_median": float(np.median(deltas)),
        "delta_p05": float(np.quantile(deltas, 0.05)),
        "delta_p95": float(np.quantile(deltas, 0.95)),
    }


# --------- P4: coupling-matrix diagnostics (per Mimic modality) -------------

def coupling_matrix_diagnostics(
    coupling: np.ndarray,
    features_e_sub: np.ndarray,
    *,
    near_duplicate_cos_eps: float = 1e-4,
) -> Dict[str, float]:
    """Diagnostics for a single entropic-GW coupling: pre-renorm marginal
    drift, numerical rank of the cosine-distance cost matrix, and count of
    near-duplicate Mimic rows under cosine distance. Intended as a targeted
    check for the Step 10 eye-only coupling which was a suspected outlier.
    """
    T = np.asarray(coupling, dtype=np.float64)
    row_drift = float(np.max(np.abs(T.sum(axis=1) - 1.0 / T.shape[0])))
    col_drift = float(np.max(np.abs(T.sum(axis=0) - 1.0 / T.shape[1])))

    norms = np.linalg.norm(features_e_sub, axis=1)
    mask = norms < 1e-12
    feats = features_e_sub.astype(np.float64).copy()
    if mask.any():
        feats[mask] = 1e-12
    # Pairwise cosine similarity, count near-duplicates above the threshold.
    normed = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    S = normed @ normed.T
    iu, ju = np.triu_indices(S.shape[0], k=1)
    near_dup = int((1.0 - S[iu, ju] < near_duplicate_cos_eps).sum())

    # Rank of the distance matrix up to numerical tolerance.
    D = 1.0 - S
    rank = int(np.linalg.matrix_rank(D, tol=1e-6))
    return {
        "coupling_row_drift": row_drift,
        "coupling_col_drift": col_drift,
        "distance_rank": rank,
        "near_duplicate_pairs": near_dup,
        "total_pairs": int(iu.size),
        "near_duplicate_fraction": float(near_dup / max(1, iu.size)),
        "n_trials": int(features_e_sub.shape[0]),
        "n_features": int(features_e_sub.shape[1]),
    }


# --------- epsilon sensitivity (optional, no null) ---------------------------

def epsilon_sensitivity(
    *,
    Cj: np.ndarray,
    Ce: np.ndarray,
    tiers_j: np.ndarray,
    tiers_e: np.ndarray,
    tier_names: Sequence[str],
    entropic_gw_fn_eps: Callable[[np.ndarray, np.ndarray, float], tuple[float, np.ndarray]],
    epsilons: Sequence[float] = (0.005, 0.01, 0.02, 0.05),
    n_perms: int = 200,
    seed: int = 1337,
) -> List[dict]:
    """For each epsilon, run entropic GW on `(Cj, Ce)` and record both the
    trace `diag_mass` and the per-cell diagonal z-scores against the
    tier-shuffle null. Per-cell z's require a small permutation loop
    (default 200 perms); the loop is cheap because the coupling is fixed."""
    tj = np.asarray(tiers_j); te = np.asarray(tiers_e)
    K = len(tier_names)
    rows: List[dict] = []
    for eps in epsilons:
        dist, T_raw = entropic_gw_fn_eps(Cj, Ce, float(eps))
        T_raw = np.asarray(T_raw, dtype=np.float64)
        row_drift = float(np.max(np.abs(T_raw.sum(axis=1) - 1.0 / T_raw.shape[0])))
        col_drift = float(np.max(np.abs(T_raw.sum(axis=0) - 1.0 / T_raw.shape[1])))
        T = (_sinkhorn_renormalize(T_raw) if max(row_drift, col_drift) > 1e-9 else T_raw)
        B = compute_block_mass(T, tj, te, tier_names)
        obs_diag = np.diag(B)

        rng = np.random.default_rng(seed)
        diag_null = np.empty((n_perms, K), dtype=np.float64)
        for k in range(n_perms):
            sj = rng.permutation(tj); se = rng.permutation(te)
            B_k = compute_block_mass(T, sj, se, tier_names)
            diag_null[k] = np.diag(B_k)
        cell_mu = diag_null.mean(axis=0)
        cell_sd = diag_null.std(axis=0, ddof=1) if n_perms > 1 else np.full(K, np.nan)
        per_cell_z = np.where(cell_sd > 1e-12,
                              (obs_diag - cell_mu) / cell_sd, np.nan)

        rows.append({
            "epsilon": float(eps),
            "gw_distance": float(dist),
            "diag_mass": float(np.trace(B)),
            "block_mass": B,
            "per_cell_z": per_cell_z.astype(np.float64),
            "per_cell_observed": obs_diag.astype(np.float64),
            "row_sum_drift": row_drift,
            "col_sum_drift": col_drift,
        })
    return rows
