"""Smoke tests for skill_manifold.trial_null."""
from __future__ import annotations

import numpy as np

from skill_manifold.trial_null import (
    compute_block_mass, diag_mass, trial_level_block_null,
    trial_level_block_null_all_trials, subsample_robustness_stratified,
    stratified_bootstrap_verdict,
)

TIER_NAMES = ("Low", "Mid", "High")


def _balanced_tiers(n_per_tier: int) -> np.ndarray:
    return np.array([*(["Low"] * n_per_tier),
                     *(["Mid"] * n_per_tier),
                     *(["High"] * n_per_tier)])


def test_block_mass_marginals_uniform():
    # Uniform 300x300 coupling: every entry = 1/N^2.
    n_per = 100; N = 3 * n_per
    T = np.full((N, N), 1.0 / (N * N))
    tj = te = _balanced_tiers(n_per)
    B = compute_block_mass(T, tj, te, TIER_NAMES)
    assert B.shape == (3, 3)
    assert abs(B.sum() - 1.0) < 1e-9
    assert np.allclose(B.sum(axis=0), 1.0 / 3.0, atol=1e-9)
    assert np.allclose(B.sum(axis=1), 1.0 / 3.0, atol=1e-9)
    # All cells should be 1/9 under uniform coupling + balanced tiers.
    assert np.allclose(B, 1.0 / 9.0, atol=1e-9)


def test_diagonal_coupling_gives_full_diag_mass():
    # Block-diagonal coupling: mass only within matching tiers.
    n_per = 50; N = 3 * n_per
    T = np.zeros((N, N))
    block = 1.0 / 3.0 / (n_per * n_per)   # each matching block has total mass 1/3
    for b in range(3):
        s = b * n_per
        T[s:s + n_per, s:s + n_per] = block
    tj = te = _balanced_tiers(n_per)
    B = compute_block_mass(T, tj, te, TIER_NAMES)
    assert abs(B.sum() - 1.0) < 1e-9
    assert abs(diag_mass(B) - 1.0) < 1e-9
    # Off-diagonals are zero.
    off = B - np.diag(np.diag(B))
    assert np.max(np.abs(off)) < 1e-12


def test_null_mean_is_one_third():
    # Under a uniform coupling, trace(B) should be 1/3 in expectation; the null
    # mean from 1000 permutations should be within 3*sigma/sqrt(K) of that.
    n_per = 100; N = 3 * n_per
    T = np.full((N, N), 1.0 / (N * N))
    tj = te = _balanced_tiers(n_per)
    K = 1000
    result = trial_level_block_null(T, tj, te, TIER_NAMES, n_perms=K, seed=0)
    # Sanity: trace is always exactly 1/3 under the uniform coupling --
    # shuffling labels does not change anything because every entry is equal.
    assert abs(result["observed"] - 1.0 / 3.0) < 1e-9
    assert abs(result["null_mean"] - 1.0 / 3.0) < 1e-9
    # Null std should be (near) zero under the uniform coupling.
    assert result["null_std"] < 1e-9 or np.isnan(result["null_std"])
    # p-value in the degenerate-zero-std case: (1 + K) / (1 + K) == 1.
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n_permutations"] == K


def test_null_mean_on_nontrivial_coupling_converges_to_one_third():
    # A non-uniform coupling (random but with uniform marginals-ish) should
    # yield a permutation null whose mean converges to 1/3 up to MC noise.
    rng = np.random.default_rng(0)
    n_per = 50; N = 3 * n_per
    T = rng.dirichlet(alpha=np.ones(N), size=N)   # each row sums to 1
    T = T / N                                      # total mass = 1; row sum = 1/N
    tj = te = _balanced_tiers(n_per)
    K = 500
    r = trial_level_block_null(T, tj, te, TIER_NAMES, n_perms=K, seed=7)
    mc_err = 3.0 * r["null_std"] / np.sqrt(K)
    assert abs(r["null_mean"] - 1.0 / 3.0) < max(mc_err, 1e-3)


def test_all_trials_expected_trace_matches_tier_count_product():
    """With unbalanced tier counts, E[trace(B)] under tier-shuffle equals
    sum_a (n_j_a * n_e_a) / (NJ * NE). Verify on a synthetic unbalanced
    setup (40/30/33 JIGSAWS-like, 200/180/130 Mimic-like) with a stub
    entropic-GW that returns the exact uniform coupling, so observed and
    expected trace coincide and only Monte Carlo noise remains."""
    nj_counts = (40, 30, 33)
    ne_counts = (200, 180, 130)
    NJ = sum(nj_counts); NE = sum(ne_counts)

    # Synthetic features (values don't matter -- the stub GW ignores them).
    rng = np.random.default_rng(0)
    feat_j = rng.normal(size=(NJ, 4))
    feat_e = rng.normal(size=(NE, 5))
    tiers_j = np.array(sum(([t] * n for t, n in zip(TIER_NAMES, nj_counts)), []))
    tiers_e = np.array(sum(([t] * n for t, n in zip(TIER_NAMES, ne_counts)), []))

    def _stub_rdm(X):
        return np.zeros((X.shape[0], X.shape[0]))

    def _stub_gw(Cj, Ce, eps):
        # Exact uniform coupling.
        M, N = Cj.shape[0], Ce.shape[0]
        T = np.full((M, N), 1.0 / (M * N))
        return 0.0, T

    r = trial_level_block_null_all_trials(
        features_j=feat_j, tiers_j=tiers_j,
        features_e=feat_e, tiers_e=tiers_e,
        tier_names=TIER_NAMES, epsilon=0.01,
        n_perms=1000, seed=42,
        entropic_gw_fn=_stub_gw, pairwise_rdm_fn=_stub_rdm,
    )

    # Sanity: coupling shape and marginals.
    assert r["coupling_shape"] == (NJ, NE)
    B = np.asarray(r["block_mass"])
    assert abs(B.sum() - 1.0) < 1e-9
    assert np.allclose(B.sum(axis=1), np.array(nj_counts) / NJ, atol=1e-9)
    assert np.allclose(B.sum(axis=0), np.array(ne_counts) / NE, atol=1e-9)

    expected_trace = (nj_counts[0] * ne_counts[0] +
                      nj_counts[1] * ne_counts[1] +
                      nj_counts[2] * ne_counts[2]) / (NJ * NE)
    # Observed under the uniform stub equals expected exactly.
    assert abs(r["observed"] - expected_trace) < 1e-12
    assert abs(r["expected_trace_under_null"] - expected_trace) < 1e-12
    # Null mean: with a *uniform* coupling, shuffling tier labels changes nothing,
    # so the null is exactly the expected value (zero variance).
    assert abs(r["null_mean"] - expected_trace) < 1e-12


def test_all_trials_null_mean_converges_with_nontrivial_coupling():
    """With a non-uniform coupling the null mean should still converge to
    expected_trace = sum_a (n_j_a n_e_a) / (NJ NE) within Monte Carlo noise."""
    nj_counts = (40, 30, 33); ne_counts = (200, 180, 130)
    NJ = sum(nj_counts); NE = sum(ne_counts)
    rng = np.random.default_rng(1)
    tiers_j = np.array(sum(([t] * n for t, n in zip(TIER_NAMES, nj_counts)), []))
    tiers_e = np.array(sum(([t] * n for t, n in zip(TIER_NAMES, ne_counts)), []))
    feat_j = rng.normal(size=(NJ, 3))
    feat_e = rng.normal(size=(NE, 3))

    # Non-uniform doubly-stochastic-ish coupling.
    base = rng.dirichlet(alpha=np.ones(NE), size=NJ)
    T0 = base / (NJ * base.sum(axis=1, keepdims=True))   # row sum = 1/NJ
    # Column drift is fine for the test; the function will renormalize.

    def _stub_gw(Cj, Ce, eps):
        return 0.0, T0.copy()

    def _stub_rdm(X):
        return np.zeros((X.shape[0], X.shape[0]))

    r = trial_level_block_null_all_trials(
        features_j=feat_j, tiers_j=tiers_j,
        features_e=feat_e, tiers_e=tiers_e,
        tier_names=TIER_NAMES, epsilon=0.01,
        n_perms=1000, seed=7,
        entropic_gw_fn=_stub_gw, pairwise_rdm_fn=_stub_rdm,
    )
    expected = (nj_counts[0] * ne_counts[0] +
                nj_counts[1] * ne_counts[1] +
                nj_counts[2] * ne_counts[2]) / (NJ * NE)
    mc_err = 3.0 * r["null_std"] / np.sqrt(r["n_permutations"])
    assert abs(r["null_mean"] - expected) < max(mc_err, 1e-3)


def test_stratified_bootstrap_preserves_tier_counts():
    """For imbalanced tier counts (49/40/14 JIGSAWS-like, 147/163/200 Mimic-like),
    every bootstrap resample must reproduce the input tier counts exactly."""
    nj_counts = (49, 40, 14)
    ne_counts = (147, 163, 200)
    NJ = sum(nj_counts); NE = sum(ne_counts)
    rng = np.random.default_rng(0)
    tiers_j = np.array(sum(([t] * n for t, n in zip(TIER_NAMES, nj_counts)), []))
    tiers_e = np.array(sum(([t] * n for t, n in zip(TIER_NAMES, ne_counts)), []))
    feat_j = rng.normal(size=(NJ, 4))
    feat_e = rng.normal(size=(NE, 4))

    def _stub_rdm(X):
        return np.zeros((X.shape[0], X.shape[0]))

    def _stub_gw(Cj, Ce, eps):
        # Uniform coupling so every resample is valid.
        M, N = Cj.shape[0], Ce.shape[0]
        return 0.0, np.full((M, N), 1.0 / (M * N))

    r = subsample_robustness_stratified(
        features_j=feat_j, tiers_j=tiers_j,
        features_e=feat_e, tiers_e=tiers_e,
        tier_names=TIER_NAMES, epsilon=0.01, n_perms=50,
        n_bootstraps=5, seed=1,
        entropic_gw_fn=_stub_gw, pairwise_rdm_fn=_stub_rdm,
    )
    assert r["n_bootstraps"] == 5
    assert r["n_degenerate"] == 0
    assert r["per_cell_z"].shape == (5, 3)
    assert tuple(r["tier_counts_j"].tolist()) == nj_counts
    assert tuple(r["tier_counts_e"].tolist()) == ne_counts
    # Every bootstrap's per-iteration tier counts match the observed counts.
    for b in range(5):
        assert tuple(r["per_bootstrap_counts_j"][b].tolist()) == nj_counts
        assert tuple(r["per_bootstrap_counts_e"][b].tolist()) == ne_counts
    # Under a uniform coupling the trace equals the expected trace on every
    # bootstrap, so trace z is ill-defined (constant observed) and we don't
    # assert on z; but the p-value distribution should be well-formed.
    assert all(0.0 <= p <= 1.0 for p in r["p_values"])
    # Fractions are in [0, 1].
    for f in (*r["frac_per_cell_bonf"], *r["frac_per_cell_positive"]):
        assert 0.0 <= f <= 1.0


def test_stratified_bootstrap_verdict_go_caution_no_go():
    """Exercise the verdict rule on synthetic per-cell z stacks."""
    tier_names = ("Low", "Mid", "High")
    # GO: Low and Mid both strongly positive, signs consistent.
    rng = np.random.default_rng(0)
    strong_pos = np.stack([
        rng.normal(loc=3.0, scale=0.5, size=30),
        rng.normal(loc=3.0, scale=0.5, size=30),
        rng.normal(loc=0.0, scale=0.5, size=30),
    ], axis=1)
    go = {
        "per_cell_z": strong_pos,
        "frac_per_cell_bonf": np.mean(np.abs(strong_pos) > 2.394, axis=0),
        "frac_per_cell_positive": np.mean(strong_pos > 0, axis=0),
    }
    assert stratified_bootstrap_verdict(go, tier_names)["verdict"] == "GO"

    # CAUTION: Mid median above +2 but sign unstable (wide variance).
    mid_noisy = np.stack([
        rng.normal(loc=0.0, scale=1.5, size=30),
        rng.normal(loc=2.5, scale=3.0, size=30),
        rng.normal(loc=0.0, scale=1.5, size=30),
    ], axis=1)
    caution = {
        "per_cell_z": mid_noisy,
        "frac_per_cell_bonf": np.mean(np.abs(mid_noisy) > 2.394, axis=0),
        "frac_per_cell_positive": np.mean(mid_noisy > 0, axis=0),
    }
    assert stratified_bootstrap_verdict(caution, tier_names)["verdict"] == "CAUTION"

    # NO-GO: all cells center at zero.
    nogo_z = rng.normal(scale=1.0, size=(30, 3))
    nogo = {
        "per_cell_z": nogo_z,
        "frac_per_cell_bonf": np.mean(np.abs(nogo_z) > 2.394, axis=0),
        "frac_per_cell_positive": np.mean(nogo_z > 0, axis=0),
    }
    assert stratified_bootstrap_verdict(nogo, tier_names)["verdict"] == "NO-GO"


def test_per_cell_z_signs_when_only_low_low_has_excess_mass():
    # Construct a coupling where only the Low-Low block is enriched relative
    # to uniform. To keep Mid-Mid and High-High untouched, we take the excess
    # mass from the four "extreme off-diagonal" blocks that involve either Low
    # or High (not Mid-Mid, not High-High).
    n_per = 60; N = 3 * n_per
    T = np.full((N, N), 1.0 / (N * N))
    slc = {t: slice(i * n_per, (i + 1) * n_per) for i, t in enumerate(TIER_NAMES)}

    # Excess in Low-Low; balance by subtracting from Mid-High + High-Mid so
    # that Mid-Mid and High-High remain exactly at 1/9.
    excess = 0.05
    T[slc["Low"], slc["Low"]] += excess / (n_per * n_per)
    T[slc["Mid"], slc["High"]] -= (excess / 2) / (n_per * n_per)
    T[slc["High"], slc["Mid"]] -= (excess / 2) / (n_per * n_per)

    tj = te = _balanced_tiers(n_per)
    r = trial_level_block_null(T, tj, te, TIER_NAMES, n_perms=500, seed=1)
    per_z = r["per_cell_z"]
    assert per_z[0] > 2.0, f"Low-Low z should be clearly positive; got {per_z[0]:.2f}"
    # Mid-Mid and High-High each contribute exactly 1/9 in the constructed T,
    # so their observed value equals the null mean -> z == 0.
    assert abs(per_z[1]) < 0.1, f"Mid-Mid z should be ~0; got {per_z[1]:.2f}"
    assert abs(per_z[2]) < 0.1, f"High-High z should be ~0; got {per_z[2]:.2f}"
