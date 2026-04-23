"""Smoke tests for skill_manifold.gw."""
from __future__ import annotations

import numpy as np

from skill_manifold.gw import (
    entropic_gromov_wasserstein, gromov_wasserstein_centroids,
    permutation_null_centroid,
)


def _identical_rdm():
    return np.array([[0.0, 1.0, 2.0],
                     [1.0, 0.0, 1.0],
                     [2.0, 1.0, 0.0]])


def test_gw_identical_rdms_is_zero():
    R = _identical_rdm()
    out = gromov_wasserstein_centroids(R, R.copy(), ("Low", "Mid", "High"))
    assert abs(out.distance) < 1e-8
    # Coupling marginals should be uniform 1/3.
    assert np.allclose(out.coupling.sum(axis=0), 1.0 / 3.0, atol=1e-6)
    assert np.allclose(out.coupling.sum(axis=1), 1.0 / 3.0, atol=1e-6)


def test_gw_different_rdms_is_positive():
    R1 = _identical_rdm()
    R2 = np.array([[0.0, 0.1, 0.2],
                   [0.1, 0.0, 0.1],
                   [0.2, 0.1, 0.0]])
    out = gromov_wasserstein_centroids(R1, R2, ("Low", "Mid", "High"))
    assert out.distance > 0


def test_permutation_null_shape_and_pvalue_in_range():
    rng = np.random.default_rng(0)
    # Build feature clouds where tier means are clearly separated on side A
    # but only weakly on side B.
    n = 15
    tiers = np.array(["Low"] * n + ["Mid"] * n + ["High"] * n)
    means_a = {"Low": -2.0, "Mid": 0.0, "High": 2.0}
    means_b = {"Low": 0.0,  "Mid": 0.05, "High": 0.1}
    fa = np.array([means_a[t] for t in tiers])[:, None] + rng.normal(size=(3 * n, 5))
    fb = np.array([means_b[t] for t in tiers])[:, None] + rng.normal(size=(3 * n, 4))

    result = permutation_null_centroid(
        fa, tiers, fb, tiers, ("Low", "Mid", "High"),
        n_perms=50, seed=1,
    )
    assert result["n_permutations"] == 50
    assert 0.0 < result["p_value"] <= 1.0
    assert np.isfinite(result["observed"])
    assert result["null"].shape == (50,)


def test_entropic_gw_runs_on_small_matrices():
    rng = np.random.default_rng(0)
    C1 = np.abs(rng.normal(size=(10, 10)))
    C1 = 0.5 * (C1 + C1.T); np.fill_diagonal(C1, 0)
    C2 = np.abs(rng.normal(size=(10, 10)))
    C2 = 0.5 * (C2 + C2.T); np.fill_diagonal(C2, 0)
    out = entropic_gromov_wasserstein(C1, C2, epsilon=0.05)
    assert out.coupling.shape == (10, 10)
    assert np.isfinite(out.distance)
