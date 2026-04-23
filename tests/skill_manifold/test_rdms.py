"""Smoke tests for skill_manifold.rdms."""
from __future__ import annotations

import numpy as np

from skill_manifold.rdms import centroid_rdm, is_valid_rdm, pairwise_cosine_rdm


def test_centroid_rdm_shape_and_properties():
    rng = np.random.default_rng(42)
    feats = rng.normal(size=(30, 8))
    tiers = np.array(["Low"] * 10 + ["Mid"] * 10 + ["High"] * 10)
    R = centroid_rdm(feats, tiers, ("Low", "Mid", "High"))
    assert R.shape == (3, 3)
    assert is_valid_rdm(R)


def test_pairwise_cosine_rdm_symmetric_zero_diagonal():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(25, 12))
    R = pairwise_cosine_rdm(feats)
    assert R.shape == (25, 25)
    assert is_valid_rdm(R)


def test_is_valid_rdm_rejects_asymmetric():
    R = np.array([[0.0, 0.1], [0.2, 0.0]])
    assert not is_valid_rdm(R)


def test_centroid_rdm_handles_empty_tier():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(10, 4))
    tiers = np.array(["Low"] * 5 + ["Mid"] * 5)  # "High" is empty
    R = centroid_rdm(feats, tiers, ("Low", "Mid", "High"))
    assert R.shape == (3, 3)
    assert is_valid_rdm(R)
