"""Centroid and pairwise RDM construction.

For the headline analysis we average residualized feature vectors within
each skill tier to get three centroids, then build a 3x3 cosine-distance
RDM. For the trial-level supporting analysis we keep all samples and
compute the full N x N cosine-distance matrix.

Cosine distance is chosen because residualized features are already
z-scored per column, so magnitudes across features are comparable and
angle-based dissimilarity captures the remaining skill-relevant direction.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return 1.0 - float(np.dot(a, b) / (na * nb))


def centroid_rdm(features: np.ndarray,
                 tiers: Sequence[str],
                 tier_order: Sequence[str]) -> np.ndarray:
    """Return a (K,K) cosine-distance RDM over K tier centroids."""
    tiers = np.asarray(tiers)
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError("features must be 2-D (n_samples, n_features)")

    centroids = []
    for t in tier_order:
        mask = tiers == t
        if mask.sum() == 0:
            centroids.append(np.zeros(feats.shape[1]))
        else:
            centroids.append(feats[mask].mean(axis=0))
    C = np.stack(centroids, axis=0)
    K = C.shape[0]
    rdm = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i + 1, K):
            d = _safe_cosine(C[i], C[j])
            rdm[i, j] = d; rdm[j, i] = d
    return rdm


def pairwise_cosine_rdm(features: np.ndarray) -> np.ndarray:
    """Full N x N cosine-distance matrix. Rows with near-zero norm are left
    as a zero row (cosine distance is ill-defined in that case)."""
    feats = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(feats, axis=1)
    mask = norms < 1e-12
    if mask.any():
        feats = feats.copy()
        feats[mask] = 1e-12   # avoid divide-by-zero inside pdist
    d = pdist(feats, metric="cosine")
    return squareform(d).astype(np.float64)


def is_valid_rdm(rdm: np.ndarray, atol: float = 1e-8) -> bool:
    """Check symmetry, zero diagonal, and finite entries."""
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        return False
    if not np.isfinite(rdm).all():
        return False
    if not np.allclose(rdm, rdm.T, atol=atol):
        return False
    if not np.allclose(np.diag(rdm), 0.0, atol=atol):
        return False
    return True
