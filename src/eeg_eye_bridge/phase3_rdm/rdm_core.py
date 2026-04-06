"""Pairwise dissimilarity matrices (RDMs) from feature rows."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr


def zscore_rows(X: npt.NDArray[np.floating], eps: float = 1e-8) -> npt.NDArray[np.floating]:
    x = np.asarray(X, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True) + eps
    return (x - mean) / std


def zscore_cols(X: npt.NDArray[np.floating], eps: float = 1e-8) -> npt.NDArray[np.floating]:
    x = np.asarray(X, dtype=np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + eps
    return (x - mean) / std


def rdm_one_minus_spearman(X: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """
    RDM where entry (i,j) = 1 - Spearman rho between row i and row j (paired features).
    Diagonal 0; symmetric.
    """
    x = np.asarray(X, dtype=np.float64)
    n = x.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i, j] = 0.0
                continue
            rij, _ = spearmanr(x[i], x[j])
            if rij is None or np.isnan(rij):
                out[i, j] = 1.0
            else:
                out[i, j] = float(1.0 - rij)
    return out


def rdm_euclidean(X: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    x = np.asarray(X, dtype=np.float64)
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=-1) + 1e-12)
    np.fill_diagonal(d, 0.0)
    return d


def rdm_cosine_one_minus(X: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    x = np.asarray(X, dtype=np.float64)
    n = x.shape[0]
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = x[i], x[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na <= 0 or nb <= 0:
                out[i, j] = 1.0
            else:
                cos = float(np.dot(a, b) / (na * nb))
                cos = max(-1.0, min(1.0, cos))
                out[i, j] = 1.0 - cos
    return out


def compute_rdm(
    X: npt.NDArray[np.floating],
    metric: str = "one_minus_spearman",
) -> npt.NDArray[np.float64]:
    m = metric.lower()
    if m in ("one_minus_spearman", "spearman", "1-spearman"):
        return rdm_one_minus_spearman(X)
    if m in ("euclidean", "l2"):
        return rdm_euclidean(X)
    if m in ("cosine", "one_minus_cosine"):
        return rdm_cosine_one_minus(X)
    raise ValueError(f"unknown_metric:{metric}")
