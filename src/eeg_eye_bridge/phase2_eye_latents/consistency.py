"""Compare Phase 1 EEG summaries to eye-derived summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from .eye_summarize import fingerprint_vector_from_summary
from .phase1_io import PHASE1_REP_KEYS, extract_prediction_error_trajectory, extract_representation_vector


def _pad_align(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(a.size, b.size)
    return a.reshape(-1)[:n], b.reshape(-1)[:n]


def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    xa, ya = _pad_align(x, y)
    if xa.size < 2:
        return float("nan")
    r, _ = spearmanr(xa, ya)
    if r is None or np.isnan(r):
        return float("nan")
    return float(r)


def occupancy_alignment_score(
    eeg_vec: np.ndarray, eye_occupancy: np.ndarray
) -> float:
    """Spearman between same-length slices of EEG summary and HMM occupancy."""
    ev = eeg_vec.reshape(-1).astype(np.float64)
    occ = eye_occupancy.reshape(-1).astype(np.float64)
    # pad to common length
    m = max(ev.size, occ.size)
    ev2 = np.zeros(m)
    occ2 = np.zeros(m)
    ev2[: ev.size] = ev
    occ2[: occ.size] = occ
    return spearman_safe(ev2, occ2)


def pe_saccade_alignment(
    pe_traj: Optional[np.ndarray],
    movement_full: np.ndarray,
) -> float:
    """
    Resample |PE| to eye length; correlate with saccade indicator (movement==2).
    """
    if pe_traj is None or len(pe_traj) < 2:
        return float("nan")
    sac = (movement_full.astype(np.int64) == 2).astype(np.float64)
    n = len(sac)
    pe = np.abs(np.asarray(pe_traj, dtype=np.float64).reshape(-1))
    pe_rs = np.interp(
        np.linspace(0.0, 1.0, n),
        np.linspace(0.0, 1.0, len(pe)),
        pe,
    )
    return spearman_safe(pe_rs, sac)


def pairwise_rdm(vectors: List[np.ndarray]) -> np.ndarray:
    """Euclidean distance RDM (n, n)."""
    m = len(vectors)
    d = max(1, max(v.size for v in vectors))
    X = np.zeros((m, d), dtype=np.float64)
    for i, v in enumerate(vectors):
        vv = v.reshape(-1)
        X[i, : vv.size] = vv
    out = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            out[i, j] = float(np.linalg.norm(X[i] - X[j]))
    return out


def rdm_upper_tri_vec(R: np.ndarray) -> np.ndarray:
    m = R.shape[0]
    vals = []
    for i in range(m):
        for j in range(i + 1, m):
            vals.append(R[i, j])
    return np.asarray(vals, dtype=np.float64)


def family_rdm_correlation(
    trial_ids: List[str],
    family_by_trial: Mapping[str, str],
    eye_fps: Mapping[str, np.ndarray],
    eeg_fps: Mapping[str, np.ndarray],
    family: str,
) -> float:
    """Spearman between upper-triangles of eye vs EEG RDM within one family (>=3 trials)."""
    tids = [t for t in trial_ids if family_by_trial.get(t) == family]
    if len(tids) < 3:
        return float("nan")
    ev = [eye_fps[t] for t in tids]
    ee = [eeg_fps[t] for t in tids]
    Re = pairwise_rdm(ev)
    Rb = pairwise_rdm(ee)
    return spearman_safe(rdm_upper_tri_vec(Re), rdm_upper_tri_vec(Rb))


def score_trial(
    trial: Mapping[str, Any],
    eye_bundle: Mapping[str, Any],
    movement_full: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Per-trial metrics for each representation.
    Keys: occupancy_spearman, pe_saccade_spearman, fingerprint_spearman (eye vs eeg fp).
    """
    occ = np.asarray(eye_bundle["eye_state_summary"]["occupancy"], dtype=np.float64)
    eye_fp = fingerprint_vector_from_summary(eye_bundle)
    pe_traj = extract_prediction_error_trajectory(trial)
    out: Dict[str, Dict[str, float]] = {}
    for rep in PHASE1_REP_KEYS:
        try:
            vec = extract_representation_vector(trial, rep)
        except KeyError:
            out[rep] = {
                "occupancy_spearman": float("nan"),
                "pe_saccade_spearman": float("nan"),
                "fingerprint_spearman": float("nan"),
            }
            continue
        out[rep] = {
            "occupancy_spearman": occupancy_alignment_score(vec, occ),
            "pe_saccade_spearman": pe_saccade_alignment(pe_traj, movement_full),
            "fingerprint_spearman": spearman_safe(vec.reshape(-1), eye_fp),
        }
    return out


def aggregate_rankings(
    scores_by_trial: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Any]:
    """Mean score per rep per metric; pick best rep per metric."""
    metrics = ["occupancy_spearman", "pe_saccade_spearman", "fingerprint_spearman"]
    sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for tid, repmap in scores_by_trial.items():
        for rep, mets in repmap.items():
            for m in metrics:
                v = mets.get(m, float("nan"))
                if v is None or np.isnan(v):
                    continue
                sums[rep][m] += float(v)
                counts[rep][m] += 1
    means: Dict[str, Dict[str, float]] = {}
    for rep in PHASE1_REP_KEYS:
        means[rep] = {}
        for m in metrics:
            c = counts[rep][m]
            means[rep][m] = sums[rep][m] / c if c else float("nan")
    best_per_metric = {}
    for m in metrics:
        best = None
        best_v = -1e18
        for rep in PHASE1_REP_KEYS:
            v = means[rep].get(m, float("nan"))
            if np.isnan(v):
                continue
            if v > best_v:
                best_v = v
                best = rep
        best_per_metric[m] = best
    # Combined: mean rank across metrics
    rank_score = {}
    for rep in PHASE1_REP_KEYS:
        vals = [means[rep].get(m, float("nan")) for m in metrics]
        vals = [v for v in vals if not np.isnan(v)]
        rank_score[rep] = float(np.mean(vals)) if vals else float("nan")
    valid = {r: v for r, v in rank_score.items() if not np.isnan(v)}
    best_combined = max(valid, key=lambda r: valid[r]) if valid else None
    return {
        "mean_by_rep": means,
        "best_per_metric": best_per_metric,
        "mean_combined_score_by_rep": rank_score,
        "best_combined": best_combined,
    }
