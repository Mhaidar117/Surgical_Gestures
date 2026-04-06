"""HMM + rolling features → eye_state_summary, transitions, pupil, events."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover
    GaussianHMM = None  # type: ignore


def _rolling_features(
    gx: np.ndarray,
    gy: np.ndarray,
    pupil: np.ndarray,
    movement: np.ndarray,
    window: int,
    stride: int,
) -> Tuple[np.ndarray, int]:
    """
    Per window: fixation ratio, gaze dispersion (std x,y), pupil std.
    Returns X of shape (n_windows, 3) and stride used.
    """
    n = len(gx)
    feats: List[np.ndarray] = []
    i = 0
    while i + window <= n:
        sl = slice(i, i + window)
        m = movement[sl]
        fix = float(np.mean(m == 1))
        disp = float(np.sqrt(np.var(gx[sl]) + np.var(gy[sl])))
        pstd = float(np.std(pupil[sl]))
        feats.append(np.array([fix, disp, pstd], dtype=np.float64))
        i += stride
    if not feats:
        return np.zeros((0, 3), dtype=np.float64), stride
    return np.stack(feats, axis=0), stride


def _fit_hmm(
    X: np.ndarray,
    n_states: int,
    seed: int = 0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (state_sequence, occupancy (n_states,), transition_empirical (n_states, n_states))
    """
    if GaussianHMM is None or X.shape[0] < max(20, n_states * 4):
        return None
    try:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=seed,
        )
        model.fit(X)
        st = model.predict(X)
    except Exception:
        return None
    occ = np.bincount(st, minlength=n_states).astype(np.float64)
    occ = occ / (occ.sum() + 1e-12)
    T = np.zeros((n_states, n_states), dtype=np.float64)
    for a, b in zip(st[:-1], st[1:]):
        T[a, b] += 1.0
    row = T.sum(axis=1, keepdims=True) + 1e-12
    T = T / row
    return st, occ, T


def _quantile_states(X: np.ndarray, n_states: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin first HMM feature by rank into n_states pseudo-states."""
    n = X.shape[0]
    ranks = np.argsort(np.argsort(X[:, 0]))
    st = (ranks * n_states // n).astype(np.int64)
    st = np.clip(st, 0, n_states - 1)
    occ = np.bincount(st, minlength=n_states).astype(np.float64)
    occ = occ / (occ.sum() + 1e-12)
    T = np.zeros((n_states, n_states), dtype=np.float64)
    for a, b in zip(st[:-1], st[1:]):
        T[a, b] += 1.0
    row = T.sum(axis=1, keepdims=True) + 1e-12
    T = T / row
    return st, occ, T


def _movement_only_fallback(movement: np.ndarray, n_states: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Last resort: map movement labels 1/2/other into min(n_states,3) then pad occupancy."""
    lab = np.where(movement == 1, 0, np.where(movement == 2, 1, 2))
    # replicate to n_states by splitting each class in time blocks
    n = len(lab)
    st = np.zeros(n, dtype=np.int64)
    for i in range(n):
        st[i] = (lab[i] + (i % max(1, n_states // 3))) % n_states
    occ = np.bincount(st, minlength=n_states).astype(np.float64)
    occ = occ / (occ.sum() + 1e-12)
    T = np.zeros((n_states, n_states), dtype=np.float64)
    for a, b in zip(st[:-1], st[1:]):
        T[a, b] += 1.0
    row = T.sum(axis=1, keepdims=True) + 1e-12
    T = T / row
    return st, occ, T


def summarize_eye_trial(
    gx: np.ndarray,
    gy: np.ndarray,
    pupil: np.ndarray,
    movement: np.ndarray,
    window_samples: int,
    window_stride: int,
    hmm_states: int,
    seed: int = 0,
) -> Dict[str, Any]:
    # Smooth inputs (Gaussian sigma ~2 samples)
    gx_s = gaussian_filter1d(gx, sigma=2.0, mode="nearest")
    gy_s = gaussian_filter1d(gy, sigma=2.0, mode="nearest")
    pu_s = gaussian_filter1d(pupil, sigma=2.0, mode="nearest")

    X, _ = _rolling_features(gx_s, gy_s, pu_s, movement, window_samples, window_stride)
    hmm_out = _fit_hmm(X, hmm_states, seed=seed) if X.shape[0] > 0 else None
    if hmm_out is not None:
        state_seq, occ, T = hmm_out
        method = "hmm"
    elif X.shape[0] >= hmm_states:
        state_seq, occ, T = _quantile_states(X, hmm_states)
        method = "quantile_windows"
    else:
        state_seq, occ, T = _movement_only_fallback(movement, hmm_states)
        method = "movement_fallback"

    dwell = []
    if len(state_seq) > 0:
        run = 1
        for i in range(1, len(state_seq)):
            if state_seq[i] == state_seq[i - 1]:
                run += 1
            else:
                dwell.append(run)
                run = 1
        dwell.append(run)
    mean_dwell = float(np.mean(dwell)) if dwell else 0.0

    eye_state_summary: Dict[str, Any] = {
        "method": method,
        "n_states": hmm_states,
        "occupancy": occ.tolist(),
        "mean_dwell_samples": mean_dwell,
    }
    eye_transition_summary: Dict[str, Any] = {
        "transition_matrix": T.tolist(),
        "vectorized_upper_tri": _upper_tri_vec(T),
    }

    blink_frac = float(np.mean(pupil <= 0)) if len(pupil) else 0.0
    pupil_summary = {
        "mean": float(np.mean(pu_s)),
        "std": float(np.std(pu_s)),
        "cv": float(np.std(pu_s) / (np.mean(np.abs(pu_s)) + 1e-8)),
        "blink_fraction": blink_frac,
    }

    # Events on full valid series (already filtered)
    n = len(movement)
    duration_s = n / 50.0  # assume 50 Hz
    n_fix = int(np.sum(movement == 1))
    n_sac = int(np.sum(movement == 2))
    # Saccade amplitude proxy: mean displacement on saccade-labeled rows
    if n_sac > 0:
        sac_mask = movement == 2
        dx = np.diff(np.r_[gx_s[0], gx_s])
        dy = np.diff(np.r_[gy_s[0], gy_s])
        amp = float(np.mean(np.sqrt(dx[sac_mask] ** 2 + dy[sac_mask] ** 2)))
    else:
        amp = 0.0
    # Fixation duration: crude run-length on fixation rows
    fix_durs = []
    run = 0
    for m in movement:
        if m == 1:
            run += 1
        else:
            if run > 0:
                fix_durs.append(run)
            run = 0
    if run > 0:
        fix_durs.append(run)
    mean_fix_dur = float(np.mean(fix_durs)) / 50.0 if fix_durs else 0.0  # seconds

    event_summary = {
        "n_fixations": n_fix,
        "n_saccades": n_sac,
        "fixation_rate_hz": n_fix / (duration_s + 1e-12),
        "saccade_rate_hz": n_sac / (duration_s + 1e-12),
        "mean_fixation_duration_s": mean_fix_dur,
        "mean_saccade_amplitude_proxy": amp,
    }

    return {
        "eye_state_summary": eye_state_summary,
        "eye_transition_summary": eye_transition_summary,
        "pupil_summary": pupil_summary,
        "event_summary": event_summary,
        "aux": {
            "hmm_feature_shape": list(X.shape),
            "state_sequence_len": int(len(state_seq)),
        },
    }


def _upper_tri_vec(T: np.ndarray) -> List[float]:
    m = T.shape[0]
    out = []
    for i in range(m):
        for j in range(i + 1, m):
            out.append(float(T[i, j]))
    return out


def fingerprint_vector_from_summary(bundle: Dict[str, Any]) -> np.ndarray:
    """Single vector for RDM / correlation: concat occupancy + upper tri + pupil + events."""
    occ = np.asarray(bundle["eye_state_summary"]["occupancy"], dtype=np.float64)
    tri = np.asarray(bundle["eye_transition_summary"]["vectorized_upper_tri"], dtype=np.float64)
    pup = bundle["pupil_summary"]
    ev = bundle["event_summary"]
    parts = [
        occ,
        tri,
        np.array(
            [pup["mean"], pup["std"], pup["cv"], pup["blink_fraction"]],
            dtype=np.float64,
        ),
        np.array(
            [
                ev["fixation_rate_hz"],
                ev["saccade_rate_hz"],
                ev["mean_saccade_amplitude_proxy"],
            ],
            dtype=np.float64,
        ),
    ]
    return np.concatenate(parts, axis=0)
