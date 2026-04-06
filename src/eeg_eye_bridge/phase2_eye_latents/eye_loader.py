"""Load and clean simulator eye CSVs (Eye/EYE/{trial_id}.csv)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class EyeSeries:
    """Aligned 1D arrays after cleaning (noise rows removed)."""

    gaze_x: np.ndarray
    gaze_y: np.ndarray
    pupil: np.ndarray  # mean of L/R where valid
    movement_type: np.ndarray  # int
    sample_rate_hz: float


def _interp_zeros(a: np.ndarray) -> np.ndarray:
    """Linear interpolate zeros (blinks) in 1D array."""
    x = a.astype(np.float64).copy()
    bad = x <= 0
    if not np.any(bad):
        return x
    good = ~bad
    if np.sum(good) < 2:
        return np.where(bad, np.nanmedian(x[good]) if np.any(good) else 0.0, x)
    idx = np.arange(len(x))
    x[bad] = np.interp(idx[bad], idx[good], x[good])
    return x


def load_eye_csv(
    path: Path,
    assumed_rate_hz: float = 50.0,
) -> EyeSeries:
    """
    Columns: 0-1 gaze, 17-18 pupil L/R, 19 movement type.
    Drops rows where movement_type in (0, 3) as noise (Exploration_Prompt).
    """
    raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 20:
        raise ValueError(f"{path}: expected >=20 columns, got {raw.shape[1]}")

    gx = raw[:, 0].astype(np.float64)
    gy = raw[:, 1].astype(np.float64)
    pl = raw[:, 17].astype(np.float64)
    pr = raw[:, 18].astype(np.float64)
    mt = raw[:, 19].astype(np.int64)

    pupil = 0.5 * (pl + pr)
    pupil = _interp_zeros(pupil)

    # Remove noise rows for primary statistics; keep copy for event counts on full series if needed
    valid = np.isfinite(gx) & np.isfinite(gy) & (mt != 0) & (mt != 3)
    if not np.any(valid):
        raise ValueError(f"{path}: no valid rows after filtering")

    return EyeSeries(
        gaze_x=gx[valid],
        gaze_y=gy[valid],
        pupil=pupil[valid],
        movement_type=mt[valid],
        sample_rate_hz=float(assumed_rate_hz),
    )


def load_eye_csv_full_for_events(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-series gaze (gx, gy) and movement_type for event stats (all rows)."""
    raw = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    gx = raw[:, 0].astype(np.float64)
    gy = raw[:, 1].astype(np.float64)
    mt = raw[:, 19].astype(np.int64)
    return gx, gy, mt
