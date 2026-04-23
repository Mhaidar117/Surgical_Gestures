"""Tertile binning of a continuous skill score into {Low, Mid, High}.

Using tertiles (np.quantile at 1/3 and 2/3) rather than equal-width bins
keeps the three tiers equally populated, which GW with uniform marginals
requires.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


TIER_NAMES = ("Low", "Mid", "High")


@dataclass
class TertileCutoffs:
    q33: float
    q66: float

    def as_dict(self) -> dict:
        return {"q33": float(self.q33), "q66": float(self.q66)}


def compute_tertiles(values: Sequence[float]) -> TertileCutoffs:
    v = np.asarray(values, dtype=np.float64)
    if v.size < 3:
        raise ValueError("need at least 3 values to compute tertile cutoffs")
    return TertileCutoffs(q33=float(np.quantile(v, 1.0 / 3.0)),
                          q66=float(np.quantile(v, 2.0 / 3.0)))


def assign_tier(values: Sequence[float], cutoffs: TertileCutoffs) -> np.ndarray:
    """Return an ndarray of tier labels from TIER_NAMES."""
    v = np.asarray(values, dtype=np.float64)
    tier = np.full(v.shape, TIER_NAMES[1], dtype=object)  # default Mid
    tier[v <= cutoffs.q33] = TIER_NAMES[0]
    tier[v >  cutoffs.q66] = TIER_NAMES[2]
    return tier


def add_tier_column(df: pd.DataFrame,
                    score_col: str,
                    tier_col: str = "tier") -> tuple[pd.DataFrame, TertileCutoffs]:
    """Add a tier column to a DataFrame using tertile cutoffs of `score_col`."""
    cutoffs = compute_tertiles(df[score_col].to_numpy(dtype=np.float64))
    out = df.copy()
    out[tier_col] = assign_tier(df[score_col].to_numpy(dtype=np.float64), cutoffs)
    return out, cutoffs


def assign_fixed_tier(values: Sequence[float],
                      low_threshold: float,
                      high_threshold: float) -> np.ndarray:
    """Assign Low/Mid/High using *hard* cutoffs instead of tertiles.

    Low: `v < low_threshold`.
    Mid: `low_threshold <= v <= high_threshold`.
    High: `v > high_threshold`.
    """
    if low_threshold > high_threshold:
        raise ValueError("low_threshold must be <= high_threshold")
    v = np.asarray(values, dtype=np.float64)
    tier = np.full(v.shape, TIER_NAMES[1], dtype=object)
    tier[v < low_threshold] = TIER_NAMES[0]
    tier[v > high_threshold] = TIER_NAMES[2]
    return tier
