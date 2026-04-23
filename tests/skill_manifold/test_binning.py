"""Smoke tests for skill_manifold.binning."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skill_manifold.binning import TIER_NAMES, add_tier_column, assign_tier, compute_tertiles


def test_tertiles_partition_into_three_roughly_equal_bins():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    cut = compute_tertiles(x)
    tiers = assign_tier(x, cut)
    counts = {t: int((tiers == t).sum()) for t in TIER_NAMES}
    assert sum(counts.values()) == 300
    for t in TIER_NAMES:
        assert 80 <= counts[t] <= 120, f"{t} has {counts[t]} items"


def test_add_tier_column_attaches_valid_labels():
    df = pd.DataFrame({"score": np.arange(60)})
    out, cut = add_tier_column(df, "score")
    assert "tier" in out.columns
    assert set(out["tier"]) == set(TIER_NAMES)
    assert cut.q33 < cut.q66
