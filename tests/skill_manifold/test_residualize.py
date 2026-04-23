"""Smoke tests for skill_manifold.residualize."""
from __future__ import annotations

import numpy as np
import pandas as pd

from skill_manifold.residualize import residualize


def test_residualize_removes_category_mean():
    # Build a DataFrame whose feature is 5 + group mean + small noise.
    rng = np.random.default_rng(0)
    groups = np.repeat(["a", "b", "c"], 30)
    noise = rng.normal(size=90, scale=0.1)
    feature = np.array([{"a": 10, "b": 30, "c": 70}[g] for g in groups]) + noise
    df = pd.DataFrame({"group": groups, "f": feature, "age": rng.normal(size=90)})

    res = residualize(df, feature_cols=["f"],
                      categorical=["group"], ordinal=["age"])
    assert "f" in res.residuals.columns
    # Most of f is explained by group, so R^2 should be high.
    assert res.r2_per_feature["f"] > 0.95
    # After residualization, refitting the nuisance model on the residual
    # should give ~0 R^2 (within numerical noise).
    assert res.post_fit_r2["f"] < 0.02
    # Residuals are re-z-scored.
    v = res.residuals["f"].to_numpy()
    assert abs(v.mean()) < 1e-8
    assert abs(v.std() - 1.0) < 1e-6


def test_residualize_preserves_non_feature_columns():
    df = pd.DataFrame({
        "g": ["a", "a", "b", "b"],
        "x": [1.0, 2.0, 3.0, 4.0],
        "extra": ["keep", "me", "as", "is"],
    })
    res = residualize(df, feature_cols=["x"],
                      categorical=["g"], ordinal=())
    assert list(res.residuals["extra"]) == ["keep", "me", "as", "is"]
    assert list(res.residuals["g"]) == ["a", "a", "b", "b"]


def test_residualize_with_only_ordinal():
    rng = np.random.default_rng(1)
    age = np.linspace(20, 40, 50)
    f = 2.0 * age + rng.normal(size=50, scale=0.5)
    df = pd.DataFrame({"age": age, "f": f})
    res = residualize(df, feature_cols=["f"], ordinal=["age"])
    assert res.r2_per_feature["f"] > 0.95
    assert res.post_fit_r2["f"] < 0.02
