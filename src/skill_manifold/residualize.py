"""Per-dataset OLS nuisance-residualization utilities.

For each feature column we fit `feature ~ nuisances` with ordinary least
squares (sklearn.linear_model.LinearRegression) and keep the residual. The
goal is to remove structure attributable to task, subject, and other
covariates so that what remains -- the residualized features -- reflects
intrinsic "skill geometry" rather than task-mean drift.

After residualization we re-z-score each feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ResidualizationResult:
    residuals: pd.DataFrame       # same rows/cols as the input features, z-scored residuals
    r2_per_feature: pd.Series     # nuisance-model R^2 on the ORIGINAL feature
    post_fit_r2: pd.Series        # nuisance-model R^2 on the RESIDUAL (should be ~0)


def _build_design_matrix(df: pd.DataFrame,
                         categorical: Sequence[str],
                         ordinal: Sequence[str]) -> np.ndarray:
    """One-hot-encode `categorical` columns and z-score `ordinal` columns
    (then horizontally stack). An all-ones intercept column is appended."""
    blocks: List[np.ndarray] = []
    if categorical:
        try:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        cat_mat = enc.fit_transform(df[list(categorical)].astype(str).to_numpy())
        blocks.append(cat_mat)
    for col in ordinal:
        v = df[col].to_numpy(dtype=np.float64)
        std = v.std()
        z = (v - v.mean()) / (std if std > 1e-12 else 1.0)
        blocks.append(z.reshape(-1, 1))
    intercept = np.ones((len(df), 1), dtype=np.float64)
    blocks.append(intercept)
    return np.hstack(blocks) if blocks else intercept


def residualize(df: pd.DataFrame,
                feature_cols: Sequence[str],
                *,
                categorical: Sequence[str] = (),
                ordinal: Sequence[str] = ()) -> ResidualizationResult:
    """Residualize `feature_cols` against the given nuisance columns.

    Returns a `ResidualizationResult` whose `residuals` frame replaces the
    feature columns with re-z-scored residuals; non-feature columns are
    preserved as-is.
    """
    X = _build_design_matrix(df, categorical, ordinal)
    out = df.copy()
    r2s: dict[str, float] = {}
    post_r2s: dict[str, float] = {}

    for col in feature_cols:
        y = df[col].to_numpy(dtype=np.float64)
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        yhat = model.predict(X)
        resid = y - yhat
        # R^2 of nuisance model on original feature.
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2s[col] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # Re-z-score the residual.
        rstd = resid.std()
        if rstd > 1e-12:
            resid_z = (resid - resid.mean()) / rstd
        else:
            resid_z = np.zeros_like(resid)
        out[col] = resid_z

        # R^2 of nuisance model refit against the residual (should be ~0).
        m2 = LinearRegression(fit_intercept=False)
        m2.fit(X, resid_z)
        yhat2 = m2.predict(X)
        ss_res2 = float(((resid_z - yhat2) ** 2).sum())
        ss_tot2 = float(((resid_z - resid_z.mean()) ** 2).sum())
        post_r2s[col] = 1.0 - ss_res2 / ss_tot2 if ss_tot2 > 1e-12 else 0.0

    return ResidualizationResult(
        residuals=out,
        r2_per_feature=pd.Series(r2s, name="r2_nuisance_on_feature"),
        post_fit_r2=pd.Series(post_r2s, name="r2_nuisance_on_residual"),
    )
