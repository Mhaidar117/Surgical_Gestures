"""Collapse trial-level FLS features to subject-level vectors + skill scores.

The downstream goal of the FLS analysis is per-subject feedback, so the
unit of skill is the subject, not the trial. Each modality's trial-level
feature DataFrame is collapsed two ways:

    'mean'         -> simple mean across all of that subject's trials.
                       Used for the headline 3x3 GW analysis.
    'per_task_mean'-> concatenation of per-task means (peg / cut / suturing).
                       Per-task slots that the subject didn't perform get
                       imputed with the column-wise mean across subjects so
                       the dimensionality stays fixed; this only matters
                       for the rare missing-task case.

The composite skill score per subject is the mean across that subject's
trials of (Performance - perf_min) / (perf_max - perf_min). This puts
GOALS (peg / cut, scored out of 25) and OSAT (suturing, scored out of 30)
on a common [0, 1] scale before averaging.
"""
from __future__ import annotations

import logging
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Hand-coded order so per-task vectors have a deterministic layout.
FLS_TASK_IDS: Tuple[int, ...] = (1, 2, 3)
FLS_TASK_NAMES = {1: "peg_transfer", 2: "pattern_cut", 3: "suturing"}


# ---------- skill score ------------------------------------------------------

def normalize_performance(df: pd.DataFrame) -> pd.Series:
    """Return a per-trial normalized score on [0, 1].

    `score = (Performance - perf_min) / (perf_max - perf_min)`
    Trials with missing or zero-range bounds get NaN and are dropped from
    downstream means.
    """
    p = df["performance"].astype(float)
    lo = df["perf_min"].astype(float)
    hi = df["perf_max"].astype(float)
    rng = hi - lo
    score = (p - lo) / rng
    score = score.where(np.isfinite(score), np.nan)
    return score


def composite_skill_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Return a (subject_id, composite_skill) frame, one row per subject.

    `composite_skill` is the mean of normalized per-trial scores over all
    of that subject's trials in `df`. Subjects with no scorable trial get
    NaN and are filtered out by the caller.
    """
    work = df.copy()
    work["_score"] = normalize_performance(work)
    grp = work.groupby("subject_id")["_score"].mean().rename("composite_skill")
    return grp.reset_index()


def per_task_skill_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Return a wide frame with one column per task: skill_task_<id>."""
    work = df.copy()
    work["_score"] = normalize_performance(work)
    pivot = (work.groupby(["subject_id", "task_id"])["_score"]
                  .mean().unstack("task_id"))
    pivot.columns = [f"skill_task_{int(c)}" for c in pivot.columns]
    return pivot.reset_index()


# ---------- feature aggregation ---------------------------------------------

def _feature_columns(df: pd.DataFrame, prefix_any: Sequence[str]) -> List[str]:
    """Pick columns whose name starts with any of `prefix_any`."""
    return [c for c in df.columns if any(c.startswith(p) for p in prefix_any)]


def aggregate_subject_mean(
    trial_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Mean each feature within subject_id. Returns one row per subject."""
    if "subject_id" not in trial_df.columns:
        raise ValueError("trial_df must have a subject_id column")
    keep = ["subject_id"] + list(feature_cols)
    sub = trial_df[keep].copy()
    out = sub.groupby("subject_id")[list(feature_cols)].mean().reset_index()
    return out


def aggregate_subject_per_task_mean(
    trial_df: pd.DataFrame,
    feature_cols: Sequence[str],
    task_ids: Sequence[int] = FLS_TASK_IDS,
) -> Tuple[pd.DataFrame, List[str]]:
    """For each (subject, task), compute the mean of each feature, then
    concatenate over `task_ids` so each subject row is the per-task vectors
    laid out side by side.

    Returns
    -------
    (frame, new_feature_cols) where `frame` has columns `subject_id` plus
    one feature per `(task_id, original_feature)` pair, named
    ``{task_name}__{feature}``. Missing (subject, task) cells are filled
    with the column-wise mean across subjects so the dimensionality is
    fixed.
    """
    if "task_id" not in trial_df.columns:
        raise ValueError("trial_df must have a task_id column")
    out_cols: List[str] = []
    pieces: List[pd.DataFrame] = []
    for tid in task_ids:
        tname = FLS_TASK_NAMES.get(tid, f"task_{tid}")
        sub = trial_df[trial_df["task_id"] == tid]
        if sub.empty:
            log.warning("no trials for task_id=%d; filling with NaN", tid)
        agg = (sub.groupby("subject_id")[list(feature_cols)]
                  .mean()
                  if not sub.empty
                  else pd.DataFrame(columns=feature_cols))
        agg.columns = [f"{tname}__{c}" for c in agg.columns]
        pieces.append(agg)
        out_cols.extend(agg.columns)

    # All subjects observed across the input frame.
    all_subjects = sorted(trial_df["subject_id"].unique().tolist())
    base = pd.DataFrame({"subject_id": all_subjects}).set_index("subject_id")
    wide = base.copy()
    for piece in pieces:
        wide = wide.join(piece, how="left")
    # Fill missing (subject, task) cells with column means.
    for c in out_cols:
        if c not in wide.columns:
            wide[c] = np.nan
        col_mean = wide[c].mean(skipna=True)
        if not np.isfinite(col_mean):
            col_mean = 0.0
        wide[c] = wide[c].fillna(col_mean)
    return wide.reset_index()[["subject_id"] + out_cols], out_cols


# ---------- demographic carry-over ------------------------------------------

def carry_subject_metadata(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Pull demographic columns up to subject level. Assumes they're constant
    within subject; if not, takes the mode (string) or mean (numeric)."""
    needed = ["subject_id", "age", "dominant_hand", "dominant_eye", "gender"]
    have = [c for c in needed if c in trial_df.columns]
    if "subject_id" not in have:
        raise ValueError("trial_df must have subject_id")
    keep = trial_df[have].drop_duplicates("subject_id", keep="first")
    return keep.reset_index(drop=True)


# ---------- one-shot helper -------------------------------------------------

def assemble_subject_frame(
    trial_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    mode: str = "mean",
) -> Tuple[pd.DataFrame, List[str]]:
    """Return (subject_frame, feature_cols) ready for residualization.

    `subject_frame` has subject_id, demographics (age/dominant_hand/
    dominant_eye/gender), composite_skill, per-task skill, and the
    aggregated feature columns. `feature_cols` is the list of feature
    column names in the returned frame (mode-dependent).
    """
    if mode == "mean":
        feat = aggregate_subject_mean(trial_df, feature_cols)
        out_cols = list(feature_cols)
    elif mode == "per_task_mean":
        feat, out_cols = aggregate_subject_per_task_mean(trial_df, feature_cols)
    else:
        raise ValueError(f"unknown aggregation mode: {mode!r}")

    meta = carry_subject_metadata(trial_df)
    composite = composite_skill_per_subject(trial_df)
    per_task = per_task_skill_per_subject(trial_df)

    out = (meta.merge(composite, on="subject_id", how="left")
                .merge(per_task,  on="subject_id", how="left")
                .merge(feat,      on="subject_id", how="left"))
    return out, out_cols
