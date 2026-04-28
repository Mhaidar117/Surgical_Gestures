#!/usr/bin/env python3
"""FLS cross-modality skill-manifold orchestrator.

Within-dataset, between-modality GW analysis on the NIBIB-RPCCC-FLS data.
Asks: do EEG and gaze, recorded simultaneously from the same 25 FLS
subjects, organize skill into compatible geometries?

Pipeline stages
---------------
    1. Build trial-level feature frames for both modalities.
       (`features_fls_gaze.build_fls_gaze_feature_frame`,
        `features_fls_eeg.build_fls_eeg_feature_frame`).
    2. Residualize each modality at the trial level against demographics
       (age, dominant_hand, dominant_eye, gender). More samples = more
       robust OLS than residualizing post-aggregation.
    3. Aggregate to subject level (mean across that subject's trials)
       and compute composite + per-task skill scores.
    4. Tertile subjects on the composite score -> Low/Mid/High.
    5. Headline GW: 3x3 centroid RDMs per modality, plain GW + tier-shuffle null.
    6. Companion GW: 25x25 cosine RDMs per modality, entropic GW +
       block-mass diagonality null with subject-label shuffles on one side.
    7. Sensitivity: per-task 3x3 GW (one analysis per FLS task).
    8. Write report markdown + JSON + plots to reports/skill_manifold_fls/.

Usage
-----
    python pipeline/skill_manifold_gw_fls.py
    python pipeline/skill_manifold_gw_fls.py --n_perms 200
    python pipeline/skill_manifold_gw_fls.py --skip-companion
    python pipeline/skill_manifold_gw_fls.py --features-cache <path>

The orchestrator is independent from `skill_manifold_gw.py`; the existing
JIGSAWS<->RAS pipeline is left untouched.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from skill_manifold.binning import (  # noqa: E402
    TIER_NAMES, add_tier_column,
)
from skill_manifold.features_fls_gaze import (  # noqa: E402
    build_fls_gaze_feature_frame, eye_feature_column_names,
    fls_data_root, fls_performance_csv, load_fls_performance_scores,
)
from skill_manifold.features_fls_eeg import (  # noqa: E402
    build_fls_eeg_feature_frame, eeg_feature_column_names,
)
from skill_manifold.gw import (  # noqa: E402
    entropic_gromov_wasserstein, gromov_wasserstein_centroids,
    permutation_null_centroid,
)
from skill_manifold.rdms import (  # noqa: E402
    centroid_rdm, is_valid_rdm, pairwise_cosine_rdm,
)
from skill_manifold.residualize import residualize  # noqa: E402
from skill_manifold.subject_aggregation import (  # noqa: E402
    FLS_TASK_IDS, FLS_TASK_NAMES, assemble_subject_frame,
    composite_skill_per_subject, normalize_performance,
)

log = logging.getLogger("skill_manifold_gw_fls")

REPORT_SUBDIR = "skill_manifold_fls"
DEFAULT_N_PERMS = 1000
DEFAULT_GW_EPSILON = 0.01
DEMOG_CATEGORICAL = ("dominant_hand", "dominant_eye", "gender")
DEMOG_ORDINAL = ("age",)
# task_id is added to the design when --residualize_task is on (default).
# Removing task-mean variance isolates within-task between-skill variance,
# which is the contrast the GW test is supposed to read.
TASK_CATEGORICAL = ("task_id",)


# ---------- logging / argparse ----------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_root", type=Path, default=REPO,
                   help="Repo root (default: this script's parent directory).")
    p.add_argument("--n_perms", type=int, default=DEFAULT_N_PERMS,
                   help="Permutations for both null tests.")
    p.add_argument("--gw_epsilon", type=float, default=DEFAULT_GW_EPSILON,
                   help="Entropic-GW regularization for the 25x25 companion.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Override the output directory (default: reports/skill_manifold_fls).")
    p.add_argument("--features_cache", type=Path, default=None,
                   help="If set, cache + reuse trial-level feature frames as parquet here.")
    p.add_argument("--skip_eeg", action="store_true",
                   help="Skip EEG feature extraction (e.g. for fast smoke tests).")
    p.add_argument("--skip_companion", action="store_true",
                   help="Skip the 25x25 entropic GW companion analysis.")
    p.add_argument("--skip_per_task", action="store_true",
                   help="Skip the per-task sensitivity analyses.")
    p.add_argument("--skip_eeg_zscore", action="store_true",
                   help="Skip per-subject EEG z-scoring (use raw "
                        "residualized features). Default behaviour z-scores "
                        "to remove residual per-subject global-gain.")
    p.add_argument("--no_residualize_task", action="store_true",
                   help="Skip residualizing features against task_id. "
                        "Default residualizes against demographics + task so "
                        "the GW test reads within-task between-skill variance.")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_level", default="INFO")
    return p.parse_args(argv)


# ---------- I/O helpers ------------------------------------------------------

def _ensure_output_dir(repo_root: Path, override: Optional[Path]) -> Tuple[Path, Path]:
    out = override if override is not None else repo_root / "reports" / REPORT_SUBDIR
    out = Path(out)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    return out, plots


def _load_or_build_feature_frame(
    *,
    repo_root: Path,
    cache_path: Optional[Path],
    cache_name: str,
    builder,
    builder_kwargs: dict,
) -> pd.DataFrame:
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        f = cache_path / f"{cache_name}.parquet"
        if f.exists():
            log.info("loading cached %s from %s", cache_name, f)
            return pd.read_parquet(f)
        df = builder(repo_root=repo_root, **builder_kwargs)
        try:
            df.to_parquet(f, index=False)
            log.info("cached %s -> %s", cache_name, f)
        except Exception as e:
            log.warning("failed to cache %s (%s); continuing without cache",
                        cache_name, e)
        return df
    return builder(repo_root=repo_root, **builder_kwargs)


# ---------- analysis steps ---------------------------------------------------

def residualize_trial_features(
    trial_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    residualize_task: bool = True,
) -> pd.DataFrame:
    """Trial-level OLS residualization against demographics (and task).

    With `residualize_task=True` (default), `task_id` is added to the
    one-hot design matrix alongside demographics so that task-mean
    variance is removed from each feature. The cross-modal alignment
    test then reads within-task between-skill variance only.

    Returns a copy of `trial_df` with `feature_cols` replaced by their
    z-scored residuals. Non-feature columns are preserved.
    """
    cat = list(DEMOG_CATEGORICAL)
    if residualize_task:
        cat.extend(TASK_CATEGORICAL)
    cat = [c for c in cat if c in trial_df.columns]
    ordin = [c for c in DEMOG_ORDINAL if c in trial_df.columns]
    if not cat and not ordin:
        log.warning("no nuisance columns found; skipping residualization")
        return trial_df.copy()
    res = residualize(trial_df, feature_cols=list(feature_cols),
                      categorical=cat, ordinal=ordin)
    return res.residuals


def drop_degenerate_subjects(
    subj_df: pd.DataFrame,
    feature_cols_gaze: Sequence[str],
    feature_cols_eeg: Sequence[str],
    *,
    norm_threshold: float = 1e-6,
) -> Tuple[pd.DataFrame, List[int]]:
    """Drop subjects whose feature vector is near-zero in either modality.

    A subject can end up with an exact zero feature vector if their raw
    features are constant across all of their trials -- the OLS step in
    `residualize_trial_features` predicts that subject's values
    perfectly from demographics, so the residual is zero. Constant
    features upstream usually mean a missing-data fallback or processing
    failure rather than biology, and at N=25 even one degenerate row
    pulls every tier centroid toward the origin.

    Returns the filtered DataFrame and the list of dropped subject IDs.
    """
    Xg = subj_df[list(feature_cols_gaze)].to_numpy(dtype=np.float64)
    Xe = subj_df[list(feature_cols_eeg)].to_numpy(dtype=np.float64)
    norm_g = np.linalg.norm(Xg, axis=1)
    norm_e = np.linalg.norm(Xe, axis=1)
    keep = (norm_g > norm_threshold) & (norm_e > norm_threshold)
    dropped = subj_df.loc[~keep, "subject_id"].astype(int).tolist()
    return subj_df.loc[keep].reset_index(drop=True), dropped


def per_subject_zscore_features(
    subj_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Per-subject z-score across feature dimensions.

    For each subject (row), subtract the mean across `feature_cols` and
    divide by the std across `feature_cols`. After this transformation
    every subject's feature vector has mean 0 and unit std along the
    feature axis, so cosine distance between centroids reflects relative
    *shape* across regions/bands rather than absolute level.

    This is the post-hoc form of adding a per-subject random intercept to
    the residualization regression. Per the postmortem, demographic
    residualization left ~83% of inter-subject EEG variance on a single
    direction (a per-subject global-gain factor that survived relative
    bandpower); this transformation removes that direction.

    Subjects whose row has zero std (e.g. all-equal features) get a
    zero row out, which the downstream degeneracy filter will catch.
    """
    out = subj_df.copy()
    cols = list(feature_cols)
    X = out[cols].to_numpy(dtype=np.float64)
    row_means = X.mean(axis=1, keepdims=True)
    row_stds = X.std(axis=1, keepdims=True)
    safe = np.where(row_stds > eps, row_stds, 1.0)
    X_z = np.where(row_stds > eps, (X - row_means) / safe, 0.0)
    out[cols] = X_z
    return out


def headline_3x3(
    subj_df: pd.DataFrame,
    feature_cols_gaze: Sequence[str],
    feature_cols_eeg: Sequence[str],
    *,
    n_perms: int,
    seed: int,
) -> Dict[str, object]:
    """3x3 tier-centroid GW between gaze and EEG, composite-skill tiers."""
    if "tier" not in subj_df.columns:
        raise ValueError("subj_df must already have a 'tier' column")
    tiers = subj_df["tier"].to_numpy()
    Xg = subj_df[list(feature_cols_gaze)].to_numpy(dtype=np.float64)
    Xe = subj_df[list(feature_cols_eeg)].to_numpy(dtype=np.float64)

    rdm_g = centroid_rdm(Xg, tiers, TIER_NAMES)
    rdm_e = centroid_rdm(Xe, tiers, TIER_NAMES)
    if not (is_valid_rdm(rdm_g) and is_valid_rdm(rdm_e)):
        raise RuntimeError("invalid centroid RDM (NaN/asymmetric); abort")

    gw = gromov_wasserstein_centroids(rdm_g, rdm_e, TIER_NAMES)

    null = permutation_null_centroid(
        Xg, tiers, Xe, tiers, TIER_NAMES,
        n_perms=n_perms, seed=seed,
    )
    return {
        "rdm_gaze": rdm_g,
        "rdm_eeg": rdm_e,
        "distance": gw.distance,
        "coupling": gw.coupling,
        "argmax_assignment": gw.argmax_assignment,
        "null": null,
    }


def companion_25x25(
    subj_df: pd.DataFrame,
    feature_cols_gaze: Sequence[str],
    feature_cols_eeg: Sequence[str],
    *,
    epsilon: float,
    n_perms: int,
    seed: int,
    sinkhorn_max_iter: int = 5000,
) -> Dict[str, object]:
    """Subject-level entropic GW with diagonal-mass null on subject labels."""
    Xg = subj_df[list(feature_cols_gaze)].to_numpy(dtype=np.float64)
    Xe = subj_df[list(feature_cols_eeg)].to_numpy(dtype=np.float64)
    subj = subj_df["subject_id"].to_numpy()

    Cg = pairwise_cosine_rdm(Xg)
    Ce = pairwise_cosine_rdm(Xe)
    gw = entropic_gromov_wasserstein(Cg, Ce, epsilon=epsilon,
                                     max_iter=sinkhorn_max_iter,
                                     sinkhorn_max_iter=sinkhorn_max_iter)
    T = np.asarray(gw.coupling, dtype=np.float64)

    # Diagonal mass = sum_i T[i, i] when row order matches col order. Both
    # Xg and Xe are aligned by subject_id, so subject i has the same row
    # index on both sides; trace(T) is the "subject-recovery" statistic.
    observed = float(np.trace(T))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perms, dtype=np.float64)
    n = T.shape[0]
    for k in range(n_perms):
        # Shuffle column subject order on the gaze side. Using POT's
        # coupling directly: T[i, j] is the mass moving subject i (EEG)
        # to subject j (gaze). A column-permutation shuffles which gaze
        # subject sits where on the column axis; the trace then measures
        # alignment under the shuffled labels.
        perm = rng.permutation(n)
        null[k] = float(np.trace(T[:, perm]))

    mu = float(null.mean())
    sd = float(null.std(ddof=1)) if n_perms > 1 else float("nan")
    p = (1.0 + float((null >= observed).sum())) / (1.0 + n_perms)
    z = (observed - mu) / sd if sd > 1e-12 else float("nan")

    # Argmax row->col assignment for diagnostic.
    argmax_match = (T.argmax(axis=1) == np.arange(n)).mean()

    return {
        "rdm_gaze": Cg,
        "rdm_eeg": Ce,
        "distance": gw.distance,
        "coupling": T,
        "subject_ids": subj,
        "diag_mass_observed": observed,
        "diag_mass_null": null,
        "diag_mass_null_mean": mu,
        "diag_mass_null_std": sd,
        "diag_mass_p_value": float(p),
        "diag_mass_z_score": float(z),
        "argmax_match_rate": float(argmax_match),
    }


def per_task_3x3(
    trial_df_resid_gaze: pd.DataFrame,
    trial_df_resid_eeg: pd.DataFrame,
    feature_cols_gaze: Sequence[str],
    feature_cols_eeg: Sequence[str],
    *,
    n_perms: int,
    seed: int,
    excluded_subjects: Optional[Sequence[int]] = None,
    zscore_eeg: bool = True,
) -> Dict[int, Dict[str, object]]:
    """Run the headline 3x3 analysis once per FLS task with per-task skill tiers.

    `excluded_subjects` filters out the same subjects the headline drops
    via `drop_degenerate_subjects()`, so per-task and headline analyses
    operate on a consistent subject set. `zscore_eeg=True` mirrors the
    headline's per-subject EEG z-score on the per-task subject means.
    """
    if excluded_subjects:
        ex = set(int(s) for s in excluded_subjects)
        trial_df_resid_gaze = trial_df_resid_gaze[
            ~trial_df_resid_gaze["subject_id"].isin(ex)
        ]
        trial_df_resid_eeg = trial_df_resid_eeg[
            ~trial_df_resid_eeg["subject_id"].isin(ex)
        ]
    out: Dict[int, Dict[str, object]] = {}
    for tid in FLS_TASK_IDS:
        sub_g = trial_df_resid_gaze[trial_df_resid_gaze["task_id"] == tid].copy()
        sub_e = trial_df_resid_eeg[trial_df_resid_eeg["task_id"] == tid].copy()
        if sub_g.empty or sub_e.empty:
            log.warning("task_id=%d has empty modality frame; skipping", tid)
            continue

        # Per-task subject feature mean. Skill = mean normalized score on
        # this task's trials per subject.
        gaze_subj = sub_g.groupby("subject_id")[list(feature_cols_gaze)].mean().reset_index()
        eeg_subj  = sub_e.groupby("subject_id")[list(feature_cols_eeg )].mean().reset_index()
        # Apply the same per-subject EEG z-score as the headline so per-task
        # cosine geometry is comparable to the headline cosine geometry.
        if zscore_eeg:
            eeg_subj = per_subject_zscore_features(eeg_subj, feature_cols_eeg)
        # Only subjects present in BOTH modalities for this task can be tiered.
        merged = gaze_subj.merge(eeg_subj, on="subject_id", how="inner")
        if len(merged) < 6:  # need >=2 per tier
            log.warning("task_id=%d has only %d subjects in both modalities; skipping",
                        tid, len(merged))
            continue

        # Skill = composite of normalized score on this task only.
        sk = composite_skill_per_subject(sub_g)
        sk = sk.merge(composite_skill_per_subject(sub_e), on="subject_id",
                      how="inner", suffixes=("_g", "_e"))
        sk["composite_skill"] = sk[["composite_skill_g", "composite_skill_e"]].mean(axis=1)
        sk = sk[["subject_id", "composite_skill"]]
        merged = merged.merge(sk, on="subject_id", how="inner").dropna(subset=["composite_skill"])
        if len(merged) < 6:
            log.warning("task_id=%d has only %d subjects with skill scores; skipping",
                        tid, len(merged))
            continue

        merged, _ = add_tier_column(merged, score_col="composite_skill")
        result = headline_3x3(
            merged,
            feature_cols_gaze=feature_cols_gaze,
            feature_cols_eeg=feature_cols_eeg,
            n_perms=n_perms, seed=seed + tid,
        )
        result["task_id"] = tid
        result["task_name"] = FLS_TASK_NAMES.get(tid, f"task_{tid}")
        result["n_subjects"] = int(len(merged))
        out[tid] = result
    return out


# ---------- plots ------------------------------------------------------------

def _heatmap(M: np.ndarray, title: str, labels: Sequence[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = M.shape[0]
    fig, ax = plt.subplots(figsize=(max(3.0, K * 0.4 + 1), max(2.8, K * 0.4 + 1)))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(K), labels, rotation=90 if K > 6 else 0, fontsize=7 if K > 8 else 9)
    ax.set_yticks(range(K), labels, fontsize=7 if K > 8 else 9)
    if K <= 8:
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < M.max() * 0.6 else "black",
                        fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _null_hist(null: np.ndarray, observed: float, title: str, path: Path,
               xlabel: str = "statistic") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=40, color="#777", edgecolor="white")
    ax.axvline(observed, color="red", linestyle="--",
               label=f"observed = {observed:.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _mds_plot(feats: np.ndarray, tiers: np.ndarray, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS

    n = feats.shape[0]
    if n < 3:
        return
    mds = MDS(n_components=2, dissimilarity="euclidean", random_state=0,
              normalized_stress="auto")
    xy = mds.fit_transform(feats)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    palette = {"Low": "#1f77b4", "Mid": "#ff7f0e", "High": "#2ca02c"}
    for t in TIER_NAMES:
        mask = tiers == t
        if mask.any():
            ax.scatter(xy[mask, 0], xy[mask, 1], s=42, alpha=0.85,
                       label=t, color=palette.get(t))
    ax.set_title(title)
    ax.set_xlabel("MDS 1"); ax.set_ylabel("MDS 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------- diagnostic plots ------------------------------------------------

TIER_PALETTE = {"Low": "#1f77b4", "Mid": "#ff7f0e", "High": "#2ca02c"}


def _pca_subject_plot(feats: np.ndarray, tiers: np.ndarray,
                      subject_ids: np.ndarray, title: str, path: Path) -> None:
    """2-component PCA scatter, colored by tier, every point labelled by subject_id.

    With N=25 the labels fit comfortably and let us see whether one tier's
    centroid is being pulled by 1-2 outlier subjects vs. a coherent group.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    n = feats.shape[0]
    if n < 3:
        return
    Xc = feats - feats.mean(axis=0, keepdims=True)
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(Xc)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(6, 5))
    for t in TIER_NAMES:
        mask = tiers == t
        if mask.any():
            ax.scatter(xy[mask, 0], xy[mask, 1], s=80, alpha=0.85,
                       label=t, color=TIER_PALETTE.get(t),
                       edgecolors="white", linewidths=1.0)
    for i, sid in enumerate(subject_ids):
        ax.annotate(str(int(sid)), (xy[i, 0], xy[i, 1]),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({100*var[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100*var[1]:.1f}% var)")
    ax.axhline(0, color="#bbb", lw=0.5); ax.axvline(0, color="#bbb", lw=0.5)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _subject_distance_to_centroid_heatmap(
    feats: np.ndarray, tiers: np.ndarray, subject_ids: np.ndarray,
    title: str, path: Path,
) -> None:
    """Heatmap of (per-subject distance to each tier centroid).

    Rows = subjects sorted by their own tier (Low, Mid, High) then by
    subject_id; cols = [Low_dist, Mid_dist, High_dist]. Each subject's
    distance to *its own* tier centroid is computed against the leave-one-
    out centroid (i.e. excluding that subject's row from the mean), so a
    subject can't appear artificially close to its own tier just by
    contributing to its own centroid.

    Reading the plot:
      - If Mid-as-outlier is coherent: every Mid row has high Low and High
        distances (the middle column is uniformly bright across the
        Mid block).
      - If Mid-as-outlier is driven by 1-2 subjects: a couple of Mid rows
        will look anomalous and the rest will look "normal" -- some Mid
        subjects close to Low, others close to High.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = feats.shape[0]
    if n < 3:
        return
    tiers = np.asarray(tiers)
    subject_ids = np.asarray(subject_ids)

    # Build full-population centroids first, plus leave-one-out variants
    # for the diagonal (own-tier) cell.
    full_centroids = {}
    for t in TIER_NAMES:
        mask = tiers == t
        if mask.any():
            full_centroids[t] = feats[mask].mean(axis=0)

    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return 1.0 - float(np.dot(a, b) / (na * nb))

    rows = []
    for i in range(n):
        my_tier = tiers[i]
        row = []
        for t in TIER_NAMES:
            mask = (tiers == t)
            if t == my_tier:
                # Leave-one-out centroid for own tier.
                others = mask.copy()
                others[i] = False
                if others.sum() == 0:
                    row.append(0.0)
                    continue
                centroid = feats[others].mean(axis=0)
            else:
                if not mask.any():
                    row.append(0.0)
                    continue
                centroid = full_centroids[t]
            row.append(_cosine_distance(feats[i], centroid))
        rows.append(row)
    M = np.asarray(rows, dtype=np.float64)

    # Sort rows by (tier_order, subject_id) for readability.
    tier_rank = np.array([TIER_NAMES.index(t) for t in tiers])
    sort_idx = np.lexsort((subject_ids, tier_rank))
    M_sorted = M[sort_idx]
    tiers_sorted = tiers[sort_idx]
    sids_sorted = subject_ids[sort_idx]

    fig, ax = plt.subplots(figsize=(5.5, max(4.5, 0.22 * n + 1.5)))
    im = ax.imshow(M_sorted, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(TIER_NAMES)), [f"d({t} centroid)" for t in TIER_NAMES])
    ax.set_yticks(range(n), [f"{int(s)} ({t})" for s, t in zip(sids_sorted, tiers_sorted)],
                  fontsize=8)

    # Underline tier blocks.
    prev = None
    for k, t in enumerate(tiers_sorted):
        if t != prev:
            ax.axhline(k - 0.5, color="white", lw=1.0)
            prev = t

    # Annotate cells.
    for i in range(n):
        for j in range(len(TIER_NAMES)):
            color = "white" if M_sorted[i, j] > M_sorted.max() * 0.6 else "black"
            ax.text(j, i, f"{M_sorted[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=7)

    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="cosine distance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def drop_one_subject_rdm_sensitivity(
    feats: np.ndarray,
    tiers: np.ndarray,
    subject_ids: np.ndarray,
) -> Dict[str, object]:
    """For each subject, recompute the centroid RDM with that subject excluded.

    Returns a dict with:
      - `subject_ids`: list of subject IDs in plot/dropout order.
      - `cells`: dict mapping each off-diagonal cell name (e.g. 'Low_High')
        to a list of N float values, one per leave-one-out.
      - `baseline_cells`: dict mapping cell names to the full-population
        cell value for visual reference.

    This is the diagnostic for the postmortem's U-shape question on the
    EEG side: if d(Low, High) stays small across most leave-one-outs, the
    "Low and High share a regional pattern" geometry is robust; if it
    jumps when any single subject drops, it's outlier-driven.
    """
    feats = np.asarray(feats, dtype=np.float64)
    tiers = np.asarray(tiers)
    subject_ids = np.asarray(subject_ids)
    n = feats.shape[0]
    if n < 4:
        return {"subject_ids": subject_ids.tolist(),
                "cells": {}, "baseline_cells": {}}

    cell_names = [(0, 1, "Low_Mid"), (0, 2, "Low_High"), (1, 2, "Mid_High")]

    # Baseline: full-population centroid RDM.
    base_rdm = centroid_rdm(feats, tiers, TIER_NAMES)
    baseline_cells = {name: float(base_rdm[i, j]) for i, j, name in cell_names}

    # Leave-one-out RDMs.
    cells: Dict[str, List[float]] = {n: [] for _, _, n in cell_names}
    for k in range(n):
        keep = np.ones(n, dtype=bool); keep[k] = False
        rdm = centroid_rdm(feats[keep], tiers[keep], TIER_NAMES)
        for i, j, name in cell_names:
            cells[name].append(float(rdm[i, j]))

    return {
        "subject_ids": [int(s) for s in subject_ids],
        "tiers": [str(t) for t in tiers],
        "cells": cells,
        "baseline_cells": baseline_cells,
    }


def _drop_one_out_plot(
    sensitivity: Dict[str, object],
    title: str,
    path: Path,
) -> None:
    """Strip+box plot of the three off-diagonal RDM cells across leave-one-out.

    X-axis: cell name. Y-axis: cosine distance. Each leave-one-out
    contributes one point per cell. The full-population value is drawn
    as a horizontal dashed line per cell.
    """
    cells = sensitivity.get("cells", {})
    baseline = sensitivity.get("baseline_cells", {})
    if not cells:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell_order = ["Low_Mid", "Low_High", "Mid_High"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    palette = ["#4878D0", "#D65F5F", "#6ACC64"]

    box_data = [cells[c] for c in cell_order]
    bp = ax.boxplot(box_data, widths=0.45, patch_artist=True,
                    medianprops={"color": "black"})
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color); patch.set_alpha(0.35)

    rng = np.random.default_rng(0)
    for k, name in enumerate(cell_order):
        ys = cells[name]
        xs = (k + 1) + (rng.random(len(ys)) - 0.5) * 0.30
        ax.scatter(xs, ys, s=24, color=palette[k], alpha=0.85,
                   edgecolors="white", linewidths=0.6)
        # Baseline line for this cell.
        if name in baseline:
            ax.hlines(baseline[name], k + 0.6, k + 1.4,
                      color=palette[k], linestyles="--", linewidths=1.2)

    ax.set_xticks(range(1, len(cell_order) + 1),
                  [c.replace("_", "↔") for c in cell_order])
    ax.set_ylabel("cosine distance between tier centroids")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(bottom=-0.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def diagnostic_plots(subj_df: pd.DataFrame,
                     feature_cols_gaze: Sequence[str],
                     feature_cols_eeg: Sequence[str],
                     plots_dir: Path) -> Dict[str, object]:
    """Write per-subject PCA scatters, tier-centroid distance heatmaps,
    and drop-one-subject RDM sensitivity plots.

    Returns the sensitivity-analysis dicts (per modality) so callers can
    serialize them into the JSON results.
    """
    if "tier" not in subj_df.columns:
        raise ValueError("subj_df must already have a 'tier' column")
    tiers = subj_df["tier"].to_numpy()
    sids = subj_df["subject_id"].to_numpy()
    Xg = subj_df[list(feature_cols_gaze)].to_numpy(dtype=np.float64)
    Xe = subj_df[list(feature_cols_eeg)].to_numpy(dtype=np.float64)

    _pca_subject_plot(Xe, tiers, sids,
                      "EEG subject features — PCA, labelled by subject_id",
                      plots_dir / "diagnostic_pca_eeg.png")
    _pca_subject_plot(Xg, tiers, sids,
                      "Gaze subject features — PCA, labelled by subject_id",
                      plots_dir / "diagnostic_pca_gaze.png")
    _subject_distance_to_centroid_heatmap(
        Xe, tiers, sids,
        "EEG: per-subject cosine distance to each tier centroid (own = leave-one-out)",
        plots_dir / "diagnostic_dist_eeg.png",
    )
    _subject_distance_to_centroid_heatmap(
        Xg, tiers, sids,
        "Gaze: per-subject cosine distance to each tier centroid (own = leave-one-out)",
        plots_dir / "diagnostic_dist_gaze.png",
    )

    # Drop-one-subject sensitivity for each modality.
    eeg_sens = drop_one_subject_rdm_sensitivity(Xe, tiers, sids)
    gaze_sens = drop_one_subject_rdm_sensitivity(Xg, tiers, sids)
    _drop_one_out_plot(
        eeg_sens,
        "EEG centroid RDM — drop-one-subject sensitivity",
        plots_dir / "diagnostic_dropone_eeg.png",
    )
    _drop_one_out_plot(
        gaze_sens,
        "Gaze centroid RDM — drop-one-subject sensitivity",
        plots_dir / "diagnostic_dropone_gaze.png",
    )
    return {"eeg_drop_one_out": eeg_sens, "gaze_drop_one_out": gaze_sens}


# ---------- main entrypoint --------------------------------------------------

def run(args: argparse.Namespace) -> Dict[str, object]:
    out_dir, plots_dir = _ensure_output_dir(args.repo_root, args.output_dir)
    log.info("outputs -> %s", out_dir)

    # Step 1: trial-level features.
    log.info("step 1: building gaze trial features ...")
    gaze_trial = _load_or_build_feature_frame(
        repo_root=args.repo_root, cache_path=args.features_cache,
        cache_name="fls_gaze_trial",
        builder=build_fls_gaze_feature_frame,
        builder_kwargs={},
    )
    log.info("  -> %d trials, %d feature cols",
             len(gaze_trial), len(eye_feature_column_names()))

    if args.skip_eeg:
        log.warning("--skip_eeg: substituting random EEG features for smoke testing")
        n = len(gaze_trial)
        rng = np.random.default_rng(0)
        rand = rng.standard_normal((n, len(eeg_feature_column_names())))
        eeg_trial = gaze_trial[
            ["trial_id", "subject_id", "task_id", "try_num",
             "age", "dominant_hand", "dominant_eye", "gender",
             "performance", "perf_min", "perf_max"]
        ].copy()
        for k, c in enumerate(eeg_feature_column_names()):
            eeg_trial[c] = rand[:, k]
    else:
        log.info("step 1b: building EEG trial features ...")
        eeg_trial = _load_or_build_feature_frame(
            repo_root=args.repo_root, cache_path=args.features_cache,
            cache_name="fls_eeg_trial",
            builder=build_fls_eeg_feature_frame,
            builder_kwargs={},
        )
        log.info("  -> %d trials, %d feature cols",
                 len(eeg_trial), len(eeg_feature_column_names()))

    feat_cols_gaze = eye_feature_column_names()
    feat_cols_eeg = eeg_feature_column_names()

    # Step 2: residualize each modality at trial level against demographics
    # (and task_id by default).
    residualize_task = not args.no_residualize_task
    log.info("step 2: residualizing trial features against demographics%s ...",
             " + task_id" if residualize_task else " only")
    gaze_resid = residualize_trial_features(
        gaze_trial, feat_cols_gaze, residualize_task=residualize_task)
    eeg_resid = residualize_trial_features(
        eeg_trial, feat_cols_eeg, residualize_task=residualize_task)

    # Step 3: aggregate to subject level (mean) + composite skill.
    log.info("step 3: aggregating to subject level ...")
    gaze_subj, _ = assemble_subject_frame(gaze_resid, feat_cols_gaze, mode="mean")
    eeg_subj, _ = assemble_subject_frame(eeg_resid, feat_cols_eeg, mode="mean")

    # Per-subject z-score the EEG features along the feature axis. The
    # postmortem run @11:45 showed PC1 = 83% of inter-subject variance
    # even after relative bandpower; that residual is a per-subject
    # global-gain offset that cosine geometry would otherwise read as
    # the dominant signal. Subtracting subject mean and dividing by
    # subject std along the 40 feature columns removes it. Gaze is left
    # alone (PC1 = 57% is a healthy spread for behavioural data).
    if not args.skip_eeg_zscore:
        log.info("step 3b: per-subject z-scoring EEG features ...")
        eeg_subj = per_subject_zscore_features(eeg_subj, feat_cols_eeg)

    # Inner-join on subject_id so both modalities have the same subject set
    # and the joint RDMs are aligned row-for-row.
    join_keys = ["subject_id", "composite_skill",
                 "age", "dominant_hand", "dominant_eye", "gender"]
    subj_df = (gaze_subj
               .merge(eeg_subj[["subject_id"] + feat_cols_eeg],
                      on="subject_id", how="inner")
               .dropna(subset=["composite_skill"])
               .sort_values("subject_id")
               .reset_index(drop=True))
    log.info("  -> %d subjects after inner-join + composite-skill filter",
             len(subj_df))

    # Drop subjects whose residualized feature vector is near-zero in
    # either modality. These are almost always upstream-pipeline failures
    # rather than biological zeros, and a single such subject biases the
    # tier centroids substantially at N=25.
    subj_df, dropped_subjects = drop_degenerate_subjects(
        subj_df,
        feature_cols_gaze=feat_cols_gaze,
        feature_cols_eeg=feat_cols_eeg,
    )
    if dropped_subjects:
        log.warning("dropped %d degenerate subject(s) (zero feature norm "
                    "in EEG or gaze): %s",
                    len(dropped_subjects), dropped_subjects)
    log.info("  -> %d subjects in joint analysis after degeneracy filter",
             len(subj_df))

    # Step 4: tertile bin.
    subj_df, cutoffs = add_tier_column(subj_df, score_col="composite_skill")
    log.info("  tertile cutoffs: q33=%.4f, q66=%.4f", cutoffs.q33, cutoffs.q66)
    tier_counts = subj_df["tier"].value_counts().to_dict()
    log.info("  tier counts: %s", tier_counts)

    # Diagnostic plots: per-subject PCA scatters + tier-centroid distance
    # heatmaps + drop-one-subject RDM sensitivity. These are independent
    # of the GW analysis and let us see whether any tier centroid is
    # being pulled by a few outlier subjects.
    log.info("writing diagnostic plots (PCA + per-subject distances + drop-one-out) ...")
    diagnostics = diagnostic_plots(subj_df,
                                   feature_cols_gaze=feat_cols_gaze,
                                   feature_cols_eeg=feat_cols_eeg,
                                   plots_dir=plots_dir)

    # Step 5: headline 3x3 GW.
    log.info("step 5: headline 3x3 GW + permutation null (n_perms=%d) ...",
             args.n_perms)
    headline = headline_3x3(
        subj_df,
        feature_cols_gaze=feat_cols_gaze,
        feature_cols_eeg=feat_cols_eeg,
        n_perms=args.n_perms, seed=args.seed,
    )

    _heatmap(headline["rdm_gaze"], "Gaze tier-centroid RDM",
             list(TIER_NAMES), plots_dir / "rdm_centroid_gaze.png")
    _heatmap(headline["rdm_eeg"], "EEG tier-centroid RDM",
             list(TIER_NAMES), plots_dir / "rdm_centroid_eeg.png")
    _heatmap(headline["coupling"], "Headline GW coupling (gaze -> EEG)",
             list(TIER_NAMES), plots_dir / "coupling_headline.png")
    _null_hist(headline["null"]["null"], headline["null"]["observed"],
               title="Headline GW: tier-shuffle null",
               path=plots_dir / "null_headline.png",
               xlabel="GW distance")

    Xg_subj = subj_df[feat_cols_gaze].to_numpy()
    Xe_subj = subj_df[feat_cols_eeg].to_numpy()
    _mds_plot(Xg_subj, subj_df["tier"].to_numpy(),
              "MDS - gaze (subject means)", plots_dir / "mds_gaze.png")
    _mds_plot(Xe_subj, subj_df["tier"].to_numpy(),
              "MDS - EEG (subject means)", plots_dir / "mds_eeg.png")

    # Step 6: subject-level companion (entropic GW + diagonal-mass null).
    companion: Optional[Dict[str, object]] = None
    if not args.skip_companion and len(subj_df) >= 6:
        log.info("step 6: %dx%d entropic GW + subject-shuffle null (eps=%.4f) ...",
                 len(subj_df), len(subj_df), args.gw_epsilon)
        companion = companion_25x25(
            subj_df,
            feature_cols_gaze=feat_cols_gaze,
            feature_cols_eeg=feat_cols_eeg,
            epsilon=args.gw_epsilon,
            n_perms=args.n_perms, seed=args.seed,
        )
        K = companion["coupling"].shape[0]
        labels = [str(s) for s in companion["subject_ids"]]
        _heatmap(companion["coupling"],
                 f"{K}x{K} entropic GW coupling (rows=EEG, cols=gaze)",
                 labels, plots_dir / "coupling_companion.png")
        _null_hist(companion["diag_mass_null"],
                   companion["diag_mass_observed"],
                   title=f"Companion {K}x{K}: subject-shuffle null on trace(T)",
                   path=plots_dir / "null_companion.png",
                   xlabel="trace(T) under subject-label shuffle")

    # Step 7: per-task sensitivity. Pass dropped_subjects through so per-task
    # operates on the same N as the headline (the previous run reported
    # per_task[*].n_subjects=25 while the headline was N=24, which made
    # the two analyses inconsistent and partly explained the per-task
    # GW blowups noted in the postmortem).
    per_task: Dict[int, Dict[str, object]] = {}
    if not args.skip_per_task:
        log.info("step 7: per-task 3x3 sensitivities (excluding %d degenerate subject(s)) ...",
                 len(dropped_subjects))
        per_task = per_task_3x3(
            gaze_resid, eeg_resid,
            feature_cols_gaze=feat_cols_gaze,
            feature_cols_eeg=feat_cols_eeg,
            n_perms=args.n_perms, seed=args.seed,
            excluded_subjects=dropped_subjects,
            zscore_eeg=not args.skip_eeg_zscore,
        )
        for tid, res in per_task.items():
            tname = res["task_name"]
            _heatmap(res["coupling"],
                     f"GW coupling - task={tname}",
                     list(TIER_NAMES),
                     plots_dir / f"coupling_task_{tname}.png")
            _null_hist(res["null"]["null"], res["null"]["observed"],
                       title=f"Per-task null - {tname}",
                       path=plots_dir / f"null_task_{tname}.png",
                       xlabel="GW distance")

    # Step 8: write report.
    summary = _summarize(subj_df, headline, companion, per_task,
                         tier_counts=tier_counts, cutoffs=cutoffs,
                         dropped_subjects=dropped_subjects,
                         diagnostics=diagnostics,
                         residualize_task=residualize_task,
                         eeg_zscore=not args.skip_eeg_zscore)
    json_path = out_dir / "results_fls.json"
    md_path = out_dir / "report_fls.md"
    json_path.write_text(json.dumps(_jsonable(summary), indent=2))
    md_path.write_text(_render_markdown(summary, plots_dir))
    log.info("wrote %s", json_path)
    log.info("wrote %s", md_path)
    return summary


# ---------- summary + markdown ----------------------------------------------

def _summarize(
    subj_df: pd.DataFrame,
    headline: Dict[str, object],
    companion: Optional[Dict[str, object]],
    per_task: Dict[int, Dict[str, object]],
    *,
    tier_counts: Dict[str, int],
    cutoffs,
    dropped_subjects: Optional[List[int]] = None,
    diagnostics: Optional[Dict[str, object]] = None,
    residualize_task: bool = True,
    eeg_zscore: bool = True,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "n_subjects": int(len(subj_df)),
        "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "tertile_cutoffs": cutoffs.as_dict(),
        "dropped_subjects": list(dropped_subjects or []),
        "config": {
            "residualize_task": bool(residualize_task),
            "eeg_zscore": bool(eeg_zscore),
        },
        "diagnostics": diagnostics or {},
        "headline": {
            "gw_distance": float(headline["distance"]),
            "argmax_assignment": dict(headline["argmax_assignment"]),
            "p_value": float(headline["null"]["p_value"]),
            "z_score": float(headline["null"]["z_score"]),
            "n_permutations": int(headline["null"]["n_permutations"]),
            "rdm_gaze": np.asarray(headline["rdm_gaze"]).tolist(),
            "rdm_eeg": np.asarray(headline["rdm_eeg"]).tolist(),
            "coupling": np.asarray(headline["coupling"]).tolist(),
        },
        "companion": None,
        "per_task": {},
    }
    if companion is not None:
        out["companion"] = {
            "n_subjects": int(companion["coupling"].shape[0]),
            "gw_distance": float(companion["distance"]),
            "diag_mass_observed": float(companion["diag_mass_observed"]),
            "diag_mass_p_value": float(companion["diag_mass_p_value"]),
            "diag_mass_z_score": float(companion["diag_mass_z_score"]),
            "argmax_match_rate": float(companion["argmax_match_rate"]),
            "subject_ids": np.asarray(companion["subject_ids"]).tolist(),
        }
    for tid, res in per_task.items():
        out["per_task"][int(tid)] = {
            "task_name": res["task_name"],
            "n_subjects": int(res["n_subjects"]),
            "gw_distance": float(res["distance"]),
            "argmax_assignment": dict(res["argmax_assignment"]),
            "p_value": float(res["null"]["p_value"]),
            "z_score": float(res["null"]["z_score"]),
        }
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _render_markdown(summary: Dict[str, object], plots_dir: Path) -> str:
    H = summary["headline"]
    dropped = summary.get("dropped_subjects") or []
    cfg = summary.get("config", {}) or {}
    lines = [
        "# FLS Cross-Modality GW Report",
        "",
        f"**Subjects:** {summary['n_subjects']}",
        f"**Tier counts:** {summary['tier_counts']}",
        f"**Tertile cutoffs:** {summary['tertile_cutoffs']}",
        f"**Dropped (degenerate) subjects:** {dropped if dropped else 'none'}",
        f"**Config:** residualize_task=`{cfg.get('residualize_task', True)}`, "
        f"eeg_zscore=`{cfg.get('eeg_zscore', True)}`",
        "",
        "## Headline 3x3 GW (composite skill)",
        f"- GW distance: `{H['gw_distance']:.6f}`",
        f"- Argmax tier assignment (gaze -> EEG): `{H['argmax_assignment']}`",
        f"- Permutation p-value: `{H['p_value']:.4f}`",
        f"- z-score: `{H['z_score']:.3f}`  (n_perms = {H['n_permutations']})",
        "",
        "![gaze RDM](plots/rdm_centroid_gaze.png)",
        "![EEG RDM](plots/rdm_centroid_eeg.png)",
        "![coupling](plots/coupling_headline.png)",
        "![null](plots/null_headline.png)",
        "",
    ]
    if summary.get("companion") is not None:
        C = summary["companion"]
        lines += [
            f"## Companion {C['n_subjects']}x{C['n_subjects']} entropic GW",
            f"- entropic GW distance: `{C['gw_distance']:.6f}`",
            f"- trace(T) observed: `{C['diag_mass_observed']:.6f}`",
            f"- subject-shuffle p-value: `{C['diag_mass_p_value']:.4f}`",
            f"- z-score: `{C['diag_mass_z_score']:.3f}`",
            f"- argmax row->col match rate: `{C['argmax_match_rate']:.3f}`",
            "",
            "![companion coupling](plots/coupling_companion.png)",
            "![companion null](plots/null_companion.png)",
            "",
        ]
    if summary.get("per_task"):
        lines += ["## Per-task sensitivity"]
        for tid, res in sorted(summary["per_task"].items()):
            lines += [
                f"### task {tid} ({res['task_name']}, n={res['n_subjects']})",
                f"- GW distance: `{res['gw_distance']:.6f}`",
                f"- p-value: `{res['p_value']:.4f}`",
                f"- z-score: `{res['z_score']:.3f}`",
                f"- argmax assignment: `{res['argmax_assignment']}`",
                "",
                f"![coupling](plots/coupling_task_{res['task_name']}.png)",
                f"![null](plots/null_task_{res['task_name']}.png)",
                "",
            ]
    lines += [
        "## Diagnostics",
        "Per-subject PCA scatters and tier-centroid distance heatmaps.",
        "If one tier is being pulled by a few outlier subjects, the PCA "
        "will show them sitting far from their tier's group of points; "
        "the distance heatmap will show one or two anomalous rows in that "
        "tier rather than a uniform band.",
        "",
        "![EEG PCA](plots/diagnostic_pca_eeg.png)",
        "![Gaze PCA](plots/diagnostic_pca_gaze.png)",
        "![EEG centroid distances](plots/diagnostic_dist_eeg.png)",
        "![Gaze centroid distances](plots/diagnostic_dist_gaze.png)",
        "",
        "### Drop-one-subject sensitivity",
        "Box+strip plots of the three off-diagonal cells of the centroid "
        "RDM across N leave-one-subject-out runs. Dashed line marks the "
        "full-population value. A tight box around the dashed line means "
        "the cell is robust to which subject is excluded; a wide spread "
        "or skewed mass means a few subjects drive the geometry.",
        "",
        "![EEG drop-one-out](plots/diagnostic_dropone_eeg.png)",
        "![Gaze drop-one-out](plots/diagnostic_dropone_gaze.png)",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    _setup_logging(args.log_level)
    run(args)


if __name__ == "__main__":
    main()
