"""Construct candidate RDMs from aggregated trial features."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Sequence

import numpy as np
import numpy.typing as npt

from .rdm_core import compute_rdm
from .schemas import RDMArtifact, TrialRecord, labels_sorted
from .task_bridge import performance_tier_from_score, subskill_family, task_family


def _zscore_vec(v: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    x = np.asarray(v, dtype=np.float64).ravel()
    s = float(x.std()) + 1e-8
    return ((x - x.mean()) / s).astype(np.float64)


def _mean_stack(vectors: list[npt.NDArray[np.floating]]) -> npt.NDArray[np.float64]:
    if not vectors:
        raise ValueError("empty_group")
    return np.stack([np.asarray(v, dtype=np.float64) for v in vectors], axis=0).mean(axis=0)


def aggregate_feature_matrix(
    records: Sequence[TrialRecord],
    label_fn: Callable[[TrialRecord], str],
    vec_fn: Callable[[TrialRecord], npt.NDArray[np.floating]],
) -> tuple[list[str], npt.NDArray[np.float64]]:
    groups: dict[str, list[npt.NDArray[np.floating]]] = defaultdict(list)
    for r in records:
        lab = label_fn(r)
        groups[lab].append(vec_fn(r))
    labs = labels_sorted(groups.keys())
    if len(labs) < 2:
        raise ValueError("need_at_least_two_units")
    # Align feature dims within each group; use first vector dim
    rows = []
    for lab in labs:
        mat = _mean_stack(groups[lab])
        rows.append(mat)
    # Pad to max dim if needed (should not happen if same source)
    max_d = max(r.shape[0] for r in rows)
    padded = []
    for r in rows:
        if r.shape[0] < max_d:
            p = np.zeros(max_d, dtype=np.float64)
            p[: r.shape[0]] = r
            padded.append(p)
        else:
            padded.append(r.astype(np.float64))
    X = np.stack(padded, axis=0)
    return labs, X


def build_rdm_artifact(
    name: str,
    rdm_type: str,
    unit_type: str,
    unit_labels: list[str],
    matrix: npt.NDArray[np.floating],
    distance_metric: str,
    source_representation: str,
    selection_metadata: dict[str, Any],
) -> RDMArtifact:
    mat = np.asarray(matrix, dtype=np.float64)
    return RDMArtifact(
        rdm_name=name,
        rdm_type=rdm_type,
        unit_type=unit_type,
        unit_labels=list(unit_labels),
        distance_metric=distance_metric,
        matrix=mat,
        source_representation=source_representation,
        selection_metadata=dict(selection_metadata),
    )


def eye_only_task_family(
    records: Sequence[TrialRecord],
    metric: str = "one_minus_spearman",
) -> RDMArtifact:
    labs, X = aggregate_feature_matrix(
        records,
        lambda r: task_family(r.task_id),
        lambda r: r.eye_vector,
    )
    mat = compute_rdm(X, metric)
    return build_rdm_artifact(
        "eye_only_task_family",
        "eye_only",
        "task_family",
        labs,
        mat,
        metric,
        "eye_summary_vector",
        {"abstraction": "task_family"},
    )


def eye_only_subskill_family(
    records: Sequence[TrialRecord],
    metric: str = "one_minus_spearman",
) -> RDMArtifact:
    labs, X = aggregate_feature_matrix(
        records,
        lambda r: subskill_family(r.task_id),
        lambda r: r.eye_vector,
    )
    mat = compute_rdm(X, metric)
    return build_rdm_artifact(
        "eye_only_subskill_family",
        "eye_only",
        "subskill_family",
        labs,
        mat,
        metric,
        "eye_summary_vector",
        {"abstraction": "subskill_family"},
    )


def eeg_latent_task_family(
    records: Sequence[TrialRecord],
    metric: str = "one_minus_spearman",
) -> RDMArtifact:
    labs, X = aggregate_feature_matrix(
        records,
        lambda r: task_family(r.task_id),
        lambda r: r.latent_summary,
    )
    mat = compute_rdm(X, metric)
    return build_rdm_artifact(
        "eeg_latent_task_family",
        "eeg_latent",
        "task_family",
        labs,
        mat,
        metric,
        "eeg_latent_summary",
        {"abstraction": "task_family"},
    )


def eeg_latent_subskill_family(
    records: Sequence[TrialRecord],
    metric: str = "one_minus_spearman",
) -> RDMArtifact:
    labs, X = aggregate_feature_matrix(
        records,
        lambda r: subskill_family(r.task_id),
        lambda r: r.latent_summary,
    )
    mat = compute_rdm(X, metric)
    return build_rdm_artifact(
        "eeg_latent_subskill_family",
        "eeg_latent",
        "subskill_family",
        labs,
        mat,
        metric,
        "eeg_latent_summary",
        {"abstraction": "subskill_family"},
    )


def eeg_pred_error_task_family(
    records: Sequence[TrialRecord],
    metric: str = "one_minus_spearman",
) -> RDMArtifact:
    labs, X = aggregate_feature_matrix(
        records,
        lambda r: task_family(r.task_id),
        lambda r: r.pred_error_summary,
    )
    mat = compute_rdm(X, metric)
    return build_rdm_artifact(
        "eeg_pred_error_task_family",
        "eeg_pred_error",
        "task_family",
        labs,
        mat,
        metric,
        "eeg_pred_error_summary",
        {"abstraction": "task_family"},
    )


def joint_feature(record: TrialRecord) -> npt.NDArray[np.float64]:
    e = _zscore_vec(record.eye_vector)
    z = _zscore_vec(record.latent_summary)
    return np.concatenate([e, z], axis=0)


def joint_eye_eeg_task_family(
    records: Sequence[TrialRecord],
    metric: str = "one_minus_spearman",
) -> RDMArtifact:
    labs, X = aggregate_feature_matrix(
        records,
        lambda r: task_family(r.task_id),
        joint_feature,
    )
    mat = compute_rdm(X, metric)
    return build_rdm_artifact(
        "joint_eye_eeg_task_family",
        "joint_eye_eeg",
        "task_family",
        labs,
        mat,
        metric,
        "concat_zscore_eye_latent",
        {"abstraction": "task_family", "fusion": "concat_zscore"},
    )


def joint_eye_eeg_subskill_family(
    records: Sequence[TrialRecord],
    metric: str = "one_minus_spearman",
) -> RDMArtifact:
    labs, X = aggregate_feature_matrix(
        records,
        lambda r: subskill_family(r.task_id),
        joint_feature,
    )
    mat = compute_rdm(X, metric)
    return build_rdm_artifact(
        "joint_eye_eeg_subskill_family",
        "joint_eye_eeg",
        "subskill_family",
        labs,
        mat,
        metric,
        "concat_zscore_eye_latent",
        {"abstraction": "subskill_family", "fusion": "concat_zscore"},
    )


def performance_tier_rdm(
    records: Sequence[TrialRecord],
    metric: str = "one_minus_spearman",
) -> RDMArtifact:
    def tier_label(r: TrialRecord) -> str:
        return performance_tier_from_score(r.performance_score)

    labs, X = aggregate_feature_matrix(
        records,
        tier_label,
        lambda r: r.eye_vector,
    )
    mat = compute_rdm(X, metric)
    return build_rdm_artifact(
        "performance_tier_rdm",
        "eye_only",
        "performance_tier",
        labs,
        mat,
        metric,
        "eye_summary_vector",
        {"abstraction": "performance_tier", "note": "eye_geometry_over_performance_bins"},
    )


def latent_phase_rdm(
    records: Sequence[TrialRecord],
    metric: str = "one_minus_spearman",
) -> RDMArtifact | None:
    usable = [r for r in records if r.latent_phase_summary is not None]
    if len(usable) < 2:
        return None

    def phase_label(r: TrialRecord) -> str:
        v = np.asarray(r.latent_phase_summary, dtype=np.float64).ravel()
        k = int(np.argmax(v))
        return f"latent_phase_{k}"

    labs, X = aggregate_feature_matrix(
        usable,
        phase_label,
        lambda r: joint_feature(r),
    )
    mat = compute_rdm(X, metric)
    return build_rdm_artifact(
        "latent_phase_rdm",
        "joint_eye_eeg",
        "latent_phase",
        labs,
        mat,
        metric,
        "latent_phase_argmax_grouped_joint",
        {"abstraction": "latent_phase", "grouping": "argmax_of_latent_phase_summary"},
    )
