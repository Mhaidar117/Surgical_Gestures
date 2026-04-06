"""RDM diagnostics, stability, and manifest ranking."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

from .rdm_builders import eye_only_task_family
from .schemas import RDMArtifact, TrialRecord, flatten_upper_tri, labels_sorted
from .task_bridge import TASK_FAMILY_BY_ID, transfer_bridge, transfer_plausibility_score


def matrix_diagnostics(mat: npt.NDArray[np.floating]) -> dict[str, float]:
    m = np.asarray(mat, dtype=np.float64)
    tri = flatten_upper_tri(m)
    s = np.linalg.svd(m, compute_uv=False)
    eff_rank = float((s.sum()) ** 2 / (np.sum(s**2) + 1e-12))
    return {
        "frobenius": float(np.linalg.norm(m, ord="fro")),
        "mean_offdiag": float(tri.mean()) if tri.size else 0.0,
        "std_offdiag": float(tri.std()) if tri.size else 0.0,
        "effective_rank_proxy": eff_rank,
    }


def spearman_between_rdms(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating]) -> float:
    va = flatten_upper_tri(np.asarray(a, dtype=np.float64))
    vb = flatten_upper_tri(np.asarray(b, dtype=np.float64))
    n = min(va.size, vb.size)
    if n == 0:
        return 0.0
    r, _ = spearmanr(va[:n], vb[:n])
    return 0.0 if r is None or np.isnan(r) else float(r)


def stability_split_spearman(
    records: Sequence[TrialRecord],
    metric: str = "one_minus_spearman",
) -> float | None:
    """
    Split trials by subject_id parity; build task-family eye RDM each; Spearman between vectors.
    """
    with_subj = [r for r in records if r.subject_id is not None]
    if len(with_subj) < 4:
        return None
    a = [r for r in with_subj if r.subject_id % 2 == 0]
    b = [r for r in with_subj if r.subject_id % 2 == 1]
    if len(a) < 2 or len(b) < 2:
        return None
    try:
        ra = eye_only_task_family(a, metric=metric)
        rb = eye_only_task_family(b, metric=metric)
    except Exception:
        return None
    # Align on common task_family labels intersection order
    la = set(ra.unit_labels)
    lb = set(rb.unit_labels)
    common = labels_sorted(la & lb)
    if len(common) < 2:
        return None
    ia = [ra.unit_labels.index(c) for c in common]
    ib = [rb.unit_labels.index(c) for c in common]
    ma = ra.matrix[np.ix_(ia, ia)]
    mb = rb.matrix[np.ix_(ib, ib)]
    return spearman_between_rdms(ma, mb)


def score_artifact(
    art: RDMArtifact,
    *,
    eye_ref: RDMArtifact | None,
    stability: float | None,
    transfer_warnings: list[str],
) -> dict[str, Any]:
    diag = matrix_diagnostics(art.matrix)
    eye_agree = None
    if eye_ref is not None and art.rdm_name != eye_ref.rdm_name:
        try:
            eye_agree = spearman_between_rdms(art.matrix, eye_ref.matrix)
        except Exception:
            eye_agree = None

    # Interpretability proxy: spread in off-diagonal distances
    interpretability = float(min(1.0, diag["std_offdiag"] / (diag["mean_offdiag"] + 1e-6)))

    # Transfer plausibility: average bridge weight from unit labels (task_family strings)
    plaus = 0.7
    meta = art.selection_metadata
    if meta.get("abstraction") == "task_family" and art.unit_labels:
        scores = []
        for lab in art.unit_labels:
            tid = next((t for t, fam in TASK_FAMILY_BY_ID.items() if fam == lab), None)
            if tid is None:
                continue
            br = transfer_bridge(int(tid))
            scores.append(
                transfer_plausibility_score(
                    br,
                    rdm_includes_eeg=art.rdm_type
                    in ("eeg_latent", "eeg_pred_error", "joint_eye_eeg"),
                )
            )
        if scores:
            plaus = float(np.mean(scores))

    if any("knot" in w.lower() for w in transfer_warnings):
        plaus *= 0.85

    combined = 0.0
    w_sum = 0.0
    if stability is not None:
        combined += 0.35 * max(0.0, stability)
        w_sum += 0.35
    if eye_agree is not None:
        combined += 0.35 * max(0.0, eye_agree)
        w_sum += 0.35
    combined += 0.15 * interpretability
    w_sum += 0.15
    combined += 0.15 * plaus
    w_sum += 0.15
    score = float(combined / w_sum) if w_sum > 0 else plaus

    return {
        # Keep report fields numeric even when a metric is unavailable for
        # the current data slice (e.g., too few trials for split stability).
        "internal_stability_spearman": float(stability) if stability is not None else 0.0,
        "eye_geometry_agreement_spearman": float(eye_agree) if eye_agree is not None else 0.0,
        "interpretability_proxy": interpretability,
        "transfer_plausibility": plaus,
        "combined_score": score,
        "matrix_diagnostics": diag,
    }


def rank_rdms(entries: list[dict[str, Any]]) -> list[str]:
    """Return rdm_name list sorted by combined_score descending."""
    def key(e: dict[str, Any]) -> float:
        return float(e.get("scores", {}).get("combined_score", 0.0))

    sorted_entries = sorted(entries, key=key, reverse=True)
    return [str(e["rdm_name"]) for e in sorted_entries]
