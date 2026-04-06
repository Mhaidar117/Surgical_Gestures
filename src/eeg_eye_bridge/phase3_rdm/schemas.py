"""
Documented input/output schemas for Phase 3 RDM construction.

Phase 1 trial pickle (``cache/eeg_eye_bridge/phase1/trials/{trial_id}.pkl``) — expected keys:

- ``trial_id`` (str)
- ``task_id`` (int, 1–27) — simulator task from Eye/Table1.csv
- ``subject_id`` (int, optional)
- ``latent_summary`` (np.ndarray) — EEG latent vector for the trial
- ``pred_error_summary`` (np.ndarray) — prediction-error vector for the trial
- ``latent_phase_summary`` (np.ndarray or dict, optional) — occupancy or embedding for latent phase

Phase 1 ``manifest.json``:

- ``trials``: list of ``{ "trial_id": str, ... }`` or list of trial_id strings

Phase 2 ``eye_summaries/{trial_id}.pkl``:

- ``trial_id`` (str)
- One of: ``eye_vector``, ``fingerprint``, ``summary_vector`` (np.ndarray) — resolved via
  ``selected_representations.json``

Phase 2 ``selected_representations.json``:

- ``eye_vector_key`` (str): key inside eye summary pickle for the primary eye feature vector
- ``eeg_latent_key`` (str, optional): override for latent field name in trial pickle
- ``eeg_pred_error_key`` (str, optional): override for pred error field name

Phase 2 ``eye_consistency_scores.pkl``:

- Dict mapping ``trial_id`` -> float score (optional; used for weighting/diagnostics)

Each Phase 3 RDM artifact (``*.pkl``) must contain:

- ``rdm_name``, ``rdm_type``, ``unit_type``, ``unit_labels``, ``distance_metric``,
  ``matrix``, ``source_representation``, ``selection_metadata``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt


@dataclass
class TrialRecord:
    """Normalized trial-level features for RDM aggregation."""

    trial_id: str
    task_id: int
    subject_id: int | None
    latent_summary: npt.NDArray[np.floating]
    pred_error_summary: npt.NDArray[np.floating]
    eye_vector: npt.NDArray[np.floating]
    latent_phase_summary: npt.NDArray[np.floating] | None
    performance_score: float | None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RDMArtifact:
    """On-disk RDM bundle (also the dict structure saved as pickle)."""

    rdm_name: str
    rdm_type: str
    unit_type: str
    unit_labels: list[str]
    distance_metric: str
    matrix: npt.NDArray[np.floating]
    source_representation: str
    selection_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "rdm_name": self.rdm_name,
            "rdm_type": self.rdm_type,
            "unit_type": self.unit_type,
            "unit_labels": list(self.unit_labels),
            "distance_metric": self.distance_metric,
            "matrix": self.matrix,
            "source_representation": self.source_representation,
            "selection_metadata": dict(self.selection_metadata),
        }


def required_rdm_keys() -> tuple[str, ...]:
    return (
        "rdm_name",
        "rdm_type",
        "unit_type",
        "unit_labels",
        "distance_metric",
        "matrix",
        "source_representation",
        "selection_metadata",
    )


def validate_rdm_dict(obj: Mapping[str, Any]) -> list[str]:
    """Return list of validation error messages (empty if ok)."""
    errors: list[str] = []
    for k in required_rdm_keys():
        if k not in obj:
            errors.append(f"missing_key:{k}")
    mat = obj.get("matrix")
    if mat is not None:
        m = np.asarray(mat, dtype=np.float64)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            errors.append("matrix_not_square")
        else:
            if not np.allclose(m, m.T):
                errors.append("matrix_not_symmetric")
            if not np.allclose(np.diag(m), 0.0, atol=1e-6):
                errors.append("diagonal_not_zero")
    return errors


@dataclass
class SelectedRepresentations:
    eye_vector_key: str = "eye_vector"
    eeg_latent_key: str = "latent_summary"
    eeg_pred_error_key: str = "pred_error_summary"
    eeg_latent_phase_key: str = "latent_phase_summary"

    @classmethod
    def from_json(cls, data: Mapping[str, Any] | None) -> "SelectedRepresentations":
        if not data:
            return cls()
        return cls(
            eye_vector_key=str(data.get("eye_vector_key", "eye_vector")),
            eeg_latent_key=str(data.get("eeg_latent_key", "latent_summary")),
            eeg_pred_error_key=str(data.get("eeg_pred_error_key", "pred_error_summary")),
            eeg_latent_phase_key=str(data.get("eeg_latent_phase_key", "latent_phase_summary")),
        )


def flatten_upper_tri(mat: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    m = mat.shape[0]
    idx = np.triu_indices(m, k=1)
    return mat[idx].astype(np.float64)


def labels_sorted(labels: Sequence[str]) -> list[str]:
    return sorted(str(x) for x in labels)
