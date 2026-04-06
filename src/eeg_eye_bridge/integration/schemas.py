"""Schema validation for phase artifact contracts (Phase 1–3)."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Phase 1 ---
PHASE1_TRIAL_KEYS = frozenset(
    {
        "trial_id",
        "participant_id",
        "task_id",
        "task_name",
        "task_family",
        "performance_score",
        "window_times",
        "baseline_embeddings",
        "pc_embeddings",
        "prediction_errors",
    }
)

# --- Phase 2 ---
PHASE2_SUMMARY_KEYS = frozenset(
    {
        "trial_id",
        "task_id",
        "task_family",
        "eye_state_summary",
        "eye_transition_summary",
        "pupil_summary",
        "event_summary",
    }
)

PHASE2_CONSISTENCY_KEYS_HINT = frozenset(
    {"metric_names", "scores_by_trial", "scores_by_family", "best_representation"}
)

# --- Phase 3 ---
PHASE3_RDM_PICKLE_KEYS = frozenset(
    {
        "rdm_name",
        "rdm_type",
        "unit_type",
        "unit_labels",
        "distance_metric",
        "matrix",
        "source_representation",
        "selection_metadata",
    }
)


@dataclass
class ValidationResult:
    ok: bool
    phase: str
    name: str
    path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_phase1_manifest(path: Path) -> ValidationResult:
    res = ValidationResult(True, "phase1", "manifest.json", str(path))
    if not path.is_file():
        res.ok = False
        res.errors.append("file missing")
        return res
    try:
        data = _load_json(path)
    except Exception as e:
        res.ok = False
        res.errors.append(f"json: {e}")
        return res
    trial_ids: List[str] = []
    if isinstance(data, dict):
        if "trials" in data and isinstance(data["trials"], list):
            for i, t in enumerate(data["trials"]):
                if isinstance(t, dict) and "trial_id" in t:
                    trial_ids.append(str(t["trial_id"]))
                elif isinstance(t, str):
                    trial_ids.append(t)
                else:
                    res.warnings.append(f"trials[{i}] unexpected shape")
        elif "trial_ids" in data and isinstance(data["trial_ids"], list):
            trial_ids = [str(x) for x in data["trial_ids"]]
        else:
            res.warnings.append("no 'trials' or 'trial_ids' list; accepting dict manifest")
    elif isinstance(data, list):
        for i, t in enumerate(data):
            if isinstance(t, dict) and "trial_id" in t:
                trial_ids.append(str(t["trial_id"]))
            else:
                res.warnings.append(f"list[{i}] missing trial_id")
    else:
        res.ok = False
        res.errors.append("manifest must be JSON object or list")
        return res
    res.details["trial_count"] = len(trial_ids)
    res.details["sample_trial_ids"] = trial_ids[:5]
    return res


def validate_phase1_trial_pkl(path: Path) -> ValidationResult:
    res = ValidationResult(True, "phase1", "trial.pkl", str(path))
    if not path.is_file():
        res.ok = False
        res.errors.append("file missing")
        return res
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        res.ok = False
        res.errors.append(f"pickle: {e}")
        return res
    if not isinstance(obj, dict):
        res.ok = False
        res.errors.append("trial artifact must be a dict")
        return res
    missing = PHASE1_TRIAL_KEYS - set(obj.keys())
    if missing:
        res.ok = False
        res.errors.append(f"missing keys: {sorted(missing)}")
    for key in ("baseline_embeddings", "pc_embeddings", "prediction_errors"):
        if key in obj and obj[key] is not None:
            arr = np.asarray(obj[key])
            res.details[f"{key}_shape"] = list(arr.shape)
    return res


def validate_phase2_selected(path: Path) -> ValidationResult:
    res = ValidationResult(True, "phase2", "selected_representations.json", str(path))
    if not path.is_file():
        res.ok = False
        res.errors.append("file missing")
        return res
    try:
        data = _load_json(path)
    except Exception as e:
        res.ok = False
        res.errors.append(f"json: {e}")
        return res
    if not isinstance(data, dict):
        res.ok = False
        res.errors.append("expected JSON object")
        return res
    res.details["top_level_keys"] = list(data.keys())[:20]
    for k in ("eye_vector_key", "eeg_latent_key", "eeg_pred_error_key", "eeg_latent_phase_key"):
        if k in data:
            res.details[k] = data[k]
    return res


def validate_phase2_eye_summary_pkl(path: Path) -> ValidationResult:
    res = ValidationResult(True, "phase2", "eye_summary.pkl", str(path))
    if not path.is_file():
        res.ok = False
        res.errors.append("file missing")
        return res
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        res.ok = False
        res.errors.append(f"pickle: {e}")
        return res
    if not isinstance(obj, dict):
        res.ok = False
        res.errors.append("expected dict")
        return res
    missing = PHASE2_SUMMARY_KEYS - set(obj.keys())
    if missing:
        res.ok = False
        res.errors.append(f"missing keys: {sorted(missing)}")
    return res


def validate_phase2_consistency_pkl(path: Path) -> ValidationResult:
    res = ValidationResult(True, "phase2", "eye_consistency_scores.pkl", str(path))
    if not path.is_file():
        res.ok = False
        res.errors.append("file missing")
        return res
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        res.ok = False
        res.errors.append(f"pickle: {e}")
        return res
    if not isinstance(obj, dict):
        res.warnings.append("consistency file is not a dict; structure may differ")
    else:
        overlap = PHASE2_CONSISTENCY_KEYS_HINT & set(obj.keys())
        if not overlap:
            res.warnings.append("no expected aggregate keys; custom keys only")
        res.details["keys"] = list(obj.keys())[:30]
    return res


def validate_rdm_matrix(matrix: np.ndarray) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if matrix.ndim != 2:
        errs.append(f"matrix must be 2D, got shape {matrix.shape}")
        return False, errs
    n, m = matrix.shape
    if n != m:
        errs.append(f"matrix must be square, got {matrix.shape}")
        return False, errs
    if not np.all(np.isfinite(matrix)):
        errs.append("matrix contains non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-5):
        errs.append("matrix not symmetric")
    diag = np.diag(matrix)
    if not np.allclose(diag, 0.0, rtol=1e-5, atol=1e-5):
        errs.append("diagonal not zero")
    return len(errs) == 0, errs


def validate_phase3_rdm_pickle(path: Path) -> ValidationResult:
    res = ValidationResult(True, "phase3", "rdm.pkl", str(path))
    if not path.is_file():
        res.ok = False
        res.errors.append("file missing")
        return res
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        res.ok = False
        res.errors.append(f"pickle: {e}")
        return res
    if not isinstance(obj, dict):
        res.ok = False
        res.errors.append("RDM artifact must be a dict")
        return res
    missing = PHASE3_RDM_PICKLE_KEYS - set(obj.keys())
    if missing:
        res.ok = False
        res.errors.append(f"missing keys: {sorted(missing)}")
    mat = obj.get("matrix")
    if mat is not None:
        arr = np.asarray(mat, dtype=np.float64)
        ok, rerrs = validate_rdm_matrix(arr)
        res.details["matrix_shape"] = list(arr.shape)
        if not ok:
            res.ok = False
            res.errors.extend(rerrs)
    labels = obj.get("unit_labels")
    if labels is not None:
        res.details["len_unit_labels"] = len(labels)
    return res


def validate_phase3_manifest(path: Path) -> ValidationResult:
    res = ValidationResult(True, "phase3", "rdm_manifest.json", str(path))
    if not path.is_file():
        res.ok = False
        res.errors.append("file missing")
        return res
    try:
        data = _load_json(path)
    except Exception as e:
        res.ok = False
        res.errors.append(f"json: {e}")
        return res
    candidates = None
    if isinstance(data, dict):
        candidates = data.get("candidates") or data.get("rdms")
        res.details["manifest_keys"] = list(data.keys())
    if candidates is None:
        res.warnings.append("no 'candidates' or 'rdms' list; checking root")
        if isinstance(data, list):
            candidates = data
    if not isinstance(candidates, list):
        res.warnings.append("could not find candidate list")
    else:
        res.details["candidate_count"] = len(candidates)
        recommended = None
        for c in candidates:
            if isinstance(c, dict) and c.get("recommended") is True:
                recommended = c.get("rdm_name") or c.get("name")
                break
        res.details["recommended_rdm_name"] = recommended
    return res


def validate_phase1_family_summaries(path: Path) -> ValidationResult:
    res = ValidationResult(True, "phase1", "family_summaries.pkl", str(path))
    if not path.is_file():
        res.ok = False
        res.errors.append("file missing")
        return res
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        res.ok = False
        res.errors.append(f"pickle: {e}")
        return res
    res.details["type"] = type(obj).__name__
    return res
