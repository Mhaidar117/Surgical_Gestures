"""
Load Phase 1 artifacts from cache/eeg_eye_bridge/phase1/.

Trial pickle contract (flexible — we try several key paths):
  - trial_id: str
  - task_id: int
  - task_family: optional str
  - baseline_embedding: (D,) float array — OR nested under representations['baseline']
  - predictive_embedding / predictive_coding_embedding
  - prediction_error_summary or prediction_error (vector)
  - prediction_error_trajectory: optional (T,) for PE-vs-eye alignment
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .task_family import parse_trial_id, task_family_for_task_id

# Canonical names Phase 2 uses for reporting and ranking
PHASE1_REP_KEYS: Tuple[str, ...] = ("baseline", "predictive", "prediction_error")


def _as_vec(x: Any) -> np.ndarray:
    if x is None:
        raise KeyError("missing array")
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = np.nanmean(arr.reshape(arr.shape[0], -1), axis=0)
    return arr.reshape(-1)


def _get_nested(d: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            raise KeyError(keys)
        cur = cur[k]
    return cur


def extract_representation_vector(trial: Mapping[str, Any], rep: str) -> np.ndarray:
    """
    Return a 1D summary vector for representation name:
    'baseline', 'predictive', 'prediction_error'.
    """
    if rep == "baseline":
        for path in (
            ("baseline_embeddings",),  # real Phase 1 export (window x dim)
            ("baseline_embedding",),
            ("representations", "baseline"),
            ("representations", "baseline_embedding"),
            ("embeddings", "baseline"),
        ):
            try:
                v = _get_nested(trial, *path) if len(path) > 1 else trial[path[0]]
                if isinstance(v, Mapping) and "vector" in v:
                    return _as_vec(v["vector"])
                if isinstance(v, Mapping) and "summary" in v:
                    return _as_vec(v["summary"])
                return _as_vec(v)
            except (KeyError, TypeError):
                continue
    if rep == "predictive":
        for path in (
            ("pc_embeddings",),  # real Phase 1 export
            ("predictive_embedding",),
            ("predictive_coding_embedding",),
            ("representations", "predictive"),
            ("representations", "predictive_embedding"),
        ):
            try:
                v = _get_nested(trial, *path) if len(path) > 1 else trial[path[0]]
                if isinstance(v, Mapping) and "vector" in v:
                    return _as_vec(v["vector"])
                return _as_vec(v)
            except (KeyError, TypeError):
                continue
    if rep == "prediction_error":
        for path in (
            ("prediction_errors",),  # real Phase 1 export (window x dim)
            ("prediction_error_summary",),
            ("prediction_error",),
            ("representations", "prediction_error"),
        ):
            try:
                v = _get_nested(trial, *path) if len(path) > 1 else trial[path[0]]
                if isinstance(v, Mapping) and "summary" in v:
                    return _as_vec(v["summary"])
                return _as_vec(v)
            except (KeyError, TypeError):
                continue
    raise KeyError(f"Could not find vector for representation {rep!r}")


def extract_prediction_error_trajectory(trial: Mapping[str, Any]) -> Optional[np.ndarray]:
    """Optional (T,) trajectory for peak alignment with eye events."""
    def _to_timecourse(x: Any) -> Optional[np.ndarray]:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0:
            return None
        if arr.ndim == 1:
            tc = arr.reshape(-1)
        else:
            t = int(arr.shape[0])
            flat = arr.reshape(t, -1)
            # Preserve the window/time axis and collapse feature dimensions.
            tc = np.nanmean(np.abs(flat), axis=1)
        tc = tc[np.isfinite(tc)]
        return tc if tc.size >= 2 else None

    for key in (
        "prediction_error_trajectory",
        "pe_trajectory",
        "prediction_error_timecourse",
        "prediction_errors",
    ):
        if key in trial and trial[key] is not None:
            tc = _to_timecourse(trial[key])
            if tc is not None:
                return tc
    rep = trial.get("representations")
    if isinstance(rep, Mapping):
        pe = rep.get("prediction_error")
        if isinstance(pe, Mapping):
            for key in ("trajectory", "timecourse", "summary"):
                if pe.get(key) is not None:
                    tc = _to_timecourse(pe[key])
                    if tc is not None:
                        return tc
    return None


def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trial_ids_from_manifest(manifest: Mapping[str, Any]) -> List[str]:
    if "trials" in manifest:
        rows = manifest["trials"]
        out = []
        for r in rows:
            if isinstance(r, str):
                out.append(r)
            elif isinstance(r, Mapping) and "trial_id" in r:
                out.append(str(r["trial_id"]))
        return out
    if "trial_ids" in manifest:
        return [str(x) for x in manifest["trial_ids"]]
    raise ValueError("manifest must contain 'trials' or 'trial_ids'")


def load_trial_pkl(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data)}")
    return data


def load_family_summaries(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_trial_metadata(trial: Mapping[str, Any], trial_id: str) -> Dict[str, Any]:
    """Ensure task_id and task_family exist for downstream."""
    out = dict(trial)
    out.setdefault("trial_id", trial.get("trial_id", trial_id))
    tid = out.get("task_id")
    if tid is None:
        _, task_id, _ = parse_trial_id(trial_id)
        out["task_id"] = task_id
    else:
        task_id = int(tid)
    out.setdefault("task_family", task_family_for_task_id(task_id))
    return out
