"""Load Phase 1/2 artifacts with defensive key resolution."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .schemas import SelectedRepresentations, TrialRecord
from .task_bridge import parse_trial_id_parts, performance_tier_from_score


def _as_vec(x: Any, label: str) -> npt.NDArray[np.floating]:
    if x is None:
        raise KeyError(f"missing_{label}")
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = np.nanmean(arr.reshape(arr.shape[0], -1), axis=0)
    arr = arr.ravel()
    if arr.size == 0:
        raise ValueError(f"empty_vector:{label}")
    return arr


def _pick_eye_vector(eye_obj: Mapping[str, Any], sel: SelectedRepresentations) -> npt.NDArray[np.floating]:
    keys = [
        sel.eye_vector_key,
        "eye_vector",
        "fingerprint",
        "summary_vector",
        "fingerprint_summary",
    ]
    for k in keys:
        if k in eye_obj and eye_obj[k] is not None:
            return _as_vec(eye_obj[k], f"eye:{k}")
    raise KeyError("no_eye_vector_found")


def load_phase1_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def trial_ids_from_manifest(manifest: Mapping[str, Any]) -> list[str]:
    trials = manifest.get("trials")
    if trials is None:
        return []
    out: list[str] = []
    for t in trials:
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, dict) and "trial_id" in t:
            out.append(str(t["trial_id"]))
    return out


def load_phase2_selection(path: Path) -> SelectedRepresentations:
    if not path.exists():
        return SelectedRepresentations()
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return SelectedRepresentations.from_json(data)


def load_phase2_consistency(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in obj.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def load_trial_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def trial_record_from_pkl(
    trial_pkl: Mapping[str, Any],
    eye_pkl: Mapping[str, Any] | None,
    sel: SelectedRepresentations,
    performance_score: float | None,
) -> TrialRecord:
    tid = str(trial_pkl.get("trial_id", ""))
    task_id = int(trial_pkl.get("task_id", -1))
    subj = trial_pkl.get("subject_id")
    subject_id = int(subj) if subj is not None else None

    lat_key = sel.eeg_latent_key
    err_key = sel.eeg_pred_error_key
    phase_key = sel.eeg_latent_phase_key

    latent = _as_vec(trial_pkl.get(lat_key), lat_key)
    pred_err = _as_vec(trial_pkl.get(err_key), err_key)

    phase_raw = trial_pkl.get(phase_key)
    if phase_raw is None:
        phase = None
    else:
        phase = np.asarray(phase_raw, dtype=np.float64).ravel()

    if eye_pkl is None:
        raise ValueError(f"missing_eye_summary_for_{tid}")
    eye_vec = _pick_eye_vector(eye_pkl, sel)

    return TrialRecord(
        trial_id=tid,
        task_id=task_id,
        subject_id=subject_id,
        latent_summary=latent,
        pred_error_summary=pred_err,
        eye_vector=eye_vec,
        latent_phase_summary=phase,
        performance_score=performance_score,
        raw=dict(trial_pkl),
    )


def load_all_trial_records(
    cache_root: Path,
    *,
    trial_ids: list[str] | None = None,
    max_trials: int | None = None,
    performance_csv: Path | None = None,
) -> tuple[list[TrialRecord], list[str]]:
    """
    Load Phase1 trials + Phase2 eye summaries from disk.

    Returns (records, warnings).
    """
    from .paths import (
        phase1_manifest,
        phase1_trials_dir,
        phase2_eye_summaries_dir,
        phase2_consistency_path,
        phase2_selected_repr_path,
    )

    warnings_out: list[str] = []
    manifest = load_phase1_manifest(phase1_manifest(cache_root))
    sel = load_phase2_selection(phase2_selected_repr_path(cache_root))
    _ = load_phase2_consistency(phase2_consistency_path(cache_root))  # optional diagnostics

    ids = trial_ids if trial_ids is not None else trial_ids_from_manifest(manifest)
    if not ids:
        warnings_out.append("phase1_manifest_empty_or_missing")
        return [], warnings_out

    if max_trials is not None:
        ids = ids[: max_trials]

    perf_lookup: dict[tuple[int, int, int], float] = {}
    if performance_csv is not None and performance_csv.exists():
        from .task_bridge import load_performance_lookup

        perf_lookup = dict(load_performance_lookup(str(performance_csv)))

    trials_dir = phase1_trials_dir(cache_root)
    eye_dir = phase2_eye_summaries_dir(cache_root)

    records: list[TrialRecord] = []
    for trial_id in tqdm(
        ids,
        desc="Phase 3: load trial records",
        unit="trial",
    ):
        tp = trials_dir / f"{trial_id}.pkl"
        ep = eye_dir / f"{trial_id}.pkl"
        if not tp.exists():
            warnings_out.append(f"missing_phase1_trial:{trial_id}")
            continue
        if not ep.exists():
            warnings_out.append(f"missing_phase2_eye:{trial_id}")
            continue
        try:
            trial_obj = load_trial_pkl(tp)
            eye_obj = load_trial_pkl(ep)
        except Exception as e:
            warnings_out.append(f"load_error:{trial_id}:{e}")
            continue

        perf_score: float | None = None
        subj, task, tr = parse_trial_id_parts(trial_id)
        if subj is not None and task is not None and tr is not None:
            perf_score = perf_lookup.get((subj, task, tr))
        if perf_score is None and "performance_score" in trial_obj:
            try:
                perf_score = float(trial_obj["performance_score"])
            except (TypeError, ValueError):
                pass

        try:
            rec = trial_record_from_pkl(trial_obj, eye_obj, sel, perf_score)
        except Exception as e:
            warnings_out.append(f"record_build_error:{trial_id}:{e}")
            continue
        records.append(rec)

    if not records:
        warnings_out.append("no_trial_records_loaded")
    return records, warnings_out
