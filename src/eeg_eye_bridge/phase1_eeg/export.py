"""
Phase 1 cache export: ``manifest.json``, per-trial pickles, ``family_summaries.pkl``.

See CLAUDE.md (EEG–Eye Bridge, Phase 1) for the contract consumed by later phases.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

CONTRACT_VERSION = "phase1_eeg_v1"


def phase1_cache_root(data_root: Path) -> Path:
    return Path(data_root) / "cache" / "eeg_eye_bridge" / "phase1"


def export_trial_pickle(
    out_path: Path,
    trial_id: str,
    participant_id: int,
    task_id: int,
    task_name: str,
    task_family: str,
    performance_score: float,
    window_times: np.ndarray,
    baseline_embeddings: np.ndarray,
    pc_embeddings: np.ndarray,
    prediction_errors: np.ndarray,
    contract_version: str = CONTRACT_VERSION,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "trial_id": trial_id,
        "participant_id": participant_id,
        "task_id": task_id,
        "task_name": task_name,
        "task_family": task_family,
        "performance_score": float(performance_score),
        "window_times": np.asarray(window_times, dtype=np.float64),
        "baseline_embeddings": np.asarray(baseline_embeddings, dtype=np.float32),
        "pc_embeddings": np.asarray(pc_embeddings, dtype=np.float32),
        "prediction_errors": np.asarray(prediction_errors, dtype=np.float32),
        "contract_version": contract_version,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def export_manifest(
    cache_root: Path,
    trials: List[Dict[str, Any]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    body: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "trials": trials,
    }
    if extra:
        body.update(extra)
    with open(cache_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)


def export_family_summaries(
    out_path: Path,
    summaries: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"contract_version": CONTRACT_VERSION, **summaries}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def aggregate_family_summaries(
    trial_records: List[Dict[str, Any]],
    baseline_dim: int,
    pc_dim: int,
) -> Dict[str, Any]:
    """
    Mean per-window embedding averaged within trial, then averaged across trials per family.
    """
    families: Dict[str, List[Dict[str, Any]]] = {}
    for rec in trial_records:
        fam = rec["task_family"]
        families.setdefault(fam, []).append(rec)

    out: Dict[str, Any] = {"families": {}}
    for fam, recs in families.items():
        b = np.stack([r["baseline_trial_mean"] for r in recs], axis=0)
        p = np.stack([r["pc_trial_mean"] for r in recs], axis=0)
        out["families"][fam] = {
            "n_trials": len(recs),
            "trial_ids": [r["trial_id"] for r in recs],
            "mean_baseline_embedding": b.mean(axis=0).astype(np.float32),
            "mean_pc_embedding": p.mean(axis=0).astype(np.float32),
            "baseline_dim": baseline_dim,
            "pc_dim": pc_dim,
        }
    return out
