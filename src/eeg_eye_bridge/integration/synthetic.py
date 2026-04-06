"""
Minimal synthetic artifacts matching phase contracts (for coordinator E2E when caches are empty).

Does not imply real simulator–JIGSAWS alignment; labels are abstract placeholders only.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from . import paths


def _symmetric_rdm(n: int, rng: np.random.Generator) -> np.ndarray:
    """Valid RDM: symmetric, zero diagonal, off-diagonal in [0, 1]."""
    a = rng.random((n, n))
    m = (a + a.T) / 2.0
    np.fill_diagonal(m, 0.0)
    return m


def write_synthetic_phase_cache(dest: Path, seed: int = 42) -> Dict[str, Any]:
    """
    Write Phase 1–3 synthetic tree under ``dest`` (e.g. temp dir / 'eeg_eye_bridge').

    Layout:
      dest/phase1/...
      dest/phase2/...
      dest/phase3/...
    """
    rng = np.random.default_rng(seed)
    dest = Path(dest)
    out: Dict[str, Any] = {"root": str(dest.resolve())}

    # --- Phase 1 ---
    p1 = dest / "phase1"
    (p1 / paths.PHASE1_TRIALS_DIR).mkdir(parents=True, exist_ok=True)
    trial_id = "synthetic_trial_0"
    manifest = {
        "version": "synthetic_phase5",
        "trials": [
            {
                "trial_id": trial_id,
                "participant_id": "synth_p1",
                "task_id": 15,
                "task_family": "needle_control",
            }
        ],
    }
    with open(p1 / paths.PHASE1_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    n_win, d_emb = 8, 16
    trial_obj = {
        "trial_id": trial_id,
        "participant_id": "synth_p1",
        "task_id": 15,
        "task_name": "synthetic_task",
        "task_family": "needle_control",
        "performance_score": 0.85,
        "window_times": np.linspace(0.0, 1.0, n_win).tolist(),
        "baseline_embeddings": rng.standard_normal((n_win, d_emb)).astype(np.float32),
        "pc_embeddings": rng.standard_normal((n_win, d_emb)).astype(np.float32),
        "prediction_errors": np.abs(rng.standard_normal(n_win)).astype(np.float32),
    }
    with open(p1 / paths.PHASE1_TRIALS_DIR / f"{trial_id}.pkl", "wb") as f:
        pickle.dump(trial_obj, f)

    family_summ: Dict[str, Any] = {
        "needle_control": {"mean_embedding": rng.standard_normal(d_emb).astype(np.float32)}
    }
    with open(p1 / paths.PHASE1_FAMILY_SUMMARIES, "wb") as f:
        pickle.dump(family_summ, f)

    # --- Phase 2 ---
    p2 = dest / "phase2"
    (p2 / paths.PHASE2_EYE_SUMMARIES).mkdir(parents=True, exist_ok=True)
    eye_summary = {
        "trial_id": trial_id,
        "task_id": 15,
        "task_family": "needle_control",
        "eye_state_summary": {"state_a": 0.4, "state_b": 0.6},
        "eye_transition_summary": {"a_to_b": 0.2},
        "pupil_summary": {"mean": 3.0, "std": 0.5},
        "event_summary": {"fixations": 10, "saccades": 12},
    }
    with open(p2 / paths.PHASE2_EYE_SUMMARIES / f"{trial_id}.pkl", "wb") as f:
        pickle.dump(eye_summary, f)

    consistency = {
        "metric_names": ["corr_baseline", "corr_pc"],
        "scores_by_trial": {trial_id: {"corr_baseline": 0.1, "corr_pc": 0.2}},
        "scores_by_family": {"needle_control": {"corr_baseline": 0.1}},
        "best_representation": {"global": "pc_embeddings"},
    }
    with open(p2 / paths.PHASE2_CONSISTENCY, "wb") as f:
        pickle.dump(consistency, f)

    selected = {
        "selected": {"task_family:needle_control": "pc_embeddings"},
        "ranking": ["pc_embeddings", "baseline_embeddings"],
    }
    with open(p2 / paths.PHASE2_SELECTED, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    # --- Phase 3 ---
    p3 = dest / "phase3"
    (p3 / paths.PHASE3_RDMS_DIR).mkdir(parents=True, exist_ok=True)
    unit_labels = ["needle_control", "needle_driving", "other_nontransfer"]
    n = len(unit_labels)
    matrix = _symmetric_rdm(n, rng)
    rdm_name = "synthetic_joint_eye_eeg_task_family"
    rdm_obj = {
        "rdm_name": rdm_name,
        "rdm_type": "joint_eye_eeg_task_family",
        "unit_type": "task_family",
        "unit_labels": unit_labels,
        "distance_metric": "pearson",
        "matrix": matrix,
        "source_representation": "pc_embeddings",
        "selection_metadata": {"recommended": True, "note": "synthetic_phase5"},
    }
    with open(p3 / paths.PHASE3_RDMS_DIR / f"{rdm_name}.pkl", "wb") as f:
        pickle.dump(rdm_obj, f)

    manifest3: Dict[str, Any] = {
        "version": "synthetic_phase5",
        "candidates": [
            {
                "rdm_name": rdm_name,
                "path": f"rdms/{rdm_name}.pkl",
                "abstraction_level": "task_family",
                "recommended": True,
            }
        ],
    }
    with open(p3 / paths.PHASE3_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest3, f, indent=2)

    out["phase1_manifest"] = str(p1 / paths.PHASE1_MANIFEST)
    out["phase3_manifest"] = str(p3 / paths.PHASE3_MANIFEST)
    out["phase3_rdm"] = str(p3 / paths.PHASE3_RDMS_DIR / f"{rdm_name}.pkl")
    out["trial_id"] = trial_id
    return out
