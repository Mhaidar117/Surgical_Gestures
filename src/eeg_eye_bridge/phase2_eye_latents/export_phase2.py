"""Write Phase 2 cache artifacts and orchestrate the pipeline."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .config import Phase2Config
from .consistency import (
    aggregate_rankings,
    family_rdm_correlation,
    score_trial,
)
from .eye_loader import load_eye_csv, load_eye_csv_full_for_events
from .eye_summarize import fingerprint_vector_from_summary, summarize_eye_trial
from .phase1_io import (
    PHASE1_REP_KEYS,
    extract_representation_vector,
    load_family_summaries,
    load_manifest,
    load_trial_pkl,
    normalize_trial_metadata,
    trial_ids_from_manifest,
)
from .task_family import parse_trial_id, task_family_for_task_id


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def run_phase2_pipeline(
    config: Phase2Config,
    phase1_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Full Phase 2 run. Returns (aggregated_scores_dict, warnings).
    """
    warnings: List[str] = []
    phase1 = Path(phase1_dir) if phase1_dir else config.phase1_dir
    out = Path(out_dir) if out_dir else config.phase2_cache_dir
    eye_root = config.eye_root
    manifest_path = phase1 / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Phase 1 manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)
    trial_ids = trial_ids_from_manifest(manifest)
    if config.trial_ids:
        trial_ids = [t for t in trial_ids if t in set(config.trial_ids)]
    if config.subset is not None:
        trial_ids = trial_ids[: config.subset]

    trials_dir = phase1 / "trials"
    fam_path = phase1 / "family_summaries.pkl"
    family_summaries = None
    if fam_path.exists():
        try:
            family_summaries = load_family_summaries(fam_path)
        except Exception as e:  # pragma: no cover
            warnings.append(f"Could not load family_summaries.pkl: {e}")
    else:
        warnings.append(f"Missing {fam_path} (optional for geometry).")

    eye_out = out / "eye_summaries"
    _ensure_dir(eye_out)

    scores_by_trial: Dict[str, Dict[str, Dict[str, float]]] = {}
    family_by_trial: Dict[str, str] = {}
    eye_fp_by_trial: Dict[str, np.ndarray] = {}
    eeg_fp_by_trial: Dict[str, np.ndarray] = {}

    for trial_id in tqdm(
        trial_ids,
        desc="Phase 2: eye summaries & EEG scores",
        unit="trial",
    ):
        pkl_path = trials_dir / f"{trial_id}.pkl"
        if not pkl_path.exists():
            warnings.append(f"Missing Phase 1 trial file: {pkl_path}")
            continue
        trial = load_trial_pkl(pkl_path)
        trial = normalize_trial_metadata(trial, trial_id)
        tid_meta = trial.get("task_id")
        if tid_meta is not None:
            tf = trial.get("task_family") or task_family_for_task_id(int(tid_meta))
        else:
            _p, tid_meta, _ = parse_trial_id(trial_id)
            tf = task_family_for_task_id(int(tid_meta))
        family_by_trial[trial_id] = str(tf)

        csv_path = eye_root / f"{trial_id}.csv"
        if not csv_path.exists():
            warnings.append(f"Missing eye CSV: {csv_path}")
            continue
        try:
            series = load_eye_csv(csv_path)
            gx_full, gy_full, mt_full = load_eye_csv_full_for_events(csv_path)
            bundle = summarize_eye_trial(
                series.gaze_x,
                series.gaze_y,
                series.pupil,
                series.movement_type,
                window_samples=config.window_samples,
                window_stride=config.window_stride,
                hmm_states=config.hmm_states,
            )
        except Exception as e:
            warnings.append(f"Eye processing failed for {trial_id}: {e}")
            continue

        fp_vec = fingerprint_vector_from_summary(bundle)
        record = {
            "trial_id": trial_id,
            "task_id": int(tid_meta),
            "task_family": family_by_trial[trial_id],
            "eye_state_summary": bundle["eye_state_summary"],
            "eye_transition_summary": bundle["eye_transition_summary"],
            "pupil_summary": bundle["pupil_summary"],
            "event_summary": bundle["event_summary"],
            # Phase 3 loaders expect a consumable vector (see loaders._pick_eye_vector).
            "fingerprint": fp_vec,
        }
        with open(eye_out / f"{trial_id}.pkl", "wb") as f:
            pickle.dump(record, f)

        eye_fp_by_trial[trial_id] = fp_vec
        try:
            eeg_fp_by_trial[trial_id] = extract_representation_vector(
                trial, "baseline"
            )
        except KeyError:
            eeg_fp_by_trial[trial_id] = np.zeros(1, dtype=np.float64)
            warnings.append(f"Missing baseline_embedding for {trial_id}; EEG fingerprint fallback.")

        scores_by_trial[trial_id] = score_trial(trial, bundle, mt_full)

    agg = aggregate_rankings(scores_by_trial)

    # Family-level RDM agreement
    scores_by_family: Dict[str, Dict[str, float]] = {}
    families = sorted(set(family_by_trial.values()))
    for fam in tqdm(
        families,
        desc="Phase 2: family-level RDM agreement",
        unit="family",
    ):
        r = family_rdm_correlation(
            list(scores_by_trial.keys()),
            family_by_trial,
            eye_fp_by_trial,
            eeg_fp_by_trial,
            fam,
        )
        scores_by_family[fam] = {"rdm_spearman_baseline_fp": r}

    metric_names = [
        "occupancy_spearman",
        "pe_saccade_spearman",
        "fingerprint_spearman",
        "rdm_spearman_baseline_fp",
    ]

    payload: Dict[str, Any] = {
        "metric_names": metric_names,
        "scores_by_trial": scores_by_trial,
        "scores_by_family": scores_by_family,
        "aggregate": agg,
        "family_summaries_phase1": family_summaries,
        "warnings": warnings,
    }

    _ensure_dir(out)
    with open(out / "eye_consistency_scores.pkl", "wb") as f:
        pickle.dump(payload, f)

    selected = {
        "eye_vector_key": "fingerprint",
        "eeg_latent_key": "pc_embeddings",
        "eeg_pred_error_key": "prediction_errors",
        "eeg_latent_phase_key": "latent_phase_summary",
        "best_per_metric": agg.get("best_per_metric", {}),
        "best_combined_representation": agg.get("best_combined"),
        "mean_combined_score_by_rep": agg.get("mean_combined_score_by_rep", {}),
        "targets": {
            "occupancy": agg.get("best_per_metric", {}).get("occupancy_spearman"),
            "prediction_error_events": agg.get("best_per_metric", {}).get(
                "pe_saccade_spearman"
            ),
            "trial_fingerprint": agg.get("best_per_metric", {}).get(
                "fingerprint_spearman"
            ),
            "combined": agg.get("best_combined"),
        },
    }
    with open(out / "selected_representations.json", "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)

    return payload, warnings
