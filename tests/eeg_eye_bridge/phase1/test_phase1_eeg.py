#!/usr/bin/env python3
"""
Phase 1 EEG: preprocessing, models, export schema, and JSON/MD reports.
Run: ``PYTHONPATH=src python tests/eeg_eye_bridge/phase1/test_phase1_eeg.py``
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
_SRC = _REPO / "src"
sys.path.insert(0, str(_SRC))

REPORT_DIR = _REPO / "reports" / "eeg_eye_bridge" / "phase1"
CACHE_DIR = _REPO / "cache" / "eeg_eye_bridge" / "phase1"


def _write_reports(payload: Dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "test_report.json"
    md_path = REPORT_DIR / "test_report.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# Phase 1 EEG test report",
        "",
        f"- files_processed: {payload.get('files_processed', 0)}",
        f"- synthetic_fallback: {payload.get('synthetic_fallback', False)}",
        f"- sample_trial_ids: {payload.get('sample_trial_ids', [])}",
        "",
        "## Tensor shapes",
        "",
        f"```\n{json.dumps(payload.get('tensor_shapes', {}), indent=2)}\n```",
        "",
        "## Prediction error stats",
        "",
        f"```\n{json.dumps(payload.get('prediction_error_stats', {}), indent=2)}\n```",
        "",
        "## Warnings / failures",
        "",
        f"```\n{json.dumps(payload.get('warnings', []), indent=2)}\n```",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")


def _select_diverse_pairs(
    pairs: List[Any],
    target_trials: int = 4,
) -> List[Any]:
    """
    Prefer a small set that spans task families so downstream Phase 3 can
    build family-level RDMs from real cache artifacts.
    """
    if len(pairs) <= target_trials:
        return pairs

    selected: List[Any] = []
    seen_families = set()

    # First pass: guarantee family diversity when available.
    for item in pairs:
        fam = item[3]
        if fam in seen_families:
            continue
        selected.append(item)
        seen_families.add(fam)
        if len(selected) >= target_trials:
            return selected

    # Second pass: fill remaining slots in stable EDF order.
    for item in pairs:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= target_trials:
            break
    return selected


def main() -> None:
    # Late imports after path
    from eeg_eye_bridge.phase1_eeg.export import (
        CONTRACT_VERSION,
        aggregate_family_summaries,
        export_family_summaries,
        export_manifest,
        export_trial_pickle,
        phase1_cache_root,
    )
    from eeg_eye_bridge.phase1_eeg.metadata import (
        load_performance_scores,
        load_task_id_to_name,
        list_edf_trials,
        task_family_for_task_id,
    )
    from eeg_eye_bridge.phase1_eeg.pipeline import run_encoders_on_windows
    from eeg_eye_bridge.phase1_eeg.preprocessing import (
        build_processor,
        load_eeg_preprocessed,
        sliding_windows,
        synthetic_trial,
    )

    warnings: List[str] = []
    failures: List[str] = []
    sample_trial_ids: List[str] = []
    tensor_shapes: Dict[str, Any] = {}
    pred_stats: Dict[str, Any] = {}
    files_processed = 0
    synthetic_fallback = False

    perf_csv = _REPO / "Eye" / "PerformanceScores.csv"
    table1 = _REPO / "Eye" / "Table1.csv"
    eeg_dir = _REPO / "EEG" / "EEG"

    cache = phase1_cache_root(_REPO)
    trials_dir = cache / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Metadata parse
    perf = load_performance_scores(perf_csv)
    id_to_name = load_task_id_to_name(table1)
    assert len(perf) > 0
    assert len(id_to_name) >= 1

    processor = build_processor()
    trials: List[Dict[str, Any]] = []

    pairs = list_edf_trials(
        eeg_dir,
        perf_csv,
        table1,
        max_trials=24,
        max_participants=None,
    )
    pairs = _select_diverse_pairs(pairs, target_trials=4)
    if not pairs:
        synthetic_fallback = True
        warnings.append("No EDF files under EEG/EEG; using synthetic trial only.")
        eeg, sfreq = synthetic_trial()
        processor.sampling_rate = sfreq
        w_np, w_times = sliding_windows(eeg, sfreq, window_sec=1.0, hop_sec=0.5)
        trials.append(
            {
                "trial_id": "synthetic_test",
                "task_name": "synthetic",
                "task_family": "other_nontransfer",
                "participant_id": 9,
                "task_id": 10,
                "performance_score": 0.0,
                "windows": w_np,
                "window_times": w_times,
            }
        )
    else:
        for edf_path, meta, task_name, task_family in pairs:
            try:
                eeg, sfreq = load_eeg_preprocessed(
                    edf_path, processor, apply_filtering=True, apply_ica=False
                )
                w_np, w_times = sliding_windows(eeg, sfreq, window_sec=1.0, hop_sec=0.5)
                trials.append(
                    {
                        "trial_id": meta.trial_id,
                        "task_name": task_name,
                        "task_family": task_family,
                        "participant_id": meta.participant_id,
                        "task_id": meta.task_id,
                        "performance_score": meta.performance_score,
                        "windows": w_np,
                        "window_times": w_times,
                    }
                )
                files_processed += 1
            except Exception as e:
                failures.append(f"{edf_path}: {e}")
                warnings.append(str(e))

    if not trials:
        eeg, sfreq = synthetic_trial()
        processor.sampling_rate = sfreq
        w_np, w_times = sliding_windows(eeg, sfreq, window_sec=1.0, hop_sec=0.5)
        trials.append(
            {
                "trial_id": "synthetic_test",
                "task_name": "synthetic",
                "task_family": "other_nontransfer",
                "participant_id": 9,
                "task_id": 10,
                "performance_score": 0.0,
                "windows": w_np,
                "window_times": w_times,
            }
        )
        synthetic_fallback = True

    manifest_trials: List[Dict[str, Any]] = []
    agg_records: List[Dict[str, Any]] = []

    for item in trials:
        tid = item["trial_id"]
        sample_trial_ids.append(tid)
        out = run_encoders_on_windows(item["windows"], device="cpu")
        tensor_shapes[tid] = {
            "windows": list(item["windows"].shape),
            "baseline_embeddings": list(out["baseline_embeddings"].shape),
            "pc_embeddings": list(out["pc_embeddings"].shape),
            "prediction_errors": list(out["prediction_errors"].shape),
        }
        pe = out["prediction_errors"]
        pred_stats[tid] = {
            "mean": float(np.mean(pe)),
            "std": float(np.std(pe)),
            "min": float(np.min(pe)),
            "max": float(np.max(pe)),
        }

        export_trial_pickle(
            trials_dir / f"{tid}.pkl",
            trial_id=tid,
            participant_id=int(item["participant_id"]),
            task_id=int(item["task_id"]),
            task_name=str(item["task_name"]),
            task_family=str(item["task_family"]),
            performance_score=float(item["performance_score"]),
            window_times=item["window_times"],
            baseline_embeddings=out["baseline_embeddings"],
            pc_embeddings=out["pc_embeddings"],
            prediction_errors=out["prediction_errors"],
        )
        with open(trials_dir / f"{tid}.pkl", "rb") as f:
            loaded = pickle.load(f)
        required = {
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
        missing = required - set(loaded.keys())
        if missing:
            failures.append(f"Missing keys in pickle {tid}: {missing}")

        manifest_trials.append(
            {
                "trial_id": tid,
                "trial_pkl": str(Path("trials") / f"{tid}.pkl"),
                "participant_id": item["participant_id"],
                "task_id": item["task_id"],
                "task_name": item["task_name"],
                "task_family": item["task_family"],
                "performance_score": item["performance_score"],
                "n_windows": int(out["baseline_embeddings"].shape[0]),
                "baseline_embed_dim": out["baseline_embed_dim"],
                "pc_embed_dim": out["pc_embed_dim"],
            }
        )
        agg_records.append(
            {
                "trial_id": tid,
                "task_family": item["task_family"],
                "baseline_trial_mean": out["baseline_trial_mean"],
                "pc_trial_mean": out["pc_trial_mean"],
            }
        )

    summaries = aggregate_family_summaries(
        agg_records,
        baseline_dim=64,
        pc_dim=64,
    )
    export_family_summaries(cache / "family_summaries.pkl", summaries)
    export_manifest(
        cache,
        manifest_trials,
        extra={"test_run": True, "contract_version": CONTRACT_VERSION},
    )

    with open(cache / "family_summaries.pkl", "rb") as f:
        fam = pickle.load(f)
    if "families" not in fam:
        failures.append("family_summaries missing 'families'")

    # task_family sanity
    assert task_family_for_task_id(15) == "needle_control"
    assert task_family_for_task_id(20) == "needle_driving"

    payload = {
        "files_processed": files_processed,
        "synthetic_fallback": synthetic_fallback,
        "sample_trial_ids": sample_trial_ids[:5],
        "tensor_shapes": tensor_shapes,
        "prediction_error_stats": pred_stats,
        "warnings": warnings + failures,
        "failures": failures,
        "contract_version": CONTRACT_VERSION,
    }
    _write_reports(payload)
    if failures:
        print("Failures:", failures)
        sys.exit(1)
    print("Phase 1 tests OK.")


if __name__ == "__main__":
    main()
