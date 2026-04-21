"""End-to-end Phase 3: build all candidate RDMs, score, write manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from . import rdm_builders as rb
from .io import write_manifest, write_rdm_pickle
from .loaders import load_all_trial_records
from .paths import phase3_manifest_path, phase3_rdms_dir
from .schemas import RDMArtifact, TrialRecord
from .task_bridge import JigsawMappingFlags, jigsaws_transfer_notes_for_bridge
from .validation import rank_rdms, score_artifact, stability_split_spearman


def build_all_candidate_rdms(
    cache_root: Path,
    *,
    metric: str = "one_minus_spearman",
    performance_scores_csv: Path | None = None,
    trial_ids: list[str] | None = None,
    max_trials: int | None = None,
) -> tuple[dict[str, RDMArtifact], dict[str, Any], list[str]]:
    """
    Load Phase 1/2 caches and produce candidate RDMs plus manifest dict.

    Returns (artifacts_by_name, manifest_dict, warnings).
    """
    repo_root = cache_root.parent.parent
    perf_csv = performance_scores_csv or (repo_root / "Eye" / "PerformanceScores.csv")

    records, load_warnings = load_all_trial_records(
        cache_root,
        trial_ids=trial_ids,
        max_trials=max_trials,
        performance_csv=perf_csv,
    )

    warnings = list(load_warnings)
    jflags = JigsawMappingFlags()

    artifacts: dict[str, RDMArtifact] = {}
    build_errors: list[str] = []

    builders = [
        ("eye_only_task_family", rb.eye_only_task_family),
        ("eye_only_subskill_family", rb.eye_only_subskill_family),
        ("eeg_latent_task_family", rb.eeg_latent_task_family),
        ("eeg_latent_subskill_family", rb.eeg_latent_subskill_family),
        ("eeg_pred_error_task_family", rb.eeg_pred_error_task_family),
        ("joint_eye_eeg_task_family", rb.joint_eye_eeg_task_family),
        ("joint_eye_eeg_subskill_family", rb.joint_eye_eeg_subskill_family),
        ("performance_tier_rdm", rb.performance_tier_rdm),
    ]

    if len(records) >= 2:
        for name, fn in tqdm(
            builders,
            desc="Phase 3: build candidate RDMs",
            unit="rdm",
        ):
            try:
                artifacts[name] = fn(records, metric=metric)
            except Exception as e:
                build_errors.append(f"{name}:{e}")

    # Optional latent phase
    try:
        lp = rb.latent_phase_rdm(records, metric=metric)
        if lp is not None:
            artifacts["latent_phase_rdm"] = lp
        else:
            warnings.append("latent_phase_rdm_skipped_missing_or_insufficient_phase_data")
    except Exception as e:
        warnings.append(f"latent_phase_rdm_failed:{e}")

    if build_errors:
        warnings.extend(build_errors)

    eye_ref = artifacts.get("eye_only_task_family")
    stability = stability_split_spearman(records, metric=metric)

    transfer_notes = {
        "jigsaws_knot_tying": jflags.knot_tying_note,
        "bridges": {
            "needle_control": jigsaws_transfer_notes_for_bridge("needle_control"),
            "needle_driving": jigsaws_transfer_notes_for_bridge("needle_driving"),
            "other_nontransfer": jigsaws_transfer_notes_for_bridge("other_nontransfer"),
        },
    }

    manifest_entries: list[dict[str, Any]] = []
    _art_items = artifacts.items()
    if artifacts:
        _art_items = tqdm(
            _art_items,
            desc="Phase 3: score & manifest entries",
            unit="rdm",
        )
    for name, art in _art_items:
        art.selection_metadata["jigsaws_knot_tying_transfer_confidence"] = "low"
        art.selection_metadata["jigsaws_knot_tying_note"] = jflags.knot_tying_note
        tw = list(warnings) if name.startswith("joint") or "eeg" in name else []
        sc = score_artifact(art, eye_ref=eye_ref, stability=stability, transfer_warnings=tw)
        entry = {
            "rdm_name": name,
            "rdm_type": art.rdm_type,
            "unit_type": art.unit_type,
            "abstraction_level": art.selection_metadata.get("abstraction"),
            "source_representation": art.source_representation,
            "distance_metric": art.distance_metric,
            "relative_path": f"rdms/{name}.pkl",
            "shape": list(art.matrix.shape),
            "unit_labels": art.unit_labels,
            "recommended_for_transfer": False,
            "scores": sc,
            "warnings": tw,
        }
        manifest_entries.append(entry)

    order = rank_rdms(manifest_entries)
    recommended = False
    for e in manifest_entries:
        e["recommended_for_transfer"] = e["rdm_name"] == order[0] if order else False
    # Mark top 3 as recommended candidates
    top = set(order[: min(3, len(order))])
    for e in manifest_entries:
        if e["rdm_name"] in top:
            e["recommended_tier"] = "high"
        else:
            e["recommended_tier"] = "secondary"

    manifest: dict[str, Any] = {
        "version": 1,
        "cache_root": str(cache_root),
        "metric": metric,
        "n_trials_used": len(records),
        "recommended_order": order,
        "transfer_notes": transfer_notes,
        "rdms": {e["rdm_name"]: e for e in manifest_entries},
        "warnings": warnings,
    }
    return artifacts, manifest, warnings


def write_phase3_outputs(
    artifacts: dict[str, RDMArtifact],
    manifest: dict[str, Any],
    cache_root: Path,
) -> None:
    out_dir = phase3_rdms_dir(cache_root)
    _write_items = artifacts.items()
    if artifacts:
        _write_items = tqdm(
            _write_items,
            desc="Phase 3: write RDM pickles",
            unit="file",
        )
    for name, art in _write_items:
        write_rdm_pickle(art, out_dir / f"{name}.pkl")
    write_manifest(manifest, phase3_manifest_path(cache_root))


def run_phase3_pipeline(
    cache_root: Path,
    *,
    metric: str = "one_minus_spearman",
    performance_scores_csv: Path | None = None,
    trial_ids: list[str] | None = None,
    max_trials: int | None = None,
    write_outputs: bool = True,
) -> tuple[dict[str, RDMArtifact], dict[str, Any], list[str]]:
    artifacts, manifest, warnings = build_all_candidate_rdms(
        cache_root,
        metric=metric,
        performance_scores_csv=performance_scores_csv,
        trial_ids=trial_ids,
        max_trials=max_trials,
    )
    manifest["warnings"] = warnings
    if write_outputs and artifacts:
        write_phase3_outputs(artifacts, manifest, cache_root)
    return artifacts, manifest, warnings
