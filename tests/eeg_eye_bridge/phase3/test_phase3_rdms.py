"""
Phase 3 RDM tests: prefer real caches; synthetic mirrors Phase 1/2 contracts; manifest semantics.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from eeg_eye_bridge.integration.schemas import validate_phase3_manifest  # noqa: E402
from eeg_eye_bridge.phase3_rdm.paths import (  # noqa: E402
    default_cache_root,
    phase1_manifest,
    phase1_trials_dir,
    phase2_eye_summaries_dir,
    phase2_selected_repr_path,
    phase3_manifest_path,
    phase3_rdms_dir,
)
from eeg_eye_bridge.phase3_rdm.loaders import load_all_trial_records  # noqa: E402
from eeg_eye_bridge.phase3_rdm.pipeline import run_phase3_pipeline  # noqa: E402
from eeg_eye_bridge.phase3_rdm.schemas import required_rdm_keys, validate_rdm_dict  # noqa: E402
from eeg_eye_bridge.phase3_rdm.validation import matrix_diagnostics  # noqa: E402

_EXPECTED_CANDIDATE_NAMES = (
    "eye_only_task_family",
    "eye_only_subskill_family",
    "eeg_latent_task_family",
    "eeg_pred_error_task_family",
    "joint_eye_eeg_task_family",
    "joint_eye_eeg_subskill_family",
    "performance_tier_rdm",
    "latent_phase_rdm",
)


def _write_synthetic_cache(cache_root: Path) -> None:
    """Mirror real Phase 1 export + Phase 2 fingerprint + Phase 3 selection keys (not a parallel fake schema)."""
    rng = np.random.default_rng(42)
    trial_specs = [
        ("9_15_1", 15),
        ("9_15_2", 16),
        ("9_17_1", 17),
        ("9_18_1", 18),
        ("9_9_1", 9),
        ("9_10_1", 10),
        ("9_23_1", 23),
        ("9_24_1", 24),
    ]
    manifest = {"trials": [{"trial_id": t[0], "task_id": t[1]} for t in trial_specs]}
    cache_root.mkdir(parents=True, exist_ok=True)
    phase1_manifest(cache_root).parent.mkdir(parents=True, exist_ok=True)
    with phase1_manifest(cache_root).open("w", encoding="utf-8") as f:
        json.dump(manifest, f)

    tdir = phase1_trials_dir(cache_root)
    edir = phase2_eye_summaries_dir(cache_root)
    tdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)

    perf_scores = [55.0, 72.0, 88.0, 91.0, 45.0, 80.0, 67.0, 93.0]
    for idx, (tid, task_id) in enumerate(trial_specs):
        n_win, d = 6, 12
        latent = rng.standard_normal((n_win, d)).astype(np.float32)
        pred_e = rng.standard_normal((n_win, 8)).astype(np.float32)
        phase_occ = rng.random(5)
        phase_occ = phase_occ / (phase_occ.sum() + 1e-9)
        trial_obj = {
            "trial_id": tid,
            "task_id": int(task_id),
            "participant_id": 9,
            "task_name": f"t{task_id}",
            "task_family": "needle_driving",
            "performance_score": perf_scores[idx],
            "window_times": np.linspace(0.0, 2.0, n_win, dtype=np.float64),
            "baseline_embeddings": latent,
            "pc_embeddings": latent,
            "prediction_errors": pred_e,
            "latent_phase_summary": phase_occ,
        }
        with (tdir / f"{tid}.pkl").open("wb") as f:
            pickle.dump(trial_obj, f)
        eye_obj = {
            "trial_id": tid,
            "fingerprint": rng.standard_normal(24),
        }
        with (edir / f"{tid}.pkl").open("wb") as f:
            pickle.dump(eye_obj, f)

    sel = {
        "eye_vector_key": "fingerprint",
        "eeg_latent_key": "pc_embeddings",
        "eeg_pred_error_key": "prediction_errors",
        "eeg_latent_phase_key": "latent_phase_summary",
    }
    with phase2_selected_repr_path(cache_root).open("w", encoding="utf-8") as f:
        json.dump(sel, f)


def _real_cache_has_trial_pair(root: Path) -> bool:
    """True if manifest exists and at least one trial has Phase1+Phase2 pickles."""
    man = phase1_manifest(root)
    if not man.exists():
        return False
    with man.open(encoding="utf-8") as f:
        data = json.load(f)
    trials = data.get("trials") or []
    for t in trials[:50]:
        tid = str(t.get("trial_id", t)) if isinstance(t, dict) else str(t)
        tp = phase1_trials_dir(root) / f"{tid}.pkl"
        ep = phase2_eye_summaries_dir(root) / f"{tid}.pkl"
        if tp.exists() and ep.exists():
            return True
    return False


def _real_cache_yields_phase3_records(root: Path) -> bool:
    """True only if Phase 1/2 contracts actually load into Phase 3 (not just files present)."""
    recs, _ = load_all_trial_records(root, max_trials=32)
    return len(recs) >= 2


def _assert_transfer_semantics(art, name: str) -> dict[str, object]:
    """Check unit_type / labels are coarse-grained, not frame-level."""
    ut = str(art.unit_type)
    labels = [str(x) for x in art.unit_labels]
    notes = {
        "unit_type": ut,
        "n_units": len(labels),
        "label_sample": labels[:5],
    }
    assert ut in (
        "task_family",
        "subskill_family",
        "performance_tier",
        "latent_phase",
        "trial",
    ), f"{name}: unexpected unit_type {ut}"
    assert len(labels) <= 32, f"{name}: too many units for coarse transfer ({len(labels)})"
    sm = art.selection_metadata or {}
    kconf = sm.get("jigsaws_knot_tying_transfer_confidence")
    if kconf is not None:
        assert str(kconf).lower() in ("low", "none"), f"{name}: knot tying must stay low-confidence, got {kconf!r}"
    return notes


def test_phase3_rdms_end_to_end(tmp_path: Path) -> None:
    real_root = default_cache_root(_REPO)
    used_real = _real_cache_has_trial_pair(real_root) and _real_cache_yields_phase3_records(
        real_root
    )
    if used_real:
        cache_root = real_root
    else:
        cache_root = tmp_path / "eeg_eye_bridge"
        _write_synthetic_cache(cache_root)

    perf_csv = _REPO / "Eye" / "PerformanceScores.csv"

    artifacts, manifest, warnings = run_phase3_pipeline(
        cache_root,
        metric="one_minus_spearman",
        performance_scores_csv=perf_csv if perf_csv.exists() else None,
        max_trials=16 if used_real else None,
        write_outputs=True,
    )

    assert artifacts, "expected at least one RDM artifact"
    out_dir = phase3_rdms_dir(cache_root)
    assert out_dir.exists()

    man_path = phase3_manifest_path(cache_root)
    assert man_path.exists()
    schema_v = validate_phase3_manifest(man_path)
    assert schema_v.ok, f"manifest schema issues: {schema_v.errors}"

    with man_path.open(encoding="utf-8") as f:
        man_loaded = json.load(f)

    assert man_loaded.get("recommended_order"), "manifest must include recommended_order"
    ro = man_loaded["recommended_order"]
    assert ro[0] in artifacts, "top recommended RDM must correspond to a built artifact"

    rdms_block = man_loaded.get("rdms")
    assert isinstance(rdms_block, dict) and rdms_block, "manifest.rdms must be a non-empty dict"

    transfer_notes = man_loaded.get("transfer_notes") or {}
    k_note = (transfer_notes.get("jigsaws_knot_tying") or "") + str(
        transfer_notes.get("bridges", {})
    )
    assert "knot" in k_note.lower(), "expected knot-tying caution in transfer_notes"

    for name, art in artifacts.items():
        err = validate_rdm_dict(art.as_dict())
        assert not err, f"{name}: {err}"
        m = np.asarray(art.matrix, dtype=np.float64)
        assert m.shape[0] == m.shape[1]
        assert np.allclose(m, m.T, atol=1e-9)
        assert np.allclose(np.diag(m), 0.0, atol=1e-5)
        assert art.source_representation, f"{name}: missing source_representation"
        assert art.distance_metric, f"{name}: missing distance_metric"
        _assert_transfer_semantics(art, name)
        if m.shape[0] > 3:
            off_diag = m[np.triu_indices_from(m, k=1)]
            assert off_diag.size == 0 or np.std(off_diag) > 1e-12, (
                f"{name}: RDM appears degenerate (no off-diagonal variation)"
            )
        _ = matrix_diagnostics(m)

    built_names = set(artifacts.keys())
    for expect in (
        "eye_only_task_family",
        "eeg_latent_task_family",
        "joint_eye_eeg_task_family",
    ):
        assert expect in built_names, f"expected candidate {expect} in {built_names}"

    for name in artifacts:
        meta = man_loaded.get("rdms", {}).get(name, {})
        if isinstance(meta, dict) and meta.get("recommended_for_transfer") is True:
            assert name == ro[0], "only first in recommended_order should be primary recommended"

    reports_dir = _REPO / "reports" / "eeg_eye_bridge" / "phase3"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "artifact_source": "real_cache" if used_real else "synthetic_contract_aligned",
        "cache_root": str(cache_root),
        "synthetic_fallback": not used_real,
        "synthetic_fallback_reason": None if used_real else "no complete phase1+phase2 trial pair in repo cache",
        "validated_manifest_keys": list(man_loaded.keys()),
        "required_rdm_keys_checked": list(required_rdm_keys()),
        "rdms_generated": list(artifacts.keys()),
        "expected_candidate_names_present": [n for n in _EXPECTED_CANDIDATE_NAMES if n in artifacts],
        "recommended_order": ro,
        "transfer_notes_present": bool(transfer_notes),
        "warnings": list(warnings),
        "per_rdm": {},
    }
    for name, art in artifacts.items():
        report["per_rdm"][name] = {
            "shape": list(art.matrix.shape),
            "unit_labels": art.unit_labels,
            "unit_type": art.unit_type,
            "distance_metric": art.distance_metric,
            "source_representation": art.source_representation,
            "selection_metadata_keys": sorted((art.selection_metadata or {}).keys()),
            "diagnostics": matrix_diagnostics(art.matrix),
            "scores": man_loaded.get("rdms", {}).get(name, {}).get("scores", {}),
        }

    with (reports_dir / "test_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Phase 3 RDM test report",
        "",
        f"- **Artifact source**: {report['artifact_source']}",
        f"- **Synthetic fallback**: {report['synthetic_fallback']}",
        f"- **Cache**: `{cache_root}`",
        "",
        "## Contract checks",
        "",
        f"- Manifest keys recorded: {', '.join(report['validated_manifest_keys'])}",
        f"- RDM pickle keys validated: {', '.join(report['required_rdm_keys_checked'])}",
        "",
        "## Generated RDMs",
        "",
    ]
    for name in artifacts:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## Recommended order (downstream)",
            "",
        ]
    )
    for r in ro:
        lines.append(f"1. `{r}`")
    lines.extend(["", "## Warnings", ""])
    for w in warnings:
        lines.append(f"- {w}")
    lines.extend(["", "## Per-RDM summaries", ""])
    for name, info in report["per_rdm"].items():
        lines.append(f"### {name}")
        lines.append(f"- shape: {info['shape']}")
        lines.append(f"- unit_type: {info['unit_type']}")
        lines.append(f"- source_representation: {info['source_representation']}")
        lines.append("")
    with (reports_dir / "test_report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        test_phase3_rdms_end_to_end(Path(td))
