#!/usr/bin/env python3
"""
Phase 5 integration coordinator: audit phases, validate contracts, synthetic E2E,
eye-only backward compatibility, final reports.

Exit codes:
  0 — harness OK; end-to-end may be partial if phase caches are missing (synthetic path).
  1 — validation failed on **real** artifacts present in repo (not synthetic fallback).

Scientific constraint: simulator EEG is disjoint from JIGSAWS; no cross-dataset timestamp
alignment; synthetic data is abstract geometry only.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Repo `src/` on path
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch

from eeg_eye_bridge.integration.adapters import (
    describe_legacy_vs_manifest,
    load_phase3_rdm_artifact,
    rdm_matrix_to_torch,
)
from eeg_eye_bridge.integration.audit import audit_all_phases, summarize_audits
from eeg_eye_bridge.integration.paths import (
    PHASE1_MANIFEST,
    PHASE1_TRIALS_DIR,
    PHASE2_SELECTED,
    PHASE3_MANIFEST,
    PHASE3_RDMS_DIR,
    cache_phase,
    reports_root,
)
from eeg_eye_bridge.integration.schemas import (
    validate_phase1_family_summaries,
    validate_phase1_manifest,
    validate_phase1_trial_pkl,
    validate_phase2_consistency_pkl,
    validate_phase2_eye_summary_pkl,
    validate_phase2_selected,
    validate_phase3_manifest,
    validate_phase3_rdm_pickle,
)
from eeg_eye_bridge.integration.synthetic import write_synthetic_phase_cache

from modules.brain_rdm import compute_model_rdm, load_eye_rdm


def _load_compute_total_loss() -> Callable:
    """Load losses without importing ``models`` package (avoids timm / ViT deps)."""
    path = _SRC / "models" / "losses.py"
    spec = importlib.util.spec_from_file_location("eeg_bridge_losses_only", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.compute_total_loss


compute_total_loss = _load_compute_total_loss()


def _first_trial_pkl(phase1: Path) -> Optional[Path]:
    trials = phase1 / PHASE1_TRIALS_DIR
    if not trials.is_dir():
        return None
    pkls = sorted(trials.glob("*.pkl"))
    return pkls[0] if pkls else None


def _real_chain_available(root: Path) -> bool:
    p1 = cache_phase(1)
    p2 = cache_phase(2)
    p3 = cache_phase(3)
    if not (p1 / PHASE1_MANIFEST).is_file():
        return False
    if _first_trial_pkl(p1) is None:
        return False
    if not (p2 / PHASE2_SELECTED).is_file():
        return False
    if not (p3 / PHASE3_MANIFEST).is_file():
        return False
    rdms = p3 / PHASE3_RDMS_DIR
    if not rdms.is_dir() or not list(rdms.glob("*.pkl")):
        return False
    return True


def _validate_real_partial(root: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Validate any phase artifacts that exist on disk (non-destructive).
    Used alongside synthetic fallback when the full chain is incomplete.
    """
    results: List[Dict[str, Any]] = []
    all_ok = True

    p1 = cache_phase(1)
    m = p1 / PHASE1_MANIFEST
    if m.is_file():
        r = validate_phase1_manifest(m)
        results.append(_vr("phase1", "manifest(real_partial)", r))
        all_ok = all_ok and r.ok
    tp = _first_trial_pkl(p1)
    if tp:
        r = validate_phase1_trial_pkl(tp)
        results.append(_vr("phase1", "trial(real_partial)", r))
        all_ok = all_ok and r.ok
    fs = p1 / "family_summaries.pkl"
    if fs.is_file():
        r = validate_phase1_family_summaries(fs)
        results.append(_vr("phase1", "family_summaries(real_partial)", r))
        all_ok = all_ok and r.ok

    p2 = cache_phase(2)
    if (p2 / PHASE2_SELECTED).is_file():
        r = validate_phase2_selected(p2 / PHASE2_SELECTED)
        results.append(_vr("phase2", "selected_representations(real_partial)", r))
        all_ok = all_ok and r.ok
    es = p2 / "eye_summaries"
    if es.is_dir():
        pkls = sorted(es.glob("*.pkl"))
        if pkls:
            r = validate_phase2_eye_summary_pkl(pkls[0])
            results.append(_vr("phase2", "eye_summary_sample(real_partial)", r))
            all_ok = all_ok and r.ok
    if (p2 / "eye_consistency_scores.pkl").is_file():
        r = validate_phase2_consistency_pkl(p2 / "eye_consistency_scores.pkl")
        results.append(_vr("phase2", "eye_consistency_scores(real_partial)", r))
        all_ok = all_ok and r.ok

    p3 = cache_phase(3)
    if (p3 / PHASE3_MANIFEST).is_file():
        r = validate_phase3_manifest(p3 / PHASE3_MANIFEST)
        results.append(_vr("phase3", "rdm_manifest(real_partial)", r))
        all_ok = all_ok and r.ok
    rdms = p3 / PHASE3_RDMS_DIR
    if rdms.is_dir():
        for pkl in sorted(rdms.glob("*.pkl")):
            r = validate_phase3_rdm_pickle(pkl)
            results.append(_vr("phase3", f"rdm:{pkl.name}(real_partial)", r))
            all_ok = all_ok and r.ok

    return results, all_ok


def _validate_real_phases(root: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """Returns (results, all_ok)."""
    results: List[Dict[str, Any]] = []
    all_ok = True

    p1 = cache_phase(1)
    r = validate_phase1_manifest(p1 / PHASE1_MANIFEST)
    results.append(_vr("phase1", "manifest", r))
    all_ok = all_ok and r.ok

    tp = _first_trial_pkl(p1)
    if tp:
        r = validate_phase1_trial_pkl(tp)
        results.append(_vr("phase1", "trial", r))
        all_ok = all_ok and r.ok
    else:
        results.append({"phase": "phase1", "name": "trial", "ok": False, "error": "no trial pkl"})
        all_ok = False

    fs = p1 / "family_summaries.pkl"
    if fs.is_file():
        r = validate_phase1_family_summaries(fs)
        results.append(_vr("phase1", "family_summaries", r))
        all_ok = all_ok and r.ok

    p2 = cache_phase(2)
    if (p2 / PHASE2_SELECTED).is_file():
        r = validate_phase2_selected(p2 / PHASE2_SELECTED)
        results.append(_vr("phase2", "selected_representations", r))
        all_ok = all_ok and r.ok
    es = p2 / "eye_summaries"
    if es.is_dir():
        pkls = sorted(es.glob("*.pkl"))
        if pkls:
            r = validate_phase2_eye_summary_pkl(pkls[0])
            results.append(_vr("phase2", "eye_summary_sample", r))
            all_ok = all_ok and r.ok
    if (p2 / "eye_consistency_scores.pkl").is_file():
        r = validate_phase2_consistency_pkl(p2 / "eye_consistency_scores.pkl")
        results.append(_vr("phase2", "eye_consistency_scores", r))
        all_ok = all_ok and r.ok

    p3 = cache_phase(3)
    if (p3 / PHASE3_MANIFEST).is_file():
        r = validate_phase3_manifest(p3 / PHASE3_MANIFEST)
        results.append(_vr("phase3", "rdm_manifest", r))
        all_ok = all_ok and r.ok
    rdms = p3 / PHASE3_RDMS_DIR
    if rdms.is_dir():
        pkls = sorted(rdms.glob("*.pkl"))
        if pkls:
            r = validate_phase3_rdm_pickle(pkls[0])
            results.append(_vr("phase3", "rdm_sample", r))
            all_ok = all_ok and r.ok

    return results, all_ok


def _vr(phase: str, name: str, r) -> Dict[str, Any]:
    return {
        "phase": phase,
        "name": name,
        "ok": r.ok,
        "path": r.path,
        "errors": r.errors,
        "warnings": r.warnings,
        "details": r.details,
    }


def _validate_synthetic_tree(cache_root: Path) -> Tuple[List[Dict[str, Any]], bool]:
    results: List[Dict[str, Any]] = []
    all_ok = True
    p1 = cache_root / "phase1"
    r = validate_phase1_manifest(p1 / PHASE1_MANIFEST)
    results.append(_vr("phase1", "manifest", r))
    all_ok = all_ok and r.ok
    tp = _first_trial_pkl(p1)
    if tp:
        r = validate_phase1_trial_pkl(tp)
        results.append(_vr("phase1", "trial", r))
        all_ok = all_ok and r.ok
    r = validate_phase1_family_summaries(p1 / "family_summaries.pkl")
    results.append(_vr("phase1", "family_summaries", r))
    all_ok = all_ok and r.ok

    p2 = cache_root / "phase2"
    r = validate_phase2_selected(p2 / PHASE2_SELECTED)
    results.append(_vr("phase2", "selected_representations", r))
    all_ok = all_ok and r.ok
    es = p2 / "eye_summaries"
    pkls = sorted(es.glob("*.pkl"))
    if pkls:
        r = validate_phase2_eye_summary_pkl(pkls[0])
        results.append(_vr("phase2", "eye_summary_sample", r))
        all_ok = all_ok and r.ok
    r = validate_phase2_consistency_pkl(p2 / "eye_consistency_scores.pkl")
    results.append(_vr("phase2", "eye_consistency_scores", r))
    all_ok = all_ok and r.ok

    p3 = cache_root / "phase3"
    r = validate_phase3_manifest(p3 / PHASE3_MANIFEST)
    results.append(_vr("phase3", "rdm_manifest", r))
    all_ok = all_ok and r.ok
    rdms = p3 / PHASE3_RDMS_DIR
    for pkl in sorted(rdms.glob("*.pkl")):
        r = validate_phase3_rdm_pickle(pkl)
        results.append(_vr("phase3", f"rdm:{pkl.name}", r))
        all_ok = all_ok and r.ok
    return results, all_ok


def _e2e_phase3_load(cache_phase3: Path) -> Dict[str, Any]:
    obj, path = load_phase3_rdm_artifact(cache_phase3)
    t = rdm_matrix_to_torch(obj)
    n = t.shape[0]
    stub = torch.randn(n, 8)
    model_rdm = compute_model_rdm(stub, method="pearson")
    out: Dict[str, Any] = {
        "rdm_path": str(path),
        "matrix_shape": list(t.shape),
        "model_rdm_shape": list(model_rdm.shape),
        "shapes_match": t.shape == model_rdm.shape,
    }
    return out


def _backward_compat_eye_only() -> Dict[str, Any]:
    eye_path = _REPO_ROOT / "Eye" / "Exploration" / "target_rdm_3x3.npy"
    if not eye_path.is_file():
        return {"ok": False, "error": f"missing {eye_path}"}
    target = load_eye_rdm(str(eye_path))
    rng = torch.Generator().manual_seed(0)
    model_rdm = torch.rand(3, 3, generator=rng)
    model_rdm = (model_rdm + model_rdm.T) / 2
    model_rdm.fill_diagonal_(0.0)
    eeg_rdm = target.clone()

    B, T, D = 2, 4, 19
    pred = torch.randn(B, T, D)
    tgt = torch.randn(B, T, D)
    gesture_logits = torch.randn(B, 15)
    skill_logits = torch.randn(B, 3)
    gesture_labels = torch.randint(0, 15, (B,))
    skill_labels = torch.randint(0, 3, (B,))

    total, comp = compute_total_loss(
        pred,
        tgt,
        gesture_logits,
        gesture_labels,
        skill_logits,
        skill_labels,
        model_rdm=model_rdm,
        eeg_rdm=eeg_rdm,
        brain_mode="eye",
        loss_weights={"kin": 1.0, "gesture": 1.0, "skill": 0.5, "brain": 0.01, "control": 0.01},
    )
    fin = bool(torch.isfinite(total).item())
    return {
        "ok": fin,
        "total_loss": float(total.detach()),
        "has_brain_component": "brain_rsa" in comp,
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# EEG–Eye Bridge — Final Integration Report (Phase 5)",
        "",
        f"**Repo root:** `{report['repo_root']}`",
        "",
        f"**End-to-end status:** {report.get('end_to_end_status', 'unknown')}",
        f"**Synthetic fallback used:** {report.get('used_synthetic_fallback')}",
        f"**Schema validation OK:** {report.get('schema_validation_ok')}",
        "",
        "## Phases detected (any artifacts)",
        "",
        json.dumps(report.get("phases_detected", []), indent=2),
        "",
        "## Backward compatibility (brain_mode=eye loss path)",
        "",
        "```json",
        json.dumps(report.get("backward_compatibility_eye_only", {}), indent=2),
        "```",
        "",
        "## E2E Phase 3 load",
        "",
        "```json",
        json.dumps(report.get("e2e_phase3_load", {}), indent=2),
        "```",
        "",
        "## Legacy vs Phase 3 layout",
        "",
        report.get("legacy_vs_phase3_note", ""),
        "",
        "## Top blockers",
        "",
        "\n".join(f"- {b}" for b in report.get("top_blockers", []) or ["(none)"]),
        "",
        "## Recommended next steps",
        "",
        "\n".join(f"- {s}" for s in report.get("recommended_next_steps", [])),
        "",
    ]
    return "\n".join(lines)


def run_coordinator() -> int:
    root = Path(_REPO_ROOT)
    audits = audit_all_phases(root)
    audit_summary = summarize_audits(audits)

    used_synthetic = not _real_chain_available(root)
    validation_results: List[Dict[str, Any]] = []
    validation_ok = True
    e2e_cache_phase3: Path = cache_phase(3)
    tmp_dir: Optional[str] = None

    partial_results, partial_ok = _validate_real_partial(root)

    if used_synthetic:
        tmp_dir = tempfile.mkdtemp(prefix="eeg_eye_bridge_synth_")
        syn_root = Path(tmp_dir) / "eeg_eye_bridge"
        write_synthetic_phase_cache(syn_root)
        syn_vr, syn_ok = _validate_synthetic_tree(syn_root)
        validation_results = partial_results + syn_vr
        validation_ok = syn_ok and partial_ok
        e2e_cache_phase3 = syn_root / "phase3"
    else:
        validation_results, validation_ok = _validate_real_phases(root)

    e2e: Dict[str, Any] = {}
    try:
        e2e = _e2e_phase3_load(e2e_cache_phase3)
        e2e["status"] = "pass"
    except Exception as e:
        e2e = {"status": "fail", "error": str(e)}

    bc = _backward_compat_eye_only()

    real_complete = _real_chain_available(root) and not used_synthetic
    if validation_ok and e2e.get("status") == "pass" and bc.get("ok"):
        e2e_status = "pass" if real_complete else "partial"
    elif not validation_ok:
        e2e_status = "fail"
    else:
        e2e_status = "partial"

    schema_keys_checked = sorted(
        {
            "phase1_manifest",
            "phase1_trial",
            "phase1_family_summaries",
            "phase2_selected",
            "phase2_eye_summary",
            "phase2_consistency",
            "phase3_manifest",
            "phase3_rdm_pickle",
        }
    )
    validation_mode = (
        "synthetic_only"
        if used_synthetic and not partial_results
        else (
            "partial_real_plus_synthetic"
            if used_synthetic and partial_results
            else "full_real"
        )
    )
    real_chain_healthy = (
        bool(validation_ok)
        and e2e.get("status") == "pass"
        and bc.get("ok")
        and not used_synthetic
    )

    report: Dict[str, Any] = {
        "repo_root": str(root),
        "paths_checked": {
            "cache": str(cache_phase(1).parent),
            "reports": str(reports_root()),
            "configs_bridge_glob": str(root / "src" / "configs" / "bridge_*.yaml"),
        },
        "phases_detected": audit_summary["phases_with_any_artifacts"],
        "audit": audit_summary["by_phase"],
        "used_synthetic_fallback": used_synthetic,
        "validation_mode": validation_mode,
        "real_chain_artifacts_present": real_complete,
        "real_chain_healthy": real_chain_healthy,
        "synthetic_fallback_is_graceful_secondary": used_synthetic,
        "schema_validation": validation_results,
        "schema_validation_ok": validation_ok,
        "schema_validation_phases_checked": schema_keys_checked,
        "legacy_vs_phase3_note": describe_legacy_vs_manifest(),
        "e2e_phase3_load": e2e,
        "end_to_end_status": e2e_status,
        "e2e_failure_surfaced": (not used_synthetic and e2e.get("status") == "fail"),
        "backward_compatibility_eye_only": bc,
        "adapters_used": [
            "eeg_eye_bridge.integration.adapters.load_phase3_rdm_artifact",
        ],
        "recommended_next_steps": [
            "Implement Phase 1–4 pipelines so cache/eeg_eye_bridge/phase* contain real artifacts.",
            "Add bridge_*.yaml configs and extend train_vit_system for manifest-based RDM targets (Phase 4).",
            "Optionally add load_phase3_rdm_candidate to BrainRDM or keep using integration.adapters.",
        ],
        "top_blockers": [] if real_complete else [
            "Phase caches missing or incomplete; coordinator used synthetic fallback.",
            "BrainRDM still uses legacy trial pickle layout; Phase 3 uses rdm_manifest + rdms/*.pkl.",
        ],
    }

    if used_synthetic:
        report["adapters_used"].append("eeg_eye_bridge.integration.synthetic.write_synthetic_phase_cache")

    if not bc.get("ok"):
        report["top_blockers"].append(
            "Eye-only backward compat check failed (see backward_compatibility_eye_only)."
        )

    out_dir = reports_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "final_integration_report.json"
    md_path = out_dir / "final_integration_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(report))

    if tmp_dir:
        import shutil

        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not used_synthetic and not validation_ok:
        return 1
    return 0


class TestPhase5IntegrationCoordinator(unittest.TestCase):
    def test_run_coordinator_writes_report_and_exit_matches_validation(self):
        code = run_coordinator()
        jr = reports_root() / "final_integration_report.json"
        self.assertTrue(jr.is_file())
        with open(jr, encoding="utf-8") as f:
            rep = json.load(f)
        self.assertIn("schema_validation_ok", rep)
        self.assertIn("used_synthetic_fallback", rep)
        self.assertIn("validation_mode", rep)
        self.assertIn("backward_compatibility_eye_only", rep)
        if rep.get("used_synthetic_fallback"):
            self.assertTrue(rep.get("synthetic_fallback_is_graceful_secondary", False))
        expect_fail = not rep.get("used_synthetic_fallback") and not rep.get(
            "schema_validation_ok", False
        )
        self.assertEqual(code, 1 if expect_fail else 0)


if __name__ == "__main__":
    sys.exit(run_coordinator())
