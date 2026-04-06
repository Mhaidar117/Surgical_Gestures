#!/usr/bin/env python3
"""
Phase 4 tests: manifest-based bridge targets, JIGSAWS alignment, coarse grouping, knot-tying caution, eye baseline.

Writes:
  reports/eeg_eye_bridge/phase4/test_report.json
  reports/eeg_eye_bridge/phase4/test_report.md
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_eye_bridge.phase4_vit.target_loader import (  # noqa: E402
    load_bridge_target_from_manifest,
    align_bridge_target_to_jigsaws_task_family,
)
from eeg_eye_bridge.phase4_vit.label_grouping import (  # noqa: E402
    expand_group_labels_for_bridge,
    default_task_label_order_matches_jigsaws,
    JIGSAWS_TASK_ORDER,
)
from modules.brain_rdm import (  # noqa: E402
    compute_centroid_rdm,
    compute_task_centroid_rdm,
    eye_rsa_loss,
)


def _load_compute_total_loss() -> Callable:
    """Load losses without importing ``models`` package (avoids ViT / timm)."""
    path = SRC / "models" / "losses.py"
    spec = importlib.util.spec_from_file_location("phase4_losses_only", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.compute_total_loss


compute_total_loss = _load_compute_total_loss()


def _write_reports(payload: Dict[str, Any], md_lines: List[str]) -> None:
    out_dir = REPO_ROOT / "reports" / "eeg_eye_bridge" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    js = out_dir / "test_report.json"
    md = out_dir / "test_report.md"
    with open(js, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def _synthetic_manifest_dir() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="phase4_test_"))
    rdms = tmp / "rdms"
    rdms.mkdir(parents=True)
    artifact = {
        "unit_type": "task_family",
        "unit_labels": list(JIGSAWS_TASK_ORDER),
        "matrix": [[0.0, 0.3, 0.6], [0.3, 0.0, 0.4], [0.6, 0.4, 0.0]],
        "rdm_type": "euclidean",
    }
    with open(rdms / "syn_eeg.pkl", "wb") as f:
        pickle.dump(artifact, f)
    manifest = {
        "version": 1,
        "recommended_order": ["syn_eeg"],
        "transfer_notes": {
            "jigsaws_knot_tying": "low_confidence: knot tying is not a strong simulator match",
        },
        "rdms": {
            "syn_eeg": {"relative_path": "rdms/syn_eeg.pkl", "unit_type": "task_family"},
        },
    }
    with open(tmp / "rdm_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return tmp


def _first_manifest_candidate_key(manifest_path: Path) -> str | None:
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    order = data.get("recommended_order") or []
    rdms = data.get("rdms")
    if isinstance(rdms, dict):
        for name in order:
            if name in rdms:
                return name
        if rdms:
            return next(iter(rdms.keys()))
    return None


def run_checks() -> Dict[str, Any]:
    warnings: List[str] = []
    results: Dict[str, Any] = {
        "artifact_source": [],
        "modes_tested": [],
        "target_shapes": {},
        "grouping_labels": {},
        "manifest_checks": {},
        "transfer_constraints": {},
        "loss_values": {},
        "backward_compat_eye_ok": False,
        "brain_mode_eye_ok": False,
        "warnings": warnings,
    }

    device = torch.device("cpu")

    syn_dir = _synthetic_manifest_dir()
    bt = load_bridge_target_from_manifest(syn_dir / "rdm_manifest.json", "syn_eeg")
    bt = align_bridge_target_to_jigsaws_task_family(bt)
    results["modes_tested"].append("bridge_loader_synthetic")
    results["target_shapes"]["syn_eeg"] = list(bt.matrix.shape)
    results["grouping_labels"]["syn_eeg"] = list(bt.unit_labels)
    assert default_task_label_order_matches_jigsaws(list(bt.unit_labels))
    syn_man = json.loads((syn_dir / "rdm_manifest.json").read_text(encoding="utf-8"))
    tn = syn_man.get("transfer_notes") or {}
    results["transfer_constraints"]["synthetic_knot_note_present"] = "knot" in str(tn).lower()

    bad_dir = Path(tempfile.mkdtemp(prefix="phase4_bad_"))
    (bad_dir / "rdms").mkdir(parents=True)
    bad_pkl = {
        "unit_type": "task_family",
        "unit_labels": ["A", "B", "C"],
        "matrix": [[0.0, 0.2, 0.3], [0.2, 0.0, 0.25], [0.3, 0.25, 0.0]],
        "rdm_type": "euclidean",
    }
    with open(bad_dir / "rdms" / "bad.pkl", "wb") as f:
        pickle.dump(bad_pkl, f)
    with open(bad_dir / "rdm_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"rdms": {"bad": {"relative_path": "rdms/bad.pkl"}}}, f)
    try:
        bt_bad = load_bridge_target_from_manifest(bad_dir / "rdm_manifest.json", "bad")
        align_bridge_target_to_jigsaws_task_family(bt_bad)
    except ValueError:
        results["modes_tested"].append("jigsaws_align_rejects_unmatched_labels")
    else:
        raise AssertionError("expected ValueError for non-JIGSAWS task labels")

    bt_use = bt
    cache_target_loaded = False
    cache_manifest = REPO_ROOT / "cache" / "eeg_eye_bridge" / "phase3" / "rdm_manifest.json"
    if cache_manifest.is_file():
        cache_key = _first_manifest_candidate_key(cache_manifest)
        if cache_key is None:
            warnings.append("phase3 rdm_manifest.json has no rdms entries")
        else:
            try:
                bt_cache = load_bridge_target_from_manifest(cache_manifest, cache_key)
                results["artifact_source"].append("repo_phase3_manifest")
                results["modes_tested"].append(f"bridge_loader_cache:{cache_key}")
                results["target_shapes"][cache_key] = list(bt_cache.matrix.shape)
                results["manifest_checks"]["cache_unit_type"] = bt_cache.unit_type
                man = json.loads(cache_manifest.read_text(encoding="utf-8"))
                results["transfer_constraints"]["knot_tying_note_in_manifest"] = (
                    "knot" in json.dumps(man.get("transfer_notes", {})).lower()
                )
                if bt_cache.num_groups == 3:
                    try:
                        bt_use = align_bridge_target_to_jigsaws_task_family(bt_cache)
                        results["grouping_labels"][cache_key] = list(bt_use.unit_labels)
                    except ValueError as e:
                        # Keep the cache target even when labels are simulator-family names
                        # that do not map 1:1 onto JIGSAWS task labels.
                        bt_use = bt_cache
                        results["manifest_checks"]["skipped_jigsaws_align"] = "label_mismatch"
                        warnings.append(f"cache label alignment skipped: {e}")
                else:
                    bt_use = bt_cache
                    results["manifest_checks"]["skipped_jigsaws_align"] = "K!=3"
                    warnings.append(
                        "cache RDM is not 3×3 task-family; centroid RSA uses synthetic target for shape match"
                    )
                cache_target_loaded = True
            except (KeyError, FileNotFoundError, ValueError) as e:
                warnings.append(f"cache manifest load failed ({cache_key}): {e}")
    else:
        warnings.append("cache/eeg_eye_bridge/phase3/rdm_manifest.json not found; cache branch skipped")

    if not cache_target_loaded:
        results["artifact_source"].append("synthetic_manifest")

    B, T, D = 2, 4, 8
    emb = torch.randn(B, T, D, device=device, requires_grad=True)
    flat = emb.view(B * T, D)
    batch = {
        "task_label": torch.tensor([0, 1], device=device),
        "gesture_label": torch.tensor([0, 1], device=device),
    }
    gl = expand_group_labels_for_bridge(batch, "task", T).to(device)
    K = bt_use.num_groups
    if K < 2:
        warnings.append("target K<2; using synthetic target for centroid test")
        bt_use = bt
        K = bt_use.num_groups
    model_rdm = compute_centroid_rdm(flat, gl, K)
    target = bt_use.matrix.to(device)
    if model_rdm.shape != target.shape:
        warnings.append(
            f"model RDM {model_rdm.shape} vs target {target.shape}; using synthetic for RSA match"
        )
        bt_use = bt
        target = bt_use.matrix.to(device)
        model_rdm = compute_centroid_rdm(flat, gl, bt_use.num_groups)
    loss_b = eye_rsa_loss(model_rdm, target)
    loss_b.backward()
    results["modes_tested"].append("bridge_centroid_rsa")
    results["loss_values"]["bridge_rsa"] = float(loss_b.detach())
    assert emb.grad is not None

    B0 = 2
    pred_k = torch.randn(B0, T, 19, device=device)
    tgt_k = torch.randn(B0, T, 19, device=device)
    g_logits = torch.randn(B0, 15, device=device)
    s_logits = torch.randn(B0, 3, device=device)
    g_lab = torch.zeros(B0, dtype=torch.long, device=device)
    s_lab = torch.zeros(B0, dtype=torch.long, device=device)
    total, comp = compute_total_loss(
        pred_k,
        tgt_k,
        g_logits,
        g_lab,
        s_logits,
        s_lab,
        model_rdm=model_rdm.detach(),
        eeg_rdm=target.detach(),
        brain_mode="bridge",
        loss_weights={"kin": 1.0, "gesture": 1.0, "skill": 0.5, "brain": 0.01, "control": 0.01},
    )
    results["modes_tested"].append("compute_total_loss_bridge")
    results["loss_values"]["total_loss_bridge"] = float(total.detach())
    results["loss_values"]["brain_rsa_component"] = float(comp["brain_rsa"].detach())

    total_eye, comp_eye = compute_total_loss(
        pred_k,
        tgt_k,
        g_logits,
        g_lab,
        s_logits,
        s_lab,
        model_rdm=model_rdm.detach(),
        eeg_rdm=target.detach(),
        brain_mode="eye",
        loss_weights={"kin": 1.0, "gesture": 1.0, "skill": 0.5, "brain": 0.01, "control": 0.01},
    )
    results["modes_tested"].append("compute_total_loss_brain_mode_eye")
    results["loss_values"]["total_loss_eye_mode"] = float(total_eye.detach())
    results["brain_mode_eye_ok"] = bool(torch.isfinite(total_eye).item())

    task_l = torch.tensor([0, 0, 1, 2, 2, 0], device=device)
    feat = torch.randn(len(task_l), D, device=device, requires_grad=True)
    m_eye = compute_task_centroid_rdm(feat, task_l)
    t_eye = torch.ones(3, 3, device=device) * 0.5
    loss_eye = eye_rsa_loss(m_eye, t_eye)
    loss_eye.backward()
    results["modes_tested"].append("eye_task_centroid_rsa")
    results["loss_values"]["eye_rsa"] = float(loss_eye.detach())
    results["backward_compat_eye_ok"] = True

    try:
        import yaml
    except ImportError:
        yaml = None
        warnings.append("PyYAML not installed; skipped config_load checks")

    if yaml is not None:
        for name in (
            "bridge_eye_baseline.yaml",
            "bridge_eeg_rdm.yaml",
            "bridge_joint_eye_eeg.yaml",
        ):
            p = SRC / "configs" / name
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    yaml.safe_load(f)
                results["modes_tested"].append(f"config_load:{name}")

    return results


def main() -> None:
    results = run_checks()
    md = [
        "# Phase 4 ViT regularizer test report",
        "",
        "## Artifact sources",
        "",
        "- " + "\n- ".join(results.get("artifact_source", []) or ["(none)"]),
        "",
        "## Modes tested",
        "",
        "- " + "\n- ".join(results["modes_tested"]),
        "",
        "## Transfer constraints checked",
        "",
        "```json",
        json.dumps(results.get("transfer_constraints", {}), indent=2),
        "```",
        "",
        "## Manifest checks",
        "",
        "```json",
        json.dumps(results.get("manifest_checks", {}), indent=2),
        "```",
        "",
        "## Target RDM shapes",
        "",
        "```json",
        json.dumps(results.get("target_shapes", {}), indent=2),
        "```",
        "",
        "## Loss values",
        "",
        "```json",
        json.dumps(results.get("loss_values", {}), indent=2),
        "```",
        "",
        "## brain_mode=eye (total loss path)",
        "",
        f"- OK: **{results.get('brain_mode_eye_ok', False)}**",
        "",
        "## Backward compatibility (eye-only centroid RSA)",
        "",
        f"- Passed: **{results.get('backward_compat_eye_ok', False)}**",
        "",
        "## Warnings",
        "",
    ]
    for w in results.get("warnings", []):
        md.append(f"- {w}")
    if not results.get("warnings"):
        md.append("- None")
    _write_reports(results, md)
    print("Wrote reports/eeg_eye_bridge/phase4/test_report.{json,md}")


def test_phase4_bridge_and_eye_paths() -> None:
    r = run_checks()
    assert r.get("backward_compat_eye_ok"), "eye-only centroid RSA path must stay valid"
    assert r.get("brain_mode_eye_ok"), "compute_total_loss(..., brain_mode='eye') must stay finite"


if __name__ == "__main__":
    main()
