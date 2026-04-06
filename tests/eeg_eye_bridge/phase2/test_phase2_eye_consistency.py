"""
Phase 2 integration test: real Phase 1 export contract, eye CSVs, Phase 3 eye vector path, reports.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from eeg_eye_bridge.integration.schemas import (  # noqa: E402
    validate_phase1_trial_pkl,
    validate_phase2_eye_summary_pkl,
    validate_phase2_selected,
)
from eeg_eye_bridge.phase1_eeg.export import CONTRACT_VERSION, export_trial_pickle  # noqa: E402
from eeg_eye_bridge.phase2_eye_latents.config import Phase2Config  # noqa: E402
from eeg_eye_bridge.phase2_eye_latents.export_phase2 import run_phase2_pipeline  # noqa: E402
from eeg_eye_bridge.phase3_rdm.loaders import (  # noqa: E402
    SelectedRepresentations,
    _pick_eye_vector,
    load_phase2_selection,
)
from eeg_eye_bridge.phase3_rdm.paths import default_cache_root, phase1_dir  # noqa: E402


def _synthetic_eye_csv(path: Path, n: int = 400, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rows = np.zeros((n, 20), dtype=np.float64)
    rows[:, 0] = np.cumsum(rng.normal(0, 1, n)) + 100
    rows[:, 1] = np.cumsum(rng.normal(0, 1, n)) + 100
    rows[:, 17] = np.clip(3.0 + 0.2 * rng.standard_normal(n), 0.1, None)
    rows[:, 18] = np.clip(3.0 + 0.2 * rng.standard_normal(n), 0.1, None)
    rows[:, 19] = rng.choice([1, 2], size=n, p=[0.55, 0.45])
    np.savetxt(path, rows, delimiter=",")


def _write_phase1_contract_dir(
    p1: Path,
    trial_specs: list[tuple[str, int, str, str]],
    seed: int = 0,
) -> None:
    """Trial pickles matching ``export_trial_pickle`` / integration PHASE1_TRIAL_KEYS."""
    rng = np.random.default_rng(seed)
    (p1 / "trials").mkdir(parents=True, exist_ok=True)
    trials_manifest: list[dict[str, object]] = []
    for trial_id, task_id, task_name, task_family in trial_specs:
        n_win = 8
        d = 12
        window_times = np.linspace(0.0, 4.0, n_win, dtype=np.float64)
        baseline_embeddings = rng.standard_normal((n_win, d)).astype(np.float32)
        pc_embeddings = rng.standard_normal((n_win, d)).astype(np.float32)
        prediction_errors = rng.standard_normal((n_win, 6)).astype(np.float32)
        part_id = int(trial_id.split("_")[0])
        export_trial_pickle(
            p1 / "trials" / f"{trial_id}.pkl",
            trial_id=trial_id,
            participant_id=part_id,
            task_id=task_id,
            task_name=task_name,
            task_family=task_family,
            performance_score=float(70.0 + 10 * rng.random()),
            window_times=window_times,
            baseline_embeddings=baseline_embeddings,
            pc_embeddings=pc_embeddings,
            prediction_errors=prediction_errors,
            contract_version=CONTRACT_VERSION,
        )
        trials_manifest.append({"trial_id": trial_id, "task_id": task_id})
    manifest = {"contract_version": CONTRACT_VERSION, "trials": trials_manifest}
    (p1 / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with open(p1 / "family_summaries.pkl", "wb") as f:
        pickle.dump({task_family: {"n_trials": 1} for *_, task_family in trial_specs}, f)


def _discover_repo_phase1() -> Path | None:
    """Prefer real ``cache/eeg_eye_bridge/phase1`` when it has schema-valid trial pkls."""
    p1 = phase1_dir(default_cache_root(_REPO))
    man = p1 / "manifest.json"
    trials = p1 / "trials"
    if not man.is_file() or not trials.is_dir():
        return None
    pkls = sorted(trials.glob("*.pkl"))
    if not pkls:
        return None
    r = validate_phase1_trial_pkl(pkls[0])
    if not r.ok:
        return None
    return p1


def _setup_eye_csvs(tmp: Path, trial_ids: list[str]) -> Path:
    eye = tmp / "Eye" / "EYE"
    eye.mkdir(parents=True)
    for i, tid in enumerate(trial_ids):
        _synthetic_eye_csv(eye / f"{tid}.csv", n=350 + i * 10, seed=3 + i)
    return eye


def test_phase2_pipeline_writes_artifacts(tmp_path: Path) -> None:
    real_p1 = _discover_repo_phase1()
    use_real = real_p1 is not None
    if use_real:
        phase1 = real_p1
        man = json.loads((phase1 / "manifest.json").read_text(encoding="utf-8"))
        rows = man.get("trials") or []
        trial_ids: list[str] = []
        for r in rows:
            if isinstance(r, dict) and "trial_id" in r:
                trial_ids.append(str(r["trial_id"]))
            elif isinstance(r, str):
                trial_ids.append(r)
        trial_ids = trial_ids[:4]
        if not trial_ids:
            use_real = False
    if not use_real:
        trial_specs = [
            ("99_17_1", 17, "task17", "needle_driving"),
            ("99_18_1", 18, "task18", "knot_tying"),
        ]
        _write_phase1_contract_dir(tmp_path / "phase1_synth", trial_specs)
        phase1 = tmp_path / "phase1_synth"
        trial_ids = [t[0] for t in trial_specs]

    _setup_eye_csvs(tmp_path, trial_ids)
    out = tmp_path / "phase2_out"
    cfg = Phase2Config(repo_root=tmp_path)
    cfg.phase1_dir = phase1
    cfg.phase2_cache_dir = out
    cfg.eye_root = tmp_path / "Eye" / "EYE"
    if use_real:
        cfg.subset = min(4, len(trial_ids))

    payload, warnings = run_phase2_pipeline(cfg, phase1_dir=phase1, out_dir=out)

    first_trial = next(iter(payload.get("scores_by_trial", {}).keys()), None)
    assert first_trial is not None, "expected at least one matched trial"

    assert (out / "eye_summaries" / f"{first_trial}.pkl").exists()
    assert (out / "eye_consistency_scores.pkl").exists()
    assert (out / "selected_representations.json").exists()

    p1_trial = phase1 / "trials" / f"{first_trial}.pkl"
    vr1 = validate_phase1_trial_pkl(p1_trial)
    assert vr1.ok, f"Phase 1 trial contract invalid: {vr1.errors}"

    with open(out / "eye_summaries" / f"{first_trial}.pkl", "rb") as f:
        eye_rec = pickle.load(f)
    for k in (
        "trial_id",
        "task_id",
        "task_family",
        "eye_state_summary",
        "eye_transition_summary",
        "pupil_summary",
        "event_summary",
        "fingerprint",
    ):
        assert k in eye_rec, f"missing eye summary key {k!r}"

    vr2 = validate_phase2_eye_summary_pkl(out / "eye_summaries" / f"{first_trial}.pkl")
    assert vr2.ok, vr2.errors

    sel_path = out / "selected_representations.json"
    vr_sel = validate_phase2_selected(sel_path)
    assert vr_sel.ok, vr_sel.errors
    sel_data = json.loads(sel_path.read_text(encoding="utf-8"))
    assert sel_data.get("eye_vector_key") == "fingerprint"
    assert sel_data.get("eeg_latent_key") == "pc_embeddings"
    assert sel_data.get("eeg_pred_error_key") == "prediction_errors"

    sel = load_phase2_selection(sel_path)
    assert isinstance(sel, SelectedRepresentations)
    vec = _pick_eye_vector(eye_rec, sel)
    assert vec.ndim == 1 and vec.size > 0

    with open(phase1 / "trials" / f"{first_trial}.pkl", "rb") as f:
        p1_obj = pickle.load(f)
    for key in ("baseline_embeddings", "pc_embeddings", "prediction_errors"):
        assert key in p1_obj, f"Phase 1 missing {key}"
        a = np.asarray(p1_obj[key])
        assert a.ndim >= 1 and np.isfinite(a).all()

    with open(out / "eye_consistency_scores.pkl", "rb") as f:
        scores = pickle.load(f)
    assert "scores_by_trial" in scores and "aggregate" in scores
    for rep in ("baseline", "predictive", "prediction_error"):
        assert rep in scores["scores_by_trial"][first_trial]

    sel_json = json.loads((out / "selected_representations.json").read_text(encoding="utf-8"))
    assert "best_combined_representation" in sel_json or "targets" in sel_json

    report_dir = _REPO / "reports" / "eeg_eye_bridge" / "phase2"
    report_dir.mkdir(parents=True, exist_ok=True)
    rep = {
        "artifact_source": "real_phase1_cache" if use_real else "synthetic_phase1_contract",
        "contract_version_checked": CONTRACT_VERSION,
        "phase1_trial_schema_ok": vr1.ok,
        "phase1_validated_keys": sorted(vr1.details.keys()) if vr1.details else [],
        "eye_summary_has_fingerprint": "fingerprint" in eye_rec,
        "phase3_eye_vector_resolvable": True,
        "phase3_eye_vector_dim": int(vec.size),
        "selected_eye_vector_key": sel_data.get("eye_vector_key"),
        "eeg_keys_for_phase3": {
            "latent": sel_data.get("eeg_latent_key"),
            "pred_error": sel_data.get("eeg_pred_error_key"),
        },
        "n_matched_trials": len(payload.get("scores_by_trial", {})),
        "sample_trial_ids": list(payload.get("scores_by_trial", {}).keys())[:5],
        "eye_summary_keys": list(eye_rec.keys()),
        "scores_by_rep": payload.get("aggregate", {}).get("mean_by_rep", {}),
        "warnings": warnings,
    }
    (report_dir / "test_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = (
        f"# Phase 2 test report\n\n"
        f"- Artifact source: **{rep['artifact_source']}**\n"
        f"- Phase 1 trial schema OK: {rep['phase1_trial_schema_ok']}\n"
        f"- Phase 3 consumable eye vector: {rep['phase3_eye_vector_resolvable']} (dim={rep['phase3_eye_vector_dim']})\n"
        f"- Matched trials: {rep['n_matched_trials']}\n"
        f"- Sample trial IDs: {rep['sample_trial_ids']}\n"
        f"- Warnings: {len(warnings)}\n"
    )
    for w in warnings:
        md += f"  - {w}\n"
    (report_dir / "test_report.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
