"""Filesystem audit for phase caches and reports."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths


@dataclass
class PhaseAudit:
    phase: int
    cache_dir: str
    reports_dir: str
    cache_exists: bool
    reports_exist: bool
    files: Dict[str, Any] = field(default_factory=dict)


def _safe_glob_count(d: Path, pattern: str) -> int:
    if not d.is_dir():
        return 0
    return len(list(d.glob(pattern)))


def audit_phase(phase: int, repo_root: Optional[Path] = None) -> PhaseAudit:
    root = repo_root or paths.repo_root()
    cache_dir = root / "cache" / "eeg_eye_bridge" / f"phase{phase}"
    reports_dir = root / "reports" / "eeg_eye_bridge" / f"phase{phase}"
    pa = PhaseAudit(
        phase=phase,
        cache_dir=str(cache_dir),
        reports_dir=str(reports_dir),
        cache_exists=cache_dir.is_dir(),
        reports_exist=reports_dir.is_dir(),
    )
    files: Dict[str, Any] = {}
    if phase == 1:
        files["manifest_json"] = (cache_dir / paths.PHASE1_MANIFEST).is_file()
        trials = cache_dir / paths.PHASE1_TRIALS_DIR
        files["trials_pkl_count"] = _safe_glob_count(trials, "*.pkl")
        files["family_summaries_pkl"] = (cache_dir / paths.PHASE1_FAMILY_SUMMARIES).is_file()
    elif phase == 2:
        es = cache_dir / paths.PHASE2_EYE_SUMMARIES
        files["eye_summary_pkl_count"] = _safe_glob_count(es, "*.pkl")
        files["eye_consistency_scores_pkl"] = (cache_dir / paths.PHASE2_CONSISTENCY).is_file()
        files["selected_representations_json"] = (cache_dir / paths.PHASE2_SELECTED).is_file()
    elif phase == 3:
        rdms = cache_dir / paths.PHASE3_RDMS_DIR
        files["rdm_pkl_count"] = _safe_glob_count(rdms, "*.pkl")
        files["rdm_manifest_json"] = (cache_dir / paths.PHASE3_MANIFEST).is_file()
    elif phase == 4:
        configs = list((root / "src" / "configs").glob("bridge_*.yaml"))
        files["bridge_yaml_count"] = len(configs)
        files["bridge_yaml_names"] = [p.name for p in configs]
    pa.files = files
    if pa.reports_exist:
        json_rep = list(reports_dir.glob("test_report.json"))
        md_rep = list(reports_dir.glob("test_report.md"))
        pa.files["test_report_json"] = len(json_rep) > 0
        pa.files["test_report_md"] = len(md_rep) > 0
    return pa


def audit_all_phases(repo_root: Optional[Path] = None) -> List[PhaseAudit]:
    return [audit_phase(n, repo_root) for n in range(1, 5)]


def summarize_audits(audits: List[PhaseAudit]) -> Dict[str, Any]:
    detected: List[int] = []
    for a in audits:
        if a.cache_exists or a.reports_exist:
            detected.append(a.phase)
    return {
        "phases_with_any_artifacts": detected,
        "by_phase": [
            {
                "phase": a.phase,
                "cache_dir": a.cache_dir,
                "reports_dir": a.reports_dir,
                "cache_exists": a.cache_exists,
                "reports_exist": a.reports_exist,
                "files": a.files,
            }
            for a in audits
        ],
    }
