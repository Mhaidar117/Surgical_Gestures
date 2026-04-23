"""Centralized filesystem conventions for the skill-manifold comparison.

All other modules import paths from here so there are no hardcoded locations.
The dataset root (JIGSAWS Gestures/, Eye/, EEG/, cache/) is resolved through
`src/dataset_paths.py`, which supports an env var and a default external root.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any

from dataset_paths import resolve_dataset_root


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "src" / "configs"
REPORTS_DIR = REPO_ROOT / "reports" / "skill_manifold"
PLOTS_DIR = REPORTS_DIR / "plots"


def data_root(cli_value: str | Path | None = None) -> Path:
    """Return the dataset root (parent of Gestures/, Eye/, cache/, ...)."""
    return resolve_dataset_root(cli_value, fallback_repo_root=REPO_ROOT)


def performance_scores_csv(root: Path) -> Path:
    return root / "Eye" / "PerformanceScores.csv"


def phase1_trials_dir(root: Path) -> Path:
    return root / "cache" / "eeg_eye_bridge" / "phase1" / "trials"


def phase1_manifest(root: Path) -> Path:
    return root / "cache" / "eeg_eye_bridge" / "phase1" / "manifest.json"


def phase2_summaries_dir(root: Path) -> Path:
    return root / "cache" / "eeg_eye_bridge" / "phase2" / "eye_summaries"


def jigsaws_task_dir(root: Path, task: str) -> Path:
    return root / "Gestures" / task


def jigsaws_meta(root: Path, task: str) -> Path:
    return jigsaws_task_dir(root, task) / f"meta_file_{task}.txt"


def jigsaws_kinematics(root: Path, task: str, trial_id: str) -> Path:
    return jigsaws_task_dir(root, task) / "kinematics" / "AllGestures" / f"{trial_id}.txt"


def jigsaws_transcription(root: Path, task: str, trial_id: str) -> Path:
    return jigsaws_task_dir(root, task) / "transcriptions" / f"{trial_id}.txt"


def load_config(name: str = "skill_manifold.yaml") -> Dict[str, Any]:
    """Load a YAML config from src/configs/."""
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs() -> tuple[Path, Path]:
    """Create reports/skill_manifold{,/plots}. Returns (reports_dir, plots_dir)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR, PLOTS_DIR


JIGSAWS_TASKS = ("Suturing", "Knot_Tying", "Needle_Passing")
JIGSAWS_OSATS_COLUMNS = (
    "respect_for_tissue",
    "suture_needle_handling",
    "time_and_motion",
    "flow_of_operation",
    "overall_performance",
    "quality_of_final_product",
)
