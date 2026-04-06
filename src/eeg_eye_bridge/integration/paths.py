"""Canonical paths for EEG–eye bridge caches and reports."""
from pathlib import Path


def repo_root() -> Path:
    """Project root: `src/eeg_eye_bridge/integration/` -> parents[3]."""
    return Path(__file__).resolve().parents[3]


def cache_root() -> Path:
    return repo_root() / "cache" / "eeg_eye_bridge"


def reports_root() -> Path:
    return repo_root() / "reports" / "eeg_eye_bridge"


def cache_phase(n: int) -> Path:
    return cache_root() / f"phase{n}"


def reports_phase(n: int) -> Path:
    return reports_root() / f"phase{n}"


# Expected filenames (contract from phase prompts)
PHASE1_MANIFEST = "manifest.json"
PHASE1_TRIALS_DIR = "trials"
PHASE1_FAMILY_SUMMARIES = "family_summaries.pkl"

PHASE2_EYE_SUMMARIES = "eye_summaries"
PHASE2_CONSISTENCY = "eye_consistency_scores.pkl"
PHASE2_SELECTED = "selected_representations.json"

PHASE3_RDMS_DIR = "rdms"
PHASE3_MANIFEST = "rdm_manifest.json"
