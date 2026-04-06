"""Default path layout for Phase 1–3 EEG–eye bridge caches."""

from __future__ import annotations

from pathlib import Path


def default_cache_root(repo_root: Path | None = None) -> Path:
    """Project root is parent of `src/` when repo_root is None."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "cache" / "eeg_eye_bridge"


def phase1_dir(cache_root: Path) -> Path:
    return cache_root / "phase1"


def phase2_dir(cache_root: Path) -> Path:
    return cache_root / "phase2"


def phase3_dir(cache_root: Path) -> Path:
    return cache_root / "phase3"


def phase1_manifest(cache_root: Path) -> Path:
    return phase1_dir(cache_root) / "manifest.json"


def phase1_trials_dir(cache_root: Path) -> Path:
    return phase1_dir(cache_root) / "trials"


def phase2_eye_summaries_dir(cache_root: Path) -> Path:
    return phase2_dir(cache_root) / "eye_summaries"


def phase2_consistency_path(cache_root: Path) -> Path:
    return phase2_dir(cache_root) / "eye_consistency_scores.pkl"


def phase2_selected_repr_path(cache_root: Path) -> Path:
    return phase2_dir(cache_root) / "selected_representations.json"


def phase3_rdms_dir(cache_root: Path) -> Path:
    return phase3_dir(cache_root) / "rdms"


def phase3_manifest_path(cache_root: Path) -> Path:
    return phase3_dir(cache_root) / "rdm_manifest.json"
