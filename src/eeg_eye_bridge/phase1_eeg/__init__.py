"""
Phase 1: simulator EEG loading, windowing, representation models, and cache export.

Later phases consume artifacts under ``cache/eeg_eye_bridge/phase1/``; see CLAUDE.md (Phase 1).
"""

from .metadata import (
    TaskFamily,
    load_performance_scores,
    load_task_id_to_name,
    normalize_edf_filename,
    task_family_for_task_id,
)
from .export import (
    CONTRACT_VERSION,
    export_family_summaries,
    export_manifest,
    export_trial_pickle,
    phase1_cache_root,
)

__all__ = [
    "TaskFamily",
    "load_performance_scores",
    "load_task_id_to_name",
    "normalize_edf_filename",
    "task_family_for_task_id",
    "CONTRACT_VERSION",
    "export_family_summaries",
    "export_manifest",
    "export_trial_pickle",
    "phase1_cache_root",
]
