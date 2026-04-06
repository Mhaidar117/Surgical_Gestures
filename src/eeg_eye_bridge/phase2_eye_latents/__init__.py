"""
Phase 2: eye-consistent latent evaluation (simulator EEG vs eye summaries).

Reads Phase 1 cache artifacts and Eye/EYE CSVs; writes phase2 cache for Phase 3.
"""

from .config import Phase2Config
from .phase1_io import (
    PHASE1_REP_KEYS,
    load_family_summaries,
    load_manifest,
    load_trial_pkl,
    trial_ids_from_manifest,
)
from .export_phase2 import run_phase2_pipeline

__all__ = [
    "Phase2Config",
    "PHASE1_REP_KEYS",
    "load_family_summaries",
    "load_manifest",
    "load_trial_pkl",
    "trial_ids_from_manifest",
    "run_phase2_pipeline",
]
