"""Phase 3: mapping-aware RDM construction for the EEG–eye bridge."""

from __future__ import annotations

from .io import load_rdm_pickle, write_manifest, write_rdm_pickle
from .paths import default_cache_root, phase3_manifest_path, phase3_rdms_dir
from .pipeline import build_all_candidate_rdms, run_phase3_pipeline, write_phase3_outputs
from .schemas import RDMArtifact, TrialRecord, validate_rdm_dict

__all__ = [
    "build_all_candidate_rdms",
    "run_phase3_pipeline",
    "write_phase3_outputs",
    "write_rdm_pickle",
    "load_rdm_pickle",
    "write_manifest",
    "default_cache_root",
    "phase3_manifest_path",
    "phase3_rdms_dir",
    "RDMArtifact",
    "TrialRecord",
    "validate_rdm_dict",
]
