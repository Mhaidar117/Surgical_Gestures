"""Integration utilities: audit, schemas, synthetic fixtures, Phase 3 loaders."""

from .adapters import (
    describe_legacy_vs_manifest,
    load_phase3_rdm_artifact,
    normalize_task_family,
    pick_recommended_candidate,
    rdm_matrix_to_torch,
)
from .audit import PhaseAudit, audit_all_phases, audit_phase, summarize_audits
from .paths import (
    cache_phase,
    cache_root,
    reports_phase,
    reports_root,
    repo_root,
)
from .schemas import ValidationResult, validate_phase3_manifest, validate_phase3_rdm_pickle

__all__ = [
    "PhaseAudit",
    "ValidationResult",
    "audit_all_phases",
    "audit_phase",
    "cache_phase",
    "cache_root",
    "describe_legacy_vs_manifest",
    "load_phase3_rdm_artifact",
    "normalize_task_family",
    "pick_recommended_candidate",
    "rdm_matrix_to_torch",
    "reports_phase",
    "reports_root",
    "repo_root",
    "summarize_audits",
    "validate_phase3_manifest",
    "validate_phase3_rdm_pickle",
]
