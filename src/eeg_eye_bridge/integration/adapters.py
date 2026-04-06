"""
Loaders and name normalization for bridge artifacts.

Phase 3 contract (manifest + rdms/*.pkl) differs from legacy BrainRDM layout
({trial_id}_rdm.pkl under eeg_rdm_cache_dir). Training integration should use
this module or extend BrainRDM explicitly — do not assume timestamp alignment
across datasets.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def normalize_task_family(name: str) -> str:
    """Lightweight canonicalization for comparison keys."""
    return str(name).strip().lower().replace(" ", "_")


def load_phase3_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_recommended_candidate(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = manifest.get("candidates")
    if isinstance(candidates, list) and candidates:
        for c in candidates:
            if isinstance(c, dict) and c.get("recommended") is True:
                return c
        c0 = candidates[0]
        return c0 if isinstance(c0, dict) else None

    rdms = manifest.get("rdms")
    if isinstance(rdms, dict) and rdms:
        order = manifest.get("recommended_order") or []
        for name in order:
            entry = rdms.get(name)
            if isinstance(entry, dict):
                out = dict(entry)
                out.setdefault("rdm_name", name)
                return out
        name, entry = next(iter(rdms.items()))
        if isinstance(entry, dict):
            out = dict(entry)
            out.setdefault("rdm_name", name)
            return out
    return None


def load_phase3_rdm_artifact(
    cache_phase3: Path,
    rdm_name: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> Tuple[Dict[str, Any], Path]:
    """
    Load one Phase 3 RDM pickle.

    If ``rdm_name`` is None, reads ``manifest_path`` (default: cache_phase3/rdm_manifest.json)
    and picks the recommended (or first) candidate.
    """
    mpath = manifest_path or (cache_phase3 / "rdm_manifest.json")
    manifest = load_phase3_manifest(mpath)
    if rdm_name is None:
        cand = pick_recommended_candidate(manifest)
        if cand is None:
            raise FileNotFoundError("No candidates in RDM manifest")
        rdm_name = cand.get("rdm_name") or cand.get("name")
        if not rdm_name:
            raise FileNotFoundError("Manifest candidate missing rdm_name")
        rel = (
            cand.get("path")
            or cand.get("relative_path")
            or f"rdms/{rdm_name}.pkl"
        )
    else:
        rel = f"rdms/{rdm_name}.pkl"
    rdm_path = cache_phase3 / rel
    if not rdm_path.is_file():
        rdm_path = cache_phase3 / "rdms" / f"{rdm_name}.pkl"
    if not rdm_path.is_file():
        raise FileNotFoundError(f"RDM file not found for {rdm_name}: {rdm_path}")
    with open(rdm_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError("RDM artifact must be a dict")
    return obj, rdm_path


def rdm_matrix_to_torch(obj: Dict[str, Any]) -> torch.Tensor:
    m = obj.get("matrix")
    if m is None:
        raise ValueError("RDM object missing 'matrix'")
    return torch.from_numpy(np.asarray(m, dtype=np.float32))


def describe_legacy_vs_manifest() -> str:
    return (
        "Legacy BrainRDM.load_eeg_rdm expects cache_dir/{trial_id}_rdm.pkl with key 'rdm'. "
        "Phase 3 contract uses cache/eeg_eye_bridge/phase3/rdm_manifest.json plus "
        "rdms/{rdm_name}.pkl with keys including matrix, unit_labels, unit_type."
    )
