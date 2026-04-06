"""
Load Phase 3 coarse RDM artifacts for ViT regularization.

Expected layout (configurable base directory):

- ``rdm_manifest.json`` at ``base_dir / rdm_manifest.json`` (or path given in config).
- Pickles under ``base_dir / rdms / {name}.pkl`` or paths relative to manifest's ``base_dir``.

Manifest formats supported:

1. **Dict ``rdms``:** keys are target names; values are objects with
   ``relative_path`` (or ``path``) to the ``.pkl`` file.
2. **List ``entries``:** each item has ``name`` and ``relative_path`` (or ``path``).

Each pickle file should load to a dict with:

- ``unit_type`` (str): e.g. ``task_family``, ``subskill_family``.
- ``unit_labels`` (list[str]): ordered row/column semantics; length must equal ``K``.
- ``matrix`` (ndarray): K×K RDM (or use ``rdm`` as alias for the same array).
- ``rdm_type`` (str): e.g. ``euclidean``, ``correlation``.

Normalization: if only ``rdm`` is present, it is used as ``matrix``.
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class BridgeRDMTarget:
    """Resolved Phase 3 RDM target for training."""

    name: str
    unit_type: str
    unit_labels: List[str]
    matrix: torch.Tensor  # (K, K), float32
    rdm_type: str
    source_path: Path = field(default_factory=Path)

    @property
    def num_groups(self) -> int:
        return len(self.unit_labels)


def _parse_manifest_entries(
    data: Dict[str, Any], manifest_dir: Path
) -> List[Tuple[str, Path]]:
    """Return list of (name, absolute_path_to_pkl)."""
    out: List[Tuple[str, Path]] = []
    if "rdms" in data and isinstance(data["rdms"], dict):
        for name, meta in data["rdms"].items():
            if not isinstance(meta, dict):
                continue
            rel = meta.get("relative_path") or meta.get("path")
            if not rel:
                continue
            out.append((name, (manifest_dir / rel).resolve()))
    if "entries" in data and isinstance(data["entries"], list):
        for entry in data["entries"]:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            rel = entry.get("relative_path") or entry.get("path")
            if not name or not rel:
                continue
            out.append((name, (manifest_dir / rel).resolve()))
    return out


def _normalize_pickle_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Accept ``matrix`` or ``rdm`` key for the K×K array."""
    arr = raw.get("matrix")
    if arr is None:
        arr = raw.get("rdm")
    if arr is None:
        raise KeyError("Pickle must contain 'matrix' or 'rdm'")
    raw = {**raw, "matrix": arr}
    return raw


def load_pickle_target(pkl_path: Path, name: str = "") -> BridgeRDMTarget:
    """Load a single Phase 3-style .pkl file."""
    pkl_path = Path(pkl_path).resolve()
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict in {pkl_path}, got {type(raw)}")
    raw = _normalize_pickle_dict(raw)
    mat = np.asarray(raw["matrix"], dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"matrix must be square K×K, got shape {mat.shape}")
    labels = list(raw["unit_labels"])
    if len(labels) != mat.shape[0]:
        raise ValueError(
            f"len(unit_labels)={len(labels)} != matrix size {mat.shape[0]}"
        )
    unit_type = str(raw.get("unit_type", "unknown"))
    rdm_type = str(raw.get("rdm_type", "unknown"))
    return BridgeRDMTarget(
        name=name or pkl_path.stem,
        unit_type=unit_type,
        unit_labels=labels,
        matrix=torch.from_numpy(mat.astype(np.float32)),
        rdm_type=rdm_type,
        source_path=pkl_path,
    )


def load_bridge_target_from_manifest(
    manifest_path: Union[str, Path],
    target_key: str,
    *,
    strict: bool = True,
) -> BridgeRDMTarget:
    """
    Load one named target from ``rdm_manifest.json`` and its pickle.

    Args:
        manifest_path: Path to ``rdm_manifest.json``.
        target_key: Key in ``rdms`` or ``name`` in ``entries``.
        strict: If True, raise when missing; if False, warn (caller should avoid training).

    Returns:
        BridgeRDMTarget with tensor matrix on CPU (move to device in training).
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = (Path.cwd() / manifest_path).resolve()
    manifest_dir = manifest_path.parent
    if not manifest_path.is_file():
        msg = f"Manifest not found: {manifest_path}"
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg)
        raise FileNotFoundError(msg)

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = _parse_manifest_entries(data, manifest_dir)
    name_to_path = {n: p for n, p in entries}
    if target_key not in name_to_path:
        raise KeyError(
            f"target_key '{target_key}' not in manifest. Available: {list(name_to_path)}"
        )
    pkl_path = name_to_path[target_key]
    if not pkl_path.is_file():
        msg = f"RDM pickle missing: {pkl_path}"
        if strict:
            raise FileNotFoundError(msg)
        warnings.warn(msg)
        raise FileNotFoundError(msg)
    return load_pickle_target(pkl_path, name=target_key)


def align_bridge_target_to_jigsaws_task_family(target: BridgeRDMTarget) -> BridgeRDMTarget:
    """
    Permute rows/columns so ``unit_labels`` match JIGSAWS task order
    (Suturing, Needle_Passing, Knot_Tying) for K=3 task-family targets.

    If alignment fails (e.g. non-standard names), raises ``ValueError``.
    """
    if target.num_groups != 3:
        return target
    from eeg_eye_bridge.phase4_vit.label_grouping import (
        JIGSAWS_TASK_ORDER,
        permutation_aligning_labels_to_jigsaws,
    )

    perm = permutation_aligning_labels_to_jigsaws(target.unit_labels)
    idx = torch.tensor(perm, dtype=torch.long)
    m = target.matrix
    m = m[idx][:, idx]
    return BridgeRDMTarget(
        name=target.name,
        unit_type=target.unit_type,
        unit_labels=list(JIGSAWS_TASK_ORDER),
        matrix=m,
        rdm_type=target.rdm_type,
        source_path=target.source_path,
    )
