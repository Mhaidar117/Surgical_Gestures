"""Write RDM pickles and manifest JSON."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

from .schemas import RDMArtifact, required_rdm_keys


def write_rdm_pickle(artifact: RDMArtifact, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.as_dict()
    for k in required_rdm_keys():
        if k not in payload:
            raise KeyError(k)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_rdm_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
