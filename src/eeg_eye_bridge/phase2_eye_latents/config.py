"""Paths and runtime flags for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Phase2Config:
    """Default layout matches Docs/Prompts/phase2_eye_consistent_latents_agent_prompt.md."""

    repo_root: Path
    phase1_dir: Path = field(init=False)
    phase2_cache_dir: Path = field(init=False)
    eye_root: Path = field(init=False)
    table1_path: Optional[Path] = None

    subset: Optional[int] = None
    trial_ids: Optional[List[str]] = None
    debug: bool = False

    hmm_states: int = 5
    window_samples: int = 25  # 500 ms at 50 Hz
    window_stride: int = 5

    def __post_init__(self) -> None:
        self.repo_root = Path(self.repo_root).resolve()
        self.phase1_dir = self.repo_root / "cache" / "eeg_eye_bridge" / "phase1"
        self.phase2_cache_dir = self.repo_root / "cache" / "eeg_eye_bridge" / "phase2"
        self.eye_root = self.repo_root / "Eye" / "EYE"
        if self.table1_path is None:
            p = self.repo_root / "Eye" / "Table1.csv"
            self.table1_path = p if p.exists() else None
