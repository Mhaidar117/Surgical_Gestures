"""
Simulator EEG trial metadata from CSV tables (no JIGSAWS alignment).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class TaskFamily:
    NEEDLE_CONTROL = "needle_control"
    NEEDLE_DRIVING = "needle_driving"
    OTHER_NONTRANSFER = "other_nontransfer"


def normalize_edf_filename(name: str) -> str:
    """Strip quotes/whitespace and ensure ``.edf`` stem for joining."""
    s = name.strip().strip("'").strip('"')
    if s.lower().endswith(".edf"):
        return s[:-4]
    return s


def task_family_for_task_id(task_id: int) -> str:
    if task_id in (15, 16):
        return TaskFamily.NEEDLE_CONTROL
    if 17 <= task_id <= 22:
        return TaskFamily.NEEDLE_DRIVING
    return TaskFamily.OTHER_NONTRANSFER


def load_task_id_to_name(table1_path: Path) -> Dict[int, str]:
    """
    Parse ``Eye/Table1.csv``: map task ID in parentheses to a short task name.
    """
    out: Dict[int, str] = {}
    text = table1_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        # Match last (id) on the line (handles ``Ring and rail 1(7)``)
        matches = list(re.finditer(r"\((\d+)\)", line))
        if not matches:
            continue
        m = matches[-1]
        tid = int(m.group(1))
        prefix = line[: m.start()]
        if "," in prefix:
            name_part = prefix.split(",", 1)[-1]
        else:
            name_part = prefix
        name_part = re.sub(r"^[\s·]+", "", name_part)
        name_part = re.sub(r"\s*\(\d+\)\s*$", "", name_part).strip()
        if name_part:
            out[tid] = name_part
    return out


@dataclass
class TrialMeta:
    trial_id: str
    participant_id: int
    task_id: int
    try_number: int
    performance_score: float
    eeg_filename: str


def load_performance_scores(csv_path: Path) -> Dict[str, TrialMeta]:
    """
    Load ``Eye/PerformanceScores.csv`` keyed by trial_id ``{subject}_{task}_{try}``.
    """
    df = pd.read_csv(csv_path)
    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]
    col_eeg = "EEG File Name"
    col_subj = "Subject ID"
    col_task = "Task ID"
    col_try = "Try"
    col_perf = "Performance(out of 100)"
    keyed: Dict[str, TrialMeta] = {}
    for _, row in df.iterrows():
        raw_name = str(row[col_eeg])
        stem = normalize_edf_filename(raw_name)
        trial_id = stem
        participant_id = int(row[col_subj])
        task_id = int(row[col_task])
        try_number = int(row[col_try])
        performance_score = float(row[col_perf])
        keyed[trial_id] = TrialMeta(
            trial_id=trial_id,
            participant_id=participant_id,
            task_id=task_id,
            try_number=try_number,
            performance_score=performance_score,
            eeg_filename=stem + ".edf",
        )
    return keyed


def resolve_task_name(task_id: int, id_to_name: Dict[int, str]) -> str:
    return id_to_name.get(task_id, f"task_{task_id}")


def list_edf_trials(
    eeg_dir: Path,
    performance_path: Path,
    table1_path: Path,
    max_trials: Optional[int] = None,
    max_participants: Optional[int] = None,
) -> List[Tuple[Path, TrialMeta, str, str]]:
    """
    Return list of (edf_path, TrialMeta, task_name, task_family) for existing EDF files.
    """
    perf = load_performance_scores(performance_path)
    id_to_name = load_task_id_to_name(table1_path)
    pairs: List[Tuple[Path, TrialMeta, str, str]] = []
    if not eeg_dir.is_dir():
        return pairs

    allowed_pids: Optional[set] = None
    if max_participants is not None:
        pids = sorted({m.participant_id for m in perf.values()})
        allowed_pids = set(pids[:max_participants])

    for p in sorted(eeg_dir.glob("*.edf")):
        stem = p.stem
        if stem not in perf:
            continue
        meta = perf[stem]
        if allowed_pids is not None and meta.participant_id not in allowed_pids:
            continue
        name = resolve_task_name(meta.task_id, id_to_name)
        fam = task_family_for_task_id(meta.task_id)
        pairs.append((p, meta, name, fam))
        if max_trials is not None and len(pairs) >= max_trials:
            break
    return pairs
