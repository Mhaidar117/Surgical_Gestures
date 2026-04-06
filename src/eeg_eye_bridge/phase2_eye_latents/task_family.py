"""Map simulator task_id to coarse family (Docs/eeg_jigsaws_task_mapping.md)."""

from __future__ import annotations

from pathlib import Path


def task_family_for_task_id(task_id: int) -> str:
    """
    needle_control: 15, 16
    needle_driving: 17-22
    other_nontransfer: 1-14, 23-27
    """
    if task_id in (15, 16):
        return "needle_control"
    if 17 <= task_id <= 22:
        return "needle_driving"
    return "other_nontransfer"


def parse_trial_id(trial_id: str) -> tuple[int, int, int]:
    """
    Parse '{participant}_{task}_{trial_index}' e.g. '10_17_1'.
    Returns (participant_id, task_id, trial_index).
    """
    parts = trial_id.strip().split("_")
    if len(parts) != 3:
        raise ValueError(f"Expected trial_id like '10_17_1', got {trial_id!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def eye_csv_path(repo_root: Path | str, trial_id: str) -> Path:
    return Path(repo_root) / "Eye" / "EYE" / f"{trial_id}.csv"
