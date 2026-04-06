"""
Simulator task ID → coarse abstractions and JIGSAWS transfer bridge.

Bridge categories (design doc): ``needle_control``, ``needle_driving``, ``other_nontransfer``.
Knot tying on JIGSAWS has no true simulator analogue in this dataset; we flag low
transfer confidence rather than silently matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Task IDs from Eye/Table1.csv (simulator modules)
TASK_FAMILY_BY_ID: dict[int, str] = {
    1: "endowrist_manipulation_1",
    2: "endowrist_manipulation_1",
    3: "endowrist_manipulation_1",
    4: "endowrist_manipulation_2",
    5: "endowrist_manipulation_2",
    6: "endowrist_manipulation_2",
    7: "endowrist_manipulation_2",
    8: "endowrist_manipulation_2",
    9: "camera_and_clutching",
    10: "camera_and_clutching",
    11: "camera_and_clutching",
    12: "camera_and_clutching",
    13: "camera_and_clutching",
    14: "camera_and_clutching",
    15: "needle_control",
    16: "needle_control",
    17: "needle_driving",
    18: "needle_driving",
    19: "needle_driving",
    20: "needle_driving",
    21: "needle_driving",
    22: "needle_driving",
    23: "energy_and_dissection",
    24: "energy_and_dissection",
    25: "energy_and_dissection",
    26: "energy_and_dissection",
    27: "energy_and_dissection",
}

# Finer subskill used for RDM units; aligns with transfer families.
SUBSKILL_FAMILY_BY_ID: dict[int, str] = {
    15: "needle_control",
    16: "needle_control",
    17: "needle_driving",
    18: "needle_driving",
    19: "needle_driving",
    20: "needle_driving",
    21: "needle_driving",
    22: "needle_driving",
}
for tid in range(1, 28):
    if tid not in SUBSKILL_FAMILY_BY_ID:
        SUBSKILL_FAMILY_BY_ID[tid] = "other"


def transfer_bridge(task_id: int) -> str:
    """Return ``needle_control``, ``needle_driving``, or ``other_nontransfer``."""
    if task_id in (15, 16):
        return "needle_control"
    if 17 <= task_id <= 22:
        return "needle_driving"
    return "other_nontransfer"


def task_family(task_id: int) -> str:
    return TASK_FAMILY_BY_ID.get(int(task_id), "unknown")


def subskill_family(task_id: int) -> str:
    return SUBSKILL_FAMILY_BY_ID.get(int(task_id), "other")


def jigsaws_transfer_notes_for_bridge(bridge: str) -> str:
    if bridge == "needle_control":
        return "Maps to JIGSAWS needle handling / positioning; moderate transfer plausibility."
    if bridge == "needle_driving":
        return "Maps to suturing and needle-passing families; strongest simulator bridge for JIGSAWS."
    return "Weak transfer to JIGSAWS task geometry; use as coarse prior only."


@dataclass(frozen=True)
class JigsawMappingFlags:
    """Explicit caution flags for JIGSAWS alignment (no silent knot-tying match)."""

    knot_tying_is_low_confidence: bool = True
    knot_tying_note: str = (
        "JIGSAWS knot tying has no direct analogue in the simulator task list; "
        "do not treat simulator 'other' tasks as knot-tying proxies."
    )


def performance_tier_from_score(score: float | None) -> str:
    """Coarse bins on 0–100 performance score."""
    if score is None:
        return "unknown"
    s = float(score)
    if s >= 85:
        return "high"
    if s >= 65:
        return "mid"
    return "low"


def transfer_plausibility_score(
    bridge: str,
    *,
    rdm_includes_eeg: bool = False,
) -> float:
    """
    Rule-based scalar in [0, 1] for manifest ranking (higher = safer coarse target).

    Needle-driving families rank highest; EEG-heavy targets are slightly downweighted
    when knot-tying caution applies (handled separately in manifest warnings).
    """
    base = {
        "needle_driving": 0.95,
        "needle_control": 0.85,
        "other_nontransfer": 0.45,
    }.get(bridge, 0.4)
    if rdm_includes_eeg:
        base *= 0.92
    return float(min(1.0, max(0.0, base)))


def load_performance_lookup(
    csv_path: str,
) -> Mapping[tuple[int, int, int], float]:
    """
    Build (subject_id, task_id, try) -> performance score from PerformanceScores.csv.

    Column names: Subject ID, Task ID, Try, Performance(out of 100)
    """
    import csv
    from pathlib import Path

    path = Path(csv_path)
    if not path.exists():
        return {}

    out: dict[tuple[int, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                sid = int(str(row.get("Subject ID", "")).strip())
                tid = int(str(row.get("Task ID", "")).strip())
                tr = int(str(row.get("Try", "")).strip())
                perf_raw = row.get("Performance(out of 100)", "")
                perf = float(str(perf_raw).strip())
            except (TypeError, ValueError):
                continue
            out[(sid, tid, tr)] = perf
    return out


def parse_trial_id_parts(trial_id: str) -> tuple[int | None, int | None, int | None]:
    """
    Parse ``{subject}_{task}_{try}`` style IDs (e.g. ``9_15_1``).

    Returns (subject_id, task_id, try) with None if parsing fails.
    """
    parts = trial_id.strip().split("_")
    if len(parts) < 3:
        return None, None, None
    try:
        subj = int(parts[0])
        task = int(parts[1])
        tr = int(parts[2])
        return subj, task, tr
    except ValueError:
        return None, None, None
