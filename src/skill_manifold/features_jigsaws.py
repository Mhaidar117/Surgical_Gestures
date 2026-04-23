"""JIGSAWS per-trial feature extraction.

For each trial we build a scalar feature vector by concatenating:

  - Gesture histogram over the pooled-task gesture set, normalized by trial
    duration (fraction of frames spent in each gesture).
  - Per-arm kinematics summary for each of Slave-Left and Slave-Right
    (12 scalars per arm: trans-speed mean/std/max, rot-speed mean/std/max,
    jerk-mean, path length, economy of motion, gripper mean, gripper std,
    gripper open-rate).
  - Trial duration in frames.

The JIGSAWS kinematics files are 76-dim per row at 30 Hz; columns 39..57 are
Slave-Left and 58..76 are Slave-Right (1-indexed) per the dataset readme.
We only use Slave channels because they are the end-effector that actually
moves in the scene -- masters are the surgeon's hands, further removed from
tissue-side skill.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from skill_manifold.io import (
    JIGSAWS_OSATS_COLUMNS,
    JIGSAWS_TASKS,
    jigsaws_kinematics,
    jigsaws_meta,
    jigsaws_transcription,
)

log = logging.getLogger(__name__)

# Slave-Left and Slave-Right column ranges (0-indexed half-open) within the 76-dim row.
SLAVE_L_SLICE = slice(38, 57)   # 1-indexed cols 39..57 -> 0-indexed 38..56
SLAVE_R_SLICE = slice(57, 76)   # 1-indexed cols 58..76 -> 0-indexed 57..75

# Sub-slices inside a single 19-dim arm block: xyz(3), R(9), trans_vel(3), rot_vel(3), gripper(1)
ARM_XYZ = slice(0, 3)
ARM_TRANS_VEL = slice(12, 15)
ARM_ROT_VEL = slice(15, 18)
ARM_GRIPPER = 18

# Trial IDs end with <letter><digits>, e.g. "Suturing_B001", "Knot_Tying_B001",
# "Needle_Passing_B001". Naive "split on first underscore" breaks the two
# multi-word task names, so we anchor on the trailing _<letter><digits> pattern.
_SURGEON_RE = re.compile(r"_([A-Z])\d+$")


def _parse_surgeon_letter(trial_id: str) -> str:
    """Extract the surgeon letter from a JIGSAWS trial id.

    Returns "?" if the pattern doesn't match -- callers should treat this as
    a parse failure rather than silently conflating unrelated trials.
    """
    m = _SURGEON_RE.search(trial_id)
    return m.group(1) if m else "?"


# Twelve per-arm scalar names, in the order produced below. Exposed for tests.
ARM_FEATURE_NAMES = (
    "trans_speed_mean", "trans_speed_std", "trans_speed_max",
    "rot_speed_mean",   "rot_speed_std",   "rot_speed_max",
    "jerk_mean",
    "path_length",
    "economy_of_motion",
    "gripper_mean", "gripper_std", "gripper_open_rate",
)


@dataclass(frozen=True)
class JigsawsTrial:
    """Parsed meta row for a single JIGSAWS trial."""
    task: str          # "Suturing" | "Knot_Tying" | "Needle_Passing"
    trial_id: str      # e.g. "Suturing_B001"
    surgeon: str       # single letter B..I
    skill: str         # "E" | "I" | "N"
    grs_total: int     # 6..30
    osats: dict        # {axis_name: int}


def parse_meta_file(meta_path: Path, task: str) -> List[JigsawsTrial]:
    """Parse meta_file_<Task>.txt into JigsawsTrial rows.

    The file is whitespace-separated with variable tab-runs; the reliable
    decoding is `split()` on each line.
    """
    trials: List[JigsawsTrial] = []
    for lineno, line in enumerate(meta_path.read_text().splitlines(), start=1):
        parts = line.split()
        if len(parts) < 9:
            continue
        trial_id = parts[0]
        skill = parts[1]
        try:
            grs_total = int(parts[2])
            osats_vals = [int(x) for x in parts[3:9]]
        except ValueError as e:
            log.warning("meta parse error %s:%d (%s)", meta_path, lineno, e)
            continue
        # Surgeon letter is the capital letter in the trailing _<letter><digits>.
        # NOTE: a naive split on the first underscore breaks for the two
        # multi-word task names -- "Knot_Tying_B001" and "Needle_Passing_B001"
        # -- so we anchor on the trial-id suffix instead.
        surgeon = _parse_surgeon_letter(trial_id)
        if surgeon == "?":
            log.warning("could not parse surgeon letter from %s", trial_id)
        osats = dict(zip(JIGSAWS_OSATS_COLUMNS, osats_vals))
        trials.append(JigsawsTrial(
            task=task, trial_id=trial_id, surgeon=surgeon,
            skill=skill, grs_total=grs_total, osats=osats,
        ))
    return trials


def load_gesture_transcription(path: Path) -> List[tuple[int, int, str]]:
    """Return rows of (start_frame, end_frame, gesture_id) inclusive."""
    out: List[tuple[int, int, str]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            start = int(parts[0]); end = int(parts[1])
        except ValueError:
            continue
        gid = parts[2].strip()
        out.append((start, end, gid))
    return out


def gesture_histogram(rows: Sequence[tuple[int, int, str]],
                      gesture_pool: Sequence[str],
                      n_frames: int) -> np.ndarray:
    """Fraction of frames spent in each gesture in `gesture_pool`.

    We divide by `n_frames` (the length of the kinematics trace), not by the
    sum of transcribed gesture durations, so an unlabeled head/tail does not
    inflate the histogram.
    """
    if n_frames <= 0:
        return np.zeros(len(gesture_pool), dtype=np.float64)
    counts = {g: 0 for g in gesture_pool}
    for start, end, gid in rows:
        if gid not in counts:
            continue
        # transcription frames are 1-indexed inclusive; convert to duration
        counts[gid] += max(0, end - start + 1)
    return np.array([counts[g] / n_frames for g in gesture_pool], dtype=np.float64)


def _arm_kinematics_features(arm_block: np.ndarray,
                             gripper_open_eps: float) -> np.ndarray:
    """Return 12 scalars for a single arm's (T, 19) trace.

    See ARM_FEATURE_NAMES for the ordering.
    """
    if arm_block.ndim != 2 or arm_block.shape[1] != 19:
        raise ValueError(f"arm_block must be (T,19); got {arm_block.shape}")
    if arm_block.shape[0] < 3:
        # Not enough frames to take two finite differences; return zeros.
        return np.zeros(len(ARM_FEATURE_NAMES), dtype=np.float64)

    xyz = arm_block[:, ARM_XYZ]                 # (T, 3)
    tvel = arm_block[:, ARM_TRANS_VEL]          # (T, 3)
    rvel = arm_block[:, ARM_ROT_VEL]            # (T, 3)
    gripper = arm_block[:, ARM_GRIPPER]         # (T,)

    tspeed = np.linalg.norm(tvel, axis=1)
    rspeed = np.linalg.norm(rvel, axis=1)

    # |jerk| = finite diff of |accel|; |accel| = finite diff of trans_vel.
    # We want a scalar summary so we use the mean of |jerk|.
    accel = np.diff(tvel, axis=0)
    jerk = np.diff(accel, axis=0)
    jerk_mag = np.linalg.norm(jerk, axis=1)
    jerk_mean = float(jerk_mag.mean()) if jerk_mag.size else 0.0

    # Path length: sum of frame-to-frame |dxyz|.
    dxyz = np.diff(xyz, axis=0)
    step_len = np.linalg.norm(dxyz, axis=1)
    path_length = float(step_len.sum())

    # Economy of motion: path_length / straight-line displacement.
    disp = float(np.linalg.norm(xyz[-1] - xyz[0]))
    economy = path_length / max(disp, 1e-9)

    # Gripper open-rate: fraction of frames where d(gripper)/dt > eps.
    dg = np.diff(gripper)
    open_rate = float((dg > gripper_open_eps).sum()) / max(1, dg.size)

    return np.array([
        float(tspeed.mean()), float(tspeed.std()), float(tspeed.max()),
        float(rspeed.mean()), float(rspeed.std()), float(rspeed.max()),
        jerk_mean,
        path_length,
        economy,
        float(gripper.mean()), float(gripper.std()),
        open_rate,
    ], dtype=np.float64)


def extract_trial_features(trial: JigsawsTrial,
                           data_root: Path,
                           gesture_pool: Sequence[str],
                           *,
                           gripper_open_eps: float = 1.0e-6,
                           economy_clip: float = 1.0e6) -> np.ndarray | None:
    """Return the per-trial feature vector, or None if any required file is missing.

    Dimensionality: len(gesture_pool) + 2*12 + 1.
    """
    kin_path = jigsaws_kinematics(data_root, trial.task, trial.trial_id)
    trans_path = jigsaws_transcription(data_root, trial.task, trial.trial_id)
    if not kin_path.exists() or not trans_path.exists():
        return None

    kin = np.loadtxt(str(kin_path))
    if kin.ndim != 2 or kin.shape[1] < 76:
        return None
    n_frames = kin.shape[0]

    trans_rows = load_gesture_transcription(trans_path)
    hist = gesture_histogram(trans_rows, gesture_pool, n_frames)

    sl = _arm_kinematics_features(kin[:, SLAVE_L_SLICE], gripper_open_eps)
    sr = _arm_kinematics_features(kin[:, SLAVE_R_SLICE], gripper_open_eps)

    # Clip economy_of_motion (it is the 9th per-arm feature, index 8).
    sl[8] = min(sl[8], economy_clip)
    sr[8] = min(sr[8], economy_clip)

    return np.concatenate([hist, sl, sr, [float(n_frames)]]).astype(np.float64)


def feature_column_names(gesture_pool: Sequence[str]) -> List[str]:
    """Column names, same order as extract_trial_features output."""
    names: List[str] = [f"gest_{g}" for g in gesture_pool]
    for side in ("SL", "SR"):
        names.extend(f"{side}_{n}" for n in ARM_FEATURE_NAMES)
    names.append("duration_frames")
    return names


def build_jigsaws_feature_frame(data_root: Path,
                                gesture_pool: Sequence[str],
                                tasks: Iterable[str] = JIGSAWS_TASKS,
                                *,
                                gripper_open_eps: float = 1.0e-6,
                                economy_clip: float = 1.0e6) -> pd.DataFrame:
    """Pool all three JIGSAWS tasks into one DataFrame keyed by trial_id.

    Columns: feature columns + metadata (task, surgeon, skill, grs_total,
    trial_index_within_surgeon_task, osats_*).
    """
    rows = []
    feature_cols = feature_column_names(gesture_pool)
    for task in tasks:
        meta_path = jigsaws_meta(data_root, task)
        if not meta_path.exists():
            log.warning("meta file missing: %s", meta_path)
            continue
        trials = parse_meta_file(meta_path, task)
        for trial in trials:
            vec = extract_trial_features(
                trial, data_root, gesture_pool,
                gripper_open_eps=gripper_open_eps, economy_clip=economy_clip,
            )
            if vec is None:
                log.info("skip %s (missing kin or transcription)", trial.trial_id)
                continue
            record = {"trial_id": trial.trial_id, "task": trial.task,
                      "surgeon": trial.surgeon, "skill": trial.skill,
                      "grs_total": trial.grs_total}
            record.update({f"osats_{k}": v for k, v in trial.osats.items()})
            record.update(dict(zip(feature_cols, vec)))
            rows.append(record)

    df = pd.DataFrame(rows)
    # Stable ordering for deterministic tests.
    df = df.sort_values(["task", "surgeon", "trial_id"]).reset_index(drop=True)
    # Per (surgeon, task) trial index, used as a nuisance during residualization.
    df["trial_index_within_surgeon_task"] = (
        df.groupby(["task", "surgeon"]).cumcount().astype(np.int64)
    )
    return df


def zscore_features(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Z-score the named feature columns in-place. Constant columns -> zeros."""
    out = df.copy()
    for col in feature_cols:
        v = out[col].to_numpy(dtype=np.float64)
        std = v.std()
        if std <= 1e-12:
            out[col] = 0.0
        else:
            out[col] = (v - v.mean()) / std
    return out
