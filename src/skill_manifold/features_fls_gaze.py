"""Per-trial gaze feature builder for the NIBIB-RPCCC-FLS dataset.

Each FLS trial is one Tobii Pro CSV under
``data/laparoscopic-surgery-fls-tasks/EYE_FLS/{subject}_{task}_{try}.csv``
with a header row and the same 20-column layout as the existing RAS Eye
files. We reuse the existing eye loader (with `has_header=True`) and
summarizer to produce an 18-d eye-summary vector per trial -- byte-for-byte
the same layout as the RAS-side `eye_*` columns in
`features_eeg_eye.feature_column_names()` so the rest of the skill-manifold
machinery (residualization, RDMs, GW) is unaware of the dataset switch.

Layout of the 18-d vector (mirrors `features_eeg_eye._build_eye_vector`):
    occupancy[0..4]              5
    transition_diag[0..4]        5
    mean_dwell                   1
    blink_fraction               1
    event_summary[6 fields]      6
    --------------------------- 18
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Reuse the loader (now header-aware) and the trial summarizer.
from eeg_eye_bridge.phase2_eye_latents.eye_loader import load_eye_csv
from eeg_eye_bridge.phase2_eye_latents.eye_summarize import summarize_eye_trial

# The 18-d vector builder lives next to the RAS feature builder so we get
# byte-identical column layout. Importing the underscore-prefixed helper is
# intentional -- there is exactly one canonical layout for the eye summary
# and we don't want a second copy drifting out of sync.
from skill_manifold.features_eeg_eye import (
    EVENT_FIELDS,
    EYE_DIM,
    _build_eye_vector,
)

log = logging.getLogger(__name__)

# Match the existing RAS pipeline's defaults so that the eye summary lives
# in the same coordinate system on both datasets (5 HMM states, 500 ms
# windows at 50 Hz, stride 5).
DEFAULT_HMM_STATES = 5
DEFAULT_WINDOW_SAMPLES = 25     # 500 ms at 50 Hz
DEFAULT_WINDOW_STRIDE = 5
TOBII_RATE_HZ = 50.0


def fls_data_root(repo_root: Path) -> Path:
    """Return the FLS dataset root inside the repo."""
    return Path(repo_root) / "data" / "laparoscopic-surgery-fls-tasks"


def fls_eye_dir(repo_root: Path) -> Path:
    return fls_data_root(repo_root) / "EYE_FLS"


def fls_eeg_dir(repo_root: Path) -> Path:
    return fls_data_root(repo_root) / "EEG_FLS"


def fls_performance_csv(repo_root: Path) -> Path:
    return fls_data_root(repo_root) / "PerformanceScores.csv"


def load_fls_performance_scores(csv_path: Path) -> pd.DataFrame:
    """Load FLS PerformanceScores.csv and normalize the column names.

    The raw file uses a UTF-8 BOM and quotes filenames with a stray ``'``,
    e.g. ``'10_1_1.edf'``. We strip the quote, derive a `trial_id`, and
    rename columns to lower_snake_case used elsewhere in the skill-manifold
    code (subject_id, task_id, age, dominant_hand, dominant_eye, gender,
    performance, perf_min, perf_max, try_num).
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for col in ("EEG File Name", "Eye File Name"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.strip("'")

    df["trial_id"] = (
        df["Eye File Name"].astype(str).str.replace(r"\.csv$", "", regex=True)
    )
    rename = {
        "Subject ID": "subject_id",
        "Task ID": "task_id",
        "Age (year)": "age",
        "Dominant Hand": "dominant_hand",
        "Dominant Eye": "dominant_eye",
        "Gender(F:Female; M:Male)": "gender",
        "Performance": "performance",
        "Minimum possible score": "perf_min",
        "Maximum Possible score": "perf_max",
        "Try": "try_num",
    }
    df = df.rename(columns=rename)
    keep = [
        "trial_id", "subject_id", "task_id", "try_num",
        "age", "dominant_hand", "dominant_eye", "gender",
        "performance", "perf_min", "perf_max",
        "EEG File Name", "Eye File Name",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def eye_feature_column_names() -> List[str]:
    """Match the eye-only subset of `features_eeg_eye.feature_column_names`."""
    names = [f"eye_occ_{i}" for i in range(5)]
    names += [f"eye_tdiag_{i}" for i in range(5)]
    names += ["eye_mean_dwell", "eye_blink_fraction"]
    names += [f"eye_{k}" for k in EVENT_FIELDS]
    if len(names) != EYE_DIM:
        raise AssertionError(
            f"FLS eye column list length {len(names)} != EYE_DIM {EYE_DIM}")
    return names


def _summarize_one_trial(csv_path: Path) -> np.ndarray:
    """Load one FLS gaze CSV and return its 18-d summary vector."""
    series = load_eye_csv(csv_path, assumed_rate_hz=TOBII_RATE_HZ, has_header=True)
    bundle = summarize_eye_trial(
        gx=series.gaze_x,
        gy=series.gaze_y,
        pupil=series.pupil,
        movement=series.movement_type,
        window_samples=DEFAULT_WINDOW_SAMPLES,
        window_stride=DEFAULT_WINDOW_STRIDE,
        hmm_states=DEFAULT_HMM_STATES,
        seed=0,
    )
    return _build_eye_vector(bundle)


def build_fls_gaze_feature_frame(
    repo_root: Path,
    *,
    scores: pd.DataFrame | None = None,
    eye_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a (n_trials, 18 + metadata) DataFrame for the FLS gaze modality.

    Parameters
    ----------
    repo_root
        Repo root used to resolve the FLS dataset directory.
    scores
        Optional pre-loaded PerformanceScores frame (use this in tests so
        that synthetic fixtures don't have to hit `data/`).
    eye_dir
        Optional override for the EYE_FLS directory.

    Returns
    -------
    DataFrame with:
        metadata cols: trial_id, subject_id, task_id, try_num, age,
            dominant_hand, dominant_eye, gender, performance,
            perf_min, perf_max
        feature cols: 18 columns from `eye_feature_column_names()`

    Trials whose CSV is missing or fails to parse are skipped with a
    warning; the row count of the returned frame may be less than the
    number of rows in PerformanceScores.csv.
    """
    if scores is None:
        scores = load_fls_performance_scores(fls_performance_csv(repo_root))
    if eye_dir is None:
        eye_dir = fls_eye_dir(repo_root)

    feat_cols = eye_feature_column_names()
    rows = []
    for _, r in scores.iterrows():
        tid = r["trial_id"]
        path = Path(eye_dir) / f"{tid}.csv"
        if not path.exists():
            log.warning("missing FLS eye CSV: %s", path)
            continue
        try:
            vec = _summarize_one_trial(path)
        except Exception as e:
            log.warning("skip %s (%s: %s)", tid, type(e).__name__, e)
            continue

        record = {
            "trial_id": tid,
            "subject_id": int(r["subject_id"]),
            "task_id": int(r["task_id"]),
            "try_num": int(r["try_num"]),
            "age": float(r["age"]),
            "dominant_hand": str(r["dominant_hand"]),
            "dominant_eye": str(r.get("dominant_eye", "")),
            "gender": str(r.get("gender", "")),
            "performance": float(r["performance"]),
            "perf_min": float(r.get("perf_min", np.nan)),
            "perf_max": float(r.get("perf_max", np.nan)),
        }
        record.update(dict(zip(feat_cols, vec)))
        rows.append(record)

    return pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)
