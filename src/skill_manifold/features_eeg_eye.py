"""EEG/Eye simulator per-trial feature extraction.

We concatenate three components into a single feature row per Try-1 trial:

  1. Phase 1 `baseline_embeddings` (n_windows, 64)  -> mean over windows (64 d).
  2. Phase 1 `pc_embeddings` (n_windows, 64)        -> mean over windows (64 d).
  3. Reconstructed eye-summary vector (18 d) built from the Phase 2 component
     dicts. The cached `fingerprint` field is absent on ~91% of trials, so we
     rebuild a fixed-shape summary from the components that *are* always
     present: occupancy(5) + transition-matrix diagonal(5) + mean_dwell(1)
     + blink_fraction(1) + event_summary(6). Pupil mean/std/cv are mostly
     NaN in the cache and are intentionally excluded.

Metadata columns attached to each row: subject_id, task_id, task_module,
age, dominant_hand, performance.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from skill_manifold.io import (
    performance_scores_csv,
    phase1_manifest,
    phase1_trials_dir,
    phase2_summaries_dir,
)

log = logging.getLogger(__name__)

EEG_BASELINE_DIM = 64
EEG_PC_DIM = 64
EYE_DIM = 18     # see _build_eye_vector

EVENT_FIELDS = (
    "n_fixations", "n_saccades",
    "fixation_rate_hz", "saccade_rate_hz",
    "mean_fixation_duration_s", "mean_saccade_amplitude_proxy",
)


def load_performance_scores(csv_path: Path) -> pd.DataFrame:
    """Load PerformanceScores.csv, strip the stray trailing ' from filenames,
    and attach a `trial_id` column derived from `Eye File Name`."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    for col in ("EEG File Name", "Eye File Name"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.rstrip("'")
    df["trial_id"] = (df["Eye File Name"]
                      .astype(str)
                      .str.replace(r"\.csv$", "", regex=True))
    rename = {
        "Subject ID": "subject_id",
        "Task ID": "task_id",
        "Age (year)": "age",
        "Dominant Hand": "dominant_hand",
        "Performance(out of 100)": "performance",
        "Try": "try_num",
    }
    df = df.rename(columns=rename)
    keep = ["trial_id", "subject_id", "task_id", "age",
            "dominant_hand", "performance", "try_num"]
    return df[keep]


def load_task_module_map(yaml_path: Path) -> Dict[int, str]:
    """Read the task-module config and invert `modules: {name: [ids]}` into
    a flat {task_id: module_name} map."""
    import yaml
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    flat: Dict[int, str] = {}
    for module_name, ids in cfg["modules"].items():
        for tid in ids:
            flat[int(tid)] = str(module_name)
    return flat


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _build_eye_vector(summary: dict) -> np.ndarray:
    """Deterministic 18-d eye-summary vector from the Phase 2 component dicts.

    Order: occupancy[0..4], transition_diag[0..4], mean_dwell, blink_fraction,
    event_summary[EVENT_FIELDS...]. Missing / NaN values are coerced to 0.0,
    which matches the Phase 3 RDM code's handling of sparse eye signals.
    """
    occ = summary.get("eye_state_summary", {}).get("occupancy") or [0.0] * 5
    occ = np.asarray(occ, dtype=np.float64)
    if occ.size < 5:
        occ = np.pad(occ, (0, 5 - occ.size))
    else:
        occ = occ[:5]

    tmat = summary.get("eye_transition_summary", {}).get("transition_matrix")
    if tmat is not None:
        tmat = np.asarray(tmat, dtype=np.float64)
        if tmat.ndim == 2 and min(tmat.shape) >= 5:
            diag = np.diag(tmat)[:5]
        else:
            diag = np.zeros(5)
    else:
        diag = np.zeros(5)

    mean_dwell = float(summary.get("eye_state_summary", {}).get("mean_dwell_samples", 0.0))
    blink = float(summary.get("pupil_summary", {}).get("blink_fraction", 0.0))

    ev = summary.get("event_summary", {}) or {}
    event_vec = np.array([float(ev.get(k, 0.0)) for k in EVENT_FIELDS])

    out = np.concatenate([occ, diag, [mean_dwell, blink], event_vec])
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    assert out.size == EYE_DIM, f"eye vector dim mismatch: {out.size}"
    return out


def _pool_phase1(phase1: dict) -> np.ndarray:
    """Return concat(mean(baseline, t), mean(pc, t)) of shape (128,)."""
    base = phase1["baseline_embeddings"]
    pc = phase1["pc_embeddings"]
    if base.ndim != 2 or base.shape[1] != EEG_BASELINE_DIM:
        raise ValueError(f"baseline shape {base.shape} unexpected")
    if pc.ndim != 2 or pc.shape[1] != EEG_PC_DIM:
        raise ValueError(f"pc shape {pc.shape} unexpected")
    return np.concatenate([base.mean(axis=0), pc.mean(axis=0)]).astype(np.float64)


def feature_column_names() -> List[str]:
    names = [f"eeg_base_{i}" for i in range(EEG_BASELINE_DIM)]
    names += [f"eeg_pc_{i}" for i in range(EEG_PC_DIM)]
    names += [f"eye_occ_{i}" for i in range(5)]
    names += [f"eye_tdiag_{i}" for i in range(5)]
    names += ["eye_mean_dwell", "eye_blink_fraction"]
    names += [f"eye_{k}" for k in EVENT_FIELDS]
    return names


def build_eeg_eye_feature_frame(
    data_root: Path,
    task_module_map: Dict[int, str],
    *,
    try_filter: int = 1,
) -> pd.DataFrame:
    """Build one row per Try == `try_filter` trial that has Phase 1 + Phase 2 caches.

    Returns a DataFrame with columns `feature_column_names()` plus metadata
    columns: trial_id, subject_id, task_id, task_module, age, dominant_hand,
    performance.
    """
    scores = load_performance_scores(performance_scores_csv(data_root))
    scores = scores[scores["try_num"] == try_filter].reset_index(drop=True)

    p1_dir = phase1_trials_dir(data_root)
    p2_dir = phase2_summaries_dir(data_root)
    rows = []
    feat_cols = feature_column_names()

    for _, r in scores.iterrows():
        tid = r["trial_id"]
        p1_path = p1_dir / f"{tid}.pkl"
        p2_path = p2_dir / f"{tid}.pkl"
        if not p1_path.exists() or not p2_path.exists():
            continue

        try:
            phase1 = _load_pickle(p1_path)
            phase2 = _load_pickle(p2_path)
            eeg_vec = _pool_phase1(phase1)
            eye_vec = _build_eye_vector(phase2)
        except Exception as e:
            log.warning("skip %s (%s: %s)", tid, type(e).__name__, e)
            continue

        feat = np.concatenate([eeg_vec, eye_vec])
        if feat.size != len(feat_cols):
            raise RuntimeError(
                f"feature dim {feat.size} != columns {len(feat_cols)}")

        module = task_module_map.get(int(r["task_id"]), "unknown")
        record = {
            "trial_id": tid,
            "subject_id": int(r["subject_id"]),
            "task_id": int(r["task_id"]),
            "task_module": module,
            "age": float(r["age"]),
            "dominant_hand": str(r["dominant_hand"]),
            "performance": float(r["performance"]),
        }
        record.update(dict(zip(feat_cols, feat)))
        rows.append(record)

    df = pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)
    return df


def mimic_modality_columns() -> Dict[str, List[str]]:
    """Partition the 146 Mimic feature columns by upstream modality.

    The feature vector built by `build_eeg_eye_feature_frame` is laid out
    deterministically by `feature_column_names()`:

      - 64 columns with prefix ``eeg_base_``   (Phase 1 baseline encoder)
      - 64 columns with prefix ``eeg_pc_``     (Phase 1 predictive-coding encoder)
      - 18 columns with prefix ``eye_``        (Phase 2 eye-summary vector)

    So a prefix-based split gives a complete and disjoint partition.
    """
    all_cols = feature_column_names()
    out: Dict[str, List[str]] = {
        "eeg_baseline":          [c for c in all_cols if c.startswith("eeg_base_")],
        "eeg_predictive_coding": [c for c in all_cols if c.startswith("eeg_pc_")],
        "eye":                   [c for c in all_cols if c.startswith("eye_")],
    }
    total = sum(len(v) for v in out.values())
    if total != len(all_cols):
        raise AssertionError(
            f"modality split is not complete: {total} != {len(all_cols)}")
    # Disjoint check: every column assigned exactly once.
    seen: set = set()
    for cols in out.values():
        for c in cols:
            if c in seen:
                raise AssertionError(f"column {c!r} assigned to >1 modality")
            seen.add(c)
    return out


def read_phase1_task_names(root: Path) -> Dict[int, str]:
    """Return {task_id: human name} by reading the Phase 1 manifest."""
    path = phase1_manifest(root)
    with path.open("r", encoding="utf-8") as f:
        mf = json.load(f)
    out: Dict[int, str] = {}
    for t in mf.get("trials", []):
        out.setdefault(int(t["task_id"]), str(t["task_name"]))
    return out
