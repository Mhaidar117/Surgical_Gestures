"""End-to-end smoke test for pipeline/skill_manifold_gw_fls.py.

Build synthetic per-trial feature parquet caches that the orchestrator
can load directly, then run `run(args)` against a temp directory and
assert the report markdown / JSON / headline plot all exist.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Load the orchestrator by path (its filename has a hyphen-free module
# name but lives under pipeline/, which isn't a package).
SPEC = importlib.util.spec_from_file_location(
    "skill_manifold_gw_fls",
    REPO / "pipeline" / "skill_manifold_gw_fls.py",
)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader is not None
SPEC.loader.exec_module(mod)

from skill_manifold.features_eeg_eye import EYE_DIM       # noqa: E402
from skill_manifold.features_fls_eeg import (             # noqa: E402
    BANDS, REGION_ORDER, eeg_feature_column_names,
)
from skill_manifold.features_fls_gaze import eye_feature_column_names  # noqa: E402


def _synth_metadata(n_subjects: int, tasks=(1, 2, 3), n_tries: int = 2,
                     seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for subj in range(1, n_subjects + 1):
        for task in tasks:
            for k in range(1, n_tries + 1):
                tid = f"{subj}_{task}_{k}"
                rows.append({
                    "trial_id": tid,
                    "subject_id": subj,
                    "task_id": task,
                    "try_num": k,
                    "age": 25 + (subj % 10),
                    "dominant_hand": "Right" if subj % 2 else "Left",
                    "dominant_eye": "Left" if subj % 3 else "Right",
                    "gender": "F" if subj % 2 else "M",
                    "performance": float(rng.uniform(8, 25)
                                          + (subj / n_subjects) * 5),
                    "perf_min": 5.0,
                    "perf_max": 25.0,
                })
    return pd.DataFrame(rows)


def _attach_features(meta: pd.DataFrame, cols: list, *,
                      seed: int, signal_per_subject: float = 0.0) -> pd.DataFrame:
    """Synthetic features. `signal_per_subject` adds a per-subject mean
    drift to the features so the GW analysis has *some* tier structure."""
    rng = np.random.default_rng(seed)
    df = meta.copy()
    base = rng.standard_normal((len(meta), len(cols)))
    if signal_per_subject:
        # Inject a 1-d skill signal: subjects with higher subject_id get a
        # larger mean shift on every feature column.
        shift = (meta["subject_id"].to_numpy() - meta["subject_id"].mean()) \
                  * signal_per_subject
        base += shift[:, None]
    for k, c in enumerate(cols):
        df[c] = base[:, k]
    return df


def _make_args(repo_root: Path, output_dir: Path, cache_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=repo_root,
        n_perms=50,                      # small for speed
        gw_epsilon=0.05,
        output_dir=output_dir,
        features_cache=cache_dir,
        skip_eeg=False,
        skip_eeg_zscore=False,
        no_residualize_task=False,
        skip_companion=False,
        skip_per_task=False,
        seed=1,
        log_level="WARNING",
    )


def _seed_caches(cache_dir: Path) -> None:
    """Pre-write synthetic gaze and EEG trial-level parquet caches so the
    orchestrator's _load_or_build_feature_frame uses them and never
    touches disk under data/laparoscopic-surgery-fls-tasks."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = _synth_metadata(n_subjects=12, tasks=(1, 2, 3), n_tries=2, seed=42)
    gaze = _attach_features(meta, eye_feature_column_names(),
                             seed=11, signal_per_subject=0.4)
    eeg  = _attach_features(meta, eeg_feature_column_names(),
                             seed=22, signal_per_subject=0.2)
    gaze.to_parquet(cache_dir / "fls_gaze_trial.parquet", index=False)
    eeg.to_parquet(cache_dir / "fls_eeg_trial.parquet", index=False)


def test_orchestrator_smoke(tmp_path: Path) -> None:
    cache = tmp_path / "feature_cache"
    out = tmp_path / "report"
    _seed_caches(cache)
    args = _make_args(repo_root=REPO, output_dir=out, cache_dir=cache)
    summary = mod.run(args)

    # Top-level shape sanity.
    assert summary["n_subjects"] >= 9   # at least 9 of 12 should land in tertiles
    assert "headline" in summary
    assert "rdm_gaze" in summary["headline"]
    assert "rdm_eeg" in summary["headline"]
    assert summary["headline"]["n_permutations"] == 50

    # Report files exist on disk.
    assert (out / "report_fls.md").exists()
    assert (out / "results_fls.json").exists()
    parsed = json.loads((out / "results_fls.json").read_text())
    assert parsed["headline"]["gw_distance"] >= 0.0

    # Required plots.
    plots = out / "plots"
    for required in ("rdm_centroid_gaze.png", "rdm_centroid_eeg.png",
                     "coupling_headline.png", "null_headline.png",
                     "mds_gaze.png", "mds_eeg.png"):
        assert (plots / required).exists(), f"missing {required}"

    # Companion ran (12 subjects >= 6 minimum).
    assert summary["companion"] is not None
    assert (plots / "coupling_companion.png").exists()
    assert (plots / "null_companion.png").exists()

    # Per-task ran for each task with enough subjects.
    assert isinstance(summary["per_task"], dict)
    assert len(summary["per_task"]) >= 1
