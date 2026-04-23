"""Shared fixtures for skill_manifold smoke tests.

These fixtures build tiny synthetic JIGSAWS and EEG/Eye caches on disk in a
tmp directory so the tests can run without the real external dataset
(~hundreds of GB in an iCloud Drive location). Each fixture returns a
`data_root` Path that mirrors the layout expected by
`src/skill_manifold/io.py`.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


# ---------- synthetic JIGSAWS ----------

SURGEONS = ("B", "C", "D", "E", "F", "G", "H", "I")
JIGSAWS_TASKS = ("Suturing", "Knot_Tying", "Needle_Passing")
# Keep per-task trials per surgeon small (2) so tests are fast.
TRIALS_PER_SURGEON_PER_TASK = 2


def _write_kinematics(path: Path, n_frames: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    # 76-dim per row; only cols 38..75 are read by our pipeline (Slave arms).
    arr = rng.normal(size=(n_frames, 76)).astype(np.float64)
    # Gripper column (18th index within arm block) should look "gripper-like"
    # (bounded near 0). Put a slow oscillation there for Slave-L and Slave-R.
    t = np.linspace(0, 4 * np.pi, n_frames)
    arr[:, 38 + 18] = 0.5 * np.sin(t) + 0.5           # Slave-L gripper
    arr[:, 57 + 18] = 0.5 * np.cos(t) + 0.5           # Slave-R gripper
    np.savetxt(str(path), arr, fmt="%.6f")


def _write_transcription(path: Path, n_frames: int, seed: int) -> None:
    """Write a transcription with 3 segments over the trial."""
    rng = np.random.default_rng(seed)
    gestures = ["G1", "G5", "G11", "G3", "G8", "G12"]
    # Three uniform thirds.
    chunks = [
        (1, n_frames // 3, rng.choice(gestures)),
        (n_frames // 3 + 1, 2 * n_frames // 3, rng.choice(gestures)),
        (2 * n_frames // 3 + 1, n_frames, rng.choice(gestures)),
    ]
    with path.open("w") as f:
        for s, e, g in chunks:
            f.write(f"{s} {e} {g}\n")


def _write_jigsaws_task(root: Path, task: str, rng: np.random.Generator) -> None:
    task_dir = root / "Gestures" / task
    (task_dir / "kinematics" / "AllGestures").mkdir(parents=True, exist_ok=True)
    (task_dir / "transcriptions").mkdir(parents=True, exist_ok=True)
    meta_rows = []
    for surgeon_idx, surgeon in enumerate(SURGEONS):
        # Assign a monotonic quality so tertile bins are populated: N / I / E
        # roughly maps to grs_total 9..27.
        skill = ("N", "I", "E")[surgeon_idx % 3]
        base_grs = {"N": 9, "I": 18, "E": 26}[skill]
        for trial_k in range(1, TRIALS_PER_SURGEON_PER_TASK + 1):
            trial_id = f"{task}_{surgeon}{trial_k:03d}"
            grs = int(np.clip(base_grs + rng.integers(-2, 3), 6, 30))
            # Split grs into 6 osats between 1 and 5 (approximate).
            osats = rng.integers(1, 6, size=6).tolist()
            n_frames = int(300 + rng.integers(-60, 120))
            kin_path = task_dir / "kinematics" / "AllGestures" / f"{trial_id}.txt"
            trans_path = task_dir / "transcriptions" / f"{trial_id}.txt"
            _write_kinematics(kin_path, n_frames,
                              seed=surgeon_idx * 97 + trial_k * 13 + hash(task) % 1000)
            _write_transcription(trans_path, n_frames,
                                 seed=surgeon_idx * 17 + trial_k * 7)
            # JIGSAWS meta row is whitespace-separated; mimic the double-tab style.
            meta_rows.append(
                f"{trial_id}\t\t{skill}\t{grs}\t\t" + "\t".join(str(x) for x in osats)
            )
    (task_dir / f"meta_file_{task}.txt").write_text("\n".join(meta_rows) + "\n")


# ---------- synthetic EEG/Eye caches ----------

N_EEG_SUBJECTS = 6
N_TASK_IDS = 27
TRIES_PER_TASK = 1  # Try==1 only, matching pipeline filter


def _synthetic_phase1_pickle(trial_id: str, task_id: int, task_name: str,
                             participant_id: int, performance: float,
                             n_windows: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "trial_id": trial_id,
        "participant_id": int(participant_id),
        "task_id": int(task_id),
        "task_name": task_name,
        "task_family": "other_nontransfer",
        "performance_score": float(performance),
        "window_times": np.stack([np.arange(n_windows),
                                    np.arange(n_windows) + 1], axis=1).astype(np.float64),
        "baseline_embeddings": rng.normal(size=(n_windows, 64)).astype(np.float32),
        "pc_embeddings": rng.normal(size=(n_windows, 64)).astype(np.float32),
        "prediction_errors": rng.normal(size=(n_windows,)).astype(np.float32),
        "contract_version": "phase1_eeg_v1",
    }


def _synthetic_phase2_pickle(trial_id: str, task_id: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # 5x5 transition matrix, each row sums to ~1.
    raw = rng.dirichlet(alpha=np.ones(5), size=5)
    diag_boost = np.diag([0.7, 0.6, 0.65, 0.7, 0.6])
    tmat = (raw + diag_boost) / (raw + diag_boost).sum(axis=1, keepdims=True)
    occ = tmat.mean(axis=0).tolist()
    return {
        "trial_id": trial_id,
        "task_id": int(task_id),
        "task_family": "other_nontransfer",
        "eye_state_summary": {
            "method": "quantile_windows",
            "n_states": 5,
            "occupancy": occ,
            "mean_dwell_samples": float(rng.uniform(3.0, 5.0)),
        },
        "eye_transition_summary": {
            "transition_matrix": tmat.tolist(),
        },
        "pupil_summary": {
            "mean": float("nan"),
            "std": float("nan"),
            "cv": float("nan"),
            "blink_fraction": float(rng.uniform(0.0, 0.05)),
        },
        "event_summary": {
            "n_fixations": int(rng.integers(500, 10000)),
            "n_saccades": int(rng.integers(100, 800)),
            "fixation_rate_hz": float(rng.uniform(30, 60)),
            "saccade_rate_hz": float(rng.uniform(1, 5)),
            "mean_fixation_duration_s": float(rng.uniform(0.3, 1.0)),
            "mean_saccade_amplitude_proxy": float(rng.uniform(10, 50)),
        },
    }


def _write_eeg_eye(root: Path, rng: np.random.Generator) -> None:
    (root / "Eye").mkdir(parents=True, exist_ok=True)
    p1 = root / "cache" / "eeg_eye_bridge" / "phase1" / "trials"
    p2 = root / "cache" / "eeg_eye_bridge" / "phase2" / "eye_summaries"
    p1.mkdir(parents=True, exist_ok=True)
    p2.mkdir(parents=True, exist_ok=True)

    # Minimal manifest.json with task names for the module map test.
    manifest = {"contract_version": "phase1_eeg_v1", "trials": []}

    task_names = [
        "Pick and place", "Peg board 1", "Peg board 2",
        "Match board 1", "Match board 2", "Match board 3",
        "Ring and rail 1", "Ring and rail 2",
        "Camera targeting 1", "Camera targeting 2", "Scaling",
        "Ring walk 1", "Ring walk 2", "Ring walk 3",
        "Needle targeting", "Thread the rings",
        "Suture sponge 1", "Suture sponge 2", "Suture sponge 3",
        "Dots and needles 1", "Dots and needles 2", "Tubes",
        "Energy switching 1", "Energy switching 2",
        "Energy dissection 1", "Energy dissection 2", "Energy dissection 3",
    ]

    rows = []
    for subj in range(1, N_EEG_SUBJECTS + 1):
        for task_id in range(1, N_TASK_IDS + 1):
            for try_num in range(1, TRIES_PER_TASK + 1):
                trial_id = f"{subj}_{task_id}_{try_num}"
                n_windows = int(rng.integers(80, 200))
                perf = float(rng.uniform(40, 100))
                seed = subj * 100 + task_id * 3 + try_num
                p1_obj = _synthetic_phase1_pickle(
                    trial_id, task_id, task_names[task_id - 1],
                    subj, perf, n_windows, seed,
                )
                p2_obj = _synthetic_phase2_pickle(trial_id, task_id, seed + 999)
                with (p1 / f"{trial_id}.pkl").open("wb") as f:
                    pickle.dump(p1_obj, f)
                with (p2 / f"{trial_id}.pkl").open("wb") as f:
                    pickle.dump(p2_obj, f)
                manifest["trials"].append({
                    "trial_id": trial_id,
                    "trial_pkl": f"trials/{trial_id}.pkl",
                    "participant_id": subj,
                    "task_id": task_id,
                    "task_name": task_names[task_id - 1],
                    "task_family": "other_nontransfer",
                    "performance_score": perf,
                    "n_windows": n_windows,
                    "baseline_embed_dim": 64,
                    "pc_embed_dim": 64,
                })
                rows.append({
                    "EEG File Name": f"{trial_id}.edf'",       # stray trailing ' mirrors real CSV
                    "Eye File Name": f"{trial_id}.csv'",
                    "Subject ID": subj,
                    "Gender(F:Female; M:Male)": "F" if subj % 2 else "M",
                    "Age (year)": 20 + (subj % 10),
                    "Dominant Hand": "Right" if subj % 3 else "Left",
                    "Task ID": task_id,
                    "Try": try_num,
                    "Performance(out of 100)": int(round(perf)),
                })

    import json as _json
    (root / "cache" / "eeg_eye_bridge" / "phase1" / "manifest.json").write_text(
        _json.dumps(manifest, indent=2))
    # PerformanceScores.csv with a UTF-8 BOM to mirror the real file.
    df = pd.DataFrame(rows)
    with (root / "Eye" / "PerformanceScores.csv").open("w", encoding="utf-8-sig") as f:
        df.to_csv(f, index=False, lineterminator="\n")


# ---------- fixture ----------

@pytest.fixture(scope="session")
def synthetic_data_root(tmp_path_factory) -> Path:
    """Build a full synthetic dataset tree rooted in a pytest tmp dir.

    Structure:
        <root>/Gestures/{task}/{meta,kinematics,transcriptions}
        <root>/Eye/PerformanceScores.csv
        <root>/cache/eeg_eye_bridge/{phase1,phase2}/...
    """
    rng = np.random.default_rng(42)
    root = tmp_path_factory.mktemp("skill_manifold_data")
    for task in JIGSAWS_TASKS:
        _write_jigsaws_task(root, task, rng)
    _write_eeg_eye(root, rng)
    return root
