#!/usr/bin/env python3
"""
Run the full EEG–eye bridge pipeline: Phase 1 → 2 → 3 → ViT training (Phase 4).

Uses tqdm on stderr for a phase-level progress bar; subprocess stdout/stderr pass through
so you still see logs from each step.

Usage (from repo root)::

    python3 pipeline/run_full_pipeline.py
    python3 pipeline/run_full_pipeline.py --phase1-synthetic --skip-train

Or with venv::

    source .venv/bin/activate
    export PYTHONPATH=src   # optional; script sets PYTHONPATH for children
    python pipeline/run_full_pipeline.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit(
        "tqdm is required. Install with: pip install tqdm"
    ) from e


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dataset_root(repo: Path, cli: Path | None) -> Path:
    from dataset_paths import resolve_dataset_root

    return resolve_dataset_root(cli, fallback_repo_root=repo)


def _py() -> str:
    return sys.executable


def _env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(repo / "src")
    prev = env.get("PYTHONPATH", "")
    if prev:
        env["PYTHONPATH"] = src + os.pathsep + prev
    else:
        env["PYTHONPATH"] = src
    return env


def main() -> None:
    repo = _repo_root()
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    p = argparse.ArgumentParser(
        description="Run full EEG–eye bridge pipeline with per-phase progress"
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root (EEG/, Eye/, Gestures/). Default: env, iCloud path if present, else repo.",
    )
    p.add_argument(
        "--phase1-synthetic",
        action="store_true",
        help="Phase 1: use --synthetic_only (no EDFs)",
    )
    p.add_argument(
        "--phase1-max-trials",
        type=int,
        default=None,
        help="Phase 1: pass --max_trials",
    )
    p.add_argument(
        "--phase2-subset",
        type=int,
        default=None,
        help="Phase 2: pass --subset",
    )
    p.add_argument(
        "--phase3-max-trials",
        type=int,
        default=None,
        help="Phase 3: pass --max-trials",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Stop after Phase 3 (skip ViT training)",
    )
    p.add_argument(
        "--train-config",
        type=Path,
        default=repo / "src" / "configs" / "bridge_eeg_rdm.yaml",
        help="YAML config for train_vit_system.py",
    )
    p.add_argument("--task", type=str, default="all")
    p.add_argument("--split", type=str, default="fold_1")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "checkpoints" / "bridge_eeg",
        help="Checkpoints directory for training",
    )
    args = p.parse_args()
    data_root = _dataset_root(repo, args.data_root)
    env = _env(repo)

    phase1_cmd = [
        _py(),
        str(repo / "pipeline" / "phase1_run_export.py"),
        "--data_root",
        str(data_root),
        "--workspace_root",
        str(repo),
    ]
    if args.phase1_synthetic:
        phase1_cmd.append("--synthetic_only")
    if args.phase1_max_trials is not None:
        phase1_cmd.extend(["--max_trials", str(args.phase1_max_trials)])

    phase2_cmd = [
        _py(),
        str(repo / "pipeline" / "phase2_run_phase2.py"),
        "--repo-root",
        str(repo),
        "--data-root",
        str(data_root),
    ]
    if args.phase2_subset is not None:
        phase2_cmd.extend(["--subset", str(args.phase2_subset)])

    phase3_cmd = [
        _py(),
        str(repo / "pipeline" / "phase3_build_rdms.py"),
        "--cache-root",
        str(repo / "cache" / "eeg_eye_bridge"),
        "--dataset-root",
        str(data_root),
    ]
    if args.phase3_max_trials is not None:
        phase3_cmd.extend(["--max-trials", str(args.phase3_max_trials)])

    train_config = Path(args.train_config).resolve()
    train_cmd = [
        _py(),
        str(repo / "src" / "training" / "train_vit_system.py"),
        "--config",
        str(train_config),
        "--data_root",
        str(data_root),
        "--task",
        args.task,
        "--split",
        args.split,
        "--output_dir",
        str(Path(args.output_dir).resolve()),
    ]

    steps: list[tuple[str, list[str]]] = [
        ("Phase 1: EEG export → cache/eeg_eye_bridge/phase1/", phase1_cmd),
        ("Phase 2: Eye consistency → cache/.../phase2/", phase2_cmd),
        ("Phase 3: RDMs + manifest → cache/.../phase3/", phase3_cmd),
    ]
    if not args.skip_train:
        steps.append(
            (
                "Phase 4: ViT training (bridge)",
                train_cmd,
            )
        )

    print(
        f"\nRepo root: {repo}\nData root: {data_root}\n",
        flush=True,
    )

    fail: Exception | None = None
    with tqdm(
        total=len(steps),
        desc="EEG–eye bridge pipeline",
        unit="phase",
        file=sys.stderr,
        dynamic_ncols=True,
    ) as bar:
        for label, cmd in steps:
            bar.set_postfix_str(label[:50] + ("…" if len(label) > 50 else ""), refresh=False)
            print("\n" + "=" * 72, flush=True)
            print(label, flush=True)
            print("=" * 72 + "\n", flush=True)
            try:
                subprocess.run(cmd, cwd=str(repo), env=env, check=True)
            except subprocess.CalledProcessError as e:
                fail = e
                print(f"\n[error] Command failed with exit code {e.returncode}", file=sys.stderr)
                break
            bar.update(1)
            bar.set_postfix_str("done", refresh=True)

    if fail is not None:
        sys.exit(fail.returncode or 1)

    print("\nPipeline finished successfully.\n", flush=True)


if __name__ == "__main__":
    main()
