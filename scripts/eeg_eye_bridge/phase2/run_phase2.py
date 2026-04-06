#!/usr/bin/env python3
"""Run Phase 2: eye summaries + EEG–eye consistency scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# repo_root = scripts/eeg_eye_bridge/phase2 -> parents[3]
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from eeg_eye_bridge.phase2_eye_latents.config import Phase2Config
from eeg_eye_bridge.phase2_eye_latents.export_phase2 import run_phase2_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 eye-consistent latent evaluation")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO,
        help="Repository root (default: auto)",
    )
    p.add_argument(
        "--phase1-dir",
        type=Path,
        default=None,
        help="Override Phase 1 cache dir (default: cache/eeg_eye_bridge/phase1)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override Phase 2 output cache dir",
    )
    p.add_argument(
        "--eye-root",
        type=Path,
        default=None,
        help="Override Eye/EYE directory",
    )
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Process only first N trials from manifest",
    )
    p.add_argument(
        "--trial-ids",
        type=str,
        default=None,
        help="Comma-separated trial ids (overrides subset)",
    )
    p.add_argument("--debug", action="store_true", help="Print verbose diagnostics")
    args = p.parse_args()

    cfg = Phase2Config(repo_root=args.repo_root, subset=args.subset, debug=args.debug)
    if args.eye_root is not None:
        cfg.eye_root = Path(args.eye_root)
    if args.trial_ids:
        cfg.trial_ids = [x.strip() for x in args.trial_ids.split(",") if x.strip()]

    payload, warnings = run_phase2_pipeline(
        cfg,
        phase1_dir=args.phase1_dir,
        out_dir=args.out_dir,
    )
    print(json.dumps({"n_trials_scored": len(payload.get("scores_by_trial", {})), "warnings": warnings}, indent=2))
    if args.debug:
        print("aggregate:", json.dumps(payload.get("aggregate"), indent=2, default=str))


if __name__ == "__main__":
    main()
