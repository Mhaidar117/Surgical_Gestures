#!/usr/bin/env python3
"""Build Phase 3 candidate RDMs from Phase 1/2 caches and write manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from eeg_eye_bridge.phase3_rdm.paths import default_cache_root
from eeg_eye_bridge.phase3_rdm.pipeline import run_phase3_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 RDM construction")
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Root cache dir (default: <repo>/cache/eeg_eye_bridge)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override Phase 3 output parent (default: same as cache-root)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="one_minus_spearman",
        help="Distance metric for RDM (one_minus_spearman, euclidean, cosine)",
    )
    parser.add_argument(
        "--performance-csv",
        type=str,
        default=None,
        help="Path to Eye/PerformanceScores.csv",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Limit number of trials from manifest (debug)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build RDMs but do not write pickles/manifest",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_root) if args.cache_root else default_cache_root(_REPO_ROOT)
    if args.output_dir:
        cache_root = Path(args.output_dir)

    perf = Path(args.performance_csv) if args.performance_csv else None

    _, manifest, warnings = run_phase3_pipeline(
        cache_root,
        metric=args.metric,
        performance_scores_csv=perf,
        max_trials=args.max_trials,
        write_outputs=not args.dry_run,
    )
    print("Phase 3 RDM pipeline finished.")
    print("recommended_order:", manifest.get("recommended_order", []))
    if warnings:
        print("warnings:")
        for w in warnings:
            print(" -", w)


if __name__ == "__main__":
    main()
