#!/usr/bin/env python3
"""
Phase 1: load simulator EEG, window, run encoders, write cache under cache/eeg_eye_bridge/phase1/.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

# Repo ``src`` on path
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

from eeg_eye_bridge.phase1_eeg.export import (
    CONTRACT_VERSION,
    aggregate_family_summaries,
    export_family_summaries,
    export_manifest,
    export_trial_pickle,
    phase1_cache_root,
)
from eeg_eye_bridge.phase1_eeg.metadata import list_edf_trials
from eeg_eye_bridge.phase1_eeg.pipeline import run_encoders_on_windows
from eeg_eye_bridge.phase1_eeg.preprocessing import (
    build_processor,
    load_eeg_preprocessed,
    sliding_windows,
    synthetic_trial,
)


def _encode_export_one_trial(
    trials_dir: Path,
    args: argparse.Namespace,
    trial_id: str,
    participant_id: int,
    task_id: int,
    task_name: str,
    task_family: str,
    performance_score: float,
    window_times: Any,
    windows: Any,
    manifest_trials: List[dict[str, Any]],
    agg_records: List[dict[str, Any]],
) -> None:
    """Run encoders and write one trial pickle; frees encoder outputs before returning."""
    out = run_encoders_on_windows(
        windows,
        device=args.device,
        baseline_embed_dim=args.baseline_embed_dim,
        pc_hidden_dim=args.pc_hidden_dim,
        pc_embed_dim=args.pc_embed_dim,
        baseline_ckpt=args.baseline_ckpt,
        pc_ckpt=args.pc_ckpt,
    )
    try:
        export_trial_pickle(
            trials_dir / f"{trial_id}.pkl",
            trial_id=trial_id,
            participant_id=int(participant_id),
            task_id=int(task_id),
            task_name=str(task_name),
            task_family=str(task_family),
            performance_score=float(performance_score),
            window_times=window_times,
            baseline_embeddings=out["baseline_embeddings"],
            pc_embeddings=out["pc_embeddings"],
            prediction_errors=out["prediction_errors"],
        )
        manifest_trials.append(
            {
                "trial_id": trial_id,
                "trial_pkl": str(Path("trials") / f"{trial_id}.pkl"),
                "participant_id": participant_id,
                "task_id": task_id,
                "task_name": task_name,
                "task_family": task_family,
                "performance_score": performance_score,
                "n_windows": int(out["baseline_embeddings"].shape[0]),
                "baseline_embed_dim": out["baseline_embed_dim"],
                "pc_embed_dim": out["pc_embed_dim"],
            }
        )
        agg_records.append(
            {
                "trial_id": trial_id,
                "task_family": task_family,
                "baseline_trial_mean": out["baseline_trial_mean"],
                "pc_trial_mean": out["pc_trial_mean"],
            }
        )
    finally:
        del out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 EEG export to cache")
    p.add_argument(
        "--data_root",
        type=Path,
        default=_REPO,
        help="Repository root (default: parent of scripts/)",
    )
    p.add_argument(
        "--eeg_dir",
        type=Path,
        default=None,
        help="Directory of EDF files (default: data_root/EEG/EEG)",
    )
    p.add_argument("--window_sec", type=float, default=1.0)
    p.add_argument("--hop_sec", type=float, default=0.5)
    p.add_argument("--max_trials", type=int, default=None)
    p.add_argument("--max_participants", type=int, default=None)
    p.add_argument("--synthetic_only", action="store_true", help="Skip EDF; one synthetic trial")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--baseline_embed_dim", type=int, default=64)
    p.add_argument("--pc_hidden_dim", type=int, default=128)
    p.add_argument("--pc_embed_dim", type=int, default=64)
    p.add_argument("--no_filter", action="store_true", help="Disable bandpass/notch after load")
    p.add_argument(
        "--strict_edf",
        action="store_true",
        help="Abort on first EDF that cannot be loaded (default: skip bad files)",
    )
    p.add_argument(
        "--baseline_ckpt",
        type=Path,
        default=None,
        help="Path to trained baseline CNN checkpoint (from train_eeg_models.py)",
    )
    p.add_argument(
        "--pc_ckpt",
        type=Path,
        default=None,
        help="Path to trained PC model checkpoint (from train_eeg_models.py)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    eeg_dir = args.eeg_dir or (data_root / "EEG" / "EEG")
    perf_csv = data_root / "Eye" / "PerformanceScores.csv"
    table1 = data_root / "Eye" / "Table1.csv"
    cache = phase1_cache_root(data_root)
    trials_dir = cache / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    processor = build_processor(sampling_rate=500.0)

    manifest_trials: list = []
    agg_records: list = []

    if args.synthetic_only:
        eeg, sfreq = synthetic_trial()
        processor.sampling_rate = sfreq
        w_np, w_times = sliding_windows(
            eeg, sfreq, window_sec=args.window_sec, hop_sec=args.hop_sec
        )
        try:
            _encode_export_one_trial(
                trials_dir,
                args,
                trial_id="synthetic_0",
                participant_id=9,
                task_id=10,
                task_name="synthetic",
                task_family="other_nontransfer",
                performance_score=0.0,
                window_times=w_times,
                windows=w_np,
                manifest_trials=manifest_trials,
                agg_records=agg_records,
            )
        finally:
            del eeg, w_np, w_times
    else:
        pairs = list_edf_trials(
            eeg_dir,
            perf_csv,
            table1,
            max_trials=args.max_trials,
            max_participants=args.max_participants,
        )
        if not pairs:
            print("No EDF files found; use --synthetic_only or place data under", eeg_dir)
            sys.exit(1)
        skipped_edf: list[tuple[str, str]] = []
        ok_count = 0
        for edf_path, meta, task_name, task_family in tqdm(
            pairs,
            desc="Phase 1: load → encode → export",
            unit="trial",
        ):
            try:
                eeg, sfreq = load_eeg_preprocessed(
                    edf_path,
                    processor,
                    apply_filtering=not args.no_filter,
                    apply_ica=False,
                )
            except (ValueError, OSError) as exc:
                tid = meta.trial_id
                msg = f"{edf_path.name}: {exc}"
                if args.strict_edf:
                    print(f"[error] EDF load failed ({tid}): {msg}", file=sys.stderr)
                    sys.exit(1)
                skipped_edf.append((tid, msg))
                tqdm.write(f"[skip] {tid}: {msg}")
                continue
            w_np, w_times = sliding_windows(
                eeg, sfreq, window_sec=args.window_sec, hop_sec=args.hop_sec
            )
            del eeg
            try:
                _encode_export_one_trial(
                    trials_dir,
                    args,
                    trial_id=meta.trial_id,
                    participant_id=int(meta.participant_id),
                    task_id=int(meta.task_id),
                    task_name=task_name,
                    task_family=task_family,
                    performance_score=float(meta.performance_score),
                    window_times=w_times,
                    windows=w_np,
                    manifest_trials=manifest_trials,
                    agg_records=agg_records,
                )
                ok_count += 1
            finally:
                del w_np, w_times
                if ok_count % 25 == 0:
                    gc.collect()

        if skipped_edf:
            print(
                f"Skipped {len(skipped_edf)} trial(s) with unreadable EDF "
                f"(see messages above). OK trials: {ok_count}."
            )
        if ok_count == 0:
            print("No trials loaded after EDF failures; fix data or use --strict_edf to debug.")
            sys.exit(1)

    summaries = aggregate_family_summaries(
        agg_records,
        baseline_dim=args.baseline_embed_dim,
        pc_dim=args.pc_embed_dim,
    )
    export_family_summaries(cache / "family_summaries.pkl", summaries)
    export_manifest(
        cache,
        manifest_trials,
        extra={
            "eeg_dir": str(eeg_dir),
            "window_sec": args.window_sec,
            "hop_sec": args.hop_sec,
            "synthetic_only": args.synthetic_only,
        },
    )
    print(f"Wrote {len(manifest_trials)} trials to {cache} ({CONTRACT_VERSION})")


if __name__ == "__main__":
    main()
