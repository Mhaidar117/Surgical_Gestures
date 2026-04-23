#!/usr/bin/env python3
"""
Debug Phase 1 EEG loading: inventory EDFs, fast MNE header scan (preload=False),
and full-load memory probes using the same path as run_export.py.

Trial list matches ``metadata.list_edf_trials`` via stdlib ``csv`` (no pandas import),
so inventory/scan runs even when pandas is missing or hangs on import.

Phase 1 currently accumulates every trial's window tensor in memory before encoding;
if this process receives SIGKILL (exit -9), suspect OOM. Streaming encode/export
is the structural fix; this script only measures.

Optional dependency: ``psutil`` for RSS (``pip install psutil``).

When piping stdout (e.g. to ``head``), use ``python -u`` or ``PYTHONUNBUFFERED=1``
so progress on stderr appears immediately.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from dataset_paths import resolve_dataset_root

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Debug Phase 1 EDF inventory, MNE scan, and memory growth (OOM diagnosis)"
    )
    p.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Dataset root (default: env or iCloud path if present, else repo)",
    )
    p.add_argument(
        "--eeg_dir",
        type=Path,
        default=None,
        help="Directory of EDF files (default: data_root/EEG/EEG)",
    )
    p.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Limit trials from list_edf_trials (same as run_export --max_trials)",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=200,
        help="Cap EDFs for fast MNE preload=False scan (default 200; 0 skips scan)",
    )
    p.add_argument(
        "--scan-all",
        action="store_true",
        help="Scan every trial with MNE preload=False (ignores --max-files; can be slow)",
    )
    p.add_argument(
        "--memory-trials",
        type=int,
        default=0,
        help="Number of trials to fully load (load_eeg_preprocessed + sliding_windows); 0 skips",
    )
    p.add_argument("--window_sec", type=float, default=1.0)
    p.add_argument("--hop_sec", type=float, default=0.5)
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable bandpass/notch after load (matches run_export --no_filter)",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write machine-readable summary JSON to this path",
    )
    return p.parse_args()


def _rss_bytes() -> Optional[int]:
    if not _HAS_PSUTIL:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


class _TaskFamily:
    NEEDLE_CONTROL = "needle_control"
    NEEDLE_DRIVING = "needle_driving"
    OTHER_NONTRANSFER = "other_nontransfer"


def _normalize_edf_filename(name: str) -> str:
    s = name.strip().strip("'").strip('"')
    if s.lower().endswith(".edf"):
        return s[:-4]
    return s


def _task_family_for_task_id(task_id: int) -> str:
    if task_id in (15, 16):
        return _TaskFamily.NEEDLE_CONTROL
    if 17 <= task_id <= 22:
        return _TaskFamily.NEEDLE_DRIVING
    return _TaskFamily.OTHER_NONTRANSFER


def _load_task_id_to_name(table1_path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    text = table1_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        matches = list(re.finditer(r"\((\d+)\)", line))
        if not matches:
            continue
        m = matches[-1]
        tid = int(m.group(1))
        prefix = line[: m.start()]
        if "," in prefix:
            name_part = prefix.split(",", 1)[-1]
        else:
            name_part = prefix
        name_part = re.sub(r"^[\s·]+", "", name_part)
        name_part = re.sub(r"\s*\(\d+\)\s*$", "", name_part).strip()
        if name_part:
            out[tid] = name_part
    return out


@dataclass
class _TrialMeta:
    trial_id: str
    participant_id: int
    task_id: int
    try_number: int
    performance_score: float
    eeg_filename: str


def _resolve_task_name(task_id: int, id_to_name: Dict[int, str]) -> str:
    return id_to_name.get(task_id, f"task_{task_id}")


def _load_performance_scores(csv_path: Path) -> Dict[str, _TrialMeta]:
    """Same keys/rows as ``metadata.load_performance_scores`` using stdlib csv."""
    col_eeg = "EEG File Name"
    col_subj = "Subject ID"
    col_task = "Task ID"
    col_try = "Try"
    col_perf = "Performance(out of 100)"
    keyed: Dict[str, _TrialMeta] = {}
    # utf-8-sig strips BOM so the first column matches "EEG File Name" (pandas does this)
    with csv_path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return keyed
        header = [h.strip() for h in header]

        def col_index(name: str) -> int:
            try:
                return header.index(name)
            except ValueError:
                return -1

        ie = col_index(col_eeg)
        isu = col_index(col_subj)
        it = col_index(col_task)
        ity = col_index(col_try)
        ip = col_index(col_perf)
        if min(ie, isu, it, ity, ip) < 0:
            return keyed
        need = max(ie, isu, it, ity, ip) + 1
        for row in reader:
            if len(row) < need:
                continue
            try:
                raw_name = str(row[ie])
                stem = _normalize_edf_filename(raw_name)
                trial_id = stem
                participant_id = int(float(row[isu]))
                task_id = int(float(row[it]))
                try_number = int(float(row[ity]))
                performance_score = float(row[ip])
            except (ValueError, TypeError):
                continue
            keyed[trial_id] = _TrialMeta(
                trial_id=trial_id,
                participant_id=participant_id,
                task_id=task_id,
                try_number=try_number,
                performance_score=performance_score,
                eeg_filename=stem + ".edf",
            )
    return keyed


def _list_edf_trials_stdlib(
    eeg_dir: Path,
    performance_path: Path,
    table1_path: Path,
    max_trials: Optional[int] = None,
    max_participants: Optional[int] = None,
) -> List[Tuple[Path, _TrialMeta, str, str]]:
    """Mirror of ``metadata.list_edf_trials`` without pandas."""
    perf = _load_performance_scores(performance_path)
    id_to_name = _load_task_id_to_name(table1_path)
    pairs: List[Tuple[Path, _TrialMeta, str, str]] = []
    if not eeg_dir.is_dir():
        return pairs

    allowed_pids: Optional[set] = None
    if max_participants is not None:
        pids = sorted({m.participant_id for m in perf.values()})
        allowed_pids = set(pids[:max_participants])

    for p in sorted(eeg_dir.glob("*.edf")):
        stem = p.stem
        if stem not in perf:
            continue
        meta = perf[stem]
        if allowed_pids is not None and meta.participant_id not in allowed_pids:
            continue
        name = _resolve_task_name(meta.task_id, id_to_name)
        fam = _task_family_for_task_id(meta.task_id)
        pairs.append((p, meta, name, fam))
        if max_trials is not None and len(pairs) >= max_trials:
            break
    return pairs


def _inventory_rows(
    pairs: List[Tuple[Path, Any, str, str]],
) -> Tuple[Dict[str, Any], List[int]]:
    sizes: List[int] = []
    for edf_path, _meta, _name, _fam in pairs:
        try:
            sizes.append(edf_path.stat().st_size)
        except OSError:
            sizes.append(0)
    n = len(sizes)
    if n == 0:
        stats = {
            "n_trials": 0,
            "total_bytes": 0,
            "min_bytes": None,
            "median_bytes": None,
            "max_bytes": None,
        }
        return stats, sizes
    stats = {
        "n_trials": n,
        "total_bytes": sum(sizes),
        "min_bytes": min(sizes),
        "median_bytes": int(statistics.median(sizes)),
        "max_bytes": max(sizes),
    }
    return stats, sizes


def _scan_edf_headers(
    pairs: List[Tuple[Path, Any, str, str]],
    max_files: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import mne

    ok: List[Dict[str, Any]] = []
    fail: List[Dict[str, Any]] = []
    subset = pairs if max_files is None else pairs[:max_files]
    for edf_path, meta, _name, _fam in subset:
        try:
            with mne.io.read_raw_edf(edf_path, preload=False, verbose=False) as raw:
                dur = float(raw.times[-1]) if raw.n_times else 0.0
                n_ch = len(raw.ch_names)
            ok.append(
                {
                    "trial_id": meta.trial_id,
                    "path": str(edf_path),
                    "n_channels": n_ch,
                    "duration_sec": dur,
                }
            )
        except Exception as e:
            fail.append(
                {
                    "trial_id": meta.trial_id,
                    "path": str(edf_path),
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    return ok, fail


def _memory_probe(
    pairs: List[Tuple[Path, Any, str, str]],
    n_trials: int,
    window_sec: float,
    hop_sec: float,
    apply_filtering: bool,
    total_for_extrap: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from eeg_eye_bridge.phase1_eeg.preprocessing import (
        build_processor,
        load_eeg_preprocessed,
        sliding_windows,
    )

    processor = build_processor(sampling_rate=500.0)
    rows: List[Dict[str, Any]] = []
    cumulative_window_bytes = 0
    rss0 = _rss_bytes()

    subset = pairs[: min(n_trials, len(pairs))]
    for edf_path, meta, _name, _fam in subset:
        try:
            eeg, sfreq = load_eeg_preprocessed(
                edf_path,
                processor,
                apply_filtering=apply_filtering,
                apply_ica=False,
            )
            w_np, _w_times = sliding_windows(
                eeg, sfreq, window_sec=window_sec, hop_sec=hop_sec
            )
        except (ValueError, OSError) as exc:
            rows.append(
                {
                    "trial_id": meta.trial_id,
                    "ok": False,
                    "error": str(exc),
                    "windows_bytes": 0,
                    "cumulative_windows_bytes": cumulative_window_bytes,
                    "rss_bytes": _rss_bytes(),
                }
            )
            continue

        wb = int(w_np.nbytes)
        cumulative_window_bytes += wb
        rows.append(
            {
                "trial_id": meta.trial_id,
                "ok": True,
                "windows_shape": list(w_np.shape),
                "windows_bytes": wb,
                "cumulative_windows_bytes": cumulative_window_bytes,
                "rss_bytes": _rss_bytes(),
            }
        )

    loaded = [r for r in rows if r.get("ok")]
    n_loaded = len(loaded)
    sum_bytes = sum(int(r["windows_bytes"]) for r in loaded)
    extrap: Dict[str, Any] = {
        "memory_trials_requested": n_trials,
        "trials_attempted": len(subset),
        "trials_loaded_ok": n_loaded,
        "sum_windows_bytes_sample": sum_bytes,
        "cumulative_windows_bytes_final": cumulative_window_bytes,
        "rss_before_bytes": rss0,
        "rss_after_bytes": _rss_bytes(),
    }
    if n_loaded > 0 and total_for_extrap > 0:
        mean_b = sum_bytes / n_loaded
        extrap["mean_windows_bytes_per_trial"] = mean_b
        extrap["linear_extrap_windows_bytes_all_trials"] = int(mean_b * total_for_extrap)
    else:
        extrap["mean_windows_bytes_per_trial"] = None
        extrap["linear_extrap_windows_bytes_all_trials"] = None

    return rows, extrap


def main() -> None:
    args = _parse_args()
    data_root = resolve_dataset_root(args.data_root, fallback_repo_root=_REPO)
    eeg_dir = args.eeg_dir or (data_root / "EEG" / "EEG")
    perf_csv = data_root / "Eye" / "PerformanceScores.csv"
    table1 = data_root / "Eye" / "Table1.csv"

    # stderr is typically line-buffered when stdout is piped; keep early progress visible
    print(
        "debug_phase1_load: building trial list (stdlib CSV + sorted EDF glob)...",
        file=sys.stderr,
        flush=True,
    )
    pairs = _list_edf_trials_stdlib(
        eeg_dir,
        perf_csv,
        table1,
        max_trials=args.max_trials,
        max_participants=None,
    )

    inv_stats, _sizes = _inventory_rows(pairs)
    out: Dict[str, Any] = {"inventory": inv_stats}

    # Stress index (~46% matches a typical mid-run OOM in long jobs)
    n_total = inv_stats["n_trials"]
    stress: Dict[str, Any] = {}
    if n_total > 0:
        idx_46 = int(0.46 * n_total)
        stress = {
            "index_at_46_percent": idx_46,
            "suggested_max_trials_for_bisect_low": max(1, idx_46 // 2),
            "suggested_max_trials_for_bisect_mid": max(1, idx_46),
        }
    out["stress_index"] = stress

    print("=== Inventory (list_edf_trials) ===")
    print(json.dumps(inv_stats, indent=2))
    if stress:
        print("\n=== Stress index (OOM bisection hints) ===")
        print(json.dumps(stress, indent=2))

    scan_ok: List[Dict[str, Any]] = []
    scan_fail: List[Dict[str, Any]] = []
    scan_cap: Optional[int]
    if args.scan_all:
        scan_cap = None
    else:
        scan_cap = args.max_files if args.max_files > 0 else None

    scan_skipped = not (args.scan_all or args.max_files > 0)
    if args.scan_all or args.max_files > 0:
        cap_desc = "all" if scan_cap is None else str(scan_cap)
        print(
            f"\n=== Fast MNE scan (preload=False), files={cap_desc} ===",
            flush=True,
        )
        if args.scan_all and len(pairs) > 500:
            print(
                "(Scanning entire list; this may take a while.)",
                file=sys.stderr,
                flush=True,
            )
        scan_ok, scan_fail = _scan_edf_headers(pairs, scan_cap)
        print(f"OK: {len(scan_ok)}, fail: {len(scan_fail)}")
        if scan_fail:
            print("Failures (first 20):")
            for f in scan_fail[:20]:
                print(f"  {f['trial_id']}: {f['error']}")
            if len(scan_fail) > 20:
                print(f"  ... and {len(scan_fail) - 20} more")
    elif args.max_files == 0:
        print("\n=== Fast MNE scan: skipped (--max-files 0) ===", flush=True)

    out["scan"] = {
        "skipped": scan_skipped,
        "scan_all": args.scan_all,
        "max_files_arg": args.max_files,
        "effective_cap": None if scan_skipped else scan_cap,
        "ok_count": len(scan_ok),
        "fail_count": len(scan_fail),
        "failures": scan_fail,
    }

    if args.memory_trials > 0:
        print(
            f"\n=== Full memory probe (first {args.memory_trials} trials, "
            f"same as Phase 1 load + sliding_windows) ==="
        )
        if not _HAS_PSUTIL:
            print(
                "(Install psutil for RSS: pip install psutil)",
                file=sys.stderr,
            )
        total_extrap = n_total
        mem_rows, extrap = _memory_probe(
            pairs,
            args.memory_trials,
            args.window_sec,
            args.hop_sec,
            apply_filtering=not args.no_filter,
            total_for_extrap=total_extrap,
        )
        for r in mem_rows:
            line = (
                f"  {r['trial_id']}: ok={r.get('ok')} "
                f"win_B={r.get('windows_bytes')} "
                f"cum_B={r.get('cumulative_windows_bytes')}"
            )
            if r.get("rss_bytes") is not None:
                line += f" rss_B={r['rss_bytes']}"
            if not r.get("ok"):
                line += f" err={r.get('error', '')}"
            print(line)
        print("\nExtrapolation:")
        print(json.dumps(extrap, indent=2))
        out["memory"] = {"rows": mem_rows, "extrapolation": extrap}
    else:
        out["memory"] = None

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
