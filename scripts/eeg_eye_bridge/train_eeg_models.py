#!/usr/bin/env python3
"""
Self-supervised training for the two EEG models used in Phase 1.

Baseline CNN  — next-window prediction: encode window t, predict flat(window t+1) via MSE.
Predictive Coding GRU — minimize the built-in per-step prediction error using
    truncated backprop through time (TBPTT).

Both models share the same conceptual objective (predict the next window) but implement
it with different architectures, making their comparison principled.

Trained weights are saved to checkpoints/eeg_models/ and picked up automatically by
run_trained_pipeline.sh, which re-exports Phase 1 and re-runs Phases 2 & 3.

Usage (from repo root, with venv active):
    python scripts/eeg_eye_bridge/train_eeg_models.py
    python scripts/eeg_eye_bridge/train_eeg_models.py --epochs 30 --max_trials 100
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from eeg_eye_bridge.phase1_eeg.metadata import list_edf_trials
from eeg_eye_bridge.phase1_eeg.models import BaselineEEGTemporalCNN, PredictiveCodingEEG
from eeg_eye_bridge.phase1_eeg.preprocessing import (
    build_processor,
    load_eeg_preprocessed,
    sliding_windows,
)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    """Pick the best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_windows(
    eeg_dir: Path,
    perf_csv: Path,
    table1: Path,
    max_trials: int,
    window_sec: float,
    hop_sec: float,
) -> List[np.ndarray]:
    """
    Load and preprocess EDF trials, returning one (n_windows, n_channels, n_time)
    array per trial.
    """
    processor = build_processor(sampling_rate=500.0)
    pairs = list_edf_trials(eeg_dir, perf_csv, table1, max_trials=max_trials)
    if not pairs:
        raise RuntimeError(f"No EDF trials found under {eeg_dir}")

    all_windows: List[np.ndarray] = []
    skipped = 0
    for edf_path, meta, _task_name, _task_family in tqdm(
        pairs, desc="Loading EDF trials", unit="trial"
    ):
        try:
            eeg, sfreq = load_eeg_preprocessed(
                edf_path, processor, apply_filtering=True, apply_ica=False
            )
        except (ValueError, OSError) as exc:
            tqdm.write(f"[skip] {meta.trial_id}: {exc}")
            skipped += 1
            continue
        wins, _ = sliding_windows(eeg, sfreq, window_sec=window_sec, hop_sec=hop_sec)
        all_windows.append(wins)
        del eeg, wins
        gc.collect()

    print(f"Loaded {len(all_windows)} trials ({skipped} skipped).")
    return all_windows


# ---------------------------------------------------------------------------
# Baseline CNN training — next-window prediction
# ---------------------------------------------------------------------------

def train_baseline(
    model: BaselineEEGTemporalCNN,
    all_windows: List[np.ndarray],
    epochs: int,
    lr: float,
    device: torch.device,
) -> BaselineEEGTemporalCNN:
    """
    Train the baseline CNN with a next-window prediction objective.

    For each window t, the CNN produces an embedding which a lightweight prediction
    head maps back to the flattened representation of window t+1. Loss is MSE.
    The prediction head is discarded after training; only the encoder weights matter.
    """
    flat_dim = model.n_channels * model.n_time
    pred_head = nn.Linear(model.embed_dim, flat_dim).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(pred_head.parameters()), lr=lr
    )

    model.train()
    pred_head.train()

    for epoch in range(epochs):
        total_loss, n_steps = 0.0, 0
        for windows in all_windows:
            if len(windows) < 2:
                continue
            wt = torch.from_numpy(windows).to(device, dtype=torch.float32)

            # Encode all windows in one batched pass
            emb = model(wt)                                      # (n_windows, embed_dim)

            # Predict each window's successor
            pred_next = pred_head(emb[:-1])                      # (n_windows-1, flat_dim)
            target = wt[1:].reshape(len(wt) - 1, flat_dim)      # (n_windows-1, flat_dim)

            loss = nn.functional.mse_loss(pred_next, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        avg = total_loss / max(n_steps, 1)
        print(f"  [Baseline] epoch {epoch + 1:>2}/{epochs}  loss={avg:.5f}")

    return model


# ---------------------------------------------------------------------------
# Predictive Coding GRU training — prediction error minimization
# ---------------------------------------------------------------------------

def train_pc(
    model: PredictiveCodingEEG,
    all_windows: List[np.ndarray],
    epochs: int,
    lr: float,
    device: torch.device,
    tbptt_steps: int = 50,
) -> PredictiveCodingEEG:
    """
    Train the PC model by minimizing its built-in per-step prediction error.

    Uses truncated backprop through time (TBPTT) with a chunk size of tbptt_steps
    to keep memory bounded for long sequences. The GRU hidden state is carried
    across chunks but detached from the gradient graph at each boundary.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss, n_steps = 0.0, 0
        for windows in all_windows:
            if len(windows) < 2:
                continue
            wt = torch.from_numpy(windows).to(device, dtype=torch.float32)
            flat = wt.reshape(len(wt), model.flat_dim)

            # Carry hidden state across TBPTT chunks, detaching at each boundary
            h = torch.zeros(model.hidden_dim, device=device, dtype=flat.dtype)
            for start in range(0, len(flat) - 1, tbptt_steps):
                chunk = flat[start : start + tbptt_steps + 1]
                h = h.detach()

                chunk_losses: list = []
                for t in range(len(chunk) - 1):
                    e = model.enc(chunk[t])
                    h = model.gru(e, h)
                    pred_next = model.predict_next(h)
                    err = torch.mean((pred_next - chunk[t + 1]) ** 2)
                    chunk_losses.append(err)

                loss = torch.stack(chunk_losses).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_steps += 1

        avg = total_loss / max(n_steps, 1)
        print(f"  [PC model] epoch {epoch + 1:>2}/{epochs}  loss={avg:.5f}")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-supervised EEG model training")
    p.add_argument("--data_root", type=Path, default=_REPO)
    p.add_argument("--max_trials", type=int, default=100,
                   help="Number of EDF trials to train on")
    p.add_argument("--epochs", type=int, default=20,
                   help="Training epochs for each model")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window_sec", type=float, default=1.0)
    p.add_argument("--hop_sec", type=float, default=0.5)
    p.add_argument("--baseline_embed_dim", type=int, default=64)
    p.add_argument("--pc_hidden_dim", type=int, default=128)
    p.add_argument("--pc_embed_dim", type=int, default=64)
    p.add_argument("--tbptt_steps", type=int, default=50,
                   help="Truncated BPTT chunk size for PC model")
    p.add_argument("--device", type=str, default=None,
                   help="cpu / cuda / mps  (auto-detects if omitted)")
    p.add_argument("--out_dir", type=Path,
                   default=_REPO / "checkpoints" / "eeg_models")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    device = torch.device(args.device or _auto_device())
    print(f"Training on device: {device}")

    eeg_dir  = data_root / "EEG" / "EEG"
    perf_csv = data_root / "Eye" / "PerformanceScores.csv"
    table1   = data_root / "Eye" / "Table1.csv"

    # ---- Load data ----
    print(f"\nLoading up to {args.max_trials} EDF trials...")
    all_windows = load_all_windows(
        eeg_dir, perf_csv, table1,
        max_trials=args.max_trials,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
    )
    if not all_windows:
        sys.exit("No windows loaded — check that EDF files are in EEG/EEG/")

    n_channels = all_windows[0].shape[1]
    n_time     = all_windows[0].shape[2]
    n_windows_avg = int(np.mean([w.shape[0] for w in all_windows]))
    print(f"Window shape: {n_channels} channels × {n_time} time steps")
    print(f"Average windows per trial: {n_windows_avg}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Train baseline CNN ----
    print(f"\n{'='*60}")
    print("  Training Baseline CNN (next-window prediction)")
    print(f"{'='*60}")
    baseline = BaselineEEGTemporalCNN(
        n_channels=n_channels,
        n_time=n_time,
        embed_dim=args.baseline_embed_dim,
    ).to(device)
    baseline = train_baseline(baseline, all_windows, args.epochs, args.lr, device)
    baseline_path = out_dir / "baseline.pt"
    torch.save({
        "state_dict":  baseline.state_dict(),
        "n_channels":  n_channels,
        "n_time":      n_time,
        "embed_dim":   args.baseline_embed_dim,
    }, baseline_path)
    print(f"\nSaved → {baseline_path}")

    # ---- Train PC model ----
    print(f"\n{'='*60}")
    print("  Training Predictive Coding GRU (error minimization)")
    print(f"{'='*60}")
    pc = PredictiveCodingEEG(
        n_channels=n_channels,
        n_time=n_time,
        hidden_dim=args.pc_hidden_dim,
        embed_dim=args.pc_embed_dim,
    ).to(device)
    pc = train_pc(pc, all_windows, args.epochs, args.lr, device, args.tbptt_steps)
    pc_path = out_dir / "pc_model.pt"
    torch.save({
        "state_dict": pc.state_dict(),
        "n_channels": n_channels,
        "n_time":     n_time,
        "hidden_dim": args.pc_hidden_dim,
        "embed_dim":  args.pc_embed_dim,
    }, pc_path)
    print(f"\nSaved → {pc_path}")

    print(f"\n{'='*60}")
    print("  Training complete.")
    print(f"  Weights in: {out_dir}")
    print("  Next step:  bash run_trained_pipeline.sh")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
