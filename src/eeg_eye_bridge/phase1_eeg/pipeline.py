"""
Run baseline + predictive-coding encoders on windowed EEG (numpy in / numpy out).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from eeg_eye_bridge.phase1_eeg.models import BaselineEEGTemporalCNN, PredictiveCodingEEG


def run_encoders_on_windows(
    windows: np.ndarray,
    device: str = "cpu",
    baseline_embed_dim: int = 64,
    pc_hidden_dim: int = 128,
    pc_embed_dim: int = 64,
    baseline_ckpt: Optional[Path] = None,
    pc_ckpt: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Args:
        windows: (n_windows, n_channels, n_time) float32
        baseline_ckpt: optional path to a checkpoint saved by train_eeg_models.py
        pc_ckpt: optional path to a checkpoint saved by train_eeg_models.py

    Returns:
        dict with numpy arrays: baseline_embeddings, pc_embeddings, prediction_errors,
        baseline_trial_mean, pc_trial_mean, baseline_embed_dim, pc_embed_dim
    """
    n_w, n_c, n_t = windows.shape
    dev = torch.device(device)
    wt = torch.from_numpy(windows).to(dev, dtype=torch.float32)

    baseline = BaselineEEGTemporalCNN(
        n_channels=n_c,
        n_time=n_t,
        embed_dim=baseline_embed_dim,
    ).to(dev)
    if baseline_ckpt is not None:
        ckpt = torch.load(baseline_ckpt, map_location=dev, weights_only=True)
        baseline.load_state_dict(ckpt["state_dict"])

    pc = PredictiveCodingEEG(
        n_channels=n_c,
        n_time=n_t,
        hidden_dim=pc_hidden_dim,
        embed_dim=pc_embed_dim,
    ).to(dev)
    if pc_ckpt is not None:
        ckpt = torch.load(pc_ckpt, map_location=dev, weights_only=True)
        pc.load_state_dict(ckpt["state_dict"])

    b_emb, b_mean = baseline.encode_sequence(wt)
    pc.eval()
    with torch.no_grad():
        pc_emb, pred_err = pc.forward_sequence(wt)
    pc_mean = pc_emb.mean(dim=0)

    return {
        "baseline_embeddings": b_emb.detach().cpu().numpy().astype(np.float32),
        "pc_embeddings": pc_emb.detach().cpu().numpy().astype(np.float32),
        "prediction_errors": pred_err.detach().cpu().numpy().astype(np.float32),
        "baseline_trial_mean": b_mean.detach().cpu().numpy().astype(np.float32),
        "pc_trial_mean": pc_mean.detach().cpu().numpy().astype(np.float32),
        "baseline_embed_dim": baseline_embed_dim,
        "pc_embed_dim": pc_embed_dim,
    }
