"""
Predictive-coding-inspired sequence model: latent state, next-window prediction, local error.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PredictiveCodingEEG(nn.Module):
    """
    Flatten each window, encode, update GRU state, predict the next window.

    After processing window ``t`` (0-based), prediction targets window ``t+1``.
    Per-step PC embedding is a linear projection of the hidden state.
    """

    def __init__(
        self,
        n_channels: int,
        n_time: int,
        hidden_dim: int = 128,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_time = n_time
        flat_dim = n_channels * n_time
        self.flat_dim = flat_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.enc = nn.Sequential(nn.Linear(flat_dim, hidden_dim), nn.ReLU(inplace=True))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.predict_next = nn.Linear(hidden_dim, flat_dim)
        self.pc_proj = nn.Linear(hidden_dim, embed_dim)

    def forward_sequence(
        self,
        windows: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            windows: (n_windows, n_channels, n_time)

        Returns:
            pc_embeddings: (n_windows, embed_dim)
            prediction_errors: (n_windows,) MSE to next window; last is 0
        """
        device = windows.device
        w = windows.shape[0]
        flat = windows.reshape(w, self.flat_dim)
        h = torch.zeros(self.hidden_dim, device=device, dtype=flat.dtype)
        pcs: list = []
        errs: list = []
        for t in range(w):
            e = self.enc(flat[t])
            h = self.gru(e, h)
            pcs.append(self.pc_proj(h))
            if t < w - 1:
                pred = self.predict_next(h)
                err = torch.mean((pred - flat[t + 1]) ** 2)
                errs.append(err)
            else:
                errs.append(torch.zeros((), device=device, dtype=flat.dtype))
        pc_emb = torch.stack(pcs, dim=0)
        pred_err = torch.stack(errs, dim=0)
        return pc_emb, pred_err
