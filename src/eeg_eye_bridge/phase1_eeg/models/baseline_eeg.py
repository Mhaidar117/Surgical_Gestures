"""
Conventional temporal CNN encoder over single windows (per-window embeddings).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BaselineEEGTemporalCNN(nn.Module):
    """
    Input: ``(batch, n_channels, n_time)`` — one segment per batch item.

    Output: embedding of shape ``(batch, embed_dim)``.
    """

    def __init__(
        self,
        n_channels: int,
        n_time: int,
        embed_dim: int = 64,
        base_channels: int = 64,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_time = n_time
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, base_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(base_channels * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_time)
        Returns:
            (batch, embed_dim)
        """
        h = self.encoder(x).squeeze(-1)
        return self.fc(h)

    def encode_sequence(self, windows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            windows: (n_windows, n_channels, n_time)
        Returns:
            per_window: (n_windows, embed_dim)
            trial_mean: (embed_dim,) mean over windows
        """
        self.eval()
        with torch.no_grad():
            emb = self(windows)
            mean = emb.mean(dim=0)
        return emb, mean
