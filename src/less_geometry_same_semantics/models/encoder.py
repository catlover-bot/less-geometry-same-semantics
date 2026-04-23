"""Lightweight point-token encoder."""

from __future__ import annotations

import torch
from torch import nn


class LightweightPointEncoder(nn.Module):
    """A small PointNet-style per-point MLP.

    The encoder intentionally avoids neighborhood search and heavy geometry
    operations. Each point becomes a token; later modules decide how many tokens
    are worth keeping.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, token_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )

    def forward(self, points: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points with shape [B, N, 3], got {tuple(points.shape)}")
        tokens = self.net(points)
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens
