"""Lightweight test-time adaptation baselines for corrupted point clouds."""

from __future__ import annotations

import torch
from torch import nn


class InputAdaptation(nn.Module):
    """Simple deterministic input adaptation before point encoding.

    Modes:
    - `normalize`: center and scale each scene with masked statistics.
    - `denoise`: additionally clips extreme radial outliers.
    """

    def __init__(self, enabled: bool = False, mode: str = "normalize", outlier_quantile: float = 0.98) -> None:
        super().__init__()
        self.enabled = enabled
        self.mode = mode
        self.outlier_quantile = outlier_quantile

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return points
        adapted = points.clone()
        weights = mask.unsqueeze(-1).to(points.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        center = (adapted * weights).sum(dim=1, keepdim=True) / denom
        adapted = (adapted - center) * weights
        scale = torch.sqrt(((adapted**2) * weights).sum(dim=(1, 2), keepdim=True) / denom.clamp_min(1.0)).clamp_min(1e-4)
        adapted = adapted / scale
        if self.mode == "denoise":
            radii = torch.linalg.norm(adapted, dim=-1).masked_fill(~mask, 0.0)
            thresholds = torch.quantile(radii, self.outlier_quantile, dim=1, keepdim=True).clamp_min(1.0)
            adapted = adapted * (radii <= thresholds).unsqueeze(-1).to(adapted.dtype)
        return adapted
