"""Token compression modules."""

from __future__ import annotations

import torch
from torch import nn


class TokenPoolingCompressor(nn.Module):
    """Reduce variable point-token sequences to a fixed token budget by chunk mean."""

    def __init__(self, compressed_tokens: int = 16) -> None:
        super().__init__()
        if compressed_tokens < 1:
            raise ValueError("compressed_tokens must be positive.")
        self.compressed_tokens = compressed_tokens

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with shape [B, N, D], got {tuple(tokens.shape)}")

        batch_size, num_tokens, token_dim = tokens.shape
        if mask is None:
            mask = torch.ones(batch_size, num_tokens, dtype=torch.bool, device=tokens.device)

        pooled = torch.zeros(
            batch_size,
            self.compressed_tokens,
            token_dim,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        pooled_mask = torch.zeros(
            batch_size,
            self.compressed_tokens,
            dtype=torch.bool,
            device=tokens.device,
        )

        for out_idx in range(self.compressed_tokens):
            start = int(round(out_idx * num_tokens / self.compressed_tokens))
            end = int(round((out_idx + 1) * num_tokens / self.compressed_tokens))
            end = max(end, start + 1)
            chunk_tokens = tokens[:, start:end]
            chunk_mask = mask[:, start:end]
            denom = chunk_mask.sum(dim=1).clamp_min(1).to(tokens.dtype).unsqueeze(-1)
            pooled[:, out_idx] = (chunk_tokens * chunk_mask.unsqueeze(-1).to(tokens.dtype)).sum(dim=1) / denom
            pooled_mask[:, out_idx] = chunk_mask.any(dim=1)

        return pooled, pooled_mask
