"""Lightweight graph-centric intermediate representation modules."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

GraphMode = Literal["no_graph", "simple_graph", "richer_graph"]


class ObjectAbstraction(nn.Module):
    """Abstract point/token features into object-like graph nodes by chunk pooling."""

    def __init__(self, object_slots: int = 12) -> None:
        super().__init__()
        if object_slots < 1:
            raise ValueError("object_slots must be positive.")
        self.object_slots = object_slots

    def forward(
        self,
        tokens: torch.Tensor,
        points: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, token_dim = tokens.shape
        node_features = torch.zeros(
            batch_size,
            self.object_slots,
            token_dim,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        node_positions = torch.zeros(
            batch_size,
            self.object_slots,
            3,
            dtype=points.dtype,
            device=points.device,
        )
        node_mask = torch.zeros(batch_size, self.object_slots, dtype=torch.bool, device=tokens.device)
        for node_idx in range(self.object_slots):
            start = int(round(node_idx * num_tokens / self.object_slots))
            end = max(start + 1, int(round((node_idx + 1) * num_tokens / self.object_slots)))
            chunk_mask = mask[:, start:end]
            denom = chunk_mask.sum(dim=1).clamp_min(1).to(tokens.dtype).unsqueeze(-1)
            node_features[:, node_idx] = (
                tokens[:, start:end] * chunk_mask.unsqueeze(-1).to(tokens.dtype)
            ).sum(dim=1) / denom
            node_positions[:, node_idx] = (
                points[:, start:end] * chunk_mask.unsqueeze(-1).to(points.dtype)
            ).sum(dim=1) / denom.to(points.dtype)
            node_mask[:, node_idx] = chunk_mask.any(dim=1)
        return node_features, node_positions, node_mask


class GraphConstruction(nn.Module):
    """Construct simple adjacency matrices from object-node positions."""

    def __init__(
        self,
        mode: GraphMode = "simple_graph",
        k_nearest: int = 4,
        edge_dropout: float = 0.0,
        edge_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.k_nearest = k_nearest
        self.edge_dropout = edge_dropout
        self.edge_noise_std = edge_noise_std

    def forward(self, positions: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, _ = positions.shape
        adjacency = torch.zeros(batch_size, node_count, node_count, dtype=positions.dtype, device=positions.device)
        if self.mode == "no_graph":
            return adjacency

        distances = torch.cdist(positions, positions)
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        distances = distances.masked_fill(~valid, float("inf"))
        eye = torch.eye(node_count, dtype=torch.bool, device=positions.device).unsqueeze(0)
        distances = distances.masked_fill(eye, float("inf"))

        if self.mode == "simple_graph":
            adjacency = torch.where(torch.isfinite(distances), torch.exp(-distances), adjacency)
        elif self.mode == "richer_graph":
            k = min(max(1, self.k_nearest), max(1, node_count - 1))
            nearest = torch.topk(distances, k=k, largest=False, dim=-1).indices
            adjacency.scatter_(dim=-1, index=nearest, value=1.0)
            adjacency = adjacency * valid.to(adjacency.dtype)
            adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))
        else:
            raise ValueError(f"Unknown graph mode: {self.mode}")

        adjacency = adjacency.masked_fill(eye, 0.0)
        if self.edge_noise_std > 0.0:
            adjacency = (adjacency + torch.randn_like(adjacency) * self.edge_noise_std).clamp_min(0.0)
        if self.edge_dropout > 0.0:
            keep = torch.rand_like(adjacency) > self.edge_dropout
            adjacency = adjacency * keep.to(adjacency.dtype)
        denom = adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return adjacency / denom


class GraphReasoner(nn.Module):
    """Tiny message-passing module for graph bottleneck reasoning."""

    def __init__(self, token_dim: int, layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = layers
        self.dropout = nn.Dropout(dropout)
        self.update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(token_dim * 2, token_dim),
                    nn.GELU(),
                    nn.LayerNorm(token_dim),
                )
                for _ in range(max(1, layers))
            ]
        )

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        features = node_features
        for update in self.update[: self.layers]:
            messages = adjacency @ features
            features = update(torch.cat([features, messages], dim=-1))
            features = self.dropout(features)
            features = features * node_mask.unsqueeze(-1).to(features.dtype)
        return features
