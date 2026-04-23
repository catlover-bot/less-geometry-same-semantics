"""End-to-end point-cloud-to-semantics model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from less_geometry_same_semantics.models.adaptation import InputAdaptation
from less_geometry_same_semantics.models.compressor import TokenPoolingCompressor
from less_geometry_same_semantics.models.decoder import DecoderThresholds, OutputMode, StructuredSemanticDecoder
from less_geometry_same_semantics.models.encoder import LightweightPointEncoder
from less_geometry_same_semantics.models.graph import GraphConstruction, GraphMode, GraphReasoner, ObjectAbstraction


class PointSemanticsModel(nn.Module):
    """Minimal baseline: point encoder -> token pooling -> structured decoder."""

    def __init__(
        self,
        encoder_hidden_dim: int = 64,
        token_dim: int = 96,
        compressed_tokens: int = 16,
        decoder_hidden_dim: int = 128,
        decoder_depth: int = 2,
        object_threshold: float = 0.45,
        attribute_threshold: float = 0.45,
        relation_threshold: float = 0.50,
        graph_mode: GraphMode = "simple_graph",
        object_slots: int = 12,
        graph_layers: int = 1,
        graph_k_nearest: int = 4,
        graph_edge_dropout: float = 0.0,
        graph_edge_noise_std: float = 0.0,
        robustness_dropout: float = 0.0,
        adaptation_enabled: bool = False,
        adaptation_mode: str = "normalize",
    ) -> None:
        super().__init__()
        self.graph_mode = graph_mode
        self.compressed_tokens = compressed_tokens
        self.encoder = LightweightPointEncoder(hidden_dim=encoder_hidden_dim, token_dim=token_dim)
        self.input_adaptation = InputAdaptation(enabled=adaptation_enabled, mode=adaptation_mode)
        self.compressor = TokenPoolingCompressor(compressed_tokens=compressed_tokens)
        self.object_abstraction = ObjectAbstraction(object_slots=object_slots)
        self.graph_construction = GraphConstruction(
            mode=graph_mode,
            k_nearest=graph_k_nearest,
            edge_dropout=graph_edge_dropout,
            edge_noise_std=graph_edge_noise_std,
        )
        self.graph_reasoner = GraphReasoner(
            token_dim=token_dim,
            layers=graph_layers if graph_mode != "no_graph" else 1,
            dropout=robustness_dropout,
        )
        self.robustness_dropout = nn.Dropout(robustness_dropout)
        self.decoder = StructuredSemanticDecoder(
            token_dim=token_dim,
            hidden_dim=decoder_hidden_dim,
            depth=decoder_depth,
            thresholds=DecoderThresholds(
                object_threshold=object_threshold,
                attribute_threshold=attribute_threshold,
                relation_threshold=relation_threshold,
            ),
        )

    def forward(self, points: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if mask is None:
            mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        points = self.input_adaptation(points, mask)
        tokens = self.encoder(points, mask)
        tokens = self.robustness_dropout(tokens)
        compressed_tokens, compressed_mask = self.compressor(tokens, mask)
        if self.graph_mode == "no_graph":
            graph_nodes = compressed_tokens
            graph_mask = compressed_mask
            graph_adjacency = torch.zeros(
                points.shape[0],
                compressed_tokens.shape[1],
                compressed_tokens.shape[1],
                dtype=compressed_tokens.dtype,
                device=compressed_tokens.device,
            )
        else:
            graph_nodes, graph_positions, graph_mask = self.object_abstraction(tokens, points, mask)
            graph_adjacency = self.graph_construction(graph_positions, graph_mask)
            graph_nodes = self.graph_reasoner(graph_nodes, graph_adjacency, graph_mask)
        logits = self.decoder(graph_nodes, graph_mask)
        logits["compressed_tokens"] = compressed_tokens
        logits["compressed_mask"] = compressed_mask
        logits["graph_nodes"] = graph_nodes
        logits["graph_mask"] = graph_mask
        logits["graph_adjacency"] = graph_adjacency
        return logits

    @torch.no_grad()
    def predict(
        self,
        points: torch.Tensor,
        mask: torch.Tensor | None = None,
        output_mode: OutputMode = "json",
        constrained: bool = True,
    ) -> list[dict[str, object] | str]:
        self.eval()
        logits = self(points, mask)
        return self.decoder.decode(logits, output_mode=output_mode, constrained=constrained)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PointSemanticsModel":
        model_cfg = config.get("model", {})
        return cls(
            encoder_hidden_dim=int(model_cfg.get("encoder_hidden_dim", 64)),
            token_dim=int(model_cfg.get("token_dim", 96)),
            compressed_tokens=int(model_cfg.get("compressed_tokens", 16)),
            decoder_hidden_dim=int(model_cfg.get("decoder_hidden_dim", 128)),
            decoder_depth=int(model_cfg.get("decoder_depth", 2)),
            object_threshold=float(model_cfg.get("object_threshold", 0.45)),
            attribute_threshold=float(model_cfg.get("attribute_threshold", 0.45)),
            relation_threshold=float(model_cfg.get("relation_threshold", 0.50)),
            graph_mode=str(model_cfg.get("graph_mode", "simple_graph")),
            object_slots=int(model_cfg.get("object_slots", 12)),
            graph_layers=int(model_cfg.get("graph_layers", 1)),
            graph_k_nearest=int(model_cfg.get("graph_k_nearest", 4)),
            graph_edge_dropout=float(model_cfg.get("graph_edge_dropout", 0.0)),
            graph_edge_noise_std=float(model_cfg.get("graph_edge_noise_std", 0.0)),
            robustness_dropout=float(model_cfg.get("robustness_dropout", 0.0)),
            adaptation_enabled=bool(model_cfg.get("adaptation_enabled", False)),
            adaptation_mode=str(model_cfg.get("adaptation_mode", "normalize")),
        )
