"""Structured semantic decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from less_geometry_same_semantics.data.constants import (
    ATTRIBUTES,
    OBJECT_CATEGORIES,
    RELATION_LABELS,
    SCENE_TYPES,
)
from less_geometry_same_semantics.schemas.schema import enforce_semantic_schema

OutputMode = Literal["json", "text"]


@dataclass(frozen=True)
class DecoderThresholds:
    """Thresholds for turning logits into structured predictions."""

    object_threshold: float = 0.45
    attribute_threshold: float = 0.45
    relation_threshold: float = 0.50


def _mlp(input_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(max(1, depth)):
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)])
        current_dim = hidden_dim
    return nn.Sequential(*layers)


class StructuredSemanticDecoder(nn.Module):
    """Predict object, attribute, relation, and scene-type labels."""

    def __init__(
        self,
        token_dim: int = 96,
        hidden_dim: int = 128,
        depth: int = 2,
        thresholds: DecoderThresholds | None = None,
    ) -> None:
        super().__init__()
        self.thresholds = thresholds or DecoderThresholds()
        self.backbone = _mlp(token_dim, hidden_dim, depth)
        self.object_head = nn.Linear(hidden_dim, len(OBJECT_CATEGORIES))
        self.attribute_head = nn.Linear(hidden_dim, len(ATTRIBUTES))
        self.relation_head = nn.Linear(hidden_dim, len(RELATION_LABELS))
        self.scene_head = nn.Linear(hidden_dim, len(SCENE_TYPES))

    def forward(self, compressed_tokens: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if compressed_tokens.ndim != 3:
            raise ValueError(
                f"Expected compressed tokens with shape [B, K, D], got {tuple(compressed_tokens.shape)}"
            )
        if mask is None:
            mask = torch.ones(
                compressed_tokens.shape[:2],
                dtype=torch.bool,
                device=compressed_tokens.device,
            )
        denom = mask.sum(dim=1).clamp_min(1).to(compressed_tokens.dtype).unsqueeze(-1)
        pooled = (compressed_tokens * mask.unsqueeze(-1).to(compressed_tokens.dtype)).sum(dim=1) / denom
        features = self.backbone(pooled)
        return {
            "object_logits": self.object_head(features),
            "attribute_logits": self.attribute_head(features),
            "relation_logits": self.relation_head(features),
            "scene_logits": self.scene_head(features),
        }

    @torch.no_grad()
    def decode(
        self,
        logits: dict[str, torch.Tensor],
        output_mode: OutputMode = "json",
        constrained: bool = True,
    ) -> list[dict[str, object] | str]:
        """Decode a batch of logits to JSON-compatible dicts or text strings."""

        object_probs = torch.sigmoid(logits["object_logits"])
        attribute_probs = torch.sigmoid(logits["attribute_logits"])
        relation_probs = torch.sigmoid(logits["relation_logits"])
        scene_idx = torch.argmax(logits["scene_logits"], dim=-1)

        outputs: list[dict[str, object] | str] = []
        for batch_idx in range(object_probs.shape[0]):
            objects = self._decode_multilabel(
                object_probs[batch_idx],
                OBJECT_CATEGORIES,
                self.thresholds.object_threshold,
                fallback=True,
            )
            attributes = self._decode_multilabel(
                attribute_probs[batch_idx],
                ATTRIBUTES,
                self.thresholds.attribute_threshold,
                fallback=False,
            )
            relations = self._decode_relations(
                relation_probs[batch_idx],
                known_objects=set(objects),
                constrained=constrained,
            )
            scene_type = SCENE_TYPES[int(scene_idx[batch_idx].item())]
            object_records = [
                {"category": category, "count": 1, "attributes": []}
                for category in objects
            ]
            payload = {
                "objects": object_records,
                "object_counts": {category: 1 for category in objects},
                "attributes": attributes,
                "relations": relations,
                "scene_type": scene_type,
                "caption": self._caption(scene_type, objects),
            }
            payload = enforce_semantic_schema(payload) if constrained else payload
            if output_mode == "text":
                outputs.append(self._free_form(payload))
            else:
                outputs.append(payload)
        return outputs

    @staticmethod
    def _decode_multilabel(
        probs: torch.Tensor,
        labels: tuple[str, ...],
        threshold: float,
        fallback: bool,
    ) -> list[str]:
        selected = [labels[i] for i, value in enumerate(probs.tolist()) if value >= threshold]
        if fallback and not selected:
            selected = [labels[int(torch.argmax(probs).item())]]
        return sorted(selected)

    def _decode_relations(
        self,
        probs: torch.Tensor,
        known_objects: set[str],
        constrained: bool,
    ) -> list[dict[str, str]]:
        selected = [
            RELATION_LABELS[i]
            for i, value in enumerate(probs.tolist())
            if value >= self.thresholds.relation_threshold
        ]
        relations: list[dict[str, str]] = []
        for label in selected[:8]:
            subject, predicate, obj = label.split(":")
            if constrained and (subject not in known_objects or obj not in known_objects):
                continue
            relations.append({"subject": subject, "predicate": predicate, "object": obj})
        return relations

    @staticmethod
    def _caption(scene_type: str, objects: list[str]) -> str:
        if not objects:
            return f"A {scene_type}."
        if len(objects) == 1:
            object_text = objects[0]
        else:
            object_text = ", ".join(objects[:-1]) + f" and {objects[-1]}"
        return f"A {scene_type} with {object_text}."

    @staticmethod
    def _free_form(payload: dict[str, object]) -> str:
        objects = payload.get("objects", [])
        categories = [
            str(item.get("category", ""))
            for item in objects
            if isinstance(item, dict) and item.get("category")
        ]
        objects_text = ", ".join(categories) or "unknown objects"
        scene_type = str(payload.get("scene_type", "scene"))
        relation_count = len(payload.get("relations", []))
        return f"{scene_type}: {objects_text}. {relation_count} coarse relations predicted."
