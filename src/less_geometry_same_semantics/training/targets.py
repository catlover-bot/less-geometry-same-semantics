"""Target encoding for structured semantic labels."""

from __future__ import annotations

from typing import Any

import torch

from less_geometry_same_semantics.data.constants import (
    ATTRIBUTES,
    OBJECT_CATEGORIES,
    RELATION_LABELS,
    SCENE_TYPES,
)
from less_geometry_same_semantics.metrics.semantic import object_categories


def relation_to_label(relation: dict[str, str]) -> str:
    """Convert a relation dict into the decoder's flat label format."""

    return f"{relation['subject']}:{relation['predicate']}:{relation['object']}"


def _multi_hot(items: list[str], labels: tuple[str, ...], device: torch.device) -> torch.Tensor:
    vector = torch.zeros(len(labels), dtype=torch.float32, device=device)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    for item in items:
        idx = label_to_idx.get(item)
        if idx is not None:
            vector[idx] = 1.0
    return vector


def encode_targets(targets: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    """Encode JSON targets into tensors used by the baseline losses."""

    object_targets = []
    attribute_targets = []
    relation_targets = []
    scene_targets = []
    scene_to_idx = {scene: i for i, scene in enumerate(SCENE_TYPES)}

    for target in targets:
        object_targets.append(_multi_hot(object_categories(target), OBJECT_CATEGORIES, device))
        attribute_targets.append(_multi_hot(list(target.get("attributes", [])), ATTRIBUTES, device))
        relation_labels = [relation_to_label(rel) for rel in target.get("relations", [])]
        relation_targets.append(_multi_hot(relation_labels, RELATION_LABELS, device))
        scene_targets.append(scene_to_idx.get(str(target.get("scene_type", "room")), 0))

    return {
        "object_targets": torch.stack(object_targets, dim=0),
        "attribute_targets": torch.stack(attribute_targets, dim=0),
        "relation_targets": torch.stack(relation_targets, dim=0),
        "scene_targets": torch.tensor(scene_targets, dtype=torch.long, device=device),
    }
