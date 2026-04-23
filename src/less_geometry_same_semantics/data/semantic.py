"""Unified semantic and graph data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from less_geometry_same_semantics.schemas.schema import enforce_semantic_schema


@dataclass(frozen=True)
class ObjectRecord:
    """Common object record used across synthetic and public ARKitScenes targets."""

    category: str
    count: int = 1
    attributes: tuple[str, ...] = ()
    instance_id: str | None = None

    def to_schema(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "count": int(self.count),
            "attributes": sorted(set(self.attributes)),
        }


@dataclass(frozen=True)
class RelationRecord:
    """Coarse relation triple."""

    subject: str
    predicate: str
    object: str
    subject_id: str | None = None
    object_id: str | None = None

    def to_schema(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }


@dataclass(frozen=True)
class SceneGraph:
    """Graph-centric semantic bottleneck representation."""

    nodes: tuple[ObjectRecord, ...]
    edges: tuple[RelationRecord, ...]
    scene_type: str = "room"
    attributes: tuple[str, ...] = ()
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_target(self) -> dict[str, Any]:
        """Convert graph records into the shared structured target schema."""

        counts: dict[str, int] = {}
        category_attributes: dict[str, set[str]] = {}
        for node in self.nodes:
            counts[node.category] = counts.get(node.category, 0) + int(node.count)
            category_attributes.setdefault(node.category, set()).update(node.attributes)

        objects = [
            {
                "category": category,
                "count": count,
                "attributes": sorted(category_attributes.get(category, set())),
            }
            for category, count in sorted(counts.items())
        ]
        attributes = sorted(set(self.attributes) | {attr for attrs in category_attributes.values() for attr in attrs})
        payload = {
            "objects": objects,
            "object_counts": {category: counts[category] for category in sorted(counts)},
            "attributes": attributes,
            "relations": [edge.to_schema() for edge in self.edges],
            "scene_type": self.scene_type,
        }
        if self.caption:
            payload["caption"] = self.caption
        return enforce_semantic_schema(payload)
