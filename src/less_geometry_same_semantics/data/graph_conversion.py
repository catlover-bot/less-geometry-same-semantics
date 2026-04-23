"""Conversion utilities between semantic targets and scene graphs."""

from __future__ import annotations

from typing import Any

from less_geometry_same_semantics.data.semantic import ObjectRecord, RelationRecord, SceneGraph
from less_geometry_same_semantics.metrics.semantic import object_count_map


def target_to_scene_graph(target: dict[str, Any], metadata: dict[str, Any] | None = None) -> SceneGraph:
    """Convert a structured target dictionary into a graph record."""

    counts = object_count_map(target)
    nodes = []
    for item in target.get("objects", []):
        if isinstance(item, dict):
            category = str(item.get("category"))
            nodes.append(
                ObjectRecord(
                    category=category,
                    count=int(item.get("count", counts.get(category, 1))),
                    attributes=tuple(str(attr) for attr in item.get("attributes", [])),
                )
            )
    edges = [
        RelationRecord(
            subject=str(relation.get("subject")),
            predicate=str(relation.get("predicate")),
            object=str(relation.get("object")),
        )
        for relation in target.get("relations", [])
    ]
    return SceneGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        scene_type=str(target.get("scene_type", "room")),
        attributes=tuple(str(attr) for attr in target.get("attributes", [])),
        caption=target.get("caption"),
        metadata=metadata or {},
    )


def scene_graph_to_target(graph: SceneGraph) -> dict[str, Any]:
    """Convert a graph record into the shared structured target schema."""

    return graph.to_target()
