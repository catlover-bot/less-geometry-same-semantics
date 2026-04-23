from __future__ import annotations

from less_geometry_same_semantics.schemas.schema import (
    enforce_semantic_schema,
    is_valid_semantic_output,
)


def test_schema_validation_accepts_expected_output() -> None:
    payload = {
        "objects": [{"category": "chair", "count": 1, "attributes": ["small"]}],
        "object_counts": {"chair": 1},
        "attributes": ["small"],
        "relations": [{"subject": "chair", "predicate": "near", "object": "table"}],
        "scene_type": "room",
        "caption": "A room with chair.",
    }

    assert is_valid_semantic_output(payload)


def test_schema_enforcement_fills_defaults() -> None:
    enforced = enforce_semantic_schema({"objects": ["chair", "chair"], "scene_type": "room"})

    assert enforced["objects"] == [{"category": "chair", "count": 1, "attributes": []}]
    assert enforced["object_counts"] == {"chair": 1}
    assert enforced["attributes"] == []
    assert enforced["relations"] == []
    assert is_valid_semantic_output(enforced)
