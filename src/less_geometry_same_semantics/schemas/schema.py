"""JSON schema loading, validation, and conservative enforcement."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from less_geometry_same_semantics.data.constants import (
    ATTRIBUTE_SET,
    OBJECT_CATEGORIES,
    OBJECT_CATEGORY_SET,
    RELATION_PREDICATE_SET,
    SCENE_TYPES,
    SCENE_TYPE_SET,
)


@lru_cache(maxsize=1)
def load_semantic_schema() -> dict[str, Any]:
    """Load the structured semantic output JSON schema."""

    schema_path = Path(__file__).with_name("semantic_output.schema.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(load_semantic_schema())


def validate_semantic_output(payload: dict[str, Any]) -> None:
    """Raise a jsonschema error if payload violates the schema."""

    _validator().validate(payload)


def is_valid_semantic_output(payload: dict[str, Any]) -> bool:
    """Return True when payload complies with the structured output schema."""

    try:
        validate_semantic_output(payload)
    except Exception:
        return False
    return True


def enforce_semantic_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a schema-compatible payload with conservative defaults.

    This is intentionally not a full constrained decoder. It is a small,
    deterministic post-processor that keeps experiments comparable.
    """

    object_counts = _normalize_object_counts(payload.get("object_counts", {}))
    objects = _normalize_objects(payload.get("objects", []), object_counts)
    for item in objects:
        object_counts[item["category"]] = max(int(item["count"]), object_counts.get(item["category"], 0))

    attributes = sorted(
        {
            str(item)
            for item in payload.get("attributes", [])
            if str(item) in ATTRIBUTE_SET
        }
    )
    relations = []
    for relation in payload.get("relations", []):
        if not isinstance(relation, dict):
            continue
        subject = relation.get("subject")
        predicate = relation.get("predicate")
        obj = relation.get("object")
        if subject is None or predicate is None or obj is None:
            continue
        if str(subject) not in OBJECT_CATEGORY_SET:
            continue
        if str(obj) not in OBJECT_CATEGORY_SET:
            continue
        if str(predicate) not in RELATION_PREDICATE_SET:
            continue
        relations.append(
            {
                "subject": str(subject),
                "predicate": str(predicate),
                "object": str(obj),
            }
        )

    scene_type = str(payload.get("scene_type", "room"))
    if scene_type not in SCENE_TYPE_SET:
        scene_type = SCENE_TYPES[0]

    enforced = {
        "objects": objects,
        "object_counts": {key: value for key, value in object_counts.items() if value > 0},
        "attributes": attributes,
        "relations": relations,
        "scene_type": scene_type,
    }
    if "caption" in payload:
        enforced["caption"] = str(payload["caption"])
    validate_semantic_output(enforced)
    return enforced


def _normalize_object_counts(raw_counts: Any) -> dict[str, int]:
    if not isinstance(raw_counts, dict):
        return {}
    counts: dict[str, int] = {}
    for category, count in raw_counts.items():
        category = str(category)
        if category not in OBJECT_CATEGORY_SET:
            continue
        try:
            value = max(0, int(count))
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            counts[category] = value
    return counts


def _normalize_objects(raw_objects: Any, object_counts: dict[str, int]) -> list[dict[str, Any]]:
    if not isinstance(raw_objects, list):
        raw_objects = []

    by_category: dict[str, dict[str, Any]] = {}
    for item in raw_objects:
        if isinstance(item, str):
            category = item
            count = object_counts.get(category, 1)
            attributes: list[str] = []
        elif isinstance(item, dict):
            category = str(item.get("category", item.get("name", item.get("label", ""))))
            try:
                count = int(item.get("count", object_counts.get(category, 1)))
            except (TypeError, ValueError):
                count = object_counts.get(category, 1)
            attributes = [
                str(attr)
                for attr in item.get("attributes", [])
                if str(attr) in ATTRIBUTE_SET
            ]
        else:
            continue

        if category not in OBJECT_CATEGORY_SET or count <= 0:
            continue
        existing = by_category.get(category)
        merged_attributes = sorted(set(attributes) | set(existing["attributes"] if existing else []))
        by_category[category] = {
            "category": category,
            "count": max(count, int(existing["count"]) if existing else 0),
            "attributes": merged_attributes,
        }

    for category, count in object_counts.items():
        if count > 0 and category not in by_category:
            by_category[category] = {"category": category, "count": count, "attributes": []}

    return [by_category[category] for category in OBJECT_CATEGORIES if category in by_category]
