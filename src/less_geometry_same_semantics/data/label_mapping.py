"""Dataset-specific label mapping into the shared coarse schema."""

from __future__ import annotations

import re

from less_geometry_same_semantics.data.constants import (
    ATTRIBUTES,
    OBJECT_CATEGORIES,
    RELATION_PREDICATES,
    SCENE_TYPES,
)

OBJECT_SYNONYMS: dict[str, str] = {
    "armchair": "chair",
    "office chair": "chair",
    "dining chair": "chair",
    "desk": "table",
    "coffee table": "table",
    "side table": "table",
    "couch": "sofa",
    "loveseat": "sofa",
    "floor lamp": "lamp",
    "desk lamp": "lamp",
    "potted plant": "plant",
    "bookshelf": "cabinet",
    "shelf": "cabinet",
    "shelves": "cabinet",
    "storage": "cabinet",
    "storage cabinet": "cabinet",
    "wardrobe": "cabinet",
    "cupboard": "cabinet",
    "stool": "chair",
    "tv stand": "cabinet",
    "tv": "lamp",
    "television": "lamp",
    "tv monitor": "lamp",
    "bed": "sofa",
    "counter": "table",
    "countertop": "table",
    "kitchen counter": "table",
    "sink": "cabinet",
    "refrigerator": "cabinet",
    "fridge": "cabinet",
    "oven": "cabinet",
    "stove": "cabinet",
    "dishwasher": "cabinet",
    "washer": "cabinet",
    "washer or dryer": "cabinet",
    "dryer": "cabinet",
    "fireplace": "cabinet",
    "bathtub": "sofa",
    "toilet": "chair",
    "stairs": "table",
}

RELATION_SYNONYMS: dict[str, str] = {
    "left": "left_of",
    "left of": "left_of",
    "right": "right_of",
    "right of": "right_of",
    "front": "in_front_of",
    "in front of": "in_front_of",
    "behind": "behind",
    "back": "behind",
    "near": "near",
    "close by": "near",
    "next to": "near",
    "beside": "near",
    "same as": "near",
    "standing on": "near",
    "supported by": "near",
    "above": "above",
    "higher than": "above",
    "below": "below",
    "under": "below",
    "overlapping": "overlapping",
    "intersecting": "overlapping",
}

SCENE_SYNONYMS: dict[str, str] = {
    "bathroom": "room",
    "bedroom": "room",
    "kitchen": "room",
    "office": "office",
    "living room": "living_room",
    "living_room": "living_room",
    "storage": "storage",
    "closet": "storage",
}


def normalize_label(label: object) -> str:
    """Normalize free-text dataset labels."""

    text = str(label).strip().lower().replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def map_object_category(label: object) -> str | None:
    """Map a dataset object label into the shared coarse object taxonomy."""

    normalized = normalize_label(label)
    if normalized in OBJECT_CATEGORIES:
        return normalized
    return OBJECT_SYNONYMS.get(normalized)


def map_relation_predicate(label: object) -> str | None:
    """Map a dataset relation label into the shared coarse predicate set."""

    normalized = normalize_label(label)
    if normalized in RELATION_PREDICATES:
        return normalized
    return RELATION_SYNONYMS.get(normalized)


def map_scene_type(label: object | None) -> str:
    """Map a dataset scene label into the shared scene-type set."""

    if label is None:
        return "room"
    normalized = normalize_label(label)
    if normalized in SCENE_TYPES:
        return normalized
    return SCENE_SYNONYMS.get(normalized, "room")


def map_attributes(labels: list[object] | tuple[object, ...] | None) -> tuple[str, ...]:
    """Keep only coarse attributes supported by the shared schema."""

    if not labels:
        return ()
    mapped = []
    for label in labels:
        normalized = normalize_label(label)
        if normalized in ATTRIBUTES:
            mapped.append(normalized)
    return tuple(sorted(set(mapped)))
