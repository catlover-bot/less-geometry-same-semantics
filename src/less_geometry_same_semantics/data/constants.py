"""Shared label spaces for the initial synthetic baseline."""

from __future__ import annotations

OBJECT_CATEGORIES: tuple[str, ...] = (
    "chair",
    "table",
    "sofa",
    "lamp",
    "plant",
    "cabinet",
)

ATTRIBUTES: tuple[str, ...] = (
    "small",
    "large",
    "tall",
    "flat",
    "round",
    "sparse",
)

RELATION_PREDICATES: tuple[str, ...] = (
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "near",
    "above",
    "below",
    "overlapping",
)

SCENE_TYPES: tuple[str, ...] = (
    "room",
    "office",
    "living_room",
    "storage",
)

RELATION_LABELS: tuple[str, ...] = tuple(
    f"{subject}:{predicate}:{obj}"
    for subject in OBJECT_CATEGORIES
    for obj in OBJECT_CATEGORIES
    if subject != obj
    for predicate in RELATION_PREDICATES
)

OBJECT_CATEGORY_SET = set(OBJECT_CATEGORIES)
ATTRIBUTE_SET = set(ATTRIBUTES)
RELATION_PREDICATE_SET = set(RELATION_PREDICATES)
SCENE_TYPE_SET = set(SCENE_TYPES)
