"""Semantic quality metrics for structured benchmark outputs."""

from __future__ import annotations

from typing import Any

from less_geometry_same_semantics.data.constants import OBJECT_CATEGORIES
from less_geometry_same_semantics.metrics.relations import relation_prf1


def precision_recall_f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision, recall, and F1 from aggregate counts."""

    precision = tp / (tp + fp) if tp + fp > 0 else 1.0 if fn == 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0 if fp == 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def object_categories(payload: dict[str, Any]) -> list[str]:
    """Extract object category labels from old or formal object representations."""

    categories: list[str] = []
    for item in payload.get("objects", []):
        if isinstance(item, str):
            categories.append(item)
        elif isinstance(item, dict) and item.get("category"):
            categories.append(str(item["category"]))
    if not categories and isinstance(payload.get("object_counts"), dict):
        categories = [
            str(category)
            for category, count in payload["object_counts"].items()
            if int(count) > 0
        ]
    return sorted(set(categories))


def object_count_map(payload: dict[str, Any]) -> dict[str, int]:
    """Extract object counts with missing categories interpreted as zero."""

    counts: dict[str, int] = {}
    raw_counts = payload.get("object_counts", {})
    if isinstance(raw_counts, dict):
        for category, count in raw_counts.items():
            try:
                value = max(0, int(count))
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                counts[str(category)] = value

    for item in payload.get("objects", []):
        if isinstance(item, str):
            counts[item] = max(1, counts.get(item, 0))
        elif isinstance(item, dict) and item.get("category"):
            category = str(item["category"])
            try:
                value = max(1, int(item.get("count", 1)))
            except (TypeError, ValueError):
                value = 1
            counts[category] = max(value, counts.get(category, 0))
    return counts


def set_prf1(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    field: str,
) -> dict[str, float]:
    """Micro-average set precision/recall/F1 for a JSON list field."""

    tp = fp = fn = 0
    for pred, ref in zip(predictions, references, strict=True):
        if field == "objects":
            pred_items = set(object_categories(pred))
            ref_items = set(object_categories(ref))
        else:
            pred_items = set(pred.get(field, []))
            ref_items = set(ref.get(field, []))
        tp += len(pred_items & ref_items)
        fp += len(pred_items - ref_items)
        fn += len(ref_items - pred_items)
    return precision_recall_f1_from_counts(tp, fp, fn)


def object_prf1(predictions: list[dict[str, Any]], references: list[dict[str, Any]]) -> dict[str, float]:
    """Object category precision/recall/F1."""

    return set_prf1(predictions, references, "objects")


def object_count_metrics(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, float]:
    """Score object-count maps with exact-match rate and mean absolute error."""

    if not predictions:
        return {"exact_match": 0.0, "mean_absolute_error": 0.0}

    exact = 0
    total_abs_error = 0.0
    total_slots = 0
    for pred, ref in zip(predictions, references, strict=True):
        pred_counts = object_count_map(pred)
        ref_counts = object_count_map(ref)
        if all(pred_counts.get(category, 0) == ref_counts.get(category, 0) for category in OBJECT_CATEGORIES):
            exact += 1
        for category in OBJECT_CATEGORIES:
            total_abs_error += abs(pred_counts.get(category, 0) - ref_counts.get(category, 0))
            total_slots += 1

    return {
        "exact_match": exact / len(predictions),
        "mean_absolute_error": total_abs_error / max(1, total_slots),
    }


def attribute_prf1(predictions: list[dict[str, Any]], references: list[dict[str, Any]]) -> dict[str, float]:
    """Coarse global attribute precision/recall/F1."""

    return set_prf1(predictions, references, "attributes")


def scene_type_accuracy(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, float]:
    """Exact scene-type accuracy."""

    if not predictions:
        return {"accuracy": 0.0}
    correct = sum(
        1
        for pred, ref in zip(predictions, references, strict=True)
        if pred.get("scene_type") == ref.get("scene_type")
    )
    return {"accuracy": correct / len(predictions)}


def semantic_quality_metrics(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate semantic quality metrics under the formal schema."""

    objects = object_prf1(predictions, references)
    attributes = attribute_prf1(predictions, references)
    relations = relation_prf1(predictions, references)
    scene = scene_type_accuracy(predictions, references)
    macro = (objects["f1"] + attributes["f1"] + relations["f1"] + scene["accuracy"]) / 4.0
    return {
        "objects": objects,
        "object_counts": object_count_metrics(predictions, references),
        "attributes": attributes,
        "relations": relations,
        "scene_type": scene,
        "semantic_macro_f1": macro,
    }
