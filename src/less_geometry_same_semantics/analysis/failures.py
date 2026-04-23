"""Failure analysis slices for semantic degradation experiments."""

from __future__ import annotations

from typing import Any

from less_geometry_same_semantics.metrics.efficiency import compression_ratio
from less_geometry_same_semantics.metrics.relations import relation_tuple
from less_geometry_same_semantics.metrics.semantic import object_categories
from less_geometry_same_semantics.schemas.schema import is_valid_semantic_output


def build_failure_report(
    examples: list[dict[str, Any]],
    max_examples: int = 5,
    strong_compression_ratio: float = 4.0,
) -> dict[str, list[dict[str, Any]]]:
    """Collect representative examples for paper failure analysis."""

    report = {
        "valid_json_wrong_semantics": [],
        "valid_json_wrong_graph_semantics": [],
        "correct_objects_wrong_relations": [],
        "object_preserved_relation_broken": [],
        "count_preserved_attribute_broken": [],
        "relation_collapse_under_severe_corruption": [],
        "semantic_preservation_under_strong_compression": [],
        "compression_preserves_coarse_semantics_loses_relations": [],
        "compression_preserves_scene_type_loses_relations": [],
        "graph_helps_under_severe_corruption": [],
        "graph_hurts_under_clean_conditions": [],
    }

    for example in examples:
        prediction = example["prediction"]
        reference = example["reference"]
        metadata = example.get("metadata", {})
        pred_objects = set(object_categories(prediction))
        ref_objects = set(object_categories(reference))
        pred_counts = prediction.get("object_counts", {})
        ref_counts = reference.get("object_counts", {})
        pred_attrs = set(prediction.get("attributes", []))
        ref_attrs = set(reference.get("attributes", []))
        pred_relations = {relation_tuple(rel) for rel in prediction.get("relations", [])}
        ref_relations = {relation_tuple(rel) for rel in reference.get("relations", [])}
        ratio = compression_ratio(
            int(metadata.get("clean_num_points", 1)),
            int(metadata.get("degraded_num_points", 1)),
        )
        preset = _preset_name(metadata)

        if is_valid_semantic_output(prediction) and (pred_objects != ref_objects or pred_relations != ref_relations):
            _append(report["valid_json_wrong_semantics"], example, max_examples)

        if pred_objects == ref_objects and pred_relations != ref_relations and ref_relations:
            _append(report["correct_objects_wrong_relations"], example, max_examples)
            _append(report["object_preserved_relation_broken"], example, max_examples)

        if is_valid_semantic_output(prediction) and pred_relations != ref_relations and ref_relations:
            _append(report["valid_json_wrong_graph_semantics"], example, max_examples)

        if pred_counts == ref_counts and pred_attrs != ref_attrs and ref_attrs:
            _append(report["count_preserved_attribute_broken"], example, max_examples)

        severe = preset in {"severe_corruption", "extreme_compression"} or ratio >= strong_compression_ratio
        if severe and ref_relations and len(pred_relations) == 0:
            _append(report["relation_collapse_under_severe_corruption"], example, max_examples)

        if ratio >= strong_compression_ratio and pred_objects == ref_objects:
            _append(report["semantic_preservation_under_strong_compression"], example, max_examples)
            if pred_relations != ref_relations and ref_relations:
                _append(report["compression_preserves_coarse_semantics_loses_relations"], example, max_examples)
        if ratio >= strong_compression_ratio and prediction.get("scene_type") == reference.get("scene_type") and pred_relations != ref_relations:
            _append(report["compression_preserves_scene_type_loses_relations"], example, max_examples)

    return report


def build_graph_help_report(
    no_graph_examples: list[dict[str, Any]],
    graph_examples: list[dict[str, Any]],
    max_examples: int = 5,
) -> list[dict[str, Any]]:
    """Find paired severe examples where graph predictions improve semantics."""

    no_graph_by_key = {_example_key(example): example for example in no_graph_examples}
    helpful = []
    for graph_example in graph_examples:
        key = _example_key(graph_example)
        baseline = no_graph_by_key.get(key)
        if baseline is None:
            continue
        if _semantic_errors(graph_example) < _semantic_errors(baseline):
            helpful.append(
                {
                    "index": graph_example.get("metadata", {}).get("index"),
                    "preset": _preset_name(graph_example.get("metadata", {})),
                    "no_graph_prediction": baseline.get("prediction"),
                    "graph_prediction": graph_example.get("prediction"),
                    "reference": graph_example.get("reference"),
                }
            )
        if len(helpful) >= max_examples:
            break
    return helpful


def _append(target: list[dict[str, Any]], example: dict[str, Any], max_examples: int) -> None:
    if len(target) < max_examples:
        target.append(_compact_example(example))


def _compact_example(example: dict[str, Any]) -> dict[str, Any]:
    metadata = example.get("metadata", {})
    return {
        "index": metadata.get("index"),
        "preset": _preset_name(metadata),
        "clean_num_points": metadata.get("clean_num_points"),
        "degraded_num_points": metadata.get("degraded_num_points"),
        "prediction": example.get("prediction"),
        "reference": example.get("reference"),
    }


def _preset_name(metadata: dict[str, Any]) -> str | None:
    corruption = metadata.get("corruption") or {}
    return corruption.get("preset") if isinstance(corruption, dict) else None


def _example_key(example: dict[str, Any]) -> tuple[object, object]:
    metadata = example.get("metadata", {})
    return metadata.get("dataset"), metadata.get("scene_id", metadata.get("index"))


def _semantic_errors(example: dict[str, Any]) -> int:
    prediction = example.get("prediction", {})
    reference = example.get("reference", {})
    errors = 0
    if set(object_categories(prediction)) != set(object_categories(reference)):
        errors += 1
    pred_relations = {relation_tuple(rel) for rel in prediction.get("relations", [])}
    ref_relations = {relation_tuple(rel) for rel in reference.get("relations", [])}
    if pred_relations != ref_relations:
        errors += 1
    if prediction.get("scene_type") != reference.get("scene_type"):
        errors += 1
    return errors
