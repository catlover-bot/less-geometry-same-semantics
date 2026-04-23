from __future__ import annotations

from less_geometry_same_semantics.metrics.relations import relation_prf1
from less_geometry_same_semantics.metrics.semantic import object_count_metrics, object_prf1, semantic_quality_metrics


def test_object_metric_computes_micro_f1() -> None:
    predictions = [{"objects": [{"category": "chair", "count": 1, "attributes": []}, {"category": "table", "count": 1, "attributes": []}]}]
    references = [{"objects": [{"category": "chair", "count": 1, "attributes": []}, {"category": "lamp", "count": 1, "attributes": []}]}]

    scores = object_prf1(predictions, references)

    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5


def test_object_count_metric_scores_explicit_counts() -> None:
    predictions = [{"object_counts": {"chair": 2, "table": 1}, "objects": []}]
    references = [{"object_counts": {"chair": 1, "table": 1}, "objects": []}]

    scores = object_count_metrics(predictions, references)

    assert scores["exact_match"] == 0.0
    assert scores["mean_absolute_error"] > 0.0


def test_relation_metric_uses_exact_triples() -> None:
    predictions = [
        {"relations": [{"subject": "chair", "predicate": "left_of", "object": "table"}]}
    ]
    references = [
        {"relations": [{"subject": "chair", "predicate": "left_of", "object": "table"}]}
    ]

    scores = relation_prf1(predictions, references)

    assert scores["f1"] == 1.0


def test_semantic_quality_metrics_has_benchmark_groups() -> None:
    predictions = [
        {
            "objects": [{"category": "chair", "count": 1, "attributes": []}],
            "object_counts": {"chair": 1},
            "attributes": ["small"],
            "relations": [],
            "scene_type": "room",
        }
    ]
    references = [
        {
            "objects": [{"category": "chair", "count": 1, "attributes": []}],
            "object_counts": {"chair": 1},
            "attributes": ["small"],
            "relations": [],
            "scene_type": "room",
        }
    ]

    scores = semantic_quality_metrics(predictions, references)

    assert scores["semantic_macro_f1"] == 1.0
    assert scores["object_counts"]["exact_match"] == 1.0
