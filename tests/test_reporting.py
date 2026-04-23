from __future__ import annotations

from less_geometry_same_semantics.analysis.failures import build_failure_report
from less_geometry_same_semantics.reporting.tables import (
    compression_latency_semantic_table,
    severity_semantic_metrics_table,
)


def test_paper_tables_read_aggregated_benchmark_record() -> None:
    record = {
        "metrics": {
            "presets": {
                "clean": {
                    "aggregate": {
                        "mean": {
                            "semantic_quality": {
                                "objects": {"f1": 1.0},
                                "object_counts": {"exact_match": 1.0},
                                "relations": {"f1": 0.5},
                                "scene_type": {"accuracy": 1.0},
                                "semantic_macro_f1": 0.875,
                            },
                            "json_validity": {"validity_rate": 1.0},
                            "compression": {"compression_ratio": 1.0, "retained_fraction": 1.0},
                            "efficiency": {"latency_ms_per_sample": 2.0},
                        }
                    }
                }
            }
        }
    }

    severity_rows = severity_semantic_metrics_table(record)
    pareto_rows = compression_latency_semantic_table(record)

    assert severity_rows[0]["semantic_macro_f1"] == 0.875
    assert pareto_rows[0]["latency_ms_per_sample"] == 2.0


def test_failure_report_finds_wrong_relations_with_valid_json() -> None:
    examples = [
        {
            "prediction": {
                "objects": [{"category": "chair", "count": 1, "attributes": []}],
                "object_counts": {"chair": 1},
                "attributes": [],
                "relations": [],
                "scene_type": "room",
            },
            "reference": {
                "objects": [{"category": "chair", "count": 1, "attributes": []}],
                "object_counts": {"chair": 1},
                "attributes": [],
                "relations": [{"subject": "chair", "predicate": "near", "object": "table"}],
                "scene_type": "room",
            },
            "metadata": {
                "index": 0,
                "clean_num_points": 512,
                "degraded_num_points": 64,
                "corruption": {"preset": "extreme_compression"},
            },
        }
    ]

    report = build_failure_report(examples)

    assert len(report["valid_json_wrong_semantics"]) == 1
    assert len(report["correct_objects_wrong_relations"]) == 1
    assert len(report["relation_collapse_under_severe_corruption"]) == 1
