from __future__ import annotations

from less_geometry_same_semantics.experiments.plan import expected_main_case_names
from less_geometry_same_semantics.reporting.freeze_claims import freeze_supported_claims
from less_geometry_same_semantics.reporting.main_tables import main_results_table


def _case(name: str, factors: dict, metrics: dict) -> tuple[str, dict]:
    return name, {
        "factors": factors,
        "runs": [{"seed": 7, "metrics": metrics}],
        "aggregate": {"mean": metrics},
    }


def test_expected_main_case_names_from_plan() -> None:
    plan = {
        "main_benchmark": {
            "expected_conditions": {
                "corruption": ["clean"],
                "point_budget": ["raw"],
                "graph": ["no_graph"],
                "constrained": [False, True],
                "adaptation": ["none"],
            }
        }
    }

    assert expected_main_case_names(plan) == [
        "clean__raw__no_graph__unconstrained__none",
        "clean__raw__no_graph__constrained__none",
    ]


def test_freeze_claims_only_supports_positive_evidence() -> None:
    base_metrics = {
        "semantic_quality": {
            "objects": {"f1": 0.8},
            "object_counts": {"exact_match": 0.7},
            "relations": {"f1": 0.4},
            "scene_type": {"accuracy": 0.8},
            "semantic_macro_f1": 0.7,
        },
        "json_validity": {"validity_rate": 1.0},
        "efficiency": {"latency_ms_per_sample": 1.0},
        "compression": {"compression_ratio": 1.0},
    }
    severe_graph = {
        **base_metrics,
        "semantic_quality": {
            **base_metrics["semantic_quality"],
            "semantic_macro_f1": 0.6,
            "relations": {"f1": 0.35},
        },
    }
    severe_no_graph = {
        **base_metrics,
        "semantic_quality": {
            **base_metrics["semantic_quality"],
            "semantic_macro_f1": 0.5,
            "relations": {"f1": 0.2},
        },
    }
    record = {
        "metrics": {
            "main_matrix": dict(
                [
                    _case("clean", {"corruption": "clean", "point_budget": "raw", "graph": "simple_graph", "constrained": True, "adaptation": "none"}, base_metrics),
                    _case("severe_graph", {"corruption": "severe_corruption", "point_budget": "raw", "graph": "simple_graph", "constrained": True, "adaptation": "none"}, severe_graph),
                    _case("severe_no_graph", {"corruption": "severe_corruption", "point_budget": "raw", "graph": "no_graph", "constrained": True, "adaptation": "none"}, severe_no_graph),
                ]
            )
        }
    }

    report = freeze_supported_claims(record)
    claims = [item["claim"] for item in report["supported_claims"]]

    assert "graph bottlenecks improve robustness under corruption" in claims


def test_main_results_table_from_main_matrix_record() -> None:
    record = {
        "metrics": {
            "main_matrix": {
                "case": {
                    "factors": {"corruption": "clean", "point_budget": "raw", "graph": "no_graph", "constrained": True, "adaptation": "none"},
                    "aggregate": {
                        "mean": {
                            "semantic_quality": {
                                "objects": {"f1": 1.0},
                                "object_counts": {"exact_match": 1.0},
                                "relations": {"f1": 1.0},
                                "scene_type": {"accuracy": 1.0},
                                "semantic_macro_f1": 1.0,
                            },
                            "json_validity": {"validity_rate": 1.0},
                            "efficiency": {"latency_ms_per_sample": 1.0},
                            "compression": {"compression_ratio": 1.0},
                        }
                    },
                }
            }
        }
    }

    rows = main_results_table(record)

    assert rows[0]["macro"] == 1.0
