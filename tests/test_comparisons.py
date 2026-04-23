from __future__ import annotations

import json
from pathlib import Path

from less_geometry_same_semantics.comparisons.adapters import build_scene_prediction_map, detector_boxes_to_semantic
from less_geometry_same_semantics.comparisons.ingest import ingest_external_manifest, load_baseline_spec
from less_geometry_same_semantics.comparisons.manifests import validate_external_manifest
from less_geometry_same_semantics.comparisons.registry import load_comparison_config
from less_geometry_same_semantics.comparisons.registry import normalize_baseline_specs
from less_geometry_same_semantics.comparisons.reporting import main_comparison_table
from less_geometry_same_semantics.training import evaluate_predictions

ROOT = Path(__file__).resolve().parents[1]


def test_detector_adapter_converts_boxes_to_shared_schema() -> None:
    payload = detector_boxes_to_semantic(
        [
            {"label": "chair", "center": [0.0, 0.0, 0.5], "dimensions": [0.5, 0.5, 0.9]},
            {"label": "table", "center": [0.9, 0.0, 0.5], "dimensions": [1.0, 0.8, 0.7]},
        ]
    )

    assert payload["object_counts"]["chair"] == 1
    assert payload["object_counts"]["table"] == 1
    assert payload["scene_type"] == "room"
    assert any(relation["predicate"] in {"left_of", "right_of", "near"} for relation in payload["relations"])


def test_normalize_baseline_specs_resolves_relative_prediction_paths(tmp_path) -> None:
    config_path = tmp_path / "comparisons.yaml"
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps({"predictions": []}), encoding="utf-8")
    config = {
        "baselines": {
            "spatiallm_import": {
                "label": "SpatialLM",
                "kind": "imported_structured",
                "scenarios": {"clean": {"prediction_path": "./predictions.json"}},
            }
        }
    }

    specs = normalize_baseline_specs(config, config_path=config_path)

    assert specs[0]["scenarios"][0]["prediction_path"] == str(prediction_path.resolve())


def test_build_scene_prediction_map_reads_structured_manifest(tmp_path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            {
                "schema_name": "external_baseline_manifest",
                "schema_version": "1.0",
                "baseline_id": "spatiallm_import",
                "kind": "imported_structured",
                "dataset": "arkitscenes",
                "split": "Validation",
                "condition": "clean",
                "efficiency": {"parameter_count": 12.0},
                "predictions": [
                    {
                        "scene_id": "scene-1",
                        "prediction": {
                            "objects": [{"category": "chair", "count": 1, "attributes": []}],
                            "object_counts": {"chair": 1},
                            "attributes": [],
                            "relations": [],
                            "scene_type": "room",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    scene_map, manifest = build_scene_prediction_map(
        path,
        kind="imported_structured",
        import_config={"baseline_id": "spatiallm_import"},
    )

    assert "scene-1" in scene_map
    assert manifest["efficiency"]["parameter_count"] == 12.0


def test_validate_external_manifest_rejects_duplicate_scene_ids() -> None:
    manifest = {
        "schema_name": "external_baseline_manifest",
        "schema_version": "1.0",
        "baseline_id": "spatiallm_import",
        "kind": "imported_structured",
        "dataset": "arkitscenes",
        "split": "Validation",
        "condition": "clean",
        "predictions": [
            {
                "scene_id": "scene-1",
                "prediction": {
                    "objects": [{"category": "chair", "count": 1, "attributes": []}],
                    "object_counts": {"chair": 1},
                    "attributes": [],
                    "relations": [],
                    "scene_type": "room",
                },
            },
            {
                "scene_id": "scene-1",
                "prediction": {
                    "objects": [{"category": "table", "count": 1, "attributes": []}],
                    "object_counts": {"table": 1},
                    "attributes": [],
                    "relations": [],
                    "scene_type": "room",
                },
            },
        ],
    }

    try:
        validate_external_manifest(manifest)
    except ValueError as exc:
        assert "Duplicate scene_id" in str(exc)
    else:
        raise AssertionError("Expected duplicate-scene-id validation error.")


def test_ingest_external_manifest_canonicalizes_spatiallm_fixture(tmp_path) -> None:
    comparison_config = load_comparison_config(ROOT / "configs" / "comparisons.yaml")
    baseline_spec = load_baseline_spec(comparison_config, baseline_id="spatiallm_import")
    output_path = tmp_path / "clean_predictions.json"

    report = ingest_external_manifest(
        baseline_spec=baseline_spec,
        scenario_name="clean",
        input_path=ROOT / "docs" / "server_baselines" / "examples" / "spatiallm_export_example.json",
        metadata_path=ROOT / "docs" / "server_baselines" / "examples" / "spatiallm_run_metadata_example.json",
        output_path=output_path,
        expected_scene_ids=["41069021", "41069025"],
    )

    canonical_manifest = json.loads(output_path.read_text(encoding="utf-8"))
    shared_schema = json.loads(output_path.with_name("clean_predictions.shared_schema.json").read_text(encoding="utf-8"))

    assert canonical_manifest["schema_name"] == "external_baseline_manifest"
    assert canonical_manifest["baseline_id"] == "spatiallm_import"
    assert report["completeness"] == "complete"
    assert sorted(shared_schema) == ["41069021", "41069025"]


def test_ingest_external_manifest_marks_incomplete_fixture(tmp_path) -> None:
    comparison_config = load_comparison_config(ROOT / "configs" / "comparisons.yaml")
    baseline_spec = load_baseline_spec(comparison_config, baseline_id="spatiallm_import")
    output_path = tmp_path / "clean_predictions.json"

    report = ingest_external_manifest(
        baseline_spec=baseline_spec,
        scenario_name="clean",
        input_path=ROOT / "docs" / "server_baselines" / "examples" / "incomplete_export_example.json",
        metadata_path=ROOT / "docs" / "server_baselines" / "examples" / "spatiallm_run_metadata_example.json",
        output_path=output_path,
        expected_scene_ids=["41069021", "41069025"],
    )

    assert report["completeness"] == "incomplete"
    assert report["missing_scene_ids"] == ["41069025"]


def test_ingest_external_manifest_rejects_malformed_fixture(tmp_path) -> None:
    comparison_config = load_comparison_config(ROOT / "configs" / "comparisons.yaml")
    baseline_spec = load_baseline_spec(comparison_config, baseline_id="spatiallm_import")
    output_path = tmp_path / "bad_predictions.json"

    try:
        ingest_external_manifest(
            baseline_spec=baseline_spec,
            scenario_name="clean",
            input_path=ROOT / "docs" / "server_baselines" / "examples" / "malformed_export_example.json",
            metadata_path=ROOT / "docs" / "server_baselines" / "examples" / "spatiallm_run_metadata_example.json",
            output_path=output_path,
        )
    except ValueError as exc:
        assert "scene_id" in str(exc)
    else:
        raise AssertionError("Expected malformed export ingestion to fail.")


def test_evaluate_predictions_supports_non_native_json_mode() -> None:
    predictions = [
        {
            "objects": [{"category": "chair", "count": 1, "attributes": []}],
            "object_counts": {"chair": 1},
            "attributes": [],
            "relations": [],
            "scene_type": "room",
        }
    ]
    references = [predictions[0]]
    metadata = [{"clean_num_points": 10, "degraded_num_points": 5, "scene_id": "scene-1"}]

    scores = evaluate_predictions(
        predictions,
        references,
        metadata,
        json_validity_mode="converted",
        efficiency_overrides={"parameter_count": 42.0},
    )

    assert scores["json_validity"]["mode"] == "converted"
    assert scores["efficiency"]["parameter_count"] == 42.0


def test_main_comparison_table_surfaces_alignment_and_na_metrics() -> None:
    record = {
        "metrics": {
            "comparisons": {
                "votenet_import__clean": {
                    "baseline": {"label": "VoteNet", "group": "main", "family": "standard_3d", "kind": "imported_detector"},
                    "condition": "clean",
                    "status": "imported",
                    "execution_mode": "external_import",
                    "task_alignment": {"json_mode": "converted", "relations_mode": "derived", "scene_type_mode": "derived"},
                    "metrics": {
                        "semantic_quality": {
                            "objects": {"f1": 0.5},
                            "relations": {"f1": 0.25},
                            "object_counts": {"exact_match": 0.0},
                            "scene_type": {"accuracy": 1.0},
                        },
                        "json_validity": {"validity_rate": 1.0, "mode": "converted"},
                        "efficiency": {"latency_ms_per_sample": 7.5},
                        "compression": {"compression_ratio": 1.0},
                    },
                },
                "spatiallm_import__severe_corruption": {
                    "baseline": {"label": "SpatialLM", "group": "main", "family": "heavy_upper_bound", "kind": "imported_structured"},
                    "condition": "severe_corruption",
                    "status": "pending_external",
                    "execution_mode": "external_import",
                    "task_alignment": {"json_mode": "converted", "relations_mode": "mapped", "scene_type_mode": "mapped"},
                    "notes": "Manifest missing.",
                },
            }
        }
    }

    rows = main_comparison_table(record)

    assert rows[0]["alignment"].startswith("json=")
    assert rows[0]["availability"] == "imported"
    assert rows[1]["status"] == "pending_external"
    assert rows[1]["availability"] == "pending_external"
    assert rows[1]["object_f1"] == "n/a"
