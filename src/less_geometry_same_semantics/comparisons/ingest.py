"""Ingestion helpers for external baseline handoff packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from less_geometry_same_semantics.comparisons.adapters import detector_boxes_to_semantic
from less_geometry_same_semantics.comparisons.manifests import (
    MANIFEST_SCHEMA_NAME,
    MANIFEST_SCHEMA_VERSION,
    RUN_METADATA_SCHEMA_NAME,
    RUN_METADATA_SCHEMA_VERSION,
    load_json_or_jsonl,
    validate_external_manifest,
    validate_external_run_metadata,
)
from less_geometry_same_semantics.comparisons.registry import normalize_baseline_specs
from less_geometry_same_semantics.schemas.schema import enforce_semantic_schema


def ingest_external_manifest(
    *,
    baseline_spec: dict[str, Any],
    scenario_name: str,
    input_path: str | Path,
    output_path: str | Path,
    metadata_path: str | Path | None = None,
    expected_scene_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Validate and canonicalize one external baseline export."""

    raw_payload = load_json_or_jsonl(input_path)
    metadata_payload = _load_optional_metadata(metadata_path, baseline_spec=baseline_spec)
    manifest = canonicalize_external_payload(
        raw_payload,
        baseline_spec=baseline_spec,
        scenario_name=scenario_name,
        metadata_payload=metadata_payload,
    )
    manifest = validate_external_manifest(
        manifest,
        expected_baseline_id=baseline_spec["baseline_id"],
        expected_kind=baseline_spec["kind"],
        source_path=input_path,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    semantic_map = shared_schema_prediction_map(manifest)
    semantic_path = output_file.with_name(output_file.stem + ".shared_schema.json")
    semantic_path.write_text(json.dumps(semantic_map, indent=2, sort_keys=True), encoding="utf-8")

    metadata_out = output_file.with_name(output_file.stem + ".metadata.json")
    metadata_out.write_text(
        json.dumps(
            {
                "schema_name": RUN_METADATA_SCHEMA_NAME,
                "schema_version": RUN_METADATA_SCHEMA_VERSION,
                "baseline_id": manifest["baseline_id"],
                "baseline_label": manifest.get("baseline_label"),
                "kind": manifest["kind"],
                "dataset": manifest["dataset"],
                "split": manifest["split"],
                "condition": manifest["condition"],
                "corruption": manifest.get("corruption"),
                "export": manifest.get("export"),
                "efficiency": manifest.get("efficiency"),
                "notes": manifest.get("notes", ""),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    scene_ids = list(semantic_map)
    expected = list(expected_scene_ids or [])
    missing_scene_ids = sorted(set(expected) - set(scene_ids))
    extra_scene_ids = sorted(set(scene_ids) - set(expected)) if expected else []
    completeness = "not_checked"
    if expected:
        completeness = "complete" if not missing_scene_ids else "incomplete"

    report = {
        "baseline_id": manifest["baseline_id"],
        "scenario": scenario_name,
        "kind": manifest["kind"],
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(output_file.resolve()),
        "shared_schema_path": str(semantic_path.resolve()),
        "metadata_output_path": str(metadata_out.resolve()),
        "num_predictions": len(scene_ids),
        "expected_scene_count": len(expected),
        "completeness": completeness,
        "missing_scene_ids": missing_scene_ids,
        "extra_scene_ids": extra_scene_ids,
        "condition": manifest["condition"],
        "split": manifest["split"],
        "dataset": manifest["dataset"],
    }
    summary_path = output_file.with_name(output_file.stem + ".summary.json")
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    report_path = output_file.with_name(output_file.stem + ".ingestion_report.md")
    lines = [
        "# External Baseline Ingestion Report",
        "",
        f"- baseline: `{manifest['baseline_id']}`",
        f"- scenario: `{scenario_name}`",
        f"- kind: `{manifest['kind']}`",
        f"- dataset: `{manifest['dataset']}`",
        f"- split: `{manifest['split']}`",
        f"- condition: `{manifest['condition']}`",
        f"- predictions: `{len(scene_ids)}`",
        f"- completeness: `{completeness}`",
        f"- canonical manifest: `{output_file}`",
        f"- shared-schema preview: `{semantic_path}`",
    ]
    if missing_scene_ids:
        lines.append(f"- missing scene ids: `{missing_scene_ids[:8]}`")
    if extra_scene_ids:
        lines.append(f"- extra scene ids: `{extra_scene_ids[:8]}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    return report


def canonicalize_external_payload(
    raw_payload: Any,
    *,
    baseline_spec: dict[str, Any],
    scenario_name: str,
    metadata_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonicalize a raw export plus optional metadata into the strict manifest schema."""

    metadata_payload = dict(metadata_payload or {})
    if isinstance(raw_payload, dict) and raw_payload.get("schema_name") == MANIFEST_SCHEMA_NAME:
        manifest = dict(raw_payload)
        if metadata_payload:
            manifest = _deep_update(manifest, _manifest_like_metadata(metadata_payload))
        return manifest

    source_mapping = dict(raw_payload) if isinstance(raw_payload, dict) else {}
    predictions_raw = _extract_prediction_entries(raw_payload)
    if not predictions_raw:
        raise ValueError(
            "No prediction entries found. Provide a canonical manifest or a raw export with a predictions list, "
            "scene-id map, or JSONL entries."
        )

    baseline_id = str(metadata_payload.get("baseline_id") or baseline_spec["baseline_id"])
    kind = str(metadata_payload.get("kind") or baseline_spec["kind"])
    dataset = str(metadata_payload.get("dataset") or source_mapping.get("dataset") or "arkitscenes")
    split = str(metadata_payload.get("split") or source_mapping.get("split") or "Validation")
    condition = str(
        metadata_payload.get("condition")
        or source_mapping.get("condition")
        or source_mapping.get("preset")
        or scenario_name
    )

    manifest = {
        "schema_name": MANIFEST_SCHEMA_NAME,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "baseline_label": str(metadata_payload.get("baseline_label") or baseline_spec.get("label", baseline_id)),
        "kind": kind,
        "dataset": dataset,
        "split": split,
        "condition": condition,
        "corruption": metadata_payload.get("corruption") or source_mapping.get("corruption") or {"preset": condition, "severity": condition},
        "export": metadata_payload.get("export") or source_mapping.get("export") or {},
        "efficiency": metadata_payload.get("efficiency") or source_mapping.get("efficiency") or {},
        "notes": str(metadata_payload.get("notes") or source_mapping.get("notes") or ""),
        "predictions": [
            _canonicalize_prediction_entry(
                entry,
                kind=kind,
                split=split,
                condition=condition,
                import_config=_scenario_import_config(baseline_spec, scenario_name),
            )
            for entry in predictions_raw
        ],
    }
    return manifest


def shared_schema_prediction_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a scene-id keyed shared-schema prediction map from a canonical manifest."""

    validate_external_manifest(
        manifest,
        expected_baseline_id=str(manifest.get("baseline_id")),
        expected_kind=str(manifest.get("kind")),
    )
    scene_map: dict[str, dict[str, Any]] = {}
    if manifest["kind"] == "imported_structured":
        for entry in manifest["predictions"]:
            scene_map[str(entry["scene_id"])] = entry["prediction"]
    else:
        for entry in manifest["predictions"]:
            scene_map[str(entry["scene_id"])] = detector_boxes_to_semantic(entry["boxes"])
    return scene_map


def scenario_output_path(
    baseline_spec: dict[str, Any],
    scenario_name: str,
    *,
    output_root: str | Path | None = None,
) -> Path:
    """Resolve the canonical manifest path for a baseline scenario."""

    for scenario in baseline_spec.get("scenarios", []):
        if scenario["name"] == scenario_name:
            prediction_path = scenario.get("prediction_path")
            if prediction_path:
                return Path(prediction_path)
    if output_root is None:
        output_root = Path("outputs") / "external_baselines" / baseline_spec["baseline_id"]
    return Path(output_root) / f"{scenario_name}_predictions.json"


def load_baseline_spec(
    comparison_config: dict[str, Any],
    *,
    baseline_id: str,
) -> dict[str, Any]:
    """Find one normalized baseline spec by id."""

    specs = normalize_baseline_specs(comparison_config)
    for spec in specs:
        if spec["baseline_id"] == baseline_id:
            return spec
    raise ValueError(f"Unknown baseline_id '{baseline_id}' in comparison config.")


def _load_optional_metadata(metadata_path: str | Path | None, *, baseline_spec: dict[str, Any]) -> dict[str, Any] | None:
    if metadata_path is None:
        return None
    payload = load_json_or_jsonl(metadata_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata file must be a JSON object: {metadata_path}")
    return validate_external_run_metadata(
        payload,
        expected_baseline_id=baseline_spec["baseline_id"],
        expected_kind=baseline_spec["kind"],
        source_path=metadata_path,
    )


def _extract_prediction_entries(raw_payload: Any) -> list[dict[str, Any]]:
    if isinstance(raw_payload, list):
        if not all(isinstance(entry, dict) for entry in raw_payload):
            raise ValueError("Prediction lists must contain only JSON objects.")
        return list(raw_payload)
    if not isinstance(raw_payload, dict):
        raise ValueError("Raw export must be a JSON object, JSON list, or JSONL file.")
    predictions = raw_payload.get("predictions", raw_payload)
    if isinstance(predictions, list):
        if not all(isinstance(entry, dict) for entry in predictions):
            raise ValueError("Prediction lists must contain only JSON objects.")
        return list(predictions)
    if isinstance(predictions, dict):
        entries = []
        for scene_id, entry in predictions.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Prediction map entry for scene '{scene_id}' is not a JSON object.")
            entries.append({"scene_id": str(scene_id), **entry})
        return entries
    raise ValueError("Could not find a predictions list or scene-id keyed prediction map.")


def _canonicalize_prediction_entry(
    entry: dict[str, Any],
    *,
    kind: str,
    split: str,
    condition: str,
    import_config: dict[str, Any],
) -> dict[str, Any]:
    scene_id = str(entry.get("scene_id") or entry.get("scene") or entry.get("id") or "").strip()
    if not scene_id:
        raise ValueError("Prediction entry is missing scene_id.")
    canonical: dict[str, Any] = {"scene_id": scene_id, "split": split, "condition": condition}
    if entry.get("raw_output_path"):
        canonical["raw_output_path"] = str(entry["raw_output_path"])
    if isinstance(entry.get("metadata"), dict):
        canonical["metadata"] = entry["metadata"]

    if kind == "imported_structured":
        payload = entry.get("prediction") or entry.get("output") or entry.get("structured_output")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Structured entry '{scene_id}' must contain a JSON object under 'prediction', 'output', or 'structured_output'."
            )
        canonical["prediction"] = enforce_semantic_schema(payload)
        return canonical

    boxes_key = str(import_config.get("boxes_key", "boxes"))
    raw_boxes = None
    for candidate_key in (boxes_key, "detections", "boxes_3d"):
        if candidate_key in entry:
            raw_boxes = entry.get(candidate_key)
            break
    if not isinstance(raw_boxes, list):
        raise ValueError(
            f"Detector entry '{scene_id}' must contain a list under '{boxes_key}', 'detections', or 'boxes_3d'."
        )
    canonical["boxes"] = [_canonicalize_detector_box(box, scene_id=scene_id, index=index) for index, box in enumerate(raw_boxes)]
    return canonical


def _canonicalize_detector_box(box: dict[str, Any], *, scene_id: str, index: int) -> dict[str, Any]:
    if not isinstance(box, dict):
        raise ValueError(f"Detector box {index} for scene '{scene_id}' is not a JSON object.")
    label = box.get("label") or box.get("category") or box.get("class") or box.get("class_name") or box.get("name")
    center = _vector3_list(box.get("center") or box.get("centroid") or box.get("translation") or box.get("position"))
    dims = _vector3_list(box.get("dimensions") or box.get("size") or box.get("extent") or box.get("axesLengths") or box.get("box_size"))
    if not label:
        raise ValueError(f"Detector box {index} for scene '{scene_id}' is missing a label.")
    if center is None:
        raise ValueError(f"Detector box {index} for scene '{scene_id}' is missing a valid 3D center.")
    if dims is None:
        raise ValueError(f"Detector box {index} for scene '{scene_id}' is missing valid 3D dimensions.")
    canonical = {"label": str(label), "center": center, "dimensions": dims}
    if box.get("score") is not None:
        canonical["score"] = float(box["score"])
    elif box.get("confidence") is not None:
        canonical["score"] = float(box["confidence"])
    if box.get("instance_id") is not None or box.get("id") is not None:
        canonical["instance_id"] = str(box.get("instance_id") or box.get("id"))
    if isinstance(box.get("metadata"), dict):
        canonical["metadata"] = box["metadata"]
    return canonical


def _vector3_list(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        ordered = [value.get("x"), value.get("y"), value.get("z")]
    elif isinstance(value, (list, tuple)):
        ordered = list(value)
    else:
        return None
    if len(ordered) < 3 or any(item is None for item in ordered[:3]):
        return None
    return [float(ordered[0]), float(ordered[1]), float(ordered[2])]


def _scenario_import_config(baseline_spec: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    for scenario in baseline_spec.get("scenarios", []):
        if scenario["name"] == scenario_name:
            return dict(scenario.get("import_config", {}))
    return {}


def _manifest_like_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "schema_name"}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
