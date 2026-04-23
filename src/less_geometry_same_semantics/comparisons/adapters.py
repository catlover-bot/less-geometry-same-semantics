"""Adapters that map external baseline outputs into the shared semantic schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from less_geometry_same_semantics.comparisons.manifests import load_json_or_jsonl, validate_external_manifest
from less_geometry_same_semantics.data.label_mapping import map_object_category, normalize_label
from less_geometry_same_semantics.data.public_datasets import derive_relations_from_boxes, infer_arkitscenes_scene_type
from less_geometry_same_semantics.schemas.schema import enforce_semantic_schema


def build_scene_prediction_map(
    prediction_path: str | Path,
    *,
    kind: str,
    import_config: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a prediction manifest and return a scene-id to semantic-output map."""

    raw_map, manifest_meta = load_prediction_manifest(
        prediction_path,
        expected_kind=kind,
        expected_baseline_id=str(import_config.get("baseline_id")) if import_config else None,
    )
    converted: dict[str, dict[str, Any]] = {}
    import_cfg = dict(import_config or {})
    for scene_id, raw_entry in raw_map.items():
        converted[scene_id] = convert_prediction_entry(raw_entry, kind=kind, import_config=import_cfg, scene_id=scene_id)
    return converted, manifest_meta


def load_prediction_manifest(
    path: str | Path,
    *,
    expected_kind: str | None = None,
    expected_baseline_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate a canonical external-baseline manifest."""

    manifest_path = Path(path)
    payload = load_json_or_jsonl(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Prediction manifest {manifest_path} must be a canonical JSON manifest. "
            "Run one of the ingestion scripts first to convert raw exports into the shared handoff format."
        )
    manifest = validate_external_manifest(
        payload,
        expected_baseline_id=expected_baseline_id,
        expected_kind=expected_kind,
        source_path=manifest_path,
    )
    raw_predictions = manifest["predictions"]
    scene_map: dict[str, Any] = {}
    for index, entry in enumerate(raw_predictions):
        if not isinstance(entry, dict):
            raise ValueError(f"Prediction entry {index} in {manifest_path} is not a JSON object.")
        scene_map[str(entry["scene_id"])] = entry
    return scene_map, manifest


def convert_prediction_entry(
    entry: dict[str, Any],
    *,
    kind: str,
    import_config: dict[str, Any],
    scene_id: str,
) -> dict[str, Any]:
    """Convert one external prediction entry to the shared semantic JSON schema."""

    if kind == "imported_structured":
        payload = entry.get("prediction") or entry.get("output") or entry.get("structured_output") or entry
        if not isinstance(payload, dict):
            raise ValueError(f"Structured prediction for scene '{scene_id}' is not a JSON object.")
        return enforce_semantic_schema(payload)
    if kind == "imported_detector":
        container = entry.get("prediction") if isinstance(entry.get("prediction"), dict) else entry
        boxes_key = str(import_config.get("boxes_key", "boxes"))
        boxes = container.get(boxes_key) or container.get("detections") or container.get("boxes_3d")
        if not isinstance(boxes, list):
            raise ValueError(f"Detector prediction for scene '{scene_id}' is missing a list of 3D boxes.")
        score_threshold = float(import_config.get("score_threshold", 0.0))
        return detector_boxes_to_semantic(boxes, score_threshold=score_threshold)
    raise ValueError(f"Unsupported external baseline kind '{kind}'.")


def detector_boxes_to_semantic(boxes: list[dict[str, Any]], *, score_threshold: float = 0.0) -> dict[str, Any]:
    """Convert detector-style 3D boxes into the shared coarse semantic JSON."""

    normalized_boxes: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    category_attributes: dict[str, set[str]] = {}
    global_attributes: set[str] = set()

    for index, raw in enumerate(boxes):
        if not isinstance(raw, dict):
            continue
        score = raw.get("score", raw.get("confidence", 1.0))
        try:
            if score is not None and float(score) < score_threshold:
                continue
        except (TypeError, ValueError):
            pass
        label = raw.get("label") or raw.get("category") or raw.get("class") or raw.get("class_name") or raw.get("name")
        category = map_object_category(label)
        if category is None:
            continue
        center = _vector3(raw.get("center") or raw.get("centroid") or raw.get("translation") or raw.get("position"))
        dims = _vector3(
            raw.get("dimensions") or raw.get("size") or raw.get("extent") or raw.get("axesLengths") or raw.get("box_size"),
            default=1.0,
        )
        if center is None or dims is None:
            continue
        dims = torch.clamp(torch.abs(dims), min=1e-3)
        half = dims / 2.0
        attributes = _attributes_from_dims(dims)
        normalized_boxes.append(
            {
                "raw_label": normalize_label(label),
                "category": category,
                "instance_id": str(raw.get("id") or raw.get("instance_id") or index),
                "center": center,
                "dims": dims,
                "min": center - half,
                "max": center + half,
                "attributes": attributes,
            }
        )
        category_counts[category] = category_counts.get(category, 0) + 1
        category_attributes.setdefault(category, set()).update(attributes)
        global_attributes.update(attributes)

    objects = [
        {
            "category": category,
            "count": category_counts[category],
            "attributes": sorted(category_attributes.get(category, set())),
        }
        for category in sorted(category_counts)
    ]
    payload = {
        "objects": objects,
        "object_counts": dict(sorted(category_counts.items())),
        "attributes": sorted(global_attributes),
        "relations": [relation.__dict__ for relation in derive_relations_from_boxes(normalized_boxes)],
        "scene_type": infer_arkitscenes_scene_type(normalized_boxes) if normalized_boxes else "room",
    }
    return enforce_semantic_schema(payload)


def _vector3(value: Any, default: float | None = None) -> torch.Tensor | None:
    if value is None:
        if default is None:
            return None
        return torch.full((3,), float(default), dtype=torch.float32)
    if isinstance(value, dict):
        ordered = [value.get(key) for key in ("x", "y", "z")]
    elif isinstance(value, (list, tuple)):
        ordered = list(value)
    else:
        ordered = []
    if len(ordered) < 3 or any(item is None for item in ordered[:3]):
        return None
    return torch.tensor([float(ordered[0]), float(ordered[1]), float(ordered[2])], dtype=torch.float32)


def _attributes_from_dims(dims: torch.Tensor) -> tuple[str, ...]:
    x, y, z = (float(value) for value in dims)
    volume = x * y * z
    attributes = []
    if volume < 0.25:
        attributes.append("small")
    if volume > 2.0:
        attributes.append("large")
    if z > max(x, y) * 1.3:
        attributes.append("tall")
    return tuple(sorted(attributes))
