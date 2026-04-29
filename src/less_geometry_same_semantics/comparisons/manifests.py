"""Canonical manifest validation for imported external baselines."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from less_geometry_same_semantics.schemas.schema import validate_semantic_output

MANIFEST_SCHEMA_NAME = "external_baseline_manifest"
MANIFEST_SCHEMA_VERSION = "1.0"
RUN_METADATA_SCHEMA_NAME = "external_baseline_run_metadata"
RUN_METADATA_SCHEMA_VERSION = "1.0"


@lru_cache(maxsize=1)
def load_external_manifest_schema() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "schemas" / "external_baseline_manifest.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_external_run_metadata_schema() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "schemas" / "external_run_metadata.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _manifest_validator() -> Draft202012Validator:
    return Draft202012Validator(load_external_manifest_schema())


@lru_cache(maxsize=1)
def _metadata_validator() -> Draft202012Validator:
    return Draft202012Validator(load_external_run_metadata_schema())


def load_json_or_jsonl(path: str | Path) -> Any:
    """Load JSON or JSONL content."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() == ".jsonl":
        entries = []
        for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_number} in {file_path}: {exc}") from exc
        return entries
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc


def validate_external_manifest(
    payload: dict[str, Any],
    *,
    expected_baseline_id: str | None = None,
    expected_kind: str | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a canonical external-baseline manifest."""

    try:
        _manifest_validator().validate(payload)
    except Exception as exc:
        origin = f" in {source_path}" if source_path is not None else ""
        raise ValueError(f"Invalid external-baseline manifest{origin}: {exc}") from exc

    baseline_id = str(payload.get("baseline_id", ""))
    kind = str(payload.get("kind", ""))
    if expected_baseline_id and baseline_id != expected_baseline_id:
        raise ValueError(
            f"Manifest baseline_id '{baseline_id}' does not match expected '{expected_baseline_id}'."
        )
    if expected_kind and kind != expected_kind:
        raise ValueError(f"Manifest kind '{kind}' does not match expected '{expected_kind}'.")

    scene_ids: set[str] = set()
    for index, entry in enumerate(payload.get("predictions", [])):
        scene_id = str(entry.get("scene_id", "")).strip()
        if not scene_id:
            raise ValueError(f"Prediction entry {index} is missing a non-empty scene_id.")
        if scene_id in scene_ids:
            raise ValueError(f"Duplicate scene_id '{scene_id}' in manifest.")
        scene_ids.add(scene_id)
        if "split" in entry and str(entry["split"]) != str(payload.get("split")):
            raise ValueError(f"Prediction entry '{scene_id}' has split '{entry['split']}' but manifest split is '{payload.get('split')}'.")
        if "condition" in entry and str(entry["condition"]) != str(payload.get("condition")):
            raise ValueError(
                f"Prediction entry '{scene_id}' has condition '{entry['condition']}' but manifest condition is '{payload.get('condition')}'."
            )

        if kind == "imported_structured":
            if "prediction" not in entry:
                raise ValueError(f"Structured manifest entry '{scene_id}' is missing 'prediction'.")
            try:
                validate_semantic_output(entry["prediction"])
            except Exception as exc:
                raise ValueError(f"Structured manifest entry '{scene_id}' has invalid shared-schema prediction: {exc}") from exc
        elif kind == "imported_detector":
            boxes = entry.get("boxes")
            if not isinstance(boxes, list):
                raise ValueError(f"Detector manifest entry '{scene_id}' must provide a 'boxes' list.")
    return payload


def validate_external_run_metadata(
    payload: dict[str, Any],
    *,
    expected_baseline_id: str | None = None,
    expected_kind: str | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an optional external baseline metadata sidecar."""

    try:
        _metadata_validator().validate(payload)
    except Exception as exc:
        origin = f" in {source_path}" if source_path is not None else ""
        raise ValueError(f"Invalid external-baseline metadata{origin}: {exc}") from exc
    baseline_id = str(payload.get("baseline_id", ""))
    kind = str(payload.get("kind", ""))
    if expected_baseline_id and baseline_id != expected_baseline_id:
        raise ValueError(
            f"Metadata baseline_id '{baseline_id}' does not match expected '{expected_baseline_id}'."
        )
    if expected_kind and kind != expected_kind:
        raise ValueError(f"Metadata kind '{kind}' does not match expected '{expected_kind}'.")
    return payload
