"""Comparison-baseline config loading and normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from less_geometry_same_semantics.utils.config import load_config

VALID_KINDS = {"internal_model", "imported_structured", "imported_detector"}
VALID_GROUPS = {"main", "supplementary"}


def load_comparison_config(path: str | Path = "configs/comparisons.yaml") -> dict[str, Any]:
    """Load a comparison-baseline config file."""

    return load_config(path)


def normalize_baseline_specs(config: dict[str, Any], *, config_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Normalize baseline specs into a list with explicit scenario entries."""

    baselines = config.get("baselines", {})
    if not isinstance(baselines, dict):
        raise ValueError("Comparison config 'baselines' must be a mapping.")

    base_dir = Path(config_path).resolve().parent if config_path is not None else Path.cwd()
    normalized = []
    for baseline_id, raw_spec in baselines.items():
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Baseline '{baseline_id}' must map to a dictionary.")
        kind = str(raw_spec.get("kind", "")).strip()
        if kind not in VALID_KINDS:
            raise ValueError(f"Baseline '{baseline_id}' has unsupported kind '{kind}'.")
        group = str(raw_spec.get("group", "main")).strip()
        if group not in VALID_GROUPS:
            raise ValueError(f"Baseline '{baseline_id}' has unsupported group '{group}'.")
        normalized.append(
            {
                "baseline_id": str(baseline_id),
                "label": str(raw_spec.get("label", baseline_id)),
                "kind": kind,
                "group": group,
                "family": str(raw_spec.get("family", "comparison")),
                "reference_url": str(raw_spec.get("reference_url", "")),
                "notes": str(raw_spec.get("notes", "")),
                "config_overrides": raw_spec.get("config_overrides", {}),
                "json_validity_mode": str(raw_spec.get("json_validity_mode", _default_json_mode(kind))),
                "task_alignment": _normalize_alignment(raw_spec, kind=kind),
                "scenarios": _normalize_scenarios(raw_spec.get("scenarios", {}), base_dir=base_dir),
            }
        )
    return normalized


def _normalize_alignment(spec: dict[str, Any], *, kind: str) -> dict[str, Any]:
    defaults = {
        "internal_model": {"json_mode": "native", "relations_mode": "native", "scene_type_mode": "native"},
        "imported_structured": {"json_mode": "native", "relations_mode": "native", "scene_type_mode": "native"},
        "imported_detector": {"json_mode": "converted", "relations_mode": "derived", "scene_type_mode": "derived"},
    }[kind]
    raw_alignment = spec.get("task_alignment", {})
    if raw_alignment is None:
        raw_alignment = {}
    if not isinstance(raw_alignment, dict):
        raise ValueError("task_alignment must be a mapping when provided.")
    alignment = dict(defaults)
    alignment.update(raw_alignment)
    alignment["status"] = str(alignment.get("status", "native" if kind == "internal_model" else "partial"))
    alignment["notes"] = str(alignment.get("notes", spec.get("notes", "")))
    return alignment


def _normalize_scenarios(raw_scenarios: Any, *, base_dir: Path) -> list[dict[str, Any]]:
    if not raw_scenarios:
        raise ValueError("Each comparison baseline must define at least one scenario.")

    scenarios: list[dict[str, Any]] = []
    items = raw_scenarios.items() if isinstance(raw_scenarios, dict) else enumerate(raw_scenarios)
    for raw_name, raw_entry in items:
        name = str(raw_name)
        if isinstance(raw_entry, str):
            raw_entry = {"prediction_path": raw_entry}
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Scenario '{name}' must be a mapping or a prediction-path string.")
        prediction_path = raw_entry.get("prediction_path")
        scenarios.append(
            {
                "name": str(raw_entry.get("name", name)),
                "prediction_path": _resolve_optional_path(prediction_path, base_dir=base_dir),
                "overrides": raw_entry.get("overrides", {}),
                "notes": str(raw_entry.get("notes", "")),
                "efficiency": raw_entry.get("efficiency", {}),
                "import_config": raw_entry.get("import_config", {}),
            }
        )
    return scenarios


def _resolve_optional_path(value: Any, *, base_dir: Path) -> str | None:
    if value in {None, ""}:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        if str(value).startswith((".", "..")):
            path = (base_dir / path).resolve()
        else:
            path = Path.cwd().joinpath(path).resolve()
    return str(path)


def _default_json_mode(kind: str) -> str:
    return "converted" if kind == "imported_detector" else "native"
