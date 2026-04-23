"""Comparison baseline helpers for lightweight robustness studies."""

from less_geometry_same_semantics.comparisons.adapters import (
    build_scene_prediction_map,
    detector_boxes_to_semantic,
)
from less_geometry_same_semantics.comparisons.ingest import (
    ingest_external_manifest,
    load_baseline_spec,
    scenario_output_path,
    shared_schema_prediction_map,
)
from less_geometry_same_semantics.comparisons.manifests import (
    validate_external_manifest,
    validate_external_run_metadata,
)
from less_geometry_same_semantics.comparisons.registry import (
    load_comparison_config,
    normalize_baseline_specs,
)
from less_geometry_same_semantics.comparisons.reporting import save_comparison_tables
from less_geometry_same_semantics.comparisons.runner import (
    collect_validation_references,
    run_internal_comparison,
)

__all__ = [
    "build_scene_prediction_map",
    "collect_validation_references",
    "detector_boxes_to_semantic",
    "ingest_external_manifest",
    "load_baseline_spec",
    "load_comparison_config",
    "normalize_baseline_specs",
    "run_internal_comparison",
    "save_comparison_tables",
    "scenario_output_path",
    "shared_schema_prediction_map",
    "validate_external_manifest",
    "validate_external_run_metadata",
]
