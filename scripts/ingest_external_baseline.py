"""Validate, canonicalize, and ingest an external baseline export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.comparisons import (
    ingest_external_manifest,
    load_baseline_spec,
    load_comparison_config,
    scenario_output_path,
)
from less_geometry_same_semantics.comparisons.manifests import load_json_or_jsonl
from less_geometry_same_semantics.utils.config import load_config, recursive_update
from less_geometry_same_semantics.data.loaders import build_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-id", required=True)
    parser.add_argument("--input-manifest", required=True, help="Raw export or canonical manifest JSON/JSONL.")
    parser.add_argument("--metadata", default=None, help="Optional metadata sidecar JSON.")
    parser.add_argument("--scenario", default=None, help="Scenario name from configs/comparisons.yaml, e.g. clean or severe_corruption.")
    parser.add_argument("--comparison-config", default="configs/comparisons.yaml")
    parser.add_argument("--config", default="configs/arkitscenes.yaml")
    parser.add_argument("--output-manifest", default=None, help="Override canonical output manifest path.")
    parser.add_argument("--skip-scene-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_config = load_comparison_config(args.comparison_config)
    baseline_spec = load_baseline_spec(comparison_config, baseline_id=args.baseline_id)
    scenario_name = args.scenario or infer_scenario_name(args.input_manifest, args.metadata)
    if not scenario_name:
        raise ValueError("Could not infer scenario name. Pass --scenario explicitly.")
    output_manifest = Path(args.output_manifest) if args.output_manifest else scenario_output_path(baseline_spec, scenario_name)
    expected_scene_ids = None if args.skip_scene_check else load_expected_scene_ids(
        args.config,
        baseline_spec=baseline_spec,
        scenario_name=scenario_name,
    )
    report = ingest_external_manifest(
        baseline_spec=baseline_spec,
        scenario_name=scenario_name,
        input_path=args.input_manifest,
        output_path=output_manifest,
        metadata_path=args.metadata,
        expected_scene_ids=expected_scene_ids,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def infer_scenario_name(input_manifest: str | Path, metadata_path: str | Path | None) -> str | None:
    for path in [metadata_path, input_manifest]:
        if path is None:
            continue
        payload = load_json_or_jsonl(path)
        if isinstance(payload, dict):
            condition = payload.get("condition") or payload.get("preset")
            if condition:
                return str(condition)
    return None


def load_expected_scene_ids(
    config_path: str | Path,
    *,
    baseline_spec: dict[str, Any],
    scenario_name: str,
) -> list[str] | None:
    base_config = load_config(config_path)
    scenario = None
    for item in baseline_spec.get("scenarios", []):
        if item["name"] == scenario_name:
            scenario = item
            break
    if scenario is None:
        raise ValueError(f"Scenario '{scenario_name}' is not defined for baseline '{baseline_spec['baseline_id']}'.")

    scenario_config = recursive_update(base_config, recursive_update(baseline_spec.get("config_overrides", {}), scenario.get("overrides", {})))
    try:
        _, val_loader = build_dataloaders(scenario_config)
    except Exception as exc:
        print(
            f"Warning: could not validate scene completeness against the local dataset split: {exc}",
            file=sys.stderr,
        )
        return None
    scene_ids: list[str] = []
    for batch in val_loader:
        scene_ids.extend(str(item.get("scene_id")) for item in batch["metadata"])
    return scene_ids


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ingestion error: {exc}", file=sys.stderr)
        raise SystemExit(1)
