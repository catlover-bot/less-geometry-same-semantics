"""Run fair comparison baselines through the shared semantic evaluation interface."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.comparisons import (
    build_scene_prediction_map,
    collect_validation_references,
    load_comparison_config,
    normalize_baseline_specs,
    run_internal_comparison,
    save_comparison_tables,
)
from less_geometry_same_semantics.training import evaluate_predictions
from less_geometry_same_semantics.utils.config import load_config, recursive_update
from less_geometry_same_semantics.utils.experiment_logging import build_run_record, save_json_record
from less_geometry_same_semantics.utils.logging import setup_logging
from less_geometry_same_semantics.utils.reproducibility import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/arkitscenes.yaml")
    parser.add_argument("--comparison-config", default="configs/comparisons.yaml")
    parser.add_argument("--output", default="outputs/comparisons/results.json")
    parser.add_argument("--artifacts-dir", default="outputs/comparisons")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    return parser.parse_args()


def parse_seeds(raw: str | None, config: dict[str, Any]) -> list[int]:
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [int(seed) for seed in config.get("benchmark", {}).get("seeds", [config.get("seed", 0)])]


def main() -> None:
    args = parse_args()
    setup_logging()
    base_config = load_config(args.config)
    comparison_config = load_comparison_config(args.comparison_config)
    specs = normalize_baseline_specs(comparison_config, config_path=args.comparison_config)
    seeds = parse_seeds(args.seeds, base_config)
    epochs = args.epochs if args.epochs is not None else int(base_config.get("training", {}).get("epochs", 1))
    device = resolve_device(str(base_config.get("training", {}).get("device", "auto")))

    results: dict[str, Any] = {}
    reference_cache: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}

    for spec in specs:
        for scenario in spec["scenarios"]:
            case_name = f"{spec['baseline_id']}__{scenario['name']}"
            comparison_overrides = recursive_update(copy.deepcopy(spec["config_overrides"]), scenario["overrides"])
            notes = _combine_notes(spec.get("notes", ""), scenario.get("notes", ""))
            baseline_meta = {
                "baseline_id": spec["baseline_id"],
                "label": spec["label"],
                "group": spec["group"],
                "family": spec["family"],
                "kind": spec["kind"],
                "reference_url": spec["reference_url"],
            }
            logging.info("Running comparison case=%s", case_name)
            if spec["kind"] == "internal_model":
                try:
                    run = run_internal_comparison(
                        base_config,
                        config_overrides=comparison_overrides,
                        seeds=seeds,
                        epochs=epochs,
                        device=device,
                    )
                    results[case_name] = {
                        "baseline": baseline_meta,
                        "condition": scenario["name"],
                        "status": "completed_local",
                        "execution_mode": "local",
                        "task_alignment": spec["task_alignment"],
                        "notes": notes,
                        **run,
                    }
                except Exception as exc:
                    results[case_name] = {
                        "baseline": baseline_meta,
                        "condition": scenario["name"],
                        "status": "local_failed",
                        "execution_mode": "local",
                        "task_alignment": spec["task_alignment"],
                        "notes": _combine_notes(notes, str(exc)),
                    }
                continue

            prediction_path = scenario.get("prediction_path")
            if not prediction_path:
                results[case_name] = {
                    "baseline": baseline_meta,
                    "condition": scenario["name"],
                    "status": "pending_external",
                    "execution_mode": "external_import",
                    "task_alignment": spec["task_alignment"],
                    "notes": _combine_notes(notes, "No prediction_path configured for this external baseline scenario."),
                }
                continue

            try:
                scenario_config = recursive_update(copy.deepcopy(base_config), comparison_overrides)
                references, metadata = _cached_references(reference_cache, scenario_config)
                scene_predictions, manifest_meta = build_scene_prediction_map(
                    prediction_path,
                    kind=spec["kind"],
                    import_config={"baseline_id": spec["baseline_id"], **scenario.get("import_config", {})},
                )
                missing_scene_ids = [
                    str(item.get("scene_id"))
                    for item in metadata
                    if str(item.get("scene_id")) not in scene_predictions
                ]
                if missing_scene_ids:
                    results[case_name] = {
                        "baseline": baseline_meta,
                        "condition": scenario["name"],
                        "status": "incomplete_import",
                        "execution_mode": "external_import",
                        "task_alignment": spec["task_alignment"],
                        "notes": _combine_notes(
                            notes,
                            f"Manifest is missing {len(missing_scene_ids)} validation scenes. "
                            f"First missing IDs: {missing_scene_ids[:5]}",
                        ),
                        "manifest": {
                            "prediction_path": str(prediction_path),
                            "missing_scene_ids": missing_scene_ids,
                        },
                    }
                    continue
                predictions = [scene_predictions[str(item.get("scene_id"))] for item in metadata]
                efficiency = _merge_efficiency(
                    scenario.get("efficiency", {}),
                    manifest_meta.get("efficiency", {}) if isinstance(manifest_meta, dict) else {},
                )
                metrics = evaluate_predictions(
                    predictions,
                    references,
                    metadata,
                    include_outputs=False,
                    json_validity_mode=spec["json_validity_mode"],
                    efficiency_overrides=efficiency,
                )
                results[case_name] = {
                    "baseline": baseline_meta,
                    "condition": scenario["name"],
                    "status": "imported",
                    "execution_mode": "external_import",
                    "task_alignment": spec["task_alignment"],
                    "notes": notes,
                    "metrics": metrics,
                    "manifest": {
                        "prediction_path": str(prediction_path),
                        "scenes_loaded": len(scene_predictions),
                    },
                }
            except (FileNotFoundError, ValueError) as exc:
                status = "pending_external" if isinstance(exc, FileNotFoundError) else "invalid_import"
                note = str(exc)
                if isinstance(exc, FileNotFoundError):
                    note = (
                        f"Canonical imported manifest not found at {_display_path(prediction_path)}. "
                        "Run the appropriate ingestion script after copying server outputs back locally."
                    )
                results[case_name] = {
                    "baseline": baseline_meta,
                    "condition": scenario["name"],
                    "status": status,
                    "execution_mode": "external_import",
                    "task_alignment": spec["task_alignment"],
                    "notes": _combine_notes(notes, note),
                    "manifest": {"prediction_path": str(prediction_path)},
                }

    record = build_run_record(
        config={"base_config": base_config, "comparison_config": comparison_config},
        metrics={"comparisons": results},
        seed=seeds[0] if seeds else 0,
        run_name="comparison_baselines",
        preset="comparison_baselines",
        extra={"seeds": seeds, "epochs": epochs},
    )
    save_json_record(record, args.output)
    save_comparison_tables(record, args.artifacts_dir)
    _write_summary(record, args.artifacts_dir)
    logging.info("Saved comparison record to %s", args.output)


def _cached_references(
    cache: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    key = json.dumps(config, sort_keys=True, default=str)
    if key not in cache:
        cache[key] = collect_validation_references(config)
    return cache[key]


def _merge_efficiency(*mappings: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        for key, value in mapping.items():
            merged[str(key)] = value
    return merged


def _combine_notes(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if part and part.strip())


def _display_path(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except Exception:
        return str(candidate)


def _write_summary(record: dict[str, Any], artifacts_dir: str | Path) -> None:
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    statuses = Counter(payload.get("status", "unknown") for payload in record.get("metrics", {}).get("comparisons", {}).values())
    lines = [
        "# Comparison Summary",
        "",
        "This directory contains main and supplementary baseline comparisons aligned to the shared coarse semantic JSON interface.",
        "",
        "Status counts:",
        "",
    ]
    for status, count in sorted(statuses.items()):
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `imported_detector` baselines are converted from 3D boxes into the shared JSON interface; relations and scene type are derived heuristically.",
            "- `imported_structured` baselines expect exported predictions already mapped to the repo schema or close enough for conservative schema enforcement.",
            "- The internal denoising adaptation baseline is a lightweight robustness comparison, not a full CloudFixer reproduction.",
            "- `pending_external` means the comparison slot is prepared but the canonical imported manifest has not been ingested yet.",
        ]
    )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Comparison setup error: {exc}", file=sys.stderr)
        raise SystemExit(1)
