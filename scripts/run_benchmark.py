"""Run a severity-preset benchmark and save robustness-under-severity curves."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.data.loaders import build_dataloaders
from less_geometry_same_semantics.analysis import build_failure_report
from less_geometry_same_semantics.metrics.aggregation import aggregate_seed_results
from less_geometry_same_semantics.metrics.robustness import robustness_curve
from less_geometry_same_semantics.models import PointSemanticsModel
from less_geometry_same_semantics.reporting.plots import plot_pareto_curve, plot_robustness_curve
from less_geometry_same_semantics.reporting.tables import (
    compression_latency_semantic_table,
    clean_vs_corrupted_table,
    save_markdown_table,
    save_table_csv,
    severity_semantic_metrics_table,
)
from less_geometry_same_semantics.training import evaluate_model, train_one_epoch
from less_geometry_same_semantics.utils.config import load_config
from less_geometry_same_semantics.utils.experiment_logging import build_run_record, save_json_record
from less_geometry_same_semantics.utils.logging import setup_logging
from less_geometry_same_semantics.utils.reproducibility import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--output", default="outputs/benchmark/benchmark_results.json")
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to benchmark.seeds.")
    return parser.parse_args()


def parse_seeds(raw: str | None, config: dict[str, Any]) -> list[int]:
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [int(seed) for seed in config.get("benchmark", {}).get("seeds", [config.get("seed", 0)])]


def run_preset(config: dict[str, Any], preset: str, seed: int, epochs: int, device: torch.device) -> dict[str, Any]:
    seed_everything(seed)
    run_config = copy.deepcopy(config)
    run_config["seed"] = seed
    run_config.setdefault("data", {}).setdefault("corruption", {})["preset"] = preset
    train_loader, val_loader = build_dataloaders(run_config)
    model = PointSemanticsModel.from_config(run_config).to(device)
    train_cfg = run_config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    for _ in range(epochs):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            relation_loss_weight=float(train_cfg.get("relation_loss_weight", 0.5)),
        )
    return evaluate_model(
        model,
        val_loader,
        device,
        output_mode=str(run_config.get("model", {}).get("output_mode", "json")),
        constrained=bool(run_config.get("model", {}).get("constrained_decoding", True)),
        include_outputs=True,
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    seeds = parse_seeds(args.seeds, config)
    presets = list(config.get("benchmark", {}).get("severity_presets", ["clean"]))
    epochs = args.epochs or int(config.get("training", {}).get("epochs", 1))
    device = resolve_device(str(config.get("training", {}).get("device", "auto")))

    preset_results: dict[str, Any] = {}
    aggregate_by_preset: dict[str, Any] = {}
    mean_metrics_by_preset: dict[str, Any] = {}
    for preset in presets:
        runs = []
        logging.info("Running preset=%s seeds=%s", preset, seeds)
        for seed in seeds:
            metrics = run_preset(config, preset, seed, epochs, device)
            examples = metrics.pop("examples", [])
            runs.append(
                {
                    "seed": seed,
                    "metrics": metrics,
                    "failure_analysis": build_failure_report(examples),
                }
            )
        eval_metrics = [run["metrics"] for run in runs]
        aggregate = aggregate_seed_results(eval_metrics)
        preset_results[preset] = {"runs": runs, "aggregate": aggregate}
        aggregate_by_preset[preset] = aggregate
        mean_metrics_by_preset[preset] = aggregate["mean"]

    curve = robustness_curve(mean_metrics_by_preset, severity_order=presets)
    record = build_run_record(
        config=config,
        metrics={
            "presets": preset_results,
            "robustness_curve": curve,
        },
        seed=seeds[0],
        run_name=str(config.get("benchmark", {}).get("name", "severity_benchmark")),
        preset="severity_sweep",
        extra={"seeds": seeds, "epochs": epochs},
    )
    save_json_record(record, args.output)
    artifact_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path(args.output).with_suffix("")
    severity_rows = severity_semantic_metrics_table(record)
    clean_rows = clean_vs_corrupted_table(record)
    compression_rows = compression_latency_semantic_table(record)
    save_table_csv(clean_rows, artifact_dir / "clean_vs_corrupted.csv")
    save_markdown_table(clean_rows, artifact_dir / "clean_vs_corrupted.md")
    save_table_csv(severity_rows, artifact_dir / "severity_semantic_metrics.csv")
    save_markdown_table(severity_rows, artifact_dir / "severity_semantic_metrics.md")
    save_table_csv(compression_rows, artifact_dir / "compression_latency_semantics.csv")
    save_markdown_table(compression_rows, artifact_dir / "compression_latency_semantics.md")
    plot_robustness_curve(record, artifact_dir / "robustness_curve.png")
    plot_pareto_curve(record, artifact_dir / "compression_semantics_latency_pareto.png")
    logging.info("Saved benchmark results to %s", args.output)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset/setup error: {exc}", file=sys.stderr)
        print("Run: python scripts/check_dataset_setup.py --plan configs/paper_plan.yaml", file=sys.stderr)
        raise SystemExit(1)
