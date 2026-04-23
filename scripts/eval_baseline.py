"""Evaluate the baseline model on synthetic degraded point clouds."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.data.loaders import build_dataloaders
from less_geometry_same_semantics.analysis import build_failure_report
from less_geometry_same_semantics.models import PointSemanticsModel
from less_geometry_same_semantics.training import evaluate_model
from less_geometry_same_semantics.utils.config import load_config
from less_geometry_same_semantics.utils.experiment_logging import build_run_record, save_json_record
from less_geometry_same_semantics.utils.logging import setup_logging
from less_geometry_same_semantics.utils.reproducibility import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="outputs/baseline/eval_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(config.get("training", {}).get("device", "auto")))

    _, val_loader = build_dataloaders(config)
    model = PointSemanticsModel.from_config(config).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        logging.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        logging.info("No checkpoint supplied; evaluating an untrained model.")

    metrics = evaluate_model(
        model,
        val_loader,
        device,
        output_mode=str(config.get("model", {}).get("output_mode", "json")),
        constrained=bool(config.get("model", {}).get("constrained_decoding", True)),
        include_outputs=True,
    )
    examples = metrics.pop("examples", [])
    output_path = Path(args.output)
    record = build_run_record(
        config=config,
        metrics={
            **metrics,
            "failure_analysis": build_failure_report(examples),
        },
        seed=seed,
        run_name=str(config.get("benchmark", {}).get("name", "baseline_eval")),
        preset=str(config.get("data", {}).get("corruption", {}).get("preset", "custom")),
        extra={"checkpoint": args.checkpoint},
    )
    save_json_record(record, output_path)
    logging.info("Metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
