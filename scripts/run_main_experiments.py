"""Run the main paper benchmark matrix with multi-seed aggregation."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.analysis import build_failure_report
from less_geometry_same_semantics.data.loaders import build_dataloaders
from less_geometry_same_semantics.metrics.aggregation import aggregate_seed_results
from less_geometry_same_semantics.models import PointSemanticsModel
from less_geometry_same_semantics.reporting.claims import save_claim_report
from less_geometry_same_semantics.training import evaluate_model, train_one_epoch
from less_geometry_same_semantics.utils.config import load_config, recursive_update
from less_geometry_same_semantics.utils.experiment_logging import build_run_record, save_json_record
from less_geometry_same_semantics.utils.logging import setup_logging
from less_geometry_same_semantics.utils.reproducibility import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/arkitscenes.yaml")
    parser.add_argument("--output", default="outputs/main_experiments/results.json")
    parser.add_argument("--artifacts-dir", default="outputs/main_experiments")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--max-cases", type=int, default=None, help="Debug helper to run only the first N matrix cases.")
    return parser.parse_args()


def parse_seeds(raw: str | None, config: dict[str, Any]) -> list[int]:
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [int(seed) for seed in config.get("benchmark", {}).get("seeds", [config.get("seed", 0)])]


def build_main_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    data_points = int(config.get("data", {}).get("num_points", config.get("data", {}).get("max_points", 32768)) or 32768)
    matrix_cfg = config.get("main_experiments", {})
    raw_tokens = int(matrix_cfg.get("raw_tokens", min(data_points, 256)))
    compressed_tokens = int(config.get("model", {}).get("compressed_tokens", 32))
    cases = []
    for corruption, point_budget, graph, constrained, adaptation in product(
        ["clean", "severe_corruption"],
        ["raw", "compressed"],
        ["no_graph", "simple_graph"],
        [False, True],
        ["none", "input_normalization"],
    ):
        overrides = {
            "data": {"corruption": {"preset": corruption}},
            "model": {
                "compressed_tokens": raw_tokens if point_budget == "raw" else compressed_tokens,
                "graph_mode": graph,
                "constrained_decoding": constrained,
                "adaptation_enabled": adaptation != "none",
                "adaptation_mode": "normalize",
            },
        }
        cases.append(
            {
                "name": f"{corruption}__{point_budget}__{graph}__{'constrained' if constrained else 'unconstrained'}__{adaptation}",
                "factors": {
                    "corruption": corruption,
                    "point_budget": point_budget,
                    "graph": graph,
                    "constrained": constrained,
                    "adaptation": adaptation,
                },
                "overrides": overrides,
            }
        )
    return cases


def run_case(config: dict[str, Any], seed: int, epochs: int, device: torch.device) -> dict[str, Any]:
    seed_everything(seed)
    run_config = copy.deepcopy(config)
    run_config["seed"] = seed
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
    metrics = evaluate_model(
        model,
        val_loader,
        device,
        output_mode=str(run_config.get("model", {}).get("output_mode", "json")),
        constrained=bool(run_config.get("model", {}).get("constrained_decoding", True)),
        include_outputs=True,
    )
    examples = metrics.pop("examples", [])
    return {"metrics": metrics, "failure_analysis": build_failure_report(examples)}


def main() -> None:
    args = parse_args()
    setup_logging()
    base_config = load_config(args.config)
    seeds = parse_seeds(args.seeds, base_config)
    epochs = args.epochs if args.epochs is not None else int(base_config.get("training", {}).get("epochs", 1))
    device = resolve_device(str(base_config.get("training", {}).get("device", "auto")))
    cases = build_main_matrix(base_config)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    results: dict[str, Any] = {}
    for case in cases:
        logging.info("Running main case=%s seeds=%s", case["name"], seeds)
        case_runs = []
        for seed in seeds:
            config = recursive_update(copy.deepcopy(base_config), case["overrides"])
            run = run_case(config, seed, epochs, device)
            case_runs.append({"seed": seed, **run})
        results[case["name"]] = {
            "factors": case["factors"],
            "runs": case_runs,
            "aggregate": aggregate_seed_results([run["metrics"] for run in case_runs]),
        }

    record = build_run_record(
        config=base_config,
        metrics={"main_matrix": results},
        seed=seeds[0],
        run_name="main_paper_matrix",
        preset="main_matrix",
        extra={"seeds": seeds, "epochs": epochs},
    )
    save_json_record(record, args.output)
    save_claim_report(record, args.artifacts_dir)
    logging.info("Saved main experiment record to %s", args.output)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset/setup error: {exc}", file=sys.stderr)
        print("Run: python scripts/check_arkitscenes_setup.py --config configs/arkitscenes.yaml", file=sys.stderr)
        raise SystemExit(1)
