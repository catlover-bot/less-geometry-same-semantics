"""Run benchmark ablations over model size, compression, decoding, and corruption families."""

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
from less_geometry_same_semantics.models import PointSemanticsModel
from less_geometry_same_semantics.reporting.plots import plot_family_degradation, plot_pareto_curve
from less_geometry_same_semantics.reporting.tables import (
    compression_latency_semantic_table,
    corruption_family_breakdown_table,
    graph_ablation_table,
    save_markdown_table,
    save_table_csv,
)
from less_geometry_same_semantics.training import evaluate_model, train_one_epoch
from less_geometry_same_semantics.utils.config import load_config, recursive_update
from less_geometry_same_semantics.utils.experiment_logging import build_run_record, save_json_record
from less_geometry_same_semantics.utils.logging import setup_logging
from less_geometry_same_semantics.utils.reproducibility import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output", default="outputs/ablation_results.json")
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to benchmark.seeds.")
    return parser.parse_args()


def parse_seeds(raw: str | None, config: dict[str, Any]) -> list[int]:
    if raw:
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [int(seed) for seed in config.get("benchmark", {}).get("seeds", [config.get("seed", 0)])]


def ablation_cases(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    ablation_cfg = config.get("ablations", {})
    cases: dict[str, dict[str, Any]] = {
        "compressed_input": {},
        "raw_input": {
            "model": {"compressed_tokens": int(ablation_cfg.get("raw_vs_compressed", {}).get("raw_tokens", data_cfg.get("num_points", 512)))}
        },
        "small_decoder": {"model": ablation_cfg.get("decoder_sizes", {}).get("small", {"decoder_hidden_dim": 64, "decoder_depth": 1})},
        "large_decoder": {"model": ablation_cfg.get("decoder_sizes", {}).get("large", {"decoder_hidden_dim": 192, "decoder_depth": 3})},
        "constrained_output": {"model": {"constrained_decoding": True}},
        "unconstrained_output": {"model": {"constrained_decoding": False}},
        "structured_decoder": {"model": {"output_mode": "json"}},
        "free_form_text_decoder": {"model": {"output_mode": "text"}},
    }

    compressed_tokens = int(ablation_cfg.get("raw_vs_compressed", {}).get("compressed_tokens", model_cfg.get("compressed_tokens", 16)))
    cases["compressed_input"] = {"model": {"compressed_tokens": compressed_tokens}}

    default_families = {
        "geometry_only": {
            "geometry_degradation": {"enabled": True, "severity": "severe"},
            "coordinate_perturbation": {"enabled": False, "severity": "none"},
            "local_structural_corruption": {"enabled": False, "severity": "none"},
            "token_point_compression": {"enabled": False, "severity": "none"},
        },
        "coordinate_only": {
            "geometry_degradation": {"enabled": False, "severity": "none"},
            "coordinate_perturbation": {"enabled": True, "severity": "severe"},
            "local_structural_corruption": {"enabled": False, "severity": "none"},
            "token_point_compression": {"enabled": False, "severity": "none"},
        },
        "local_structure_only": {
            "geometry_degradation": {"enabled": False, "severity": "none"},
            "coordinate_perturbation": {"enabled": False, "severity": "none"},
            "local_structural_corruption": {"enabled": True, "severity": "severe"},
            "token_point_compression": {"enabled": False, "severity": "none"},
        },
        "compression_only": {
            "geometry_degradation": {"enabled": False, "severity": "none"},
            "coordinate_perturbation": {"enabled": False, "severity": "none"},
            "local_structural_corruption": {"enabled": False, "severity": "none"},
            "token_point_compression": {"enabled": True, "severity": "severe"},
        },
    }
    for name, corruption in (ablation_cfg.get("corruption_families") or default_families).items():
        cases[f"family_{name}"] = {
            "data": {
                "corruption": {
                    "preset": f"ablation_{name}",
                    **corruption,
                }
            }
        }
    default_graph_modes = {
        "no_graph": {"graph_mode": "no_graph"},
        "simple_graph": {"graph_mode": "simple_graph", "graph_layers": 1},
        "richer_graph": {"graph_mode": "richer_graph", "graph_layers": 2, "graph_k_nearest": 4},
        "graph_with_relation_dropout": {"graph_mode": "simple_graph", "graph_layers": 1, "graph_edge_dropout": 0.25},
        "graph_with_noisy_edges": {"graph_mode": "simple_graph", "graph_layers": 1, "graph_edge_noise_std": 0.10},
    }
    for name, model_overrides in (ablation_cfg.get("graph_modes") or default_graph_modes).items():
        cases[f"graph_{name}"] = {"model": model_overrides}
    for name, model_overrides in ablation_cfg.get("robustness_mechanism", {}).items():
        cases[f"robustness_{name}"] = {"model": model_overrides}
    default_adaptation = {
        "none": {"adaptation_enabled": False},
        "input_normalization": {"adaptation_enabled": True, "adaptation_mode": "normalize"},
    }
    for name, model_overrides in (ablation_cfg.get("adaptation") or default_adaptation).items():
        cases[f"adaptation_{name}"] = {"model": model_overrides}
    return cases


def run_one_case(config: dict[str, Any], seed: int, epochs: int, device: torch.device) -> dict[str, Any]:
    seed_everything(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed
    train_loader, val_loader = build_dataloaders(config)
    model = PointSemanticsModel.from_config(config).to(device)
    train_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    train_history = [
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            relation_loss_weight=float(train_cfg.get("relation_loss_weight", 0.5)),
        )
        for _ in range(epochs)
    ]
    metrics = evaluate_model(
        model,
        val_loader,
        device,
        output_mode=str(config.get("model", {}).get("output_mode", "json")),
        constrained=bool(config.get("model", {}).get("constrained_decoding", True)),
        include_outputs=True,
    )
    return {"train": train_history, "eval": metrics}


def main() -> None:
    args = parse_args()
    setup_logging()
    base_config = load_config(args.config)
    seeds = parse_seeds(args.seeds, base_config)
    epochs = args.epochs or int(base_config.get("training", {}).get("epochs", 1))
    device = resolve_device(str(base_config.get("training", {}).get("device", "auto")))
    results: dict[str, Any] = {}

    for case_name, overrides in ablation_cases(base_config).items():
        case_runs = []
        logging.info("Running ablation case=%s seeds=%s", case_name, seeds)
        for seed in seeds:
            case_config = recursive_update(copy.deepcopy(base_config), overrides)
            run = run_one_case(case_config, seed, epochs, device)
            examples = run["eval"].pop("examples", [])
            case_runs.append(
                {
                    "seed": seed,
                    **run,
                    "failure_analysis": build_failure_report(examples),
                }
            )
        eval_metrics = [run["eval"] for run in case_runs]
        results[case_name] = {
            "runs": case_runs,
            "aggregate": aggregate_seed_results(eval_metrics),
        }
        mean_quality = results[case_name]["aggregate"]["mean"]["semantic_quality"]["semantic_macro_f1"]
        logging.info("case=%s semantic_macro_f1_mean=%.3f", case_name, mean_quality)

    record = build_run_record(
        config=base_config,
        metrics={"ablation_cases": results},
        seed=seeds[0],
        run_name="ablation",
        preset=str(base_config.get("data", {}).get("corruption", {}).get("preset", "custom")),
        extra={"seeds": seeds, "epochs": epochs},
    )
    save_json_record(record, args.output)
    artifact_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path(args.output).with_suffix("")
    compression_rows = compression_latency_semantic_table(record)
    family_rows = corruption_family_breakdown_table(record)
    graph_rows = graph_ablation_table(record)
    save_table_csv(compression_rows, artifact_dir / "ablation_compression_latency_semantics.csv")
    save_markdown_table(compression_rows, artifact_dir / "ablation_compression_latency_semantics.md")
    save_table_csv(family_rows, artifact_dir / "corruption_family_breakdown.csv")
    save_markdown_table(family_rows, artifact_dir / "corruption_family_breakdown.md")
    if graph_rows:
        save_table_csv(graph_rows, artifact_dir / "graph_ablation.csv")
        save_markdown_table(graph_rows, artifact_dir / "graph_ablation.md")
    plot_pareto_curve(record, artifact_dir / "ablation_pareto.png")
    if family_rows:
        plot_family_degradation(record, artifact_dir / "corruption_family_degradation.png")
    logging.info("Saved ablation results to %s", args.output)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset/setup error: {exc}", file=sys.stderr)
        print("Run: python scripts/check_arkitscenes_setup.py --config configs/arkitscenes.yaml", file=sys.stderr)
        raise SystemExit(1)
