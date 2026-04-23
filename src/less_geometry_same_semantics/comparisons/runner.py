"""Execution helpers for internal and imported comparison baselines."""

from __future__ import annotations

import copy
from typing import Any

import torch

from less_geometry_same_semantics.data.loaders import build_dataloaders
from less_geometry_same_semantics.metrics.aggregation import aggregate_seed_results
from less_geometry_same_semantics.models import PointSemanticsModel
from less_geometry_same_semantics.training import evaluate_model, train_one_epoch
from less_geometry_same_semantics.utils.config import recursive_update
from less_geometry_same_semantics.utils.reproducibility import seed_everything


def collect_validation_references(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect validation targets and metadata for imported-baseline evaluation."""

    _, val_loader = build_dataloaders(config)
    references: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for batch in val_loader:
        references.extend(batch["targets"])
        metadata.extend(batch["metadata"])
    return references, metadata


def run_internal_comparison(
    base_config: dict[str, Any],
    *,
    config_overrides: dict[str, Any],
    seeds: list[int],
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train and evaluate one internal baseline across multiple seeds."""

    runs = []
    for seed in seeds:
        run_config = recursive_update(copy.deepcopy(base_config), config_overrides)
        run_config["seed"] = seed
        seed_everything(seed)
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
        runs.append(
            {
                "seed": seed,
                "metrics": evaluate_model(
                    model,
                    val_loader,
                    device,
                    output_mode=str(run_config.get("model", {}).get("output_mode", "json")),
                    constrained=bool(run_config.get("model", {}).get("constrained_decoding", True)),
                    include_outputs=False,
                ),
            }
        )
    return {"runs": runs, "aggregate": aggregate_seed_results([run["metrics"] for run in runs])}
