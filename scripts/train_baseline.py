"""Train the minimal less-geometry-same-semantics baseline."""

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
from less_geometry_same_semantics.models import PointSemanticsModel
from less_geometry_same_semantics.training import evaluate_model, train_one_epoch
from less_geometry_same_semantics.utils.config import load_config
from less_geometry_same_semantics.utils.experiment_logging import build_run_record, save_json_record
from less_geometry_same_semantics.utils.logging import setup_logging
from less_geometry_same_semantics.utils.reproducibility import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    seed_everything(seed)

    train_cfg = config.get("training", {})
    output_dir = Path(args.output_dir or train_cfg.get("output_dir", "outputs/baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(train_cfg.get("device", "auto")))
    logging.info("Using device: %s", device)

    train_loader, val_loader = build_dataloaders(config)
    model = PointSemanticsModel.from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = args.epochs or int(train_cfg.get("epochs", 3))
    history = []
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            relation_loss_weight=float(train_cfg.get("relation_loss_weight", 0.5)),
        )
        eval_stats = evaluate_model(
            model,
            val_loader,
            device,
            output_mode=str(config.get("model", {}).get("output_mode", "json")),
            constrained=bool(config.get("model", {}).get("constrained_decoding", True)),
        )
        record = {"epoch": epoch, "train": train_stats, "eval": eval_stats}
        history.append(record)
        logging.info(
            "epoch=%d loss=%.4f obj_f1=%.3f rel_f1=%.3f json=%.3f",
            epoch,
            train_stats["loss"],
            eval_stats["semantic_quality"]["objects"]["f1"],
            eval_stats["semantic_quality"]["relations"]["f1"],
            eval_stats["json_validity"]["validity_rate"],
        )

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": config, "history": history}, checkpoint_path)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    run_record = build_run_record(
        config=config,
        metrics={"history": history, "final": history[-1]["eval"] if history else {}},
        seed=seed,
        run_name=str(config.get("benchmark", {}).get("name", "baseline_train")),
        preset=str(config.get("data", {}).get("corruption", {}).get("preset", "custom")),
        extra={"checkpoint": str(checkpoint_path)},
    )
    save_json_record(run_record, output_dir / "run.json")
    logging.info("Saved checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
