"""Generate paper-friendly dataset diagnostics for configured train/val splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.data.diagnostics import dataset_diagnostics, save_diagnostics_artifacts
from less_geometry_same_semantics.data.loaders import build_dataloaders
from less_geometry_same_semantics.utils.config import load_config
from less_geometry_same_semantics.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/arkitscenes.yaml")
    parser.add_argument("--output-dir", default="outputs/diagnostics")
    parser.add_argument("--max-scenes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)
    train_loader, val_loader = build_dataloaders(config)
    output_root = Path(args.output_dir)
    for split_name, dataset in [("train", train_loader.dataset), ("val", val_loader.dataset)]:
        summary = dataset_diagnostics(dataset, max_scenes=args.max_scenes)
        save_diagnostics_artifacts(summary, output_root / split_name)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset/setup error: {exc}", file=sys.stderr)
        print("Run: python scripts/check_arkitscenes_setup.py --config configs/arkitscenes.yaml", file=sys.stderr)
        raise SystemExit(1)
