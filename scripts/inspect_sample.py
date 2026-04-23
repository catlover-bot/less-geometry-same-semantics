"""Print one synthetic scene before and after corruption."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.data.loaders import corruption_from_config
from less_geometry_same_semantics.data.synthetic import SyntheticSceneDataset
from less_geometry_same_semantics.metrics.efficiency import compression_ratio, retained_fraction
from less_geometry_same_semantics.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config.get("data", {})
    benchmark_cfg = config.get("benchmark", {})
    dataset = SyntheticSceneDataset(
        num_samples=max(args.index + 1, 1),
        num_points=int(data_cfg.get("num_points", 512)),
        corruption=corruption_from_config(data_cfg.get("corruption"), benchmark_cfg.get("presets")),
        seed=int(config.get("seed", 0)),
        return_clean=True,
    )
    sample = dataset[args.index]
    clean_n = sample["metadata"]["clean_num_points"]
    degraded_n = sample["metadata"]["degraded_num_points"]
    report = {
        "target": sample["target"],
        "clean_num_points": clean_n,
        "degraded_num_points": degraded_n,
        "compression_ratio": compression_ratio(clean_n, degraded_n),
        "retained_fraction": retained_fraction(clean_n, degraded_n),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
