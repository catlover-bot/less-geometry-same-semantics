"""Reproducible paper experiment runner.

Runs multi-seed severity benchmarks and, optionally, ablations. Each called
runner writes aggregated JSON plus CSV/Markdown tables and plots.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--ablation-config", default="configs/baseline.yaml")
    parser.add_argument("--output-dir", default="outputs/paper")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to config benchmark.seeds.")
    parser.add_argument("--skip-ablations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_output = output_dir / "severity_benchmark.json"
    _run(
        [
            sys.executable,
            "scripts/run_benchmark.py",
            "--config",
            args.config,
            "--output",
            str(benchmark_output),
            "--artifacts-dir",
            str(output_dir / "severity_benchmark"),
            *_optional_pair("--epochs", args.epochs),
            *_optional_pair("--seeds", args.seeds),
        ]
    )

    if not args.skip_ablations:
        ablation_output = output_dir / "ablations.json"
        _run(
            [
                sys.executable,
                "scripts/run_ablation.py",
                "--config",
                args.ablation_config,
                "--output",
                str(ablation_output),
                "--artifacts-dir",
                str(output_dir / "ablations"),
                *_optional_pair("--epochs", args.epochs),
                *_optional_pair("--seeds", args.seeds),
            ]
        )


def _optional_pair(flag: str, value: object | None) -> list[str]:
    return [flag, str(value)] if value is not None else []


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
