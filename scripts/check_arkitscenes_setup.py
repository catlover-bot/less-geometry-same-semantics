"""Focused ARKitScenes setup check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.data.preflight import (
    build_setup_report,
    format_setup_report_markdown,
    save_setup_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/arkitscenes.yaml")
    parser.add_argument("--output-dir", default="outputs/preflight/arkitscenes")
    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument("--no-strict", action="store_true", help="Return exit code 0 even when checks fail.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_setup_report([args.config], max_scenes=args.max_scenes)
    save_setup_report(report, args.output_dir)
    print(format_setup_report_markdown(report))
    if report["status"] == "fail" and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
