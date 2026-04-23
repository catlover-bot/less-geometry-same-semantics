"""Check local Python and public-dataset setup before real benchmark runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.data.preflight import (
    build_setup_report,
    check_environment,
    format_setup_report_markdown,
    save_setup_report,
)
from less_geometry_same_semantics.experiments.plan import load_experiment_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=None, help="Dataset config to check. May be repeated.")
    parser.add_argument("--plan", default="configs/paper_plan.yaml", help="Plan used when --config is omitted.")
    parser.add_argument("--output-dir", default="outputs/preflight", help="Where JSON/Markdown reports are saved.")
    parser.add_argument("--max-scenes", type=int, default=3, help="Number of scene ids per split to sample-check.")
    parser.add_argument("--environment-only", action="store_true", help="Only check Python/package environment.")
    parser.add_argument("--no-strict", action="store_true", help="Return exit code 0 even when checks fail.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.environment_only:
        report = {"status": "pass", "environment": check_environment(), "datasets": []}
        report["status"] = "fail" if report["environment"]["status"] == "fail" else "pass"
    else:
        configs = args.config or _configs_from_plan(args.plan)
        report = build_setup_report(configs, max_scenes=args.max_scenes)
    save_setup_report(report, args.output_dir)
    print(format_setup_report_markdown(report))
    if report["status"] == "fail" and not args.no_strict:
        raise SystemExit(1)


def _configs_from_plan(plan_path: str | Path) -> list[str]:
    plan = load_experiment_plan(plan_path)
    configs = [plan["datasets"]["primary"]["config"]]
    secondary = plan.get("datasets", {}).get("secondary", {}).get("config")
    if secondary:
        configs.append(secondary)
    return configs


if __name__ == "__main__":
    main()
