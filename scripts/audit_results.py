"""Audit a paper package for missing results, metrics, and artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.experiments.plan import load_experiment_plan, plan_path
from less_geometry_same_semantics.reporting.audit import audit_paper_outputs, save_audit_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/paper_plan.yaml")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_experiment_plan(args.plan)
    output_dir = Path(args.output_dir) if args.output_dir else plan_path(plan, plan.get("paper_artifacts", {}).get("audit_dir", "audit"))
    report = audit_paper_outputs(plan)
    save_audit_report(report, output_dir)


if __name__ == "__main__":
    main()
