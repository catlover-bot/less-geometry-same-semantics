"""Run the full paper package workflow from an explicit experiment plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.experiments.plan import load_experiment_plan, plan_path
from less_geometry_same_semantics.data.preflight import build_setup_report, format_setup_report_markdown, save_setup_report
from less_geometry_same_semantics.comparisons.reporting import save_comparison_tables
from less_geometry_same_semantics.reporting.audit import audit_paper_outputs, save_audit_report
from less_geometry_same_semantics.reporting.freeze_claims import freeze_supported_claims, save_frozen_claims
from less_geometry_same_semantics.reporting.main_figures import save_main_figures
from less_geometry_same_semantics.reporting.main_tables import save_main_tables
from less_geometry_same_semantics.reporting.paper_support import generate_paper_support_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/paper_plan.yaml")
    parser.add_argument("--primary-config", default=None, help="Override plan primary dataset config.")
    parser.add_argument("--secondary-config", default=None, help="Override plan secondary dataset config.")
    parser.add_argument("--output-dir", default=None, help="Override plan paper.output_dir.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-supplementary", action="store_true")
    parser.add_argument("--skip-comparisons", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_experiment_plan(args.plan)
    if args.output_dir:
        plan.setdefault("paper", {})["output_dir"] = args.output_dir
    if args.primary_config:
        plan.setdefault("datasets", {}).setdefault("primary", {})["config"] = args.primary_config
    if args.secondary_config:
        plan.setdefault("datasets", {}).setdefault("secondary", {})["config"] = args.secondary_config
    output_root = Path(plan["paper"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds or ",".join(str(seed) for seed in plan.get("runs", {}).get("seeds", []))
    epochs = args.epochs if args.epochs is not None else int(plan.get("runs", {}).get("epochs", 1))
    primary_config = plan["datasets"]["primary"]["config"]
    secondary_config = plan.get("datasets", {}).get("secondary", {}).get("config")
    supplementary_enabled = bool(plan.get("supplementary", {}).get("enabled", False))
    if not args.skip_preflight:
        preflight_configs = [primary_config]
        if secondary_config and supplementary_enabled and not args.skip_supplementary:
            preflight_configs.append(secondary_config)
        report = build_setup_report(preflight_configs)
        save_setup_report(report, output_root / "preflight")
        if report["status"] == "fail":
            print(format_setup_report_markdown(report))
            raise SystemExit(
                "Dataset preflight failed. Fix the errors above or rerun with --skip-preflight if you intentionally want raw loader errors."
            )

    main_record_path = output_root / plan["main_benchmark"]["output"]
    severity_record_path = output_root / plan["severity_benchmark"]["output"]
    ablation_record_path = output_root / plan["ablations"]["output"]
    comparison_cfg = plan.get("comparisons", {})
    comparison_enabled = bool(comparison_cfg.get("enabled", False)) and not args.skip_comparisons
    comparison_record_path = output_root / comparison_cfg.get("output", "comparisons/results.json")

    if not args.skip_runs:
        _run(
            [
                sys.executable,
                "scripts/run_main_experiments.py",
                "--config",
                primary_config,
                "--output",
                str(main_record_path),
                "--artifacts-dir",
                str(output_root / plan["main_benchmark"]["artifacts_dir"]),
                "--epochs",
                str(epochs),
                "--seeds",
                seeds,
                *_optional_pair("--max-cases", args.max_cases),
            ]
        )
        _run(
            [
                sys.executable,
                "scripts/run_benchmark.py",
                "--config",
                primary_config,
                "--output",
                str(severity_record_path),
                "--artifacts-dir",
                str(output_root / plan["severity_benchmark"]["artifacts_dir"]),
                "--epochs",
                str(epochs),
                "--seeds",
                seeds,
            ]
        )
        if not args.skip_ablations:
            _run(
                [
                    sys.executable,
                    "scripts/run_ablation.py",
                    "--config",
                    primary_config,
                    "--output",
                    str(ablation_record_path),
                    "--artifacts-dir",
                    str(output_root / plan["ablations"]["artifacts_dir"]),
                    "--epochs",
                    str(epochs),
                    "--seeds",
                    seeds,
                ]
            )
        if comparison_enabled:
            _run(
                [
                    sys.executable,
                    "scripts/run_comparison_baselines.py",
                    "--config",
                    primary_config,
                    "--comparison-config",
                    comparison_cfg.get("config", "configs/comparisons.yaml"),
                    "--output",
                    str(comparison_record_path),
                    "--artifacts-dir",
                    str(output_root / comparison_cfg.get("artifacts_dir", "comparisons")),
                    "--epochs",
                    str(epochs),
                    "--seeds",
                    seeds,
                ]
            )
        if secondary_config and supplementary_enabled and not args.skip_supplementary:
            _run(
                [
                    sys.executable,
                    "scripts/run_benchmark.py",
                    "--config",
                    secondary_config,
                    "--output",
                    str(output_root / plan["supplementary"]["output"]),
                    "--artifacts-dir",
                    str(output_root / plan["supplementary"]["artifacts_dir"]),
                    "--epochs",
                    str(epochs),
                    "--seeds",
                    seeds,
                ]
            )

    main_record = _load_record(main_record_path)
    severity_record = _load_record(severity_record_path) if severity_record_path.exists() else None
    ablation_record = _load_record(ablation_record_path) if ablation_record_path.exists() else None
    comparison_record = _load_record(comparison_record_path) if comparison_enabled and comparison_record_path.exists() else None

    artifact_cfg = plan.get("paper_artifacts", {})
    figures_dir = plan_path(plan, artifact_cfg.get("figures_dir", "figures"))
    tables_dir = plan_path(plan, artifact_cfg.get("tables_dir", "tables"))
    claims_dir = plan_path(plan, artifact_cfg.get("claims_dir", "claims"))
    support_dir = plan_path(plan, artifact_cfg.get("support_dir", "paper_support"))
    audit_dir = plan_path(plan, artifact_cfg.get("audit_dir", "audit"))

    save_main_figures(main_record=main_record, severity_record=severity_record, output_dir=figures_dir)
    save_main_tables(main_record=main_record, ablation_record=ablation_record, output_dir=tables_dir)
    if comparison_record is not None:
        save_comparison_tables(comparison_record, tables_dir)
    frozen = freeze_supported_claims(main_record, severity_record)
    save_frozen_claims(frozen, claims_dir)
    generate_paper_support_package(
        support_dir,
        context={
            "figures": str(figures_dir),
            "tables": str(tables_dir),
            "claims": str(claims_dir),
            "main_record": str(main_record_path),
            "severity_record": str(severity_record_path),
            "comparisons": str(comparison_record_path) if comparison_record is not None else "not_generated",
        },
    )
    if not args.skip_audit:
        audit = audit_paper_outputs(plan)
        save_audit_report(audit, audit_dir)


def _optional_pair(flag: str, value: object | None) -> list[str]:
    return [flag, str(value)] if value is not None else []


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def _load_record(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required record missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
