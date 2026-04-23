"""Audit completed paper artifacts for draft readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from less_geometry_same_semantics.experiments.plan import expected_main_case_names, plan_path
from less_geometry_same_semantics.comparisons.reporting import comparison_rows
from less_geometry_same_semantics.reporting.claims import main_matrix_table
from less_geometry_same_semantics.reporting.tables import severity_semantic_metrics_table


REQUIRED_MAIN_METRICS = [
    "object_f1",
    "count_exact",
    "relation_f1",
    "scene_accuracy",
    "semantic_macro_f1",
    "json_validity",
    "latency_ms_per_sample",
    "compression_ratio",
]


def audit_paper_outputs(plan: dict[str, Any]) -> dict[str, Any]:
    """Check whether completed outputs are sufficient for a paper draft."""

    findings: list[dict[str, str]] = []
    output_root = Path(plan.get("paper", {}).get("output_dir", "outputs/paper_package"))
    main_path = output_root / plan["main_benchmark"]["output"]
    severity_path = output_root / plan["severity_benchmark"]["output"]
    ablation_path = output_root / plan["ablations"]["output"]
    comparison_cfg = plan.get("comparisons", {})
    comparison_enabled = bool(comparison_cfg.get("enabled", False))
    comparison_path = output_root / comparison_cfg.get("output", "comparisons/results.json")

    main_record = _load_json(main_path, findings, "main benchmark record")
    severity_record = _load_json(severity_path, findings, "severity benchmark record")
    _load_json(ablation_path, findings, "ablation record")
    comparison_record = _load_json(comparison_path, findings, "comparison record") if comparison_enabled else None

    if main_record:
        _audit_main_record(plan, main_record, findings)
    if severity_record:
        _audit_severity_record(plan, severity_record, findings)
    if comparison_enabled and comparison_record:
        _audit_comparison_record(comparison_record, findings)
    _audit_expected_artifacts(plan, findings)

    status = "pass" if not any(item["severity"] == "error" for item in findings) else "fail"
    return {"status": status, "findings": findings}


def save_audit_report(report: dict[str, Any], output_dir: str | Path) -> None:
    """Save audit report as JSON and Markdown."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Result Audit", "", f"Status: **{report['status']}**", ""]
    for finding in report["findings"]:
        lines.append(f"- **{finding['severity']}** `{finding['check']}`: {finding['message']}")
    (out / "result_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json(path: Path, findings: list[dict[str, str]], label: str) -> dict[str, Any] | None:
    if not path.exists():
        findings.append({"severity": "error", "check": "artifact_exists", "message": f"Missing {label}: {path}"})
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        findings.append({"severity": "error", "check": "json_parse", "message": f"Could not parse {path}: {exc}"})
        return None


def _audit_main_record(plan: dict[str, Any], record: dict[str, Any], findings: list[dict[str, str]]) -> None:
    expected_cases = set(expected_main_case_names(plan))
    actual_cases = set(record.get("metrics", {}).get("main_matrix", {}))
    missing = sorted(expected_cases - actual_cases)
    if missing:
        findings.append({"severity": "error", "check": "expected_main_cases", "message": f"Missing main cases: {missing[:8]}"})
    extra = sorted(actual_cases - expected_cases)
    if extra:
        findings.append({"severity": "warning", "check": "unexpected_main_cases", "message": f"Unexpected main cases: {extra[:8]}"})

    expected_seeds = [int(seed) for seed in plan.get("runs", {}).get("seeds", [])]
    for case_name, payload in record.get("metrics", {}).get("main_matrix", {}).items():
        seeds = sorted(int(run["seed"]) for run in payload.get("runs", []))
        if expected_seeds and seeds != sorted(expected_seeds):
            findings.append({"severity": "error", "check": "expected_seeds", "message": f"{case_name} has seeds {seeds}, expected {expected_seeds}"})

    for row in main_matrix_table(record):
        missing_metrics = [metric for metric in REQUIRED_MAIN_METRICS if row.get(metric) is None]
        if missing_metrics:
            findings.append({"severity": "error", "check": "main_table_metrics", "message": f"{row.get('case')} missing metrics {missing_metrics}"})


def _audit_severity_record(plan: dict[str, Any], record: dict[str, Any], findings: list[dict[str, str]]) -> None:
    expected = set(plan.get("severity_benchmark", {}).get("expected_presets", []))
    actual = {row["preset"] for row in severity_semantic_metrics_table(record)}
    missing = sorted(expected - actual)
    if missing:
        findings.append({"severity": "error", "check": "severity_presets", "message": f"Missing severity presets: {missing}"})


def _audit_expected_artifacts(plan: dict[str, Any], findings: list[dict[str, str]]) -> None:
    artifact_cfg = plan.get("paper_artifacts", {})
    expected = [
        plan_path(plan, artifact_cfg.get("figures_dir", "figures"), "severity_metrics.png"),
        plan_path(plan, artifact_cfg.get("figures_dir", "figures"), "compression_latency_semantics.png"),
        plan_path(plan, artifact_cfg.get("figures_dir", "figures"), "graph_vs_no_graph_severe.png"),
        plan_path(plan, artifact_cfg.get("tables_dir", "tables"), "main_results.csv"),
        plan_path(plan, artifact_cfg.get("tables_dir", "tables"), "graph_ablation.csv"),
        plan_path(plan, artifact_cfg.get("tables_dir", "tables"), "compression_efficiency.csv"),
        plan_path(plan, artifact_cfg.get("claims_dir", "claims"), "frozen_claims.json"),
        plan_path(plan, artifact_cfg.get("support_dir", "paper_support"), "problem_statement.md"),
    ]
    if bool(plan.get("comparisons", {}).get("enabled", False)):
        expected.extend(
            [
                plan_path(plan, artifact_cfg.get("tables_dir", "tables"), "main_comparisons.csv"),
                plan_path(plan, artifact_cfg.get("tables_dir", "tables"), "supplementary_comparisons.csv"),
                plan_path(plan, artifact_cfg.get("tables_dir", "tables"), "heavy_vs_lightweight.csv"),
                plan_path(plan, artifact_cfg.get("tables_dir", "tables"), "robustness_vs_compute.csv"),
            ]
        )
    for path in expected:
        if not path.exists():
            findings.append({"severity": "warning", "check": "paper_artifact_exists", "message": f"Missing expected paper artifact: {path}"})


def _audit_comparison_record(record: dict[str, Any], findings: list[dict[str, str]]) -> None:
    rows = comparison_rows(record)
    if not rows:
        findings.append({"severity": "warning", "check": "comparison_rows", "message": "Comparison record exists but contains no rows."})
        return
    pending = [row for row in rows if row.get("group") == "main" and row.get("availability") not in {"available_now", "imported"}]
    if pending:
        findings.append(
            {
                "severity": "warning",
                "check": "comparison_status",
                "message": f"Main comparison baselines pending external import or with import issues: {[row['case'] for row in pending[:8]]}",
            }
        )
