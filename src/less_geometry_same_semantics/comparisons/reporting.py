"""Paper-facing tables for baseline comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from less_geometry_same_semantics.reporting.tables import save_latex_table, save_markdown_table, save_table_csv


def comparison_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten comparison results into human-readable table rows."""

    rows = []
    for case_name, payload in record.get("metrics", {}).get("comparisons", {}).items():
        metrics = payload.get("aggregate", {}).get("mean") or payload.get("metrics", {})
        baseline = payload.get("baseline", {})
        alignment = payload.get("task_alignment", {})
        rows.append(
            {
                "case": case_name,
                "baseline": baseline.get("label", payload.get("baseline_id", case_name)),
                "group": baseline.get("group", "main"),
                "family": baseline.get("family", "comparison"),
                "kind": baseline.get("kind", "unknown"),
                "execution": payload.get("execution_mode", _execution_mode(baseline)),
                "availability": _availability_from_status(payload.get("status", "unknown"), payload.get("execution_mode", _execution_mode(baseline))),
                "condition": payload.get("condition", ""),
                "status": payload.get("status", "unknown"),
                "alignment": _alignment_summary(alignment),
                "object_f1": _format_metric(metrics, "semantic_quality.objects.f1"),
                "relation_f1": _format_metric(metrics, "semantic_quality.relations.f1"),
                "count_exact": _format_metric(metrics, "semantic_quality.object_counts.exact_match"),
                "scene_accuracy": _format_metric(metrics, "semantic_quality.scene_type.accuracy"),
                "json_validity": _format_metric(metrics, "json_validity.validity_rate"),
                "json_mode": _format_value(_get(metrics, "json_validity.mode") or alignment.get("json_mode")),
                "latency_ms": _format_metric(metrics, "efficiency.latency_ms_per_sample"),
                "memory_mb": _format_metric(metrics, "efficiency.process_memory_mb"),
                "parameter_count": _format_metric(metrics, "efficiency.parameter_count"),
                "compression_ratio": _format_metric(metrics, "compression.compression_ratio"),
                "notes": payload.get("notes", ""),
            }
        )
    return sorted(rows, key=lambda row: (row["group"], row["availability"], row["baseline"], row["condition"]))


def main_comparison_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in comparison_rows(record) if row["group"] == "main"]


def supplementary_comparison_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in comparison_rows(record) if row["group"] == "supplementary"]


def heavy_vs_lightweight_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in comparison_rows(record):
        if row["status"] != "completed":
            continue
        if row["family"] not in {"lightweight", "heavy_upper_bound", "standard_3d", "robustness"}:
            continue
        rows.append(
            {
                "baseline": row["baseline"],
                "family": row["family"],
                "execution": row["execution"],
                "availability": row["availability"],
                "condition": row["condition"],
                "macro": _format_metric_from_row(row, "object_f1", "relation_f1", "scene_accuracy"),
                "object_f1": row["object_f1"],
                "relation_f1": row["relation_f1"],
                "latency_ms": row["latency_ms"],
                "parameter_count": row["parameter_count"],
                "alignment": row["alignment"],
            }
        )
    return rows


def robustness_vs_compute_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in comparison_rows(record):
        if row["status"] != "completed":
            continue
        if row["condition"] not in {"severe_corruption", "extreme_compression", "clean"}:
            continue
        rows.append(
            {
                "baseline": row["baseline"],
                "condition": row["condition"],
                "family": row["family"],
                "execution": row["execution"],
                "availability": row["availability"],
                "object_f1": row["object_f1"],
                "relation_f1": row["relation_f1"],
                "count_exact": row["count_exact"],
                "scene_accuracy": row["scene_accuracy"],
                "latency_ms": row["latency_ms"],
                "memory_mb": row["memory_mb"],
                "parameter_count": row["parameter_count"],
                "compression_ratio": row["compression_ratio"],
            }
        )
    return rows


def comparison_status_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "baseline": row["baseline"],
            "condition": row["condition"],
            "group": row["group"],
            "execution": row["execution"],
            "availability": row["availability"],
            "status": row["status"],
            "alignment": row["alignment"],
            "notes": row["notes"],
        }
        for row in comparison_rows(record)
    ]


def save_comparison_tables(record: dict[str, Any], output_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Save comparison tables in CSV, Markdown, and LaTeX-friendly text."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "main_comparisons": main_comparison_table(record),
        "supplementary_comparisons": supplementary_comparison_table(record),
        "heavy_vs_lightweight": heavy_vs_lightweight_table(record),
        "robustness_vs_compute": robustness_vs_compute_table(record),
        "comparison_status": comparison_status_table(record),
    }
    for name, rows in tables.items():
        save_table_csv(rows, out / f"{name}.csv")
        save_markdown_table(rows, out / f"{name}.md")
        save_latex_table(rows, out / f"{name}.tex", caption=name.replace("_", " ").title())
    return tables


def _alignment_summary(alignment: dict[str, Any]) -> str:
    return ", ".join(
        [
            f"json={alignment.get('json_mode', 'n/a')}",
            f"relations={alignment.get('relations_mode', 'n/a')}",
            f"scene={alignment.get('scene_type_mode', 'n/a')}",
        ]
    )


def _format_metric(mapping: dict[str, Any], path: str) -> str:
    value = _get(mapping, path)
    return _format_value(value)


def _format_metric_from_row(row: dict[str, Any], *keys: str) -> str:
    numeric = []
    for key in keys:
        try:
            numeric.append(float(row[key]))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return "n/a"
    return f"{sum(numeric) / len(numeric):.4f}"


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


def _get(mapping: dict[str, Any], path: str) -> Any:
    cursor: Any = mapping
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _execution_mode(baseline: dict[str, Any]) -> str:
    return "local" if baseline.get("kind") == "internal_model" else "external_import"


def _availability_from_status(status: str, execution_mode: str) -> str:
    if execution_mode == "local":
        return "available_now" if status == "completed_local" else "local_issue"
    if status == "imported":
        return "imported"
    if status in {"pending_external", "incomplete_import"}:
        return "pending_external"
    if status == "invalid_import":
        return "import_problem"
    return "unknown"
