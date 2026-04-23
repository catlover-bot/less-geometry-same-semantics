"""Generate standard paper tables from experiment records."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def severity_semantic_metrics_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rows for severity-vs-semantic-metrics tables."""

    rows = []
    for name, metrics in _iter_metric_groups(record, "presets"):
        rows.append(
            {
                "preset": name,
                "object_f1": _get(metrics, "semantic_quality.objects.f1"),
                "count_exact": _get(metrics, "semantic_quality.object_counts.exact_match"),
                "relation_f1": _get(metrics, "semantic_quality.relations.f1"),
                "scene_accuracy": _get(metrics, "semantic_quality.scene_type.accuracy"),
                "semantic_macro_f1": _get(metrics, "semantic_quality.semantic_macro_f1"),
                "json_validity": _get(metrics, "json_validity.validity_rate"),
            }
        )
    return rows


def clean_vs_corrupted_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a compact main table contrasting clean with all corrupted settings."""

    rows = []
    for row in severity_semantic_metrics_table(record):
        setting = "clean" if row["preset"] == "clean" else "corrupted"
        rows.append(
            {
                "setting": setting,
                "preset": row["preset"],
                "object_f1": row["object_f1"],
                "relation_f1": row["relation_f1"],
                "semantic_macro_f1": row["semantic_macro_f1"],
                "json_validity": row["json_validity"],
            }
        )
    return rows


def compression_latency_semantic_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rows linking compression, latency, and semantic quality."""

    rows = []
    groups = list(_iter_metric_groups(record, "presets")) or list(_iter_metric_groups(record, "ablation_cases"))
    for name, metrics in groups:
        rows.append(
            {
                "setting": name,
                "compression_ratio": _get(metrics, "compression.compression_ratio"),
                "retained_fraction": _get(metrics, "compression.retained_fraction"),
                "latency_ms_per_sample": _get(metrics, "efficiency.latency_ms_per_sample"),
                "semantic_macro_f1": _get(metrics, "semantic_quality.semantic_macro_f1"),
            }
        )
    return rows


def corruption_family_breakdown_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rows for per-corruption-family ablation breakdowns."""

    rows = []
    for name, metrics in _iter_metric_groups(record, "ablation_cases"):
        if not name.startswith("family_"):
            continue
        rows.append(
            {
                "family": name.removeprefix("family_"),
                "object_f1": _get(metrics, "semantic_quality.objects.f1"),
                "relation_f1": _get(metrics, "semantic_quality.relations.f1"),
                "semantic_macro_f1": _get(metrics, "semantic_quality.semantic_macro_f1"),
                "compression_ratio": _get(metrics, "compression.compression_ratio"),
            }
        )
    return rows


def graph_ablation_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rows for graph vs no-graph ablations."""

    rows = []
    for name, metrics in _iter_metric_groups(record, "ablation_cases"):
        if not name.startswith("graph_"):
            continue
        rows.append(
            {
                "graph_setting": name.removeprefix("graph_"),
                "object_f1": _get(metrics, "semantic_quality.objects.f1"),
                "relation_f1": _get(metrics, "semantic_quality.relations.f1"),
                "semantic_macro_f1": _get(metrics, "semantic_quality.semantic_macro_f1"),
                "latency_ms_per_sample": _get(metrics, "efficiency.latency_ms_per_sample"),
                "parameter_count": _get(metrics, "efficiency.parameter_count"),
            }
        )
    return rows


def save_table_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Save table rows to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def save_markdown_table(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Save table rows to a compact Markdown table."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(header)) for header in headers) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def save_latex_table(rows: list[dict[str, Any]], path: str | Path, caption: str = "") -> Path:
    """Save a simple LaTeX-friendly tabular fragment."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("% Empty table\n", encoding="utf-8")
        return output_path
    headers = list(rows[0].keys())
    lines = [
        "% Auto-generated table fragment",
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\toprule",
        " & ".join(_latex_escape(header) for header in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(_format_cell(row.get(header))) for header in headers) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if caption:
        lines.insert(0, f"% {caption}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _iter_metric_groups(record: dict[str, Any], group_name: str) -> list[tuple[str, dict[str, Any]]]:
    groups = record.get("metrics", {}).get(group_name, {})
    rows = []
    for name, payload in groups.items():
        if "aggregate" in payload:
            rows.append((name, payload["aggregate"].get("mean", {})))
        elif "metrics" in payload:
            rows.append((name, payload["metrics"]))
        else:
            rows.append((name, payload))
    return rows


def _get(mapping: dict[str, Any], path: str, default: float = 0.0) -> float:
    cursor: Any = mapping
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    try:
        return float(cursor)
    except (TypeError, ValueError):
        return default


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
