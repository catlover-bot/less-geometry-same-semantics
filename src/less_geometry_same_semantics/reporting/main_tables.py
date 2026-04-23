"""Main paper table construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from less_geometry_same_semantics.reporting.claims import main_matrix_table
from less_geometry_same_semantics.reporting.tables import (
    corruption_family_breakdown_table,
    save_latex_table,
    save_markdown_table,
    save_table_csv,
)


def main_results_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Compact main results table from the main benchmark matrix."""

    rows = []
    for row in main_matrix_table(record):
        rows.append(
            {
                "condition": row["case"],
                "corruption": row["corruption"],
                "budget": row["point_budget"],
                "graph": row["graph"],
                "constrained": row["constrained"],
                "adaptation": row["adaptation"],
                "obj_f1": row["object_f1"],
                "rel_f1": row["relation_f1"],
                "macro": row["semantic_macro_f1"],
                "json": row["json_validity"],
            }
        )
    return rows


def graph_comparison_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Graph vs no-graph table under severe corruption."""

    rows = []
    for row in main_matrix_table(record):
        if row["corruption"] != "severe_corruption":
            continue
        if row["graph"] not in {"no_graph", "simple_graph"}:
            continue
        rows.append(
            {
                "graph": row["graph"],
                "budget": row["point_budget"],
                "constrained": row["constrained"],
                "adaptation": row["adaptation"],
                "object_f1": row["object_f1"],
                "count_exact": row["count_exact"],
                "relation_f1": row["relation_f1"],
                "scene_accuracy": row["scene_accuracy"],
                "semantic_macro_f1": row["semantic_macro_f1"],
            }
        )
    return rows


def compression_efficiency_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Compression, latency, and semantic-quality table."""

    rows = []
    for row in main_matrix_table(record):
        rows.append(
            {
                "condition": row["case"],
                "budget": row["point_budget"],
                "compression_ratio": row["compression_ratio"],
                "latency_ms": row["latency_ms_per_sample"],
                "object_f1": row["object_f1"],
                "relation_f1": row["relation_f1"],
                "semantic_macro_f1": row["semantic_macro_f1"],
            }
        )
    return rows


def save_main_tables(
    *,
    main_record: dict[str, Any],
    output_dir: str | Path,
    ablation_record: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Save all main paper tables in CSV, Markdown, and LaTeX-friendly text."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "main_results": main_results_table(main_record),
        "graph_ablation": graph_comparison_table(main_record),
        "compression_efficiency": compression_efficiency_table(main_record),
        "corruption_family_breakdown": corruption_family_breakdown_table(ablation_record) if ablation_record else [],
    }
    for name, rows in tables.items():
        save_table_csv(rows, out / f"{name}.csv")
        save_markdown_table(rows, out / f"{name}.md")
        save_latex_table(rows, out / f"{name}.tex", caption=name.replace("_", " ").title())
    return tables
