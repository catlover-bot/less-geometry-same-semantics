"""Claim-oriented reporting for paper main results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from less_geometry_same_semantics.reporting.tables import save_markdown_table, save_table_csv


def main_matrix_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten main benchmark matrix results into table rows."""

    rows = []
    cases = record.get("metrics", {}).get("main_matrix", {})
    for case_name, payload in cases.items():
        metrics = payload.get("aggregate", {}).get("mean", {})
        factors = payload.get("factors", {})
        rows.append(
            {
                "case": case_name,
                "corruption": factors.get("corruption"),
                "point_budget": factors.get("point_budget"),
                "graph": factors.get("graph"),
                "constrained": factors.get("constrained"),
                "adaptation": factors.get("adaptation"),
                "object_f1": _get(metrics, "semantic_quality.objects.f1"),
                "count_exact": _get(metrics, "semantic_quality.object_counts.exact_match"),
                "relation_f1": _get(metrics, "semantic_quality.relations.f1"),
                "scene_accuracy": _get(metrics, "semantic_quality.scene_type.accuracy"),
                "semantic_macro_f1": _get(metrics, "semantic_quality.semantic_macro_f1"),
                "json_validity": _get(metrics, "json_validity.validity_rate"),
                "latency_ms_per_sample": _get(metrics, "efficiency.latency_ms_per_sample"),
                "compression_ratio": _get(metrics, "compression.compression_ratio"),
            }
        )
    return rows


def claim_interpretations(record: dict[str, Any]) -> list[str]:
    """Generate short textual interpretations for paper drafts."""

    rows = main_matrix_table(record)
    interpretations = []
    if not rows:
        return interpretations
    clean = _best(rows, corruption="clean")
    severe = _best(rows, corruption="severe_corruption")
    if clean and severe:
        drop = clean["semantic_macro_f1"] - severe["semantic_macro_f1"]
        interpretations.append(
            f"Severe corruption changes semantic macro score by {drop:.3f} relative to the best clean setting in this matrix."
        )
    graph = _best(rows, graph="simple_graph", corruption="severe_corruption")
    no_graph = _best(rows, graph="no_graph", corruption="severe_corruption")
    if graph and no_graph:
        interpretations.append(
            f"Under severe corruption, the graph bottleneck changes semantic macro score by {graph['semantic_macro_f1'] - no_graph['semantic_macro_f1']:.3f} versus no graph."
        )
    constrained = _best(rows, constrained=True)
    unconstrained = _best(rows, constrained=False)
    if constrained and unconstrained:
        interpretations.append(
            f"Schema-constrained decoding changes JSON validity by {constrained['json_validity'] - unconstrained['json_validity']:.3f} versus unconstrained decoding."
        )
    compressed = _best(rows, point_budget="compressed")
    raw = _best(rows, point_budget="raw")
    if compressed and raw:
        interpretations.append(
            f"Compressed input keeps object F1 within {raw['object_f1'] - compressed['object_f1']:.3f} of raw input, while relation F1 changes by {compressed['relation_f1'] - raw['relation_f1']:.3f}."
        )
    return interpretations


def save_claim_report(record: dict[str, Any], output_dir: str | Path) -> None:
    """Save claim-oriented tables and a short Markdown interpretation report."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = main_matrix_table(record)
    save_table_csv(rows, out / "main_matrix.csv")
    save_markdown_table(rows, out / "main_matrix.md")
    interpretations = claim_interpretations(record)
    (out / "claim_interpretations.json").write_text(json.dumps(interpretations, indent=2), encoding="utf-8")
    lines = ["# Claim-Oriented Interpretation", ""]
    lines.extend(f"- {item}" for item in interpretations)
    (out / "claim_interpretations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _best(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if all(row.get(key) == value for key, value in filters.items())
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["semantic_macro_f1"])
