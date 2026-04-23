"""Generate CSV/Markdown tables and plots from a saved experiment JSON record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.reporting.plots import (
    plot_family_degradation,
    plot_pareto_curve,
    plot_robustness_curve,
)
from less_geometry_same_semantics.reporting.tables import (
    compression_latency_semantic_table,
    clean_vs_corrupted_table,
    corruption_family_breakdown_table,
    graph_ablation_table,
    save_markdown_table,
    save_table_csv,
    severity_semantic_metrics_table,
)
from less_geometry_same_semantics.reporting.claims import save_claim_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", help="Path to a benchmark or ablation JSON record.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_path = Path(args.record)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir) if args.output_dir else record_path.with_suffix("")

    severity_rows = severity_semantic_metrics_table(record)
    if severity_rows:
        clean_rows = clean_vs_corrupted_table(record)
        save_table_csv(clean_rows, output_dir / "clean_vs_corrupted.csv")
        save_markdown_table(clean_rows, output_dir / "clean_vs_corrupted.md")
        save_table_csv(severity_rows, output_dir / "severity_semantic_metrics.csv")
        save_markdown_table(severity_rows, output_dir / "severity_semantic_metrics.md")
        plot_robustness_curve(record, output_dir / "robustness_curve.png")

    compression_rows = compression_latency_semantic_table(record)
    if compression_rows:
        save_table_csv(compression_rows, output_dir / "compression_latency_semantics.csv")
        save_markdown_table(compression_rows, output_dir / "compression_latency_semantics.md")
        plot_pareto_curve(record, output_dir / "compression_semantics_latency_pareto.png")

    family_rows = corruption_family_breakdown_table(record)
    if family_rows:
        save_table_csv(family_rows, output_dir / "corruption_family_breakdown.csv")
        save_markdown_table(family_rows, output_dir / "corruption_family_breakdown.md")
        plot_family_degradation(record, output_dir / "corruption_family_degradation.png")

    graph_rows = graph_ablation_table(record)
    if graph_rows:
        save_table_csv(graph_rows, output_dir / "graph_ablation.csv")
        save_markdown_table(graph_rows, output_dir / "graph_ablation.md")

    if "main_matrix" in record.get("metrics", {}):
        save_claim_report(record, output_dir / "claims")


if __name__ == "__main__":
    main()
