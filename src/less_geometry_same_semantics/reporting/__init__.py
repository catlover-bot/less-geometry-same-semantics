"""Paper table and plotting utilities."""

from less_geometry_same_semantics.reporting.tables import (
    compression_latency_semantic_table,
    clean_vs_corrupted_table,
    corruption_family_breakdown_table,
    graph_ablation_table,
    save_markdown_table,
    save_table_csv,
    severity_semantic_metrics_table,
)
from less_geometry_same_semantics.reporting.claims import main_matrix_table, save_claim_report
from less_geometry_same_semantics.reporting.main_figures import save_main_figures
from less_geometry_same_semantics.reporting.main_tables import save_main_tables

__all__ = [
    "compression_latency_semantic_table",
    "clean_vs_corrupted_table",
    "corruption_family_breakdown_table",
    "graph_ablation_table",
    "main_matrix_table",
    "save_claim_report",
    "save_main_figures",
    "save_main_tables",
    "save_markdown_table",
    "save_table_csv",
    "severity_semantic_metrics_table",
]
