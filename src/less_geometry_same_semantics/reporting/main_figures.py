"""Dedicated main paper figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from less_geometry_same_semantics.reporting.claims import main_matrix_table
from less_geometry_same_semantics.reporting.tables import severity_semantic_metrics_table


def _setup_matplotlib() -> Any:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )
    return plt


def save_main_figures(
    *,
    main_record: dict[str, Any],
    severity_record: dict[str, Any] | None,
    output_dir: str | Path,
) -> list[Path]:
    """Generate the exact main paper figures."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    if severity_record is not None:
        paths.append(plot_severity_metrics(severity_record, out / "severity_metrics.png"))
    paths.append(plot_compression_latency_semantics(main_record, out / "compression_latency_semantics.png"))
    paths.append(plot_graph_vs_no_graph_severe(main_record, out / "graph_vs_no_graph_severe.png"))
    if _has_adaptation(main_record):
        paths.append(plot_adaptation_comparison(main_record, out / "adaptation_comparison.png"))
    return paths


def plot_severity_metrics(record: dict[str, Any], path: str | Path) -> Path:
    """Severity vs object F1 / relation F1 / JSON validity."""

    plt = _setup_matplotlib()
    rows = severity_semantic_metrics_table(record)
    x = list(range(len(rows)))
    labels = [row["preset"].replace("_", "\n") for row in rows]
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(x, [row["object_f1"] for row in rows], marker="o", label="Object F1")
    ax.plot(x, [row["relation_f1"] for row in rows], marker="s", label="Relation F1")
    ax.plot(x, [row["json_validity"] for row in rows], marker="^", label="JSON validity")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Semantic robustness across degradation severity")
    ax.legend()
    return _save(fig, path)


def plot_compression_latency_semantics(record: dict[str, Any], path: str | Path) -> Path:
    """Compression ratio vs latency vs semantic quality."""

    plt = _setup_matplotlib()
    rows = main_matrix_table(record)
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = [row["compression_ratio"] for row in rows]
    y = [row["semantic_macro_f1"] for row in rows]
    sizes = [35.0 + max(1.0, row["latency_ms_per_sample"]) * 6.0 for row in rows]
    colors = ["#2f6fbb" if row["point_budget"] == "compressed" else "#c55a11" for row in rows]
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.72)
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Semantic macro score")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Compression, latency, and semantic fidelity")
    return _save(fig, path)


def plot_graph_vs_no_graph_severe(record: dict[str, Any], path: str | Path) -> Path:
    """Graph vs no-graph under severe corruption."""

    plt = _setup_matplotlib()
    rows = [
        row for row in main_matrix_table(record)
        if row["corruption"] == "severe_corruption" and row["graph"] in {"no_graph", "simple_graph"}
    ]
    best_by_graph = {}
    for row in rows:
        key = row["graph"]
        if key not in best_by_graph or row["semantic_macro_f1"] > best_by_graph[key]["semantic_macro_f1"]:
            best_by_graph[key] = row
    labels = list(best_by_graph)
    metrics = ["object_f1", "count_exact", "relation_f1", "scene_accuracy"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    if not labels:
        ax.text(0.5, 0.5, "No severe graph comparison available", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, path)
    x = list(range(len(metrics)))
    width = 0.35
    for idx, label in enumerate(labels):
        row = best_by_graph[label]
        offset = (idx - (len(labels) - 1) / 2) * width
        ax.bar([value + offset for value in x], [row[metric] for metric in metrics], width=width, label=label)
    ax.set_xticks(x, [metric.replace("_", "\n") for metric in metrics])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Graph bottleneck under severe corruption")
    if labels:
        ax.legend()
    return _save(fig, path)


def plot_adaptation_comparison(record: dict[str, Any], path: str | Path) -> Path:
    """Optional adaptation comparison if runs include adaptation factors."""

    plt = _setup_matplotlib()
    rows = [row for row in main_matrix_table(record) if row["corruption"] == "severe_corruption"]
    best_by_adaptation = {}
    for row in rows:
        key = row["adaptation"]
        if key not in best_by_adaptation or row["semantic_macro_f1"] > best_by_adaptation[key]["semantic_macro_f1"]:
            best_by_adaptation[key] = row
    labels = list(best_by_adaptation)
    values = [best_by_adaptation[label]["semantic_macro_f1"] for label in labels]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.bar(labels, values, color=["#777777", "#2f6fbb"][: len(labels)])
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Semantic macro score")
    ax.set_title("Input adaptation under severe corruption")
    return _save(fig, path)


def _has_adaptation(record: dict[str, Any]) -> bool:
    return len({row["adaptation"] for row in main_matrix_table(record)}) > 1


def _save(fig: Any, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    return output_path
