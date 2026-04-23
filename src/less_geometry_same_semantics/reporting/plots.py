"""Publication-oriented plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from less_geometry_same_semantics.reporting.tables import (
    compression_latency_semantic_table,
    corruption_family_breakdown_table,
    severity_semantic_metrics_table,
)


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
        }
    )
    return plt


def plot_robustness_curve(record: dict[str, Any], path: str | Path) -> Path:
    """Plot semantic robustness as severity increases."""

    plt = _setup_matplotlib()
    rows = severity_semantic_metrics_table(record)
    x = list(range(len(rows)))
    y = [row["semantic_macro_f1"] for row in rows]
    labels = [row["preset"].replace("_", "\n") for row in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Semantic macro score")
    ax.set_xlabel("Corruption severity preset")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Robustness under point-cloud degradation")
    return _save(fig, path)


def plot_pareto_curve(record: dict[str, Any], path: str | Path) -> Path:
    """Plot compression ratio vs semantic fidelity with latency as marker size."""

    plt = _setup_matplotlib()
    rows = compression_latency_semantic_table(record)
    x = [row["compression_ratio"] for row in rows]
    y = [row["semantic_macro_f1"] for row in rows]
    latency = [max(1.0, row["latency_ms_per_sample"]) for row in rows]
    sizes = [30.0 + value * 8.0 for value in latency]
    labels = [str(row["setting"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.scatter(x, y, s=sizes, alpha=0.75)
    for xi, yi, label in zip(x, y, labels, strict=True):
        ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Compression ratio (higher is smaller input)")
    ax.set_ylabel("Semantic macro score")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Compression/semantics/latency Pareto view")
    return _save(fig, path)


def plot_family_degradation(record: dict[str, Any], path: str | Path) -> Path:
    """Plot degradation patterns for corruption-family ablations."""

    plt = _setup_matplotlib()
    rows = corruption_family_breakdown_table(record)
    labels = [row["family"].replace("_", "\n") for row in rows]
    object_scores = [row["object_f1"] for row in rows]
    relation_scores = [row["relation_f1"] for row in rows]
    x = list(range(len(rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar([value - width / 2 for value in x], object_scores, width=width, label="Object F1")
    ax.bar([value + width / 2 for value in x], relation_scores, width=width, label="Relation F1")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Per-family semantic degradation")
    ax.legend(frameon=False)
    return _save(fig, path)


def _save(fig: Any, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    return output_path
