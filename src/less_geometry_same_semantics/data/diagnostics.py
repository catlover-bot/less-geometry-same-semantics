"""Dataset diagnostics for paper-ready public benchmark summaries."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


def dataset_diagnostics(dataset: Dataset, max_scenes: int | None = None) -> dict[str, Any]:
    """Summarize a dataset split without requiring a model forward pass."""

    limit = len(dataset) if max_scenes is None else min(len(dataset), max_scenes)
    object_hist: Counter[str] = Counter()
    relation_hist: Counter[str] = Counter()
    point_counts = []
    object_counts = []
    relation_counts = []
    for index in range(limit):
        sample = dataset[index]
        target = sample["target"]
        point_counts.append(int(sample["metadata"].get("clean_num_points", sample["points"].shape[0])))
        object_counts.append(len(target.get("objects", [])))
        relation_counts.append(len(target.get("relations", [])))
        object_hist.update(item["category"] for item in target.get("objects", []) if isinstance(item, dict))
        relation_hist.update(rel["predicate"] for rel in target.get("relations", []) if isinstance(rel, dict))

    preprocessing = getattr(dataset, "preprocessing_summary", {})
    return {
        "dataset": getattr(dataset, "dataset_name", dataset.__class__.__name__),
        "split": getattr(dataset, "split", "unknown"),
        "split_size": len(dataset),
        "diagnosed_scenes": limit,
        "invalid_or_skipped": preprocessing.get("scenes_skipped", 0),
        "average_points_per_scene": _mean(point_counts),
        "average_objects_per_scene": _mean(object_counts),
        "average_relations_per_scene": _mean(relation_counts),
        "object_category_histogram": dict(sorted(object_hist.items())),
        "relation_category_histogram": dict(sorted(relation_hist.items())),
        "preprocessing_summary": preprocessing,
    }


def save_diagnostics_artifacts(summary: dict[str, Any], output_dir: str | Path) -> None:
    """Save diagnostics as JSON, CSV, Markdown, and plots."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _save_summary_csv(summary, out / "summary.csv")
    _save_hist_csv(summary["object_category_histogram"], out / "object_category_histogram.csv", "object_category")
    _save_hist_csv(summary["relation_category_histogram"], out / "relation_category_histogram.csv", "relation")
    _save_markdown(summary, out / "diagnostics.md")
    _plot_hist(summary["object_category_histogram"], out / "object_category_histogram.png", "Object Category Histogram")
    _plot_hist(summary["relation_category_histogram"], out / "relation_category_histogram.png", "Relation Category Histogram")


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _save_summary_csv(summary: dict[str, Any], path: Path) -> None:
    rows = [
        {"metric": "split_size", "value": summary["split_size"]},
        {"metric": "invalid_or_skipped", "value": summary["invalid_or_skipped"]},
        {"metric": "average_points_per_scene", "value": summary["average_points_per_scene"]},
        {"metric": "average_objects_per_scene", "value": summary["average_objects_per_scene"]},
        {"metric": "average_relations_per_scene", "value": summary["average_relations_per_scene"]},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def _save_hist_csv(hist: dict[str, int], path: Path, label_name: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[label_name, "count"])
        writer.writeheader()
        for key, value in hist.items():
            writer.writerow({label_name: key, "count": value})


def _save_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Dataset Diagnostics: {summary['dataset']} ({summary['split']})",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Split size | {summary['split_size']} |",
        f"| Invalid/skipped | {summary['invalid_or_skipped']} |",
        f"| Avg points/scene | {summary['average_points_per_scene']:.2f} |",
        f"| Avg objects/scene | {summary['average_objects_per_scene']:.2f} |",
        f"| Avg relations/scene | {summary['average_relations_per_scene']:.2f} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_hist(hist: dict[str, int], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = list(hist)
    values = [hist[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
