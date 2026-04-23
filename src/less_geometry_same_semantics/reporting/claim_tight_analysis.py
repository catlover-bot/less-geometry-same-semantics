"""Claim-tight paper analysis for completed ARKitScenes runs."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from less_geometry_same_semantics.data.constants import OBJECT_CATEGORIES
from less_geometry_same_semantics.metrics.relations import relation_tuple
from less_geometry_same_semantics.metrics.semantic import object_categories, object_count_map
from less_geometry_same_semantics.reporting.claims import main_matrix_table
from less_geometry_same_semantics.reporting.tables import save_markdown_table, save_table_csv


SELECTED_PRESETS = ["clean", "severe_corruption", "extreme_compression"]


def save_claim_tight_analysis(
    main_record: dict[str, Any],
    severity_record: dict[str, Any],
    output_dir: str | Path,
    ablation_record: dict[str, Any] | None = None,
    frozen_claims: dict[str, Any] | None = None,
    diagnostics_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate paper-facing analysis grounded in completed result records."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    severity_rows = _severity_table(severity_record)
    seed_rows = _seed_rows(severity_record)
    seed_summary_rows = _seed_summary_rows(seed_rows)
    relation_rows = _relation_fragility_rows(severity_rows)
    efficiency_rows = _efficiency_rows(main_record)
    failure_rows = _failure_example_rows([("severity", severity_record), ("main", main_record)])
    class_rows = _per_class_reference_rows(diagnostics_dir) if diagnostics_dir else []
    representative_class_rows = _representative_failure_class_rows(failure_rows)

    _save_table_bundle(relation_rows, out / "relation_fragility", "relation_fragility")
    _save_table_bundle(seed_rows, out / "object_f1_anomaly", "seedwise_metrics")
    _save_table_bundle(seed_summary_rows, out / "object_f1_anomaly", "seed_variance_summary")
    _save_table_bundle(class_rows, out / "object_f1_anomaly", "reference_class_distribution")
    _save_table_bundle(representative_class_rows, out / "object_f1_anomaly", "representative_failure_class_counts")
    _save_table_bundle(failure_rows, out / "object_f1_anomaly", "representative_failure_examples")
    _save_table_bundle(efficiency_rows, out / "efficiency", "lightweight_efficiency")

    claim_status = _claim_status(frozen_claims or {}, severity_rows, main_record)
    _save_claim_status(claim_status, out / "claim_status")

    object_anomaly = _object_f1_anomaly_summary(severity_rows, seed_rows, class_rows, failure_rows)
    (out / "object_f1_anomaly").mkdir(parents=True, exist_ok=True)
    (out / "object_f1_anomaly" / "object_f1_anomaly.md").write_text(object_anomaly, encoding="utf-8")
    (out / "object_f1_anomaly" / "per_scene_breakdown.md").write_text(
        _per_scene_breakdown_note(failure_rows),
        encoding="utf-8",
    )

    relation_summary = _relation_fragility_summary(relation_rows)
    (out / "relation_fragility" / "relation_fragility_summary.md").write_text(relation_summary, encoding="utf-8")

    efficiency_summary = _efficiency_summary(efficiency_rows)
    (out / "efficiency" / "lightweight_efficiency_summary.md").write_text(efficiency_summary, encoding="utf-8")

    _save_plots(out, relation_rows, efficiency_rows, seed_rows, class_rows)
    _save_paper_draft(out / "paper_draft", severity_rows, relation_rows, efficiency_rows, claim_status)
    _save_readme(out, claim_status)

    summary = {
        "output_dir": str(out),
        "strong_claim_count": len(claim_status["strongly_supported"]),
        "weak_claim_count": len(claim_status["weakly_supported"]),
        "unsupported_claim_count": len(claim_status["unsupported"]),
        "negative_result_count": len(claim_status["negative_results"]),
        "full_per_scene_predictions_available": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save_table_bundle(rows: list[dict[str, Any]], directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_table_csv(rows, directory / f"{stem}.csv")
    save_markdown_table(rows, directory / f"{stem}.md")


def _severity_table(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    groups = record.get("metrics", {}).get("presets", {})
    for preset, payload in groups.items():
        metrics = payload.get("aggregate", {}).get("mean", {})
        rows.append(
            {
                "preset": preset,
                "object_f1": _get(metrics, "semantic_quality.objects.f1"),
                "count_exact": _get(metrics, "semantic_quality.object_counts.exact_match"),
                "relation_f1": _get(metrics, "semantic_quality.relations.f1"),
                "scene_accuracy": _get(metrics, "semantic_quality.scene_type.accuracy"),
                "semantic_macro_f1": _get(metrics, "semantic_quality.semantic_macro_f1"),
                "json_validity": _get(metrics, "json_validity.validity_rate"),
            }
        )
    order = {name: index for index, name in enumerate(["clean", "mild_corruption", "medium_corruption", "severe_corruption", "extreme_compression"])}
    return sorted(rows, key=lambda row: order.get(str(row["preset"]), 99))


def _seed_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for preset, payload in record.get("metrics", {}).get("presets", {}).items():
        for run in payload.get("runs", []):
            metrics = run.get("metrics", {})
            rows.append(
                {
                    "preset": preset,
                    "seed": run.get("seed"),
                    "object_f1": _get(metrics, "semantic_quality.objects.f1"),
                    "count_exact": _get(metrics, "semantic_quality.object_counts.exact_match"),
                    "relation_f1": _get(metrics, "semantic_quality.relations.f1"),
                    "scene_accuracy": _get(metrics, "semantic_quality.scene_type.accuracy"),
                    "semantic_macro_f1": _get(metrics, "semantic_quality.semantic_macro_f1"),
                    "json_validity": _get(metrics, "json_validity.validity_rate"),
                }
            )
    return rows


def _seed_summary_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_preset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        by_preset[str(row["preset"])].append(row)
    for preset, rows_for_preset in by_preset.items():
        output = {"preset": preset, "num_seeds": len(rows_for_preset)}
        for metric in ["object_f1", "relation_f1", "count_exact", "scene_accuracy", "semantic_macro_f1", "json_validity"]:
            values = [float(row[metric]) for row in rows_for_preset]
            output[f"{metric}_mean"] = statistics.mean(values) if values else 0.0
            output[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        rows.append(output)
    order = {name: index for index, name in enumerate(["clean", "mild_corruption", "medium_corruption", "severe_corruption", "extreme_compression"])}
    return sorted(rows, key=lambda row: order.get(str(row["preset"]), 99))


def _relation_fragility_rows(severity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in severity_rows if row["preset"] in SELECTED_PRESETS]
    clean = next((row for row in selected if row["preset"] == "clean"), None)
    clean_object = float(clean["object_f1"]) if clean else 0.0
    clean_relation = float(clean["relation_f1"]) if clean else 0.0
    rows = []
    for row in selected:
        object_f1 = float(row["object_f1"])
        relation_f1 = float(row["relation_f1"])
        rows.append(
            {
                "preset": row["preset"],
                "object_f1": object_f1,
                "relation_f1": relation_f1,
                "count_exact": row["count_exact"],
                "scene_accuracy": row["scene_accuracy"],
                "semantic_macro_f1": row["semantic_macro_f1"],
                "json_validity": row["json_validity"],
                "object_retention_vs_clean": _ratio(object_f1, clean_object),
                "relation_retention_vs_clean": _ratio(relation_f1, clean_relation),
                "object_relation_gap": object_f1 - relation_f1,
            }
        )
    return rows


def _efficiency_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case_name, payload in record.get("metrics", {}).get("main_matrix", {}).items():
        metrics = payload.get("aggregate", {}).get("mean", {})
        factors = payload.get("factors", {})
        rows.append(
            {
                "condition": case_name,
                "corruption": factors.get("corruption"),
                "budget": factors.get("point_budget"),
                "graph": factors.get("graph"),
                "constrained": factors.get("constrained"),
                "adaptation": factors.get("adaptation"),
                "parameter_count": _get(metrics, "efficiency.parameter_count"),
                "compressed_token_budget": _get(metrics, "efficiency.compressed_token_budget"),
                "latency_ms_per_sample": _get(metrics, "efficiency.latency_ms_per_sample"),
                "input_tensor_memory_mb": _get(metrics, "efficiency.input_tensor_memory_mb"),
                "process_memory_mb": _get(metrics, "efficiency.process_memory_mb"),
                "compression_ratio": _get(metrics, "compression.compression_ratio"),
                "object_f1": _get(metrics, "semantic_quality.objects.f1"),
                "relation_f1": _get(metrics, "semantic_quality.relations.f1"),
                "semantic_macro_f1": _get(metrics, "semantic_quality.semantic_macro_f1"),
                "json_validity": _get(metrics, "json_validity.validity_rate"),
            }
        )
    return rows


def _per_class_reference_rows(diagnostics_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(diagnostics_dir)
    rows = []
    for split in ["train", "val"]:
        path = root / split / "object_category_histogram.csv"
        if not path.exists():
            continue
        total = 0
        counts: dict[str, int] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                count = int(row.get("count", 0))
                counts[str(row.get("object_category", ""))] = count
                total += count
        for category, count in sorted(counts.items()):
            rows.append(
                {
                    "split": split,
                    "category": category,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                }
            )
    return rows


def _failure_example_rows(records: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for record_name, record in records:
        for group_name, group_payload in _iter_record_groups(record):
            for run in group_payload.get("runs", []):
                seed = run.get("seed")
                failures = run.get("failure_analysis", {})
                for bucket, examples in failures.items():
                    for example in examples:
                        prediction = example.get("prediction", {})
                        reference = example.get("reference", {})
                        pred_objects = set(object_categories(prediction))
                        ref_objects = set(object_categories(reference))
                        pred_relations = {relation_tuple(item) for item in prediction.get("relations", [])}
                        ref_relations = {relation_tuple(item) for item in reference.get("relations", [])}
                        rows.append(
                            {
                                "record": record_name,
                                "group": group_name,
                                "seed": seed,
                                "bucket": bucket,
                                "example_index": example.get("index"),
                                "preset": example.get("preset"),
                                "object_f1": _set_f1(pred_objects, ref_objects),
                                "relation_f1": _set_f1(pred_relations, ref_relations),
                                "pred_objects": " ".join(sorted(pred_objects)),
                                "ref_objects": " ".join(sorted(ref_objects)),
                                "pred_object_count_total": sum(object_count_map(prediction).values()),
                                "ref_object_count_total": sum(object_count_map(reference).values()),
                                "pred_relation_count": len(pred_relations),
                                "ref_relation_count": len(ref_relations),
                                "pred_scene_type": prediction.get("scene_type"),
                                "ref_scene_type": reference.get("scene_type"),
                            }
                        )
    return rows


def _representative_failure_class_rows(failure_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in failure_rows:
        pred = set(str(row.get("pred_objects", "")).split())
        ref = set(str(row.get("ref_objects", "")).split())
        for category in OBJECT_CATEGORIES:
            if category in pred:
                counts[category]["pred_present"] += 1
            if category in ref:
                counts[category]["ref_present"] += 1
            if category in pred and category in ref:
                counts[category]["tp"] += 1
            if category in pred and category not in ref:
                counts[category]["fp"] += 1
            if category in ref and category not in pred:
                counts[category]["fn"] += 1
    rows = []
    for category in OBJECT_CATEGORIES:
        item = counts[category]
        rows.append(
            {
                "scope": "representative_failure_examples_only",
                "category": category,
                "pred_present": item["pred_present"],
                "ref_present": item["ref_present"],
                "tp": item["tp"],
                "fp": item["fp"],
                "fn": item["fn"],
                "precision": _safe_div(item["tp"], item["tp"] + item["fp"]),
                "recall": _safe_div(item["tp"], item["tp"] + item["fn"]),
                "f1": _f1(item["tp"], item["fp"], item["fn"]),
            }
        )
    return rows


def _claim_status(
    frozen_claims: dict[str, Any],
    severity_rows: list[dict[str, Any]],
    main_record: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    strong = []
    weak = []
    unsupported = []
    negative = []

    for item in frozen_claims.get("supported_claims", []):
        strong.append({"claim": item.get("claim"), "evidence": item.get("evidence", {})})
    for item in frozen_claims.get("unsupported_or_unfrozen_claims", []):
        unsupported.append({"claim": item.get("claim"), "evidence": item.get("evidence", {})})

    validity = {str(row["preset"]): row["json_validity"] for row in severity_rows}
    if validity and all(float(value) == 1.0 for value in validity.values()):
        strong.append(
            {
                "claim": "structured JSON outputs remain valid across the completed severity sweep",
                "evidence": {"json_validity_by_preset": validity},
            }
        )

    clean = _row_by_preset(severity_rows, "clean")
    severe = _row_by_preset(severity_rows, "severe_corruption")
    extreme = _row_by_preset(severity_rows, "extreme_compression")
    if clean and severe and extreme:
        weak.append(
            {
                "claim": "object category F1 is stable under severe and extreme degradation in this subset",
                "evidence": {
                    "clean_object_f1": clean["object_f1"],
                    "severe_object_f1": severe["object_f1"],
                    "extreme_object_f1": extreme["object_f1"],
                    "caution": "Object F1 is category-set based and does not measure instance counts.",
                },
            }
        )

    rows = main_matrix_table(main_record)
    if rows:
        best = max(rows, key=lambda row: row["semantic_macro_f1"])
        weak.append(
            {
                "claim": "the baseline is lightweight enough for CPU-scale experimentation",
                "evidence": {
                    "best_condition": best["case"],
                    "latency_ms_per_sample": best["latency_ms_per_sample"],
                    "compression_ratio": best["compression_ratio"],
                    "caution": "Latency is local CPU timing and should be reported with hardware context.",
                },
            }
        )

    if clean:
        relation_gap = float(clean["object_f1"]) - float(clean["relation_f1"])
        negative.append(
            {
                "claim": "coarse relation prediction is much weaker than object category prediction",
                "evidence": {
                    "clean_object_f1": clean["object_f1"],
                    "clean_relation_f1": clean["relation_f1"],
                    "clean_gap": relation_gap,
                },
            }
        )
    if clean and float(clean["count_exact"]) == 0.0:
        negative.append(
            {
                "claim": "object count exact match is not solved by the current baseline",
                "evidence": {"clean_count_exact": clean["count_exact"]},
            }
        )

    return {
        "strongly_supported": strong,
        "weakly_supported": weak,
        "unsupported": unsupported,
        "negative_results": negative,
    }


def _save_claim_status(status: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "claim_status.json").write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Claim-Tight Status", ""]
    labels = [
        ("strongly_supported", "Strongly Supported"),
        ("weakly_supported", "Weakly Supported"),
        ("unsupported", "Unsupported"),
        ("negative_results", "Negative Results To Report"),
    ]
    for key, label in labels:
        lines.extend([f"## {label}", ""])
        if not status[key]:
            lines.extend(["- None.", ""])
            continue
        for item in status[key]:
            lines.append(f"- {item['claim']}")
            if item.get("evidence"):
                lines.append(f"  - Evidence: `{json.dumps(item['evidence'], sort_keys=True)}`")
        lines.append("")
    (output_dir / "claim_status.md").write_text("\n".join(lines), encoding="utf-8")


def _object_f1_anomaly_summary(
    severity_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> str:
    clean = _row_by_preset(severity_rows, "clean")
    severe = _row_by_preset(severity_rows, "severe_corruption")
    extreme = _row_by_preset(severity_rows, "extreme_compression")
    lines = [
        "# Object-F1 Stability Check",
        "",
        "The expanded ARKitScenes run shows object category F1 at severe/extreme settings similar to or slightly above clean.",
        "This should be treated as an observed metric behavior, not as a broad robustness claim by itself.",
        "",
    ]
    if clean and severe and extreme:
        lines.extend(
            [
                "## Aggregate Pattern",
                "",
                f"- Clean object F1: {float(clean['object_f1']):.4f}",
                f"- Severe object F1: {float(severe['object_f1']):.4f}",
                f"- Extreme-compression object F1: {float(extreme['object_f1']):.4f}",
                f"- Severe minus clean object F1: {float(severe['object_f1']) - float(clean['object_f1']):.4f}",
                f"- Extreme minus clean object F1: {float(extreme['object_f1']) - float(clean['object_f1']):.4f}",
                "",
            ]
        )
    selected = [row for row in seed_rows if row["preset"] in SELECTED_PRESETS]
    if selected:
        lines.extend(["## Seed-Wise Variance", ""])
        for preset in SELECTED_PRESETS:
            values = [float(row["object_f1"]) for row in selected if row["preset"] == preset]
            if values:
                lines.append(f"- {preset}: mean={statistics.mean(values):.4f}, std={statistics.pstdev(values) if len(values) > 1 else 0.0:.4f}, values={_format_values(values)}")
        lines.append("")
    if class_rows:
        val_classes = [row for row in class_rows if row["split"] == "val"]
        lines.extend(
            [
                "## Class-Distribution Context",
                "",
                "The validation subset has a small, coarse label vocabulary. Category-set F1 can remain stable when common categories are predicted consistently.",
            ]
        )
        for row in val_classes:
            lines.append(f"- {row['category']}: {row['count']} validation references ({float(row['fraction']):.3f})")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- The object metric is category-set F1. It checks whether each object category appears, not whether all instances are counted correctly.",
            "- Count exact match is 0.0 in the selected severity table, so stable object F1 does not imply stable object counting.",
            "- Relation F1 stays near zero, so the result supports coarse object-category retention more than structured relation recovery.",
            "- The current saved records keep compact failure examples, not complete per-scene prediction/reference traces. A complete per-scene and prediction-side per-class audit requires rerunning with full example retention.",
            "- Because only 10 validation scenes were used, the object-F1 improvement over clean could still be subset size, metric behavior, or decoder prior rather than true corruption-induced improvement.",
            "",
            "## Available Per-Scene Evidence",
            "",
            f"- Representative compact failure/example rows available: {len(failure_rows)}.",
            "- Complete per-scene metrics available from current artifacts: no.",
            "- Complete prediction-side per-class metrics available from current artifacts: no.",
        ]
    )
    return "\n".join(lines) + "\n"


def _per_scene_breakdown_note(failure_rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Per-Scene Breakdown Availability",
            "",
            "Complete per-scene prediction/reference traces are not present in the current saved experiment records.",
            "The available records contain aggregate metrics by seed plus compact representative failure examples.",
            "",
            "What is available now:",
            "",
            f"- Representative compact examples: {len(failure_rows)} rows.",
            "- Representative example table: `representative_failure_examples.md` and `.csv`.",
            "- Example indices are retained, but full scene IDs and all non-failure examples are not retained in compact failure reports.",
            "",
            "What this means for the object-F1 anomaly:",
            "",
            "- Seed-wise aggregate variance can be analyzed from current records.",
            "- Reference-side class distribution can be analyzed from diagnostics.",
            "- Complete per-scene and prediction-side per-class metrics cannot be computed without rerunning evaluation with full example retention.",
            "",
            "Claim guidance: do not claim the object-F1 increase is explained at the per-scene level from these artifacts alone.",
            "",
        ]
    )


def _relation_fragility_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["# Relation-Fragility Analysis", ""]
    if not rows:
        return "\n".join(lines + ["No rows available.", ""])
    clean = _row_by_preset(rows, "clean")
    severe = _row_by_preset(rows, "severe_corruption")
    extreme = _row_by_preset(rows, "extreme_compression")
    for row in rows:
        lines.append(
            f"- {row['preset']}: object F1={float(row['object_f1']):.4f}, relation F1={float(row['relation_f1']):.4f}, "
            f"count exact={float(row['count_exact']):.4f}, scene accuracy={float(row['scene_accuracy']):.4f}."
        )
    lines.extend(["", "## Claim-Tight Reading", ""])
    if clean and severe and extreme:
        lines.append(
            "Relation semantics are fragile in absolute terms: relation F1 is far below object F1 for clean, severe, and extreme settings."
        )
        lines.append(
            "However, the data do not show a clean monotonic relation drop under degradation because relation F1 is already very low on clean input."
        )
        lines.append(
            "A safe statement is: coarse object categories are retained much better than relation triples; relation recovery remains weak across the whole severity sweep."
        )
    return "\n".join(lines) + "\n"


def _efficiency_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["# Lightweight Efficiency Summary", ""]
    if not rows:
        return "\n".join(lines + ["No efficiency rows available.", ""])
    param_values = sorted({int(row["parameter_count"]) for row in rows})
    token_values = sorted({int(row["compressed_token_budget"]) for row in rows})
    latencies = [float(row["latency_ms_per_sample"]) for row in rows]
    ratios = [float(row["compression_ratio"]) for row in rows]
    best_macro = max(rows, key=lambda row: float(row["semantic_macro_f1"]))
    fastest = min(rows, key=lambda row: float(row["latency_ms_per_sample"]))
    lines.extend(
        [
            f"- Parameter-count values observed: {', '.join(str(value) for value in param_values)}.",
            f"- Token budgets observed: {', '.join(str(value) for value in token_values)}.",
            f"- Latency range: {min(latencies):.2f} to {max(latencies):.2f} ms/sample on this CPU run.",
            f"- Compression-ratio range: {min(ratios):.2f} to {max(ratios):.2f}.",
            f"- Best macro-F1 condition: {best_macro['condition']} at {float(best_macro['semantic_macro_f1']):.4f} macro F1.",
            f"- Fastest condition: {fastest['condition']} at {float(fastest['latency_ms_per_sample']):.2f} ms/sample.",
            "",
            "These numbers make the paper's lightweight framing concrete, but latency should be reported with hardware and CPU/GPU context.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_plots(
    out: Path,
    relation_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    if relation_rows:
        _plot_relation_fragility(relation_rows, figures / "object_relation_fragility.png")
    if efficiency_rows:
        _plot_efficiency_tradeoff(efficiency_rows, figures / "lightweight_efficiency_tradeoff.png")
    if seed_rows:
        _plot_seed_object_f1(seed_rows, figures / "seedwise_object_f1.png")
    if class_rows:
        _plot_class_distribution(class_rows, figures / "reference_class_distribution.png")


def _plot_relation_fragility(rows: list[dict[str, Any]], output_path: Path) -> None:
    labels = [str(row["preset"]) for row in rows]
    metrics = ["object_f1", "relation_f1", "count_exact", "scene_accuracy"]
    x = range(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, metric in enumerate(metrics):
        offsets = [value + (idx - 1.5) * width for value in x]
        ax.bar(offsets, [float(row[metric]) for row in rows], width=width, label=metric)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Object Semantics vs Relation Fragility")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_efficiency_tradeoff(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for budget, marker in [("raw", "o"), ("compressed", "s")]:
        subset = [row for row in rows if row["budget"] == budget]
        if not subset:
            continue
        sizes = [40.0 + 25.0 * float(row["compression_ratio"]) for row in subset]
        ax.scatter(
            [float(row["latency_ms_per_sample"]) for row in subset],
            [float(row["semantic_macro_f1"]) for row in subset],
            s=sizes,
            alpha=0.75,
            marker=marker,
            label=budget,
        )
    ax.set_xlabel("Latency (ms/sample)")
    ax.set_ylabel("Semantic macro F1")
    ax.set_title("Lightweight Semantic Retention vs Latency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_seed_object_f1(rows: list[dict[str, Any]], output_path: Path) -> None:
    selected = [row for row in rows if row["preset"] in SELECTED_PRESETS]
    by_seed: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_seed[row["seed"]].append(row)
    order = {name: index for index, name in enumerate(SELECTED_PRESETS)}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for seed, seed_rows in sorted(by_seed.items()):
        sorted_rows = sorted(seed_rows, key=lambda row: order.get(str(row["preset"]), 99))
        ax.plot([row["preset"] for row in sorted_rows], [float(row["object_f1"]) for row in sorted_rows], marker="o", label=f"seed {seed}")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Object F1")
    ax.set_title("Seed-wise Object-F1 Stability")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_class_distribution(rows: list[dict[str, Any]], output_path: Path) -> None:
    val_rows = [row for row in rows if row["split"] == "val"]
    if not val_rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([row["category"] for row in val_rows], [int(row["count"]) for row in val_rows])
    ax.set_ylabel("Validation references")
    ax.set_title("ARKitScenes Validation Class Distribution")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _save_paper_draft(
    out: Path,
    severity_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, Any]],
    claim_status: dict[str, list[dict[str, Any]]],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    clean = _row_by_preset(severity_rows, "clean")
    severe = _row_by_preset(severity_rows, "severe_corruption")
    extreme = _row_by_preset(severity_rows, "extreme_compression")
    best_eff = max(efficiency_rows, key=lambda row: float(row["semantic_macro_f1"])) if efficiency_rows else {}

    (out / "title_candidates.md").write_text(
        "\n".join(
            [
                "# Title Candidates",
                "",
                "1. Lightweight 3D Semantic Understanding Under Aggressive Point-Cloud Degradation",
                "2. Less Geometry, Stable Semantics: A Lightweight ARKitScenes Robustness Study",
                "3. Semantic Retention from Degraded Point Clouds with Lightweight Structured Decoding",
                "4. Robust Coarse 3D Semantics Without High-Fidelity Geometry",
                "5. A Lightweight Benchmark Study of Semantic Stability Under Point-Cloud Corruption",
                "",
            ]
        ),
        encoding="utf-8",
    )
    abstract = _abstract_text(clean, severe, extreme, best_eff, claim_status)
    (out / "abstract.md").write_text(abstract, encoding="utf-8")
    (out / "outline.md").write_text(_outline_text(), encoding="utf-8")
    (out / "limitations.md").write_text(_limitations_text(claim_status), encoding="utf-8")
    (out / "negative_results.md").write_text(_negative_results_text(claim_status), encoding="utf-8")
    (out / "future_work.md").write_text(_future_work_text(), encoding="utf-8")


def _abstract_text(
    clean: dict[str, Any] | None,
    severe: dict[str, Any] | None,
    extreme: dict[str, Any] | None,
    best_eff: dict[str, Any],
    claim_status: dict[str, list[dict[str, Any]]],
) -> str:
    clean_macro = float(clean["semantic_macro_f1"]) if clean else 0.0
    severe_macro = float(severe["semantic_macro_f1"]) if severe else 0.0
    extreme_macro = float(extreme["semantic_macro_f1"]) if extreme else 0.0
    retention = _ratio(severe_macro, clean_macro)
    for item in claim_status.get("strongly_supported", []):
        if item.get("claim") == "aggressive degradation preserves a substantial fraction of coarse semantic performance":
            evidence = item.get("evidence", {})
            clean_macro = float(evidence.get("clean_semantic_macro_f1", clean_macro))
            severe_macro = float(evidence.get("severe_semantic_macro_f1", severe_macro))
            retention = float(evidence.get("relative_retention", retention))
            break
    param_count = int(best_eff.get("parameter_count", 0)) if best_eff else 0
    return "\n".join(
        [
            "# Abstract Draft",
            "",
            "We study lightweight 3D scene understanding when point-cloud geometry is aggressively degraded.",
            "Rather than preserving high-fidelity geometry, the system targets task-relevant structured semantics: object categories, coarse counts, attributes, relations, scene type, and a short optional caption.",
            f"On an ARKitScenes 3DOD subset with 11 training scenes and 10 validation scenes, the completed robustness sweep retains {retention:.1%} of clean semantic macro F1 under severe corruption ({severe_macro:.3f} vs {clean_macro:.3f}) and maintains valid structured JSON outputs across all severity presets.",
            f"Extreme compression yields semantic macro F1 {extreme_macro:.3f}.",
            f"The current lightweight baseline uses approximately {param_count:,} parameters in the strongest reported condition.",
            "The results support a cautious claim of coarse semantic retention under degradation, while also revealing clear limitations: relation prediction remains weak, exact object counting is not solved, graph bottlenecks do not improve robustness in the current run, and constrained decoding does not improve validity because unconstrained outputs are already valid.",
            "These findings frame the contribution as a claim-tight robustness and efficiency study, not as evidence that graph bottlenecks are the primary source of robustness.",
            "",
        ]
    )


def _outline_text() -> str:
    return "\n".join(
        [
            "# Paper Outline",
            "",
            "## Introduction",
            "- Motivate degraded point-cloud settings where high-fidelity geometry is unavailable or unnecessary.",
            "- State the paper question: can lightweight models retain coarse task-relevant semantics under aggressive degradation?",
            "- Position graph/constrained-decoding results as secondary ablations, not the main success story.",
            "- Preview the supported claim: coarse semantic macro performance is substantially retained under severe degradation on the completed ARKitScenes subset.",
            "",
            "## Method",
            "- Describe the point-cloud input, corruption/compression presets, and structured semantic target.",
            "- Describe the lightweight encoder, token compression, object abstraction, optional graph bottleneck, and structured decoder at a high level.",
            "- Emphasize that no high-fidelity reconstruction objective is used.",
            "",
            "## Experiments",
            "- Dataset: ARKitScenes 3DOD subset, 11 training scenes and 10 validation scenes.",
            "- Metrics: object F1, count exact match, relation F1, scene accuracy, semantic macro F1, JSON validity, latency, memory, compression ratio, parameter count.",
            "- Main comparisons: clean vs severe vs extreme compression, raw vs compressed, graph/no-graph as secondary, constrained/unconstrained as secondary.",
            "",
            "## Results",
            "- Lead with semantic retention under severity and JSON validity.",
            "- Then present efficiency/latency/parameter-count tradeoffs.",
            "- Then report relation fragility and count failures honestly.",
            "- Close with secondary/null ablations for graph and constrained decoding.",
            "",
        ]
    )


def _limitations_text(claim_status: dict[str, list[dict[str, Any]]]) -> str:
    return "\n".join(
        [
            "# Limitations",
            "",
            "- The completed real-data run uses 11 ARKitScenes training scenes and 10 validation scenes, so claims should be framed as a first public-dataset robustness pass rather than final benchmark-scale evidence.",
            "- ARKitScenes provides 3D object boxes, not human scene-graph triplets. Relation targets are derived by heuristic geometry rules.",
            "- Object F1 is category-set based and does not measure instance-count correctness; count exact match remains weak.",
            "- The saved artifacts do not contain full per-scene prediction/reference traces, so complete per-scene and prediction-side per-class analysis requires one more run with full example retention.",
            "- The current graph bottleneck and constrained-decoding ablations are null or negative; they should not be presented as primary contributions.",
            "",
        ]
    )


def _negative_results_text(claim_status: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# Negative Results Paragraph", ""]
    lines.append(
        "The expanded ARKitScenes run also gives useful negative evidence. The graph bottleneck does not improve severe-corruption semantic macro F1 under the current criterion, with the frozen-claim delta effectively zero. Schema-constrained decoding does not improve JSON validity or semantic stability because validity is already perfect for the unconstrained structured outputs in this run. Relation prediction remains much weaker than object-category prediction across clean, severe, and extreme settings, and exact object-count matching is not solved. We therefore treat graph and constrained decoding as secondary ablations and frame the main result around lightweight coarse semantic retention under degraded input."
    )
    lines.append("")
    return "\n".join(lines)


def _future_work_text() -> str:
    return "\n".join(
        [
            "# Future Work",
            "",
            "- Run a larger ARKitScenes subset with retained full prediction/reference traces for complete per-scene and per-class analysis.",
            "- Improve relation supervision and relation metrics before making stronger claims about structured scene-graph recovery.",
            "- Add calibrated count modeling within the existing lightweight family.",
            "- Report hardware-normalized latency and memory measurements on both CPU and a single GPU.",
            "",
        ]
    )


def _save_readme(out: Path, claim_status: dict[str, list[dict[str, Any]]]) -> None:
    lines = [
        "# Claim-Tight Analysis Package",
        "",
        "This package recenters the current paper pass around lightweight robustness and semantic retention.",
        "It is generated only from completed ARKitScenes outputs and does not add model families.",
        "",
        "Inspect in this order:",
        "",
        "1. `claim_status/claim_status.md`",
        "2. `relation_fragility/relation_fragility_summary.md`",
        "3. `object_f1_anomaly/object_f1_anomaly.md`",
        "4. `efficiency/lightweight_efficiency_summary.md`",
        "5. `paper_draft/abstract.md` and `paper_draft/outline.md`",
        "",
        "Main framing:",
        "",
        "- Primary: lightweight 3D semantic retention under aggressive point-cloud degradation.",
        "- Secondary/null: graph bottleneck and constrained decoding ablations.",
        "",
        f"Strongly supported claims: {len(claim_status['strongly_supported'])}.",
        f"Unsupported claims: {len(claim_status['unsupported'])}.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _iter_record_groups(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    metrics = record.get("metrics", {})
    rows = []
    for group_key in ["presets", "main_matrix", "ablation_cases"]:
        for name, payload in metrics.get(group_key, {}).items():
            rows.append((name, payload))
    return rows


def _row_by_preset(rows: list[dict[str, Any]], preset: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("preset") == preset), None)


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


def _ratio(value: float, baseline: float) -> float:
    return value / baseline if baseline else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _set_f1(pred: set[Any], ref: set[Any]) -> float:
    return _f1(len(pred & ref), len(pred - ref), len(ref - pred))


def _format_values(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"
