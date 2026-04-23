"""Robustness-under-severity curve utilities."""

from __future__ import annotations

from typing import Any

DEFAULT_SEVERITY_ORDER = [
    "clean",
    "mild_corruption",
    "medium_corruption",
    "severe_corruption",
    "extreme_compression",
]


def robustness_curve(
    results_by_preset: dict[str, dict[str, Any]],
    severity_order: list[str] | None = None,
) -> dict[str, Any]:
    """Build a compact semantic robustness curve from preset results."""

    order = severity_order or DEFAULT_SEVERITY_ORDER
    clean_score = _semantic_score(results_by_preset.get("clean", {}))
    points = []
    for severity_index, preset in enumerate(order):
        if preset not in results_by_preset:
            continue
        result = results_by_preset[preset]
        score = _semantic_score(result)
        compression = _compression_ratio(result)
        points.append(
            {
                "preset": preset,
                "severity_index": severity_index,
                "semantic_macro_f1": score,
                "relative_to_clean": score / clean_score if clean_score > 0.0 else 0.0,
                "compression_ratio": compression,
            }
        )
    return {
        "points": points,
        "mean_semantic_macro_f1": sum(point["semantic_macro_f1"] for point in points) / max(1, len(points)),
        "mean_relative_to_clean": sum(point["relative_to_clean"] for point in points) / max(1, len(points)),
    }


def _semantic_score(result: dict[str, Any]) -> float:
    quality = result.get("semantic_quality", result)
    return float(quality.get("semantic_macro_f1", 0.0))


def _compression_ratio(result: dict[str, Any]) -> float:
    compression = result.get("compression", {})
    return float(compression.get("compression_ratio", 1.0))
