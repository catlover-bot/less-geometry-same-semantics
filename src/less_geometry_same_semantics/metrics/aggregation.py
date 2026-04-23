"""Aggregation helpers for multi-seed benchmark runs."""

from __future__ import annotations

import math
from typing import Any


def aggregate_seed_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Recursively aggregate numeric metric leaves across random seeds."""

    if not results:
        return {"num_runs": 0, "mean": {}, "std": {}}
    numeric_paths: dict[tuple[str, ...], list[float]] = {}
    for result in results:
        _collect_numeric_paths(result, (), numeric_paths)
    return {
        "num_runs": len(results),
        "mean": _unflatten({path: _mean(values) for path, values in numeric_paths.items()}),
        "std": _unflatten({path: _std(values) for path, values in numeric_paths.items()}),
    }


def _collect_numeric_paths(
    value: Any,
    prefix: tuple[str, ...],
    output: dict[tuple[str, ...], list[float]],
) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        output.setdefault(prefix, []).append(float(value))
    elif isinstance(value, dict):
        for key, child in value.items():
            _collect_numeric_paths(child, prefix + (str(key),), output)


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _unflatten(flat: dict[tuple[str, ...], float]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path, value in flat.items():
        cursor = root
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        if path:
            cursor[path[-1]] = value
    return root
