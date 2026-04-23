"""Latency, memory, and compression metrics."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import psutil
import torch

T = TypeVar("T")


def compression_ratio(original_points: int, degraded_points: int) -> float:
    """Return original/degraded point count ratio."""

    return float(original_points) / max(1.0, float(degraded_points))


def retained_fraction(original_points: int, degraded_points: int) -> float:
    """Return degraded/original point count fraction."""

    return max(0.0, float(degraded_points)) / max(1.0, float(original_points))


def compression_ratio_from_metadata(metadata: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate compression ratios from sample metadata."""

    if not metadata:
        return {"compression_ratio": 1.0, "retained_fraction": 1.0}
    ratios = [
        compression_ratio(int(item["clean_num_points"]), int(item["degraded_num_points"]))
        for item in metadata
    ]
    retained = [
        retained_fraction(int(item["clean_num_points"]), int(item["degraded_num_points"]))
        for item in metadata
    ]
    return {
        "compression_ratio": sum(ratios) / len(ratios),
        "retained_fraction": sum(retained) / len(retained),
    }


def tensor_memory_mb(*tensors: torch.Tensor) -> float:
    """Estimate tensor memory in MiB."""

    bytes_used = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    return bytes_used / (1024.0 * 1024.0)


def process_memory_mb() -> float:
    """Return current process resident memory in MiB."""

    process = psutil.Process()
    return process.memory_info().rss / (1024.0 * 1024.0)


def parameter_count(model: torch.nn.Module) -> int:
    """Return trainable parameter count."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def measure_latency_ms(fn: Callable[[], T], repeats: int = 10, warmup: int = 1) -> dict[str, float]:
    """Measure average callable latency in milliseconds."""

    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {"latency_ms": (elapsed / max(1, repeats)) * 1000.0}
