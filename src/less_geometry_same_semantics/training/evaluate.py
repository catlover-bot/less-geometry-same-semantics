"""Evaluation helpers for structured semantic outputs."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from less_geometry_same_semantics.metrics.efficiency import (
    compression_ratio_from_metadata,
    parameter_count,
    process_memory_mb,
    tensor_memory_mb,
)
from less_geometry_same_semantics.metrics.corruption_audit import corruption_audit
from less_geometry_same_semantics.metrics.json_validity import json_validity_rate
from less_geometry_same_semantics.metrics.semantic import semantic_quality_metrics


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    metadata: list[dict[str, Any]],
    *,
    include_outputs: bool = False,
    json_validity_mode: str = "native",
    efficiency_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate already-produced semantic predictions against JSON targets."""

    result = {
        "semantic_quality": semantic_quality_metrics(predictions, references),
        "json_validity": _json_validity_payload(predictions, mode=json_validity_mode),
        "compression": compression_ratio_from_metadata(metadata),
        "corruption_audit": corruption_audit(metadata),
        "efficiency": {
            "latency_ms_per_batch": None,
            "latency_ms_per_sample": None,
            "input_tensor_memory_mb": None,
            "process_memory_mb": None,
            "parameter_count": None,
            "compressed_token_budget": None,
        },
        "num_samples": len(references),
    }
    if efficiency_overrides:
        result["efficiency"].update(efficiency_overrides)
    if include_outputs:
        result["examples"] = [
            {
                "prediction": prediction,
                "reference": reference,
                "metadata": item_metadata,
            }
            for prediction, reference, item_metadata in zip(predictions, references, metadata, strict=True)
        ]
    return result


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_mode: str = "json",
    constrained: bool = True,
    include_outputs: bool = False,
) -> dict[str, Any]:
    """Evaluate model predictions against JSON targets."""

    model.eval()
    predictions: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    input_memory_mb: list[float] = []

    for batch in dataloader:
        points = batch["points"].to(device)
        mask = batch["mask"].to(device)
        input_memory_mb.append(tensor_memory_mb(points, mask))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        batch_predictions = model.predict(
            points,
            mask,
            output_mode="json" if output_mode == "json" else "text",
            constrained=constrained,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms.append((time.perf_counter() - start) * 1000.0)
        if output_mode != "json":
            batch_predictions = [
                {
                    "objects": [],
                    "object_counts": {},
                    "attributes": [],
                    "relations": [],
                    "scene_type": "free_form",
                    "caption": str(item),
                }
                for item in batch_predictions
            ]
        predictions.extend(batch_predictions)  # type: ignore[arg-type]
        references.extend(batch["targets"])
        metadata.extend(batch["metadata"])
        if include_outputs:
            examples.extend(
                {
                    "prediction": prediction,
                    "reference": reference,
                    "metadata": item_metadata,
                }
                for prediction, reference, item_metadata in zip(
                    batch_predictions,
                    batch["targets"],
                    batch["metadata"],
                    strict=True,
                )
            )

    result = evaluate_predictions(
        predictions,
        references,
        metadata,
        include_outputs=include_outputs,
        json_validity_mode="native",
        efficiency_overrides={
            "latency_ms_per_batch": sum(latency_ms) / max(1, len(latency_ms)),
            "latency_ms_per_sample": sum(latency_ms) / max(1, len(references)),
            "input_tensor_memory_mb": sum(input_memory_mb) / max(1, len(input_memory_mb)),
            "process_memory_mb": process_memory_mb(),
            "parameter_count": float(parameter_count(model)),
            "compressed_token_budget": float(getattr(model, "compressed_tokens", 0)),
        },
    )
    if include_outputs:
        result["examples"] = examples
    return result


def _json_validity_payload(payloads: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    if mode == "not_applicable":
        return {"applicable": False, "mode": mode, "valid": None, "invalid": None, "validity_rate": None}
    scores = json_validity_rate(payloads)
    scores["applicable"] = True
    scores["mode"] = mode
    return scores
