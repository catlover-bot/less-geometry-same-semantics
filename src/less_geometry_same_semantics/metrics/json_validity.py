"""JSON validity and schema-compliance metrics."""

from __future__ import annotations

from typing import Any

from less_geometry_same_semantics.schemas.schema import is_valid_semantic_output


def json_validity_rate(payloads: list[dict[str, Any]]) -> dict[str, float]:
    """Return schema compliance rate for a list of model outputs."""

    if not payloads:
        return {"valid": 0.0, "invalid": 0.0, "validity_rate": 0.0}
    valid_count = sum(1 for payload in payloads if is_valid_semantic_output(payload))
    invalid_count = len(payloads) - valid_count
    return {
        "valid": float(valid_count),
        "invalid": float(invalid_count),
        "validity_rate": valid_count / len(payloads),
    }
