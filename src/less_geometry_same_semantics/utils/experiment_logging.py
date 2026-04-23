"""Structured experiment logging utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    """Return a compact ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def build_run_record(
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    seed: int,
    run_name: str,
    preset: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable benchmark run record."""

    record = {
        "run_name": run_name,
        "timestamp": utc_timestamp(),
        "seed": seed,
        "preset": preset,
        "config": config,
        "metrics": metrics,
    }
    if extra:
        record["extra"] = extra
    return record


def save_json_record(record: dict[str, Any], output_path: str | Path) -> Path:
    """Write a structured record as pretty JSON and return its path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path
