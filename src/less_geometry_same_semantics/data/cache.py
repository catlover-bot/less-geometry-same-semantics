"""Deterministic preprocessing cache helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch


def cache_key(payload: dict[str, Any]) -> str:
    """Stable cache key for preprocessing settings."""

    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_or_build_cached_example(
    cache_dir: str | Path | None,
    key_payload: dict[str, Any],
    builder: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Load a processed example from cache or build and store it."""

    if cache_dir is None:
        return builder()
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_root / f"{cache_key(key_payload)}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    example = builder()
    torch.save(example, path)
    return example
