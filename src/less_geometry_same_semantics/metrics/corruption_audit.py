"""Corruption audit summaries from evaluation metadata."""

from __future__ import annotations

from collections import Counter
from typing import Any


def corruption_audit(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize what corruption changed in a run."""

    if not metadata:
        return {}
    removed_fractions = []
    presets: Counter[str] = Counter()
    family_severities: Counter[str] = Counter()
    for item in metadata:
        clean = max(1, int(item.get("clean_num_points", 1)))
        degraded = max(1, int(item.get("degraded_num_points", clean)))
        removed_fractions.append(1.0 - degraded / clean)
        corruption = item.get("corruption") or {}
        if isinstance(corruption, dict):
            presets[str(corruption.get("preset", "unknown"))] += 1
            for family in [
                "geometry_degradation",
                "coordinate_perturbation",
                "local_structural_corruption",
                "token_point_compression",
            ]:
                cfg = corruption.get(family, {})
                if isinstance(cfg, dict) and cfg.get("enabled"):
                    family_severities[f"{family}:{cfg.get('severity', 'unknown')}"] += 1
    return {
        "num_samples": len(metadata),
        "mean_removed_fraction": sum(removed_fractions) / len(removed_fractions),
        "max_removed_fraction": max(removed_fractions),
        "preset_counts": dict(sorted(presets.items())),
        "family_severity_counts": dict(sorted(family_severities.items())),
    }
