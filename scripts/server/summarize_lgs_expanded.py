#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

p = Path("outputs/main_50_30/results.json")
obj = json.loads(p.read_text())

matrix = obj["metrics"]["main_matrix"]

def get(d, path, default="n/a"):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def fmt(x):
    if x == "n/a" or x is None:
        return "n/a"
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)

print("# LGS Expanded Summary")
print()
print("| case | corruption | budget | graph | constrained | adaptation | object_f1 | relation_f1 | scene_accuracy | json_validity | latency_ms | params |")
print("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|")

for case, rec in sorted(matrix.items()):
    parts = case.split("__")
    if len(parts) != 5:
        corruption, budget, graph, constrained, adaptation = case, "n/a", "n/a", "n/a", "n/a"
    else:
        corruption, budget, graph, constrained, adaptation = parts

    mean = rec["aggregate"]["mean"]

    print(
        f"| {case} | "
        f"{corruption} | {budget} | {graph} | {constrained} | {adaptation} | "
        f"{fmt(get(mean, 'semantic_quality.objects.f1'))} | "
        f"{fmt(get(mean, 'semantic_quality.relations.f1'))} | "
        f"{fmt(get(mean, 'semantic_quality.scene_type.accuracy'))} | "
        f"{fmt(get(mean, 'json_validity.validity_rate'))} | "
        f"{fmt(get(mean, 'efficiency.latency_ms_per_sample'))} | "
        f"{fmt(get(mean, 'efficiency.parameter_count'))} |"
    )
