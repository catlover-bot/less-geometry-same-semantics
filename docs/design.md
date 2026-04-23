# Design Notes

This repository is a lightweight, paper-oriented experimental framework rather than a production 3D understanding stack.

The guiding claim is: we study whether task-relevant semantics can be preserved under aggressive point-cloud degradation and lightweight language modeling, even when geometric fidelity is not preserved.

The main real-world target is now ARKitScenes 3DOD. Previous 3RScan/3DSSG and ScanNet configs are archived as legacy material and are not part of the default paper workflow.

## Main Separation

- `data/` owns synthetic generation, ARKitScenes loading, YAML preset resolution, and degradation.
- `analysis/` owns qualitative failure report slices.
- `models/` owns neural components only.
- `training/` owns target encoding, optimization, and evaluation loops.
- `metrics/` owns semantic quality scoring, resource measurement, seed aggregation, and robustness curves.
- `schemas/` owns the structured semantic output contract.
- `reporting/` owns CSV/Markdown paper tables and publication-style plots.

The baseline intentionally avoids heavyweight 3D backbones and language models. The first question is whether stable semantics survive geometric loss, so the model stays small and the output deterministic.

## Graph Bottleneck

The graph-centric path is:

```text
point cloud -> point tokens -> object-like nodes -> graph reasoning -> structured JSON
```

`graph_mode` controls the bottleneck:

- `no_graph`: pooled tokens go directly to the decoder.
- `simple_graph`: dense distance-weighted object-node graph.
- `richer_graph`: k-nearest object-node graph with message passing.

The current object abstraction is deliberately simple chunk pooling over point tokens. It is a strong baseline interface rather than a final proposal module.

## ARKitScenes Adapter

ARKitScenes 3DOD provides object-oriented 3D bounding boxes and scene meshes/point sources. It does not provide explicit scene-graph triplets. The adapter derives a coarse graph target as follows:

- node categories from 3DOD box labels mapped into the shared coarse taxonomy
- object counts from category counts
- coarse attributes from box dimensions
- relation triplets from deterministic bbox/centroid heuristics
- scene type from coarse object-label cues

The derived relations are heuristic supervision and must be reported as such.

Processed examples can be cached with `data.cache_dir`. Malformed samples are skipped by default with explicit warnings. Dataset diagnostics record split sizes, skipped scenes, point statistics, object/relation counts, and histograms.

## Structured Output First

JSON is the primary output because it is easy to validate and score. The schema explicitly includes object records, object counts, coarse global attributes, relation triples, and scene type. A free-form text mode exists for comparison, but it is derived from the same structured predictions.

## Corruption Families

Corruptions are grouped into benchmark categories:

- geometry degradation
- coordinate perturbation
- local structural corruption
- token/point compression

Each family has an enabled flag, severity level, optional parameter overrides, and deterministic seed support. Presets such as `clean`, `mild_corruption`, `medium_corruption`, `severe_corruption`, and `extreme_compression` are resolved from YAML first, then from built-in defaults.

## Synthetic Benchmark Role

The synthetic scene generator remains a controlled harness for testing corruption operators, model interfaces, metrics, and quick CPU experiments. Real datasets should return the same sample format:

```python
{
    "points": Tensor[num_points, 3],
    "target": {
        "objects": [{"category": "chair", "count": 1, "attributes": ["small"]}],
        "object_counts": {"chair": 1},
        "attributes": [...],
        "relations": [...],
        "scene_type": "...",
        "caption": "..."
    },
    "metadata": {...}
}
```

## Paper Outputs

The standard records include config, seed, timestamp, metrics, aggregation, and compact failure examples. Reporting utilities operate on saved JSON records, which keeps plotting/table generation separate from model execution and makes paper artifacts reproducible.
