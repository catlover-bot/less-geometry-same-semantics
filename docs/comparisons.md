# Comparison Baselines

This repo's main paper framing remains lightweight 3D semantic understanding
under aggressive point-cloud degradation. The comparison layer exists to make
that claim more credible, not to turn the project into a survey of unrelated
model families.

## Baseline Groups

Main comparisons:

- `lightweight_structured`: the repo's native lightweight structured baseline
- `lightweight_input_denoising`: lightweight test-time input adaptation baseline
- `votenet_import`: standard non-LLM 3D detector baseline after conversion
- `spatiallm_import`: heavy upper-bound path using exported structured outputs

Supplementary comparisons:

- `3dgraphllm_import`
- `minigpt3d_import`
- `pointllm_import`

Supplementary baselines stay supplementary because their native task alignment
is broader or looser than this benchmark's coarse semantic JSON target.

## Shared Evaluation Interface

All baselines report into the same evaluation bundle when possible:

- object F1
- relation F1
- object-count exact match
- scene-type accuracy
- JSON validity
- latency
- memory usage
- parameter count
- compression ratio

For non-native structured baselines, the record also stores alignment notes such
as `json=converted`, `relations=derived`, or `scene=mapped`.

The import handoff is defined by:

- `src/less_geometry_same_semantics/schemas/external_baseline_manifest.schema.json`
- `src/less_geometry_same_semantics/schemas/external_run_metadata.schema.json`
- `docs/server_baselines/`

## Manifest Formats

### Imported Structured Predictions

Canonical imported manifests require:

- `schema_name: external_baseline_manifest`
- `schema_version: 1.0`
- `baseline_id`
- `kind`
- `dataset`
- `split`
- `condition`
- `predictions`

Structured predictions should already match the shared schema closely enough for
conservative enforcement.

```json
{
  "schema_name": "external_baseline_manifest",
  "schema_version": "1.0",
  "baseline_id": "spatiallm_import",
  "kind": "imported_structured",
  "dataset": "arkitscenes",
  "split": "Validation",
  "condition": "clean",
  "efficiency": {
    "latency_ms_per_sample": 120.5,
    "process_memory_mb": 14500,
    "parameter_count": 720000000
  },
  "predictions": [
    {
      "scene_id": "41069021",
      "prediction": {
        "objects": [{"category": "chair", "count": 2, "attributes": ["small"]}],
        "object_counts": {"chair": 2},
        "attributes": ["small"],
        "relations": [{"subject": "chair", "predicate": "near", "object": "table"}],
        "scene_type": "room"
      }
    }
  ]
}
```

### Imported Detector Predictions

Canonical detector manifests use the same top-level fields and store one `boxes`
list per scene. The adapter maps labels into the shared object categories and
derives coarse relations and scene type from 3D box geometry.

```json
{
  "schema_name": "external_baseline_manifest",
  "schema_version": "1.0",
  "baseline_id": "votenet_import",
  "kind": "imported_detector",
  "dataset": "arkitscenes",
  "split": "Validation",
  "condition": "clean",
  "efficiency": {
    "latency_ms_per_sample": 35.2,
    "process_memory_mb": 2100,
    "parameter_count": 13800000
  },
  "predictions": [
    {
      "scene_id": "41069021",
      "boxes": [
        {
          "label": "chair",
          "center": [0.2, 0.1, 0.6],
          "dimensions": [0.5, 0.5, 0.9],
          "score": 0.93
        }
      ]
    }
  ]
}
```

The detector adapter is intentionally simple and explicit:

- objects and counts come from mapped detector categories
- coarse attributes come from box dimensions
- relations come from coarse box-center and overlap heuristics
- scene type comes from coarse object-label cues

## Running Comparisons

```powershell
uv run python scripts/run_comparison_baselines.py `
  --config configs/arkitscenes.yaml `
  --comparison-config configs/comparisons.yaml `
  --output outputs/comparisons/results.json `
  --artifacts-dir outputs/comparisons
```

The full paper package will also run comparisons automatically when
`configs/paper_plan.yaml` has `comparisons.enabled: true`.

Prediction paths in `configs/comparisons.yaml` are treated as repo-root relative
unless they start with `./` or `../`, in which case they are resolved relative
to the comparison config file.

## Ingestion

Raw server exports should be ingested first:

```powershell
uv run python scripts/ingest_external_baseline.py `
  --baseline-id spatiallm_import `
  --input-manifest docs/server_baselines/examples/spatiallm_export_example.json `
  --metadata docs/server_baselines/examples/spatiallm_run_metadata_example.json `
  --scenario clean
```

This writes canonical local artifacts under `outputs/external_baselines/` and
lets the comparison runner treat missing imports as `pending_external` instead
of crashing the workflow.

## Honest Caveats

- `SpatialLM` is integrated as a heavy upper-bound import path, not as a full
  in-repo reproduction.
- `VoteNet` is evaluated after conversion from 3D detections to the shared JSON
  interface; relations and scene type are therefore derived, not native.
- The denoising adaptation baseline is a lightweight practical robustness
  baseline. It should not be described as a faithful reproduction of CloudFixer.
- `3DGraphLLM`, `MiniGPT-3D`, and `PointLLM` are supplementary hooks only unless
  their exported outputs can be mapped cleanly into the benchmark target space.
