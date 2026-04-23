# SpatialLM Server Handoff

Role in this repo: heavy upper-bound comparison, imported after server-side execution.

## What To Run On The Server

1. Run SpatialLM inference on ARKitScenes scenes for one benchmark condition at a time.
2. Convert each scene output into the repo's shared coarse semantic JSON fields:
   - `objects`
   - `object_counts`
   - `attributes`
   - `relations`
   - `scene_type`
   - optional `caption`
3. Save one export JSON plus one metadata JSON.

This repo does not assume a specific SpatialLM launcher. Preserve the final
scene ids and the condition name used for the run.

## What Files To Save

- prediction export JSON:
  - list of scene predictions or a `predictions` array
- metadata JSON:
  - baseline id
  - kind
  - dataset
  - split
  - condition
  - optional efficiency fields
  - export provenance

See:

- `examples/spatiallm_export_example.json`
- `examples/spatiallm_run_metadata_example.json`

## Local Ingestion

```powershell
uv run python scripts/ingest_spatiallm_results.py `
  --input-manifest <server_copy>\spatiallm_export.json `
  --metadata <server_copy>\spatiallm_run_metadata.json `
  --scenario clean
```

## Required Fields

- per-scene `scene_id`
- per-scene `prediction`
- top-level `condition`
- top-level `split`

## Notes

- SpatialLM should stay a heavy upper-bound comparison, not the main story.
- The imported outputs should already be mapped into this repo's shared schema
  before ingestion.
