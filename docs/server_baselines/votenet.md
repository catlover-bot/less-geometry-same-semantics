# VoteNet Server Handoff

Role in this repo: standard non-LLM 3D detector comparison, imported after
server-side execution.

## What To Run On The Server

1. Run VoteNet inference on ARKitScenes scenes for one benchmark condition at a time.
2. Save per-scene 3D detections with:
   - label
   - center
   - dimensions
   - score if available
3. Save one metadata JSON sidecar with condition, split, and efficiency info if available.

## Conversion Rule

VoteNet does not natively output:

- relations
- scene type
- JSON structured semantics

So this repo derives them locally from the imported 3D boxes:

- objects and counts from mapped detector categories
- coarse attributes from box dimensions
- coarse relations from box geometry heuristics
- scene type from coarse object-label cues

## Local Ingestion

```powershell
uv run python scripts/ingest_votenet_results.py `
  --input-manifest <server_copy>\votenet_export.json `
  --metadata <server_copy>\votenet_run_metadata.json `
  --scenario severe_corruption
```

## Files To Preserve

- `scene_id`
- detection boxes
- split
- condition
- corruption setting name

Optional but useful:

- latency
- memory usage
- parameter count
- source command
