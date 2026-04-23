# MiniGPT-3D Server Handoff

Role in this repo: supplementary imported comparison only.

MiniGPT-3D is not natively aligned to the coarse structured benchmark target, so
the local import path expects already-mapped structured outputs.

## What To Preserve

- `scene_id`
- mapped shared-schema prediction per scene
- split
- condition
- optional efficiency metadata

## Local Ingestion

```powershell
uv run python scripts/ingest_external_baseline.py `
  --baseline-id minigpt3d_import `
  --input-manifest <server_copy>\minigpt3d_export.json `
  --metadata <server_copy>\minigpt3d_run_metadata.json `
  --scenario clean
```

## Caveat

Treat as supplementary unless the task mapping is clearly documented.
