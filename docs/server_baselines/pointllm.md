# PointLLM Server Handoff

Role in this repo: supplementary imported comparison only.

PointLLM should be exported into the shared coarse semantic JSON schema before
local ingestion.

## What To Save

- one prediction export JSON
- one metadata JSON
- scene ids matching ARKitScenes split discovery

## Local Ingestion

```powershell
uv run python scripts/ingest_external_baseline.py `
  --baseline-id pointllm_import `
  --input-manifest <server_copy>\pointllm_export.json `
  --metadata <server_copy>\pointllm_run_metadata.json `
  --scenario clean
```

## Caveat

Keep this supplementary unless the exported outputs align cleanly with the
shared benchmark target.
