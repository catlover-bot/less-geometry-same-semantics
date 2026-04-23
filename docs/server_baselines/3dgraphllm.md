# 3DGraphLLM Server Handoff

Role in this repo: supplementary imported comparison only.

## Server Output Expectation

Map each scene output into the repo's shared coarse semantic JSON schema before
copying results back:

- `objects`
- `object_counts`
- `attributes`
- `relations`
- `scene_type`

If the native output is broader than this target, keep only the fields that can
be mapped cleanly and document the mapping in the metadata `notes`.

## Local Ingestion

```powershell
uv run python scripts/ingest_external_baseline.py `
  --baseline-id 3dgraphllm_import `
  --input-manifest <server_copy>\3dgraphllm_export.json `
  --metadata <server_copy>\3dgraphllm_run_metadata.json `
  --scenario clean
```

## Caveat

This comparison should remain supplementary unless the mapping into the shared
benchmark target is clean and auditable.
