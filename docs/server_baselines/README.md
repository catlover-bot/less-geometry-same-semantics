# University Server Baseline Handoff

These docs define the external-baseline handoff for this repo.

Goal:

1. run a heavy baseline on a university server
2. save a small export package with predictions and metadata
3. copy the export package back to this repo
4. ingest it locally into `outputs/external_baselines/`
5. evaluate and compare it through the shared paper workflow

This keeps the main paper centered on lightweight robustness while still making
heavy comparisons reproducible and auditable.

## Canonical Local Layout

After local ingestion, canonical artifacts live under:

```text
outputs/external_baselines/<baseline_id>/
  <condition>_predictions.json
  <condition>_predictions.shared_schema.json
  <condition>_predictions.metadata.json
  <condition>_predictions.summary.json
  <condition>_predictions.ingestion_report.md
```

The comparison config already points at these canonical manifest locations.

## Required Handoff Fields

Every imported baseline should preserve:

- `baseline_id`
- `kind`
- `dataset`
- `split`
- `condition`
- `scene_id`
- predictions or boxes per scene

Recommended if available:

- latency per sample
- memory usage
- parameter count
- source command
- code commit
- creation timestamp

## Local Ingestion

Generic ingestion:

```powershell
uv run python scripts/ingest_external_baseline.py `
  --baseline-id spatiallm_import `
  --input-manifest docs/server_baselines/examples/spatiallm_export_example.json `
  --metadata docs/server_baselines/examples/spatiallm_run_metadata_example.json `
  --scenario clean
```

Convenience wrappers:

```powershell
uv run python scripts/ingest_spatiallm_results.py `
  --input-manifest <server_copy>\spatiallm_export.json `
  --metadata <server_copy>\spatiallm_run_metadata.json `
  --scenario severe_corruption

uv run python scripts/ingest_votenet_results.py `
  --input-manifest <server_copy>\votenet_export.json `
  --metadata <server_copy>\votenet_run_metadata.json `
  --scenario clean
```

## Dry-Run Fixtures

Use these before touching a real server:

- `docs/server_baselines/examples/spatiallm_export_example.json`
- `docs/server_baselines/examples/spatiallm_run_metadata_example.json`
- `docs/server_baselines/examples/votenet_export_example.json`
- `docs/server_baselines/examples/votenet_run_metadata_example.json`
- `docs/server_baselines/examples/malformed_export_example.json`
- `docs/server_baselines/examples/incomplete_export_example.json`

## Baseline Docs

- [SpatialLM](./spatiallm.md)
- [VoteNet](./votenet.md)
- [3DGraphLLM](./3dgraphllm.md)
- [MiniGPT-3D](./minigpt3d.md)
- [PointLLM](./pointllm.md)
