# less-geometry-same-semantics

A benchmark and experimental framework for semantics-preserving 3D scene understanding under aggressive point-cloud degradation, using graph-centric intermediate representations and lightweight structured decoding.

Core claim:

> We study whether task-relevant semantics can be preserved under aggressive point-cloud degradation and lightweight language modeling, even when geometric fidelity is not preserved.

The main pipeline is unchanged:

```text
point cloud -> objects / relations / scene graph -> structured semantic output
```

The current real-dataset target is **ARKitScenes 3DOD**. Previous 3RScan/3DSSG and ScanNet configs are archived under `configs/legacy/` and are no longer part of the main workflow.

## ARKitScenes First

ARKitScenes is now the primary public benchmark target. The active adapter expects the official ARKitScenes layout, especially the 3D object detection subset:

```text
ARKitScenes/
  3dod/
    Training/
      <video_id>/
        <video_id>_3dod_annotation.json
        <video_id>_3dod_mesh.ply
    Validation/
      <video_id>/
        <video_id>_3dod_annotation.json
        <video_id>_3dod_mesh.ply
```

The loader also supports prepared `*_pc.npy` point clouds and a deterministic annotation-box fallback for lightweight debug runs.

ARKitScenes provides object boxes, not explicit scene graphs. We derive the graph target with documented coarse heuristics:

- object nodes from 3DOD box labels
- object counts from category counts
- coarse attributes from box dimensions
- coarse spatial relations from box centroids and axis-aligned extents
- scene type from coarse object-label cues

These derived relations are heuristic supervision and should be described that way in any paper text.

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

On Windows, if bare `python` only prints `Python` and exits, use the local
environment through `uv run python ...` or activate `.venv` first:

```powershell
uv run python --version
.\.venv\Scripts\python.exe --version
```

Prepare or document an ARKitScenes download:

```powershell
uv run python scripts/setup_arkitscenes.py `
  --download-dir C:\datasets\ARKitScenes `
  --subset 3dod `
  --fetch-official-script
```

For a small debug download, pass explicit video ids and add `--run-download`.
On Windows, `--resumable-3dod-download` is recommended for explicit 3DOD scene
downloads because large Apple asset downloads can be interrupted:

```powershell
uv run python scripts/setup_arkitscenes.py `
  --download-dir C:\datasets\ARKitScenes `
  --subset 3dod `
  --split Training `
  --video-id 47333462 `
  --fetch-official-script `
  --run-download `
  --resumable-3dod-download
```

Then set the dataset root in the current PowerShell session:

```powershell
$env:ARKITSCENES_ROOT = "C:\datasets\ARKitScenes"
```

For this local Windows machine, the verified root is:

```powershell
$env:ARKITSCENES_ROOT = "C:\Users\Owner\datasets\ARKitScenes"
```

## Minimal Run Path

Run these in order:

```powershell
uv run python scripts/check_arkitscenes_setup.py --config configs/arkitscenes.yaml

uv run python scripts/run_dataset_diagnostics.py `
  --config configs/arkitscenes.yaml `
  --output-dir outputs/diagnostics/arkitscenes `
  --max-scenes 5

uv run python scripts/run_main_experiments.py `
  --config configs/arkitscenes.yaml `
  --epochs 0 `
  --seeds 7 `
  --max-cases 1 `
  --output outputs/debug_one_case/results.json `
  --artifacts-dir outputs/debug_one_case

uv run python scripts/run_paper_package.py --plan configs/paper_plan.yaml
```

The default paper plan is ARKitScenes-first:

- `configs/arkitscenes.yaml`
- `configs/paper_plan.yaml`
- `configs/paper_plan_arkitscenes.yaml`

## Outputs

Dataset checks:

- `outputs/preflight/arkitscenes/dataset_setup_check.md`
- `outputs/setup/arkitscenes/arkitscenes_setup_report.md`

Diagnostics:

- `outputs/diagnostics/arkitscenes/train/diagnostics.md`
- `outputs/diagnostics/arkitscenes/val/diagnostics.md`
- JSON, CSV, and histogram PNG files in the same folders

Paper package:

- `outputs/paper_package/audit/result_audit.md`
- `outputs/paper_package/claims/frozen_claims.md`
- `outputs/paper_package/tables/main_results.md`
- `outputs/paper_package/tables/graph_ablation.md`
- `outputs/paper_package/tables/compression_efficiency.md`
- `outputs/paper_package/tables/main_comparisons.md`
- `outputs/paper_package/tables/heavy_vs_lightweight.md`
- `outputs/paper_package/tables/robustness_vs_compute.md`
- `outputs/paper_package/figures/severity_metrics.png`
- `outputs/paper_package/figures/graph_vs_no_graph_severe.png`

Claim-tight lightweight-robustness analysis:

```powershell
uv run python scripts/generate_claim_tight_analysis.py
```

Key outputs:

- `outputs/paper_package/claim_tight_analysis/claim_status/claim_status.md`
- `outputs/paper_package/claim_tight_analysis/object_f1_anomaly/object_f1_anomaly.md`
- `outputs/paper_package/claim_tight_analysis/relation_fragility/relation_fragility_summary.md`
- `outputs/paper_package/claim_tight_analysis/efficiency/lightweight_efficiency_summary.md`
- `outputs/paper_package/claim_tight_analysis/paper_draft/abstract.md`

## Corruptions

Corruptions are grouped into research families with deterministic seeds and severity levels:

- `geometry_degradation`
- `coordinate_perturbation`
- `local_structural_corruption`
- `token_point_compression`

Preset severities include `clean`, `mild_corruption`, `medium_corruption`, `severe_corruption`, and `extreme_compression`.

## Model and Ablations

The baseline remains intentionally lightweight:

1. Lightweight point encoder.
2. Optional point/token compressor.
3. Object abstraction stage.
4. Graph construction stage.
5. Lightweight graph reasoning module.
6. Structured semantic decoder.

Supported comparisons:

- clean vs corrupted input
- raw point/token budget vs compressed budget
- no graph vs graph bottleneck
- schema-constrained vs unconstrained decoding
- no adaptation vs lightweight input normalization

No new model families are introduced by the ARKitScenes pivot.

## Fair Comparison Layer

The repo now includes a comparison lane that keeps the paper framing centered on
lightweight robustness rather than on graph claims.

Main comparison groups:

- native lightweight structured baseline from this repo
- lightweight input denoising adaptation baseline for robustness comparison
- VoteNet-style imported 3D detections converted into the shared coarse JSON space
- SpatialLM-style imported structured outputs as a heavy upper-bound path

Supplementary hooks:

- `3DGraphLLM`
- `MiniGPT-3D`
- `PointLLM`

These supplementary hooks are deliberately marked as supplementary because their
native task alignment is weaker than the benchmark's coarse semantic JSON target.

Run the comparison package:

```powershell
uv run python scripts/run_comparison_baselines.py `
  --config configs/arkitscenes.yaml `
  --comparison-config configs/comparisons.yaml `
  --output outputs/comparisons/results.json `
  --artifacts-dir outputs/comparisons
```

Important comparison notes:

- `imported_detector` baselines do not natively predict relations or scene type. The repo derives those fields from 3D boxes using the same coarse heuristics used for ARKitScenes graph targets.
- `imported_structured` baselines expect exported per-scene predictions already mapped into the shared JSON schema, or close enough for conservative schema enforcement.
- The input denoising baseline is a lightweight practical robustness comparison. It should not be described as a full CloudFixer reproduction.
- Missing heavy baseline exports are reported as `pending_external`; they do not break the local paper workflow.

External handoff and ingestion:

```powershell
uv run python scripts/ingest_external_baseline.py `
  --baseline-id spatiallm_import `
  --input-manifest docs/server_baselines/examples/spatiallm_export_example.json `
  --metadata docs/server_baselines/examples/spatiallm_run_metadata_example.json `
  --scenario clean
```

University-server handoff docs:

- `docs/server_baselines/README.md`
- `docs/server_baselines/spatiallm.md`
- `docs/server_baselines/votenet.md`
- `docs/server_baselines/3dgraphllm.md`
- `docs/server_baselines/minigpt3d.md`
- `docs/server_baselines/pointllm.md`

See [docs/comparisons.md](docs/comparisons.md) for the manifest format and comparison caveats.

## Repository Structure

```text
configs/                         Active synthetic and ARKitScenes configs
configs/legacy/                  Deprecated 3RScan/3DSSG and ScanNet configs
docs/                            Design notes and paper-support templates
outputs/                         Run outputs, caches, diagnostics, paper artifacts
scripts/                         Setup, diagnostics, experiment, and reporting scripts
src/less_geometry_same_semantics/
  analysis/                       Failure analysis
  comparisons/                    Baseline import adapters and comparison reporting
  data/                           ARKitScenes loader, graph conversion, corruptions
  metrics/                        Semantic, robustness, efficiency, validity metrics
  models/                         Point encoder, object abstraction, graph reasoning, decoder
  reporting/                      Tables, plots, claims, paper package
  schemas/                        Unified semantic output schema
  training/                       Training/evaluation loops
  utils/                          Config, logging, reproducibility
tests/                            Lightweight tests
```

## Troubleshooting

- `Unresolved environment variable(s): ARKITSCENES_ROOT` means the variable was not set in the current PowerShell session.
- `Dataset root does not exist` means `ARKITSCENES_ROOT` points to the wrong directory.
- `No ARKitScenes scenes found` means the root does not contain a recognized split CSV or downloaded `3dod/Training` / `3dod/Validation` folders.
- `Missing annotation` means the scene folder lacks `<video_id>_3dod_annotation.json`.
- `Missing point cloud/mesh` means the scene lacks `<video_id>_3dod_mesh.ply` or prepared point arrays. The config enables annotation-box fallback for debug runs, but real experiments should use meshes or prepared point clouds when available.
- If `python --version` only prints `Python`, disable the Windows App Execution
  Alias for Python or run commands as `uv run python ...`.
- If the official ARKitScenes download exits after an interrupted transfer, retry
  explicit `--video-id` downloads with `--resumable-3dod-download`.

PowerShell variables are session-local:

```powershell
$env:ARKITSCENES_ROOT = "C:\datasets\ARKitScenes"
```

## Legacy Status

The previous 3RScan/3DSSG and ScanNet workflow was blocked by dataset access and local path friction. Those configs are archived in `configs/legacy/` for reproducibility, but the README, preflight path, and paper plan no longer use them.

## Tests

```powershell
pytest
```

## Future Work

- Run the full ARKitScenes paper package once local data is available.
- Report ARKitScenes-derived relations as heuristic graph supervision.
- Consider richer public datasets only after the ARKitScenes pipeline produces stable paper artifacts.
