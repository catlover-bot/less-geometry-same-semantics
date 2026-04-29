#!/usr/bin/env python
from __future__ import annotations

import csv
import os
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/is/hirotaka-m/workspace/less-geometry-same-semantics")
SPATIALLM_ROOT = Path("/home/is/hirotaka-m/workspace/SpatialLM")
ARKIT_ROOT = Path(os.environ["ARKITSCENES_ROOT"])

IDS_FILE = ROOT / "outputs/setup/arkitscenes_expand/validation_all30_ids.txt"
OUT_DIR = ROOT / "outputs/external_baselines/spatiallm/latency_probe_outputs"
CSV_PATH = ROOT / "outputs/external_baselines/spatiallm/latency_probe/spatiallm_cli_latency.csv"
MD_PATH = ROOT / "outputs/external_baselines/spatiallm/latency_probe/spatiallm_cli_latency.md"

MODEL_ID = "ysmao/SpatialLM1.1-Qwen-0.5B-ARKitScenes-SFT"

# 最初は3件だけ。全30件測るなら [:3] を外す。
ids = [x.strip() for x in IDS_FILE.read_text().splitlines() if x.strip()][:3]

OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

rows = []

env = os.environ.copy()
env["HF_HOME"] = f"/cl/work11/{os.environ['USER']}/hf_cache"
env["TRANSFORMERS_CACHE"] = env["HF_HOME"]

for i, vid in enumerate(ids, 1):
    ply = ARKIT_ROOT / "3dod" / "Validation" / vid / f"{vid}_3dod_mesh.ply"
    out = OUT_DIR / f"{vid}.txt"

    cmd = [
        "python",
        "inference.py",
        "-d",
        "object",
        "-p",
        str(ply),
        "-o",
        str(out),
        "--model_path",
        MODEL_ID,
    ]

    print(f"=== [{i}/{len(ids)}] SpatialLM latency probe: {vid} ===")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(SPATIALLM_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    t1 = time.perf_counter()

    elapsed = t1 - t0
    ok = proc.returncode == 0

    print("ok:", ok, "elapsed_sec:", elapsed)
    if not ok:
        print(proc.stdout[-4000:])

    rows.append({
        "scene_id": vid,
        "ok": ok,
        "elapsed_sec": elapsed,
        "elapsed_ms": elapsed * 1000.0,
        "output_file": str(out),
    })

with CSV_PATH.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

valid = [r["elapsed_ms"] for r in rows if r["ok"]]
mean_ms = statistics.mean(valid) if valid else 0.0
median_ms = statistics.median(valid) if valid else 0.0

MD_PATH.write_text(
    "# SpatialLM CLI End-to-End Latency Probe\n\n"
    "This measures `python inference.py` wall-clock time per scene, including process startup, model loading, point-cloud loading, generation, and file writing.\n\n"
    f"- scenes: {len(rows)}\n"
    f"- successful scenes: {len(valid)}\n"
    f"- mean_ms: {mean_ms:.2f}\n"
    f"- median_ms: {median_ms:.2f}\n\n"
    "This is not warm inference latency. It is an operational end-to-end CLI latency.\n",
    encoding="utf-8",
)

print("wrote", CSV_PATH)
print("wrote", MD_PATH)
print("mean_ms", mean_ms)
print("median_ms", median_ms)
