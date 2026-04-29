#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BBOX_RE = re.compile(r"bbox_(\d+)=Bbox\((.*?)\)\s*$")

# Current shared semantic schema categories.
# Keep this conservative to avoid schema invalid labels.
ALLOWED = ["chair", "table", "sofa", "lamp", "plant", "cabinet"]

LABEL_MAP = {
    "chair": "chair",
    "table": "table",
    "sofa": "sofa",
    "lamp": "lamp",
    "plant": "plant",
    "cabinet": "cabinet",

    # Map some SpatialLM/ScanNet-style labels into the coarse schema if reasonable.
    "desk": "table",
    "counter": "cabinet",
    "bookshelf": "cabinet",
    "shelf": "cabinet",
    "stool": "chair",

    # Drop labels not supported by the current shared schema:
    # bed, bathtub, toilet, sink, oven, stove, refrigerator, tv_monitor, etc.
}


def parse_label(line: str) -> str | None:
    m = BBOX_RE.match(line.strip())
    if not m:
        return None

    parts = [p.strip() for p in m.group(2).split(",")]
    if len(parts) < 8:
        return None

    raw_label = parts[0]
    return LABEL_MAP.get(raw_label)


def structured_prediction_from_txt(path: Path) -> dict:
    counts = Counter()

    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            mapped = parse_label(line)
            if mapped in ALLOWED:
                counts[mapped] += 1

    object_counts = {cat: int(counts.get(cat, 0)) for cat in ALLOWED}

    objects = [
        {
            "category": cat,
            "count": count,
            "attributes": [],
        }
        for cat, count in object_counts.items()
        if count > 0
    ]

    return {
        "objects": objects,
        "object_counts": object_counts,
        "attributes": [],
        "relations": [],
        "scene_type": "room",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--output-manifest", required=True)
    ap.add_argument("--output-metadata", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--split", default="Validation")
    args = ap.parse_args()

    ids = [x.strip() for x in Path(args.ids_file).read_text().splitlines() if x.strip()]
    inp = Path(args.input_dir)

    predictions = []
    missing = []
    empty = 0
    total_supported_objects = 0

    for vid in ids:
        p = inp / f"{vid}.txt"
        if not p.exists():
            missing.append(vid)

        pred = structured_prediction_from_txt(p)

        if len(pred["objects"]) == 0:
            empty += 1

        total_supported_objects += sum(pred["object_counts"].values())

        predictions.append({
            "scene_id": vid,
            "prediction": pred,
        })

    if missing:
        raise FileNotFoundError(f"Missing SpatialLM txt files: {missing}")

    manifest = {
        "schema_name": "external_baseline_manifest",
        "schema_version": "1.0",
        "baseline_id": "spatiallm_import",
        "baseline_label": "SpatialLM imported structured outputs",
        "kind": "imported_structured",
        "dataset": "arkitscenes",
        "split": args.split,
        "condition": args.condition,
        "export": {
            "source_system": "university_server",
            "source_hostname": "elm43",
            "source_root": "~/workspace/SpatialLM",
            "command": "python inference.py -d object -p <ply> -o <txt> --model_path ysmao/SpatialLM1.1-Qwen-0.5B-ARKitScenes-SFT",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": "ysmao/SpatialLM1.1-Qwen-0.5B-ARKitScenes-SFT",
            "notes": "SpatialLM Bbox(...) text outputs mapped into the shared semantic schema. Unsupported labels are dropped.",
        },
        "efficiency": {
            "latency_ms_per_sample": 0.0,
            "process_memory_mb": 0.0,
            "parameter_count": 0.0,
        },
        "notes": (
            f"Converted SpatialLM object-box text outputs for {args.condition}. "
            "Relations are empty because this run used `-d object`. "
            "Empty outputs are valid zero-output model behavior."
        ),
        "predictions": predictions,
    }

    metadata = {
        "schema_name": "external_baseline_run_metadata",
        "schema_version": "1.0",
        "baseline_id": manifest["baseline_id"],
        "baseline_label": manifest["baseline_label"],
        "kind": manifest["kind"],
        "dataset": manifest["dataset"],
        "split": manifest["split"],
        "condition": manifest["condition"],
        "export": manifest["export"],
        "efficiency": manifest["efficiency"],
        "notes": manifest["notes"],
    }

    Path(args.output_manifest).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(args.output_metadata).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("wrote", args.output_manifest)
    print("wrote", args.output_metadata)
    print("ids", len(ids), "empty", empty, "total_supported_objects", total_supported_objects)


if __name__ == "__main__":
    main()
