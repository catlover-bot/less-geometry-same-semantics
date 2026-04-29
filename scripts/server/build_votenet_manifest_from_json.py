#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--output-manifest", required=True)
    ap.add_argument("--output-metadata", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--split", default="Validation")
    args = ap.parse_args()

    ids = [x.strip() for x in Path(args.ids_file).read_text().splitlines() if x.strip()]
    base = Path(args.json_dir)

    predictions = []
    latencies = []
    empty = 0
    total_raw = 0
    total_mapped = 0

    for vid in ids:
        p = base / f"{vid}.json"
        if not p.exists():
            raise FileNotFoundError(f"missing VoteNet json: {p}")

        obj = json.loads(p.read_text())
        boxes = []

        for det in obj.get("detections", []):
            boxes.append({
                "label": det["label"],
                "center": det["center"],
                "dimensions": det.get("dimensions", det.get("size")),
                "score": det.get("score", 0.0),
                "metadata": {
                    "source_label": det.get("source_label", det["label"]),
                },
            })

        if not boxes:
            empty += 1

        total_raw += int(obj.get("raw_detection_count", len(boxes)))
        total_mapped += int(obj.get("mapped_detection_count", len(boxes)))

        if obj.get("latency_seconds") is not None:
            latencies.append(float(obj["latency_seconds"]) * 1000.0)

        predictions.append({
            "scene_id": vid,
            "boxes": boxes,
            "split": args.split,
            "condition": args.condition,
            "metadata": {
                "source_file": str(p),
                "raw_detection_count": obj.get("raw_detection_count", len(boxes)),
                "mapped_detection_count": obj.get("mapped_detection_count", len(boxes)),
                "latency_seconds": obj.get("latency_seconds"),
            },
        })

    avg_latency = mean(latencies) if latencies else 0.0

    manifest = {
        "schema_name": "external_baseline_manifest",
        "schema_version": "1.0",
        "baseline_id": "votenet_import",
        "baseline_label": "VoteNet imported detections",
        "kind": "imported_detector",
        "dataset": "arkitscenes",
        "split": args.split,
        "condition": args.condition,
        "export": {
            "source_system": "university_server",
            "source_hostname": "elm43",
            "source_root": "~/workspace/votenet",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": "VoteNet ScanNet pretrained",
            "command": "python scripts_arkit/run_votenet_arkit_export.py --input-ply <xyz_only_ply> --scene-id <scene_id> --output-json <scene>.json",
        },
        "efficiency": {
            "latency_ms_per_sample": avg_latency,
            "process_memory_mb": 0.0,
            "parameter_count": 0.0,
        },
        "notes": f"VoteNet imported detections for {args.condition}. Empty boxes are valid zero-detection outputs.",
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

    Path(args.output_manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path(args.output_metadata).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("wrote", args.output_manifest)
    print("wrote", args.output_metadata)
    print("ids", len(ids), "empty", empty, "total_raw", total_raw, "total_mapped", total_mapped, "avg_latency_ms", avg_latency)


if __name__ == "__main__":
    main()
