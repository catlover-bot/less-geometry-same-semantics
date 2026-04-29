#!/usr/bin/env python
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean


ROOT = Path(".")
OUT = ROOT / "outputs/paper_package/final_tables"
OUT.mkdir(parents=True, exist_ok=True)

MAIN_COMPARISONS = ROOT / "outputs/paper_package/tables/main_comparisons.md"


def parse_markdown_table(path: Path) -> list[dict[str, str]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        raise ValueError(f"No markdown table found in {path}")

    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def to_float(x: str | None) -> float | None:
    if x is None or x == "" or x.lower() == "n/a":
        return None
    return float(x)


def fmt(x: float | None, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def get_case(rows: list[dict[str, str]], case: str) -> dict[str, str]:
    for r in rows:
        if r.get("case") == case:
            return r
    raise KeyError(f"missing case: {case}")


def spatiallm_bbox_stats() -> list[dict[str, object]]:
    base = ROOT / "outputs/external_baselines/spatiallm"
    dirs = {
        "clean": base / "raw_clean",
        "mild": base / "raw_mild",
        "medium": base / "raw_medium",
        "severe": base / "raw_severe",
    }
    bbox_re = re.compile(r"^bbox_\d+=Bbox\(")
    rows = []
    for condition, d in dirs.items():
        files = sorted(d.glob("*.txt"))
        empty = 0
        total_bbox = 0
        for p in files:
            text = p.read_text(encoding="utf-8", errors="ignore")
            bbox_count = sum(1 for line in text.splitlines() if bbox_re.match(line.strip()))
            total_bbox += bbox_count
            if p.stat().st_size == 0 or bbox_count == 0:
                empty += 1
        rows.append({
            "model": "SpatialLM",
            "condition": condition,
            "files": len(files),
            "empty_outputs": empty,
            "total_outputs": total_bbox,
            "avg_outputs_per_scene": total_bbox / len(files) if files else 0.0,
            "note": "bbox text lines",
        })
    return rows


def votenet_json_stats() -> list[dict[str, object]]:
    base = ROOT / "outputs/external_baselines/votenet"
    dirs = {
        "clean": base / "json_clean",
        "severe": base / "json_severe",
    }
    rows = []
    for condition, d in dirs.items():
        files = sorted(d.glob("*.json"))
        empty = 0
        total_raw = 0
        total_mapped = 0
        latencies = []
        for p in files:
            obj = json.loads(p.read_text(encoding="utf-8"))
            raw = int(obj.get("raw_detection_count", len(obj.get("detections", []))))
            mapped = int(obj.get("mapped_detection_count", len(obj.get("detections", []))))
            total_raw += raw
            total_mapped += mapped
            if mapped == 0:
                empty += 1
            if obj.get("latency_seconds") is not None:
                latencies.append(float(obj["latency_seconds"]) * 1000.0)

        rows.append({
            "model": "VoteNet",
            "condition": condition,
            "files": len(files),
            "empty_outputs": empty,
            "total_outputs": total_mapped,
            "avg_outputs_per_scene": total_mapped / len(files) if files else 0.0,
            "avg_latency_ms": mean(latencies) if latencies else None,
            "note": f"raw detections={total_raw}, mapped detections={total_mapped}",
        })
    return rows


def write_final_claim_table(rows: list[dict[str, str]]) -> None:
    lgs_clean = get_case(rows, "lightweight_structured__clean")
    lgs_severe = get_case(rows, "lightweight_structured__severe_corruption")
    spatial_clean = get_case(rows, "spatiallm_import__clean")
    spatial_severe = get_case(rows, "spatiallm_import__severe_corruption")
    votenet_clean = get_case(rows, "votenet_import__clean")
    votenet_severe = get_case(rows, "votenet_import__severe_corruption")

    lgs_clean_obj = to_float(lgs_clean["object_f1"])
    lgs_severe_obj = to_float(lgs_severe["object_f1"])
    spatial_clean_obj = to_float(spatial_clean["object_f1"])
    spatial_severe_obj = to_float(spatial_severe["object_f1"])
    votenet_clean_obj = to_float(votenet_clean["object_f1"])
    votenet_severe_obj = to_float(votenet_severe["object_f1"])

    lgs_latency = to_float(lgs_clean["latency_ms"])
    votenet_latency = to_float(votenet_clean["latency_ms"])
    speedup = (votenet_latency / lgs_latency) if lgs_latency and votenet_latency else None

    out = OUT / "final_claim_table.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# Final Claim Table\n\n")
        f.write("This table states what the current experiments support and how strongly each claim should be worded.\n\n")

        f.write("| ID | Claim | Evidence | Status | Safe paper wording | Caveat |\n")
        f.write("|---|---|---|---|---|---|\n")

        f.write(
            "| C1 | Lightweight LGS is competitive on clean object semantics. | "
            f"Clean object F1: LGS={fmt(lgs_clean_obj)}, SpatialLM={fmt(spatial_clean_obj)}, VoteNet={fmt(votenet_clean_obj)}. | "
            "Supported | "
            "On clean ARKitScenes inputs, the lightweight structured model reaches object-level performance comparable to heavier or standard external baselines. | "
            "This is a coarse semantic evaluation after mapping external outputs into the shared label space. |\n"
        )

        f.write(
            "| C2 | LGS is substantially more stable under severe input corruption. | "
            f"Severe object F1: LGS={fmt(lgs_severe_obj)}, SpatialLM={fmt(spatial_severe_obj)}, VoteNet={fmt(votenet_severe_obj)}. | "
            "Strongly supported | "
            "Under the severe coordinate perturbation and quantization protocol, LGS preserves object-level semantics while the imported external baselines collapse. | "
            "Do not claim universal robustness of all LLMs or all 3D detectors; this claim is limited to the tested corruption protocol and imported baselines. |\n"
        )

        f.write(
            "| C3 | LGS is much faster than VoteNet in this evaluation path. | "
            f"Latency: LGS={fmt(lgs_latency, 2)} ms, VoteNet={fmt(votenet_latency, 2)} ms, speedup≈{fmt(speedup, 1)}x. | "
            "Supported | "
            "The lightweight model runs about two orders of magnitude faster than the VoteNet import path on the evaluated setup. | "
            "VoteNet latency is measured through the export script; memory and parameter count are placeholders in current metadata. |\n"
        )

        f.write(
            "| C4 | Output format validity is not the bottleneck. | "
            f"JSON validity is {lgs_clean['json_validity']} for LGS clean, {lgs_severe['json_validity']} for LGS severe, "
            f"{spatial_clean['json_validity']} for SpatialLM clean, {spatial_severe['json_validity']} for SpatialLM severe, "
            f"{votenet_clean['json_validity']} for VoteNet clean, and {votenet_severe['json_validity']} for VoteNet severe. | "
            "Supported | "
            "All compared systems produce valid converted or native JSON under the evaluation pipeline. | "
            "Validity does not imply semantic correctness; it only confirms schema compatibility. |\n"
        )

        f.write(
            "| C5 | Relations remain a weak point for the lightweight model. | "
            f"Clean relation F1: LGS={lgs_clean['relation_f1']}, VoteNet={votenet_clean['relation_f1']}; severe relation F1 remains low across systems. | "
            "Caveat / limitation | "
            "The current lightweight model primarily supports robust coarse object semantics; relation semantics require further work. | "
            "External relation scores are derived or mapped, not always native model outputs. |\n"
        )

        f.write(
            "| C6 | Exact object counting is not solved. | "
            f"count_exact is {lgs_clean['count_exact']} for LGS clean and remains {lgs_severe['count_exact']} for LGS severe. | "
            "Limitation | "
            "The benchmark should not claim exact count recovery; the current evidence supports category-level semantic retention instead. | "
            "Count-sensitive metrics need a separate analysis or improved model objective. |\n"
        )

    print(f"wrote {out}")


def write_failure_analysis_table(rows: list[dict[str, str]]) -> None:
    spatial = spatiallm_bbox_stats()
    votenet = votenet_json_stats()

    out = OUT / "failure_analysis_table.md"
    with out.open("w", encoding="utf-8") as f:
        f.write("# Failure Analysis Table\n\n")
        f.write("This table summarizes how external baselines fail under the tested corruption protocol.\n\n")

        f.write("## Raw output collapse\n\n")
        f.write("| model | condition | scenes | empty/no-output scenes | total mapped outputs | avg outputs per scene | latency ms | note |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|\n")

        for r in spatial + votenet:
            latency = r.get("avg_latency_ms")
            f.write(
                f"| {r['model']} | {r['condition']} | {r['files']} | {r['empty_outputs']} | "
                f"{r['total_outputs']} | {float(r['avg_outputs_per_scene']):.2f} | "
                f"{fmt(latency, 2) if latency is not None else 'n/a'} | {r['note']} |\n"
            )

        f.write("\n## Severe-condition semantic comparison\n\n")
        f.write("| case | object_f1 | relation_f1 | scene_accuracy | json_validity | interpretation |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")

        case_names = [
            ("lightweight_structured__severe_corruption", "LGS keeps coarse object semantics under severe corruption."),
            ("spatiallm_import__severe_corruption", "SpatialLM emits no usable object boxes under this severe protocol."),
            ("votenet_import__severe_corruption", "VoteNet mostly collapses; only one validation scene produced mapped detections."),
        ]

        for case, interp in case_names:
            r = get_case(rows, case)
            f.write(
                f"| {case} | {r['object_f1']} | {r['relation_f1']} | {r['scene_accuracy']} | {r['json_validity']} | {interp} |\n"
            )

        f.write("\n## Notes\n\n")
        f.write("- Empty detector outputs are treated as valid zero-detection model outputs, not as missing files.\n")
        f.write("- SpatialLM mild/medium/severe collapse is based on bbox text output counts.\n")
        f.write("- VoteNet severe collapse is based on exported detector JSON counts.\n")
        f.write("- The severe corruption protocol here is coordinate perturbation plus quantization.\n")

    print(f"wrote {out}")


def write_readme() -> None:
    out = OUT / "README.md"
    out.write_text(
        "# Final Paper Tables\n\n"
        "Generated from the current paper-package comparison outputs and external baseline raw outputs.\n\n"
        "- `final_claim_table.md`: claim-by-claim paper wording and evidence.\n"
        "- `failure_analysis_table.md`: collapse/failure behavior for SpatialLM and VoteNet.\n",
        encoding="utf-8",
    )
    print(f"wrote {out}")


def main() -> None:
    rows = parse_markdown_table(MAIN_COMPARISONS)
    write_final_claim_table(rows)
    write_failure_analysis_table(rows)
    write_readme()


if __name__ == "__main__":
    main()
