#!/usr/bin/env python
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import os

from less_geometry_same_semantics.data.public_datasets import (
    discover_arkitscenes_annotation_file,
    _parse_arkitscenes_annotation_boxes,
)
from less_geometry_same_semantics.data.label_mapping import map_object_category

ROOT = Path(".")
ARKIT_ROOT = Path(os.environ["ARKITSCENES_ROOT"])
IDS_FILE = ROOT / "outputs/setup/arkitscenes_expand/validation_all30_ids.txt"
OUT_DIR = ROOT / "outputs/paper_package_50_30/fair_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED = {"chair", "table", "sofa", "lamp", "plant", "cabinet"}

SPATIALLM_LABEL_MAP = {
    "chair": "chair",
    "dining_chair": "chair",
    "bar_chair": "chair",
    "stool": "chair",
    "sofa": "sofa",
    "combination_sofa": "sofa",
    "table": "table",
    "coffee_table": "table",
    "dining_table": "table",
    "side_table": "table",
    "desk": "table",
    "leisure_table_and_chair_combination": "table",
    "lamp": "lamp",
    "illumination": "lamp",
    "chandelier": "lamp",
    "floor-standing_lamp": "lamp",
    "plants": "plant",
    "plant": "plant",
    "potted_bonsai": "plant",
    "cabinet": "cabinet",
    "wardrobe": "cabinet",
    "nightstand": "cabinet",
    "tv_cabinet": "cabinet",
    "wine_cabinet": "cabinet",
    "bathroom_cabinet": "cabinet",
    "shoe_cabinet": "cabinet",
    "entrance_cabinet": "cabinet",
    "decorative_cabinet": "cabinet",
    "washing_cabinet": "cabinet",
    "wall_cabinet": "cabinet",
    "sideboard": "cabinet",
    "cupboard": "cabinet",
    "bookcase": "cabinet",
    "bookshelf": "cabinet",
    "shelf": "cabinet",
    "counter": "cabinet",
}

BBOX_RE = re.compile(r"bbox_(\d+)=Bbox\((.*?)\)\s*$")


def ids() -> list[str]:
    return [x.strip() for x in IDS_FILE.read_text().splitlines() if x.strip()]


def safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def normalize_label(label: Any, source: str) -> str | None:
    if label is None:
        return None
    raw = str(label).strip()
    if source == "spatiallm":
        mapped = SPATIALLM_LABEL_MAP.get(raw)
    else:
        mapped = map_object_category(raw) or raw
    if mapped in SUPPORTED:
        return mapped
    return None


def center_dims_to_aabb(center: list[float], dims: list[float]) -> tuple[list[float], list[float]]:
    c = [float(x) for x in center[:3]]
    d = [abs(float(x)) for x in dims[:3]]
    lo = [c[i] - d[i] / 2.0 for i in range(3)]
    hi = [c[i] + d[i] / 2.0 for i in range(3)]
    return lo, hi


def box_to_aabb(box: dict[str, Any]) -> tuple[list[float], list[float]] | None:
    # Common direct min/max formats.
    for min_key, max_key in [
        ("min", "max"),
        ("mins", "maxs"),
        ("min_corner", "max_corner"),
        ("bbox_min", "bbox_max"),
    ]:
        if min_key in box and max_key in box:
            lo = [float(x) for x in box[min_key]]
            hi = [float(x) for x in box[max_key]]
            return lo, hi

    # Nested bbox dict.
    if isinstance(box.get("bbox"), dict):
        nested = box["bbox"]
        got = box_to_aabb(nested)
        if got is not None:
            return got

    # center + dimensions/size/extents.
    center = box.get("center") or box.get("centroid")
    dims = box.get("dimensions") or box.get("size") or box.get("extent") or box.get("extents")
    if center is not None and dims is not None:
        return center_dims_to_aabb(center, dims)

    # Some ARKit/OBB parsed variants.
    obb = box.get("obb") or box.get("obbAligned") or box.get("segments.obb")
    if isinstance(obb, dict):
        center = obb.get("center")
        dims = obb.get("dimensions") or obb.get("size") or obb.get("extent") or obb.get("extents")
        if center is not None and dims is not None:
            return center_dims_to_aabb(center, dims)

    return None


def aabb_center(lo: list[float], hi: list[float]) -> list[float]:
    return [(lo[i] + hi[i]) / 2.0 for i in range(3)]


def aabb_dims(lo: list[float], hi: list[float]) -> list[float]:
    return [max(0.0, hi[i] - lo[i]) for i in range(3)]


def iou3d(a: tuple[list[float], list[float]], b: tuple[list[float], list[float]]) -> float:
    alo, ahi = a
    blo, bhi = b
    inter = 1.0
    va = 1.0
    vb = 1.0
    for i in range(3):
        inter_len = max(0.0, min(ahi[i], bhi[i]) - max(alo[i], blo[i]))
        inter *= inter_len
        va *= max(0.0, ahi[i] - alo[i])
        vb *= max(0.0, bhi[i] - blo[i])
    denom = va + vb - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def load_gt_boxes(scene_id: str) -> list[dict[str, Any]]:
    ann = discover_arkitscenes_annotation_file(ARKIT_ROOT, scene_id, "Validation", "3dod")
    raw_boxes = _parse_arkitscenes_annotation_boxes(ann, require_mapped=True)
    out = []
    for b in raw_boxes:
        label = b.get("category")
        if label not in SUPPORTED:
            continue
        aabb = box_to_aabb(b)
        if aabb is None:
            continue
        out.append({
            "label": label,
            "aabb": aabb,
            "score": 1.0,
            "source": "gt",
        })
    return out


def parse_spatiallm_txt(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = BBOX_RE.match(line.strip())
        if not m:
            continue
        parts = [p.strip() for p in m.group(2).split(",")]
        if len(parts) < 8:
            continue
        label = normalize_label(parts[0], "spatiallm")
        if label is None:
            continue
        nums = []
        ok = True
        for x in parts[1:]:
            y = safe_float(x)
            if y is None:
                ok = False
                break
            nums.append(y)
        if not ok:
            continue
        center = nums[:3]
        dims = nums[-3:]
        out.append({
            "label": label,
            "aabb": center_dims_to_aabb(center, dims),
            "score": 1.0,
            "source": "spatiallm",
        })
    return out


def parse_votenet_json(path: Path) -> list[dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    obj = json.loads(path.read_text())

    # VoteNet export uses a different axis convention from ARKitScenes.
    # Coordinate audit selected perm=021, signs=++-:
    # VoteNet [x, y, z] -> ARKitScenes [x, z, -y].
    perm = [0, 2, 1]
    signs = [1.0, 1.0, -1.0]

    for det in obj.get("detections", []):
        label = normalize_label(det.get("label"), "votenet")
        if label is None:
            continue
        center = det.get("center")
        dims = det.get("dimensions") or det.get("size")
        if center is None or dims is None:
            continue

        center = [float(center[i]) * signs[j] for j, i in enumerate(perm)]
        dims = [float(dims[i]) for i in perm]

        out.append({
            "label": label,
            "aabb": center_dims_to_aabb(center, dims),
            "score": float(det.get("score", 0.0)),
            "source": "votenet",
        })
    return out


def match_boxes(pred: list[dict[str, Any]], gt: list[dict[str, Any]], threshold: float) -> tuple[int, int, int]:
    used = set()
    tp = 0

    pred_sorted = sorted(pred, key=lambda x: x.get("score", 0.0), reverse=True)

    for p in pred_sorted:
        best_i = None
        best_iou = 0.0
        for i, g in enumerate(gt):
            if i in used:
                continue
            if p["label"] != g["label"]:
                continue
            score = iou3d(p["aabb"], g["aabb"])
            if score > best_iou:
                best_iou = score
                best_i = i
        if best_i is not None and best_iou >= threshold:
            tp += 1
            used.add(best_i)

    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f


def derive_relations(boxes: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    rels = set()
    for i, a in enumerate(boxes):
        alo, ahi = a["aabb"]
        ac = aabb_center(alo, ahi)
        ad = aabb_dims(alo, ahi)
        for j, b in enumerate(boxes):
            if i == j:
                continue
            if a["label"] == b["label"]:
                continue
            blo, bhi = b["aabb"]
            bc = aabb_center(blo, bhi)
            bd = aabb_dims(blo, bhi)

            dx = ac[0] - bc[0]
            dy = ac[1] - bc[1]
            dz = ac[2] - bc[2]
            horiz = math.sqrt(dx * dx + dy * dy)

            # Conservative coarse rules. This is diagnostic only.
            near_thr = 0.5 * (max(ad[0], ad[1]) + max(bd[0], bd[1])) + 0.5
            z_thr = 0.35 * (ad[2] + bd[2])

            if dz > z_thr:
                pred = "above"
            elif dz < -z_thr:
                pred = "below"
            elif horiz <= near_thr:
                pred = "near"
            else:
                continue

            rels.add((a["label"], pred, b["label"]))
    return rels


def set_prf(pred: set, gt: set) -> tuple[int, int, int]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    return tp, fp, fn


def semantic_presence(gt: list[dict[str, Any]], pred: list[dict[str, Any]]) -> tuple[int, int, int]:
    gt_set = {x["label"] for x in gt}
    pred_set = {x["label"] for x in pred}
    return set_prf(pred_set, gt_set)


def evaluate_model(model: str, condition: str, pred_loader):
    scene_rows = []
    total_gt = 0
    total_pred = 0
    empty = 0
    supported_counter = Counter()
    gt_counter = Counter()

    box_totals = {0.25: [0, 0, 0], 0.50: [0, 0, 0]}
    sem_total = [0, 0, 0]
    rel_total = [0, 0, 0]

    for scene_id in ids():
        gt = load_gt_boxes(scene_id)
        pred = pred_loader(scene_id)

        total_gt += len(gt)
        total_pred += len(pred)
        if len(pred) == 0:
            empty += 1

        gt_counter.update(x["label"] for x in gt)
        supported_counter.update(x["label"] for x in pred)

        for thr in box_totals:
            tp, fp, fn = match_boxes(pred, gt, thr)
            box_totals[thr][0] += tp
            box_totals[thr][1] += fp
            box_totals[thr][2] += fn

        tp, fp, fn = semantic_presence(gt, pred)
        sem_total[0] += tp
        sem_total[1] += fp
        sem_total[2] += fn

        gt_rel = derive_relations(gt)
        pred_rel = derive_relations(pred)
        rtp, rfp, rfn = set_prf(pred_rel, gt_rel)
        rel_total[0] += rtp
        rel_total[1] += rfp
        rel_total[2] += rfn

        scene_rows.append({
            "scene_id": scene_id,
            "gt_boxes": len(gt),
            "pred_boxes": len(pred),
            "pred_empty": int(len(pred) == 0),
            "gt_labels": " ".join(sorted({x["label"] for x in gt})),
            "pred_labels": " ".join(sorted({x["label"] for x in pred})),
            "gt_relations": len(gt_rel),
            "pred_relations": len(pred_rel),
        })

    sem_p, sem_r, sem_f = prf(*sem_total)
    rel_p, rel_r, rel_f = prf(*rel_total)

    row = {
        "model": model,
        "condition": condition,
        "scenes": len(ids()),
        "empty_scenes": empty,
        "empty_rate": empty / len(ids()),
        "gt_boxes": total_gt,
        "pred_boxes": total_pred,
        "presence_precision": sem_p,
        "presence_recall": sem_r,
        "presence_f1": sem_f,
        "derived_relation_precision": rel_p,
        "derived_relation_recall": rel_r,
        "derived_relation_f1": rel_f,
    }

    for thr, vals in box_totals.items():
        p, r, f = prf(*vals)
        key = str(thr).replace(".", "")
        row[f"box_iou{key}_precision"] = p
        row[f"box_iou{key}_recall"] = r
        row[f"box_iou{key}_f1"] = f

    return row, scene_rows, dict(gt_counter), dict(supported_counter)


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    path.write_text(
        ",".join(keys) + "\n" +
        "\n".join(",".join(str(r.get(k, "")) for k in keys) for r in rows) + "\n",
        encoding="utf-8",
    )


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def main():
    rows = []
    all_scene_rows = []

    specs = [
        (
            "SpatialLM",
            "clean",
            lambda sid: parse_spatiallm_txt(ROOT / "outputs/external_baselines/spatiallm/raw_clean" / f"{sid}.txt"),
        ),
        (
            "SpatialLM",
            "severe_corruption",
            lambda sid: parse_spatiallm_txt(ROOT / "outputs/external_baselines/spatiallm/raw_severe" / f"{sid}.txt"),
        ),
        (
            "VoteNet",
            "clean",
            lambda sid: parse_votenet_json(ROOT / "outputs/external_baselines/votenet/json_clean" / f"{sid}.json"),
        ),
        (
            "VoteNet",
            "severe_corruption",
            lambda sid: parse_votenet_json(ROOT / "outputs/external_baselines/votenet/json_severe" / f"{sid}.json"),
        ),
    ]

    for model, condition, loader in specs:
        row, scene_rows, gt_hist, pred_hist = evaluate_model(model, condition, loader)
        rows.append(row)
        for sr in scene_rows:
            sr = dict(sr)
            sr["model"] = model
            sr["condition"] = condition
            all_scene_rows.append(sr)

    write_csv(OUT_DIR / "external_native_metrics.csv", rows)
    write_csv(OUT_DIR / "external_native_per_scene.csv", all_scene_rows)

    cols = [
        "model", "condition", "scenes", "empty_scenes", "empty_rate",
        "gt_boxes", "pred_boxes",
        "presence_f1",
        "box_iou025_f1", "box_iou05_f1",
        "derived_relation_f1",
    ]
    md = "# External-Native Fairness Diagnostics\n\n"
    md += "These metrics are diagnostic and are more favorable to box-producing external baselines than the shared semantic JSON metric.\n\n"
    md += md_table(rows, cols)
    md += "\n## Interpretation\n\n"
    md += "- `presence_f1` measures coarse category presence from boxes.\n"
    md += "- `box_iou025_f1` and `box_iou05_f1` are detector-style greedy box matching diagnostics.\n"
    md += "- `derived_relation_f1` derives relations from predicted boxes for both SpatialLM and VoteNet using the same coarse heuristic.\n"
    md += "- LGS is not included here because it does not output boxes; this table complements, not replaces, the shared schema evaluation.\n"
    (OUT_DIR / "external_native_metrics.md").write_text(md, encoding="utf-8")

    print("wrote", OUT_DIR / "external_native_metrics.csv")
    print("wrote", OUT_DIR / "external_native_per_scene.csv")
    print("wrote", OUT_DIR / "external_native_metrics.md")


if __name__ == "__main__":
    main()
