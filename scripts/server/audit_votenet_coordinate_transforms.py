#!/usr/bin/env python
from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Any

import os
import numpy as np

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


def ids() -> list[str]:
    return [x.strip() for x in IDS_FILE.read_text().splitlines() if x.strip()]


def arr(x):
    return np.array([float(v) for v in x], dtype=float)


def normalize_label(label: Any) -> str | None:
    if label is None:
        return None
    mapped = map_object_category(str(label)) or str(label)
    return mapped if mapped in SUPPORTED else None


def center_dims_to_aabb(center: np.ndarray, dims: np.ndarray):
    dims = np.abs(dims)
    lo = center - dims / 2.0
    hi = center + dims / 2.0
    return lo, hi


def iou3d(a, b) -> float:
    alo, ahi = a
    blo, bhi = b
    inter = np.maximum(0.0, np.minimum(ahi, bhi) - np.maximum(alo, blo))
    inter_vol = float(np.prod(inter))
    va = float(np.prod(np.maximum(0.0, ahi - alo)))
    vb = float(np.prod(np.maximum(0.0, bhi - blo)))
    denom = va + vb - inter_vol
    return inter_vol / denom if denom > 0 else 0.0


def load_gt(scene_id: str):
    ann = discover_arkitscenes_annotation_file(ARKIT_ROOT, scene_id, "Validation", "3dod")
    boxes = _parse_arkitscenes_annotation_boxes(ann, require_mapped=True)
    out = []
    for b in boxes:
        label = b.get("category")
        if label not in SUPPORTED:
            continue
        c = arr(b["center"])
        d = arr(b["dims"])
        out.append({"label": label, "aabb": center_dims_to_aabb(c, d)})
    return out


def transform_center_dims(
    c: np.ndarray,
    d: np.ndarray,
    perm: tuple[int, int, int],
    signs: tuple[int, int, int],
):
    cc = c[list(perm)] * np.array(signs, dtype=float)
    dd = d[list(perm)]
    return cc, dd


def load_votenet(scene_id: str, perm: tuple[int, int, int], signs: tuple[int, int, int]):
    p = ROOT / f"outputs/external_baselines/votenet/json_clean/{scene_id}.json"
    obj = json.loads(p.read_text())
    out = []

    for det in obj.get("detections", []):
        label = normalize_label(det.get("label"))
        if label is None:
            continue

        c = arr(det["center"])
        d = arr(det.get("dimensions", det.get("size")))
        c2, d2 = transform_center_dims(c, d, perm, signs)

        out.append({
            "label": label,
            "aabb": center_dims_to_aabb(c2, d2),
            "score": float(det.get("score", 0.0)),
        })

    return sorted(out, key=lambda x: x["score"], reverse=True)


def match(pred, gt, thr: float):
    used = set()
    tp = 0

    for p in pred:
        best_i = None
        best = 0.0

        for i, g in enumerate(gt):
            if i in used:
                continue
            if p["label"] != g["label"]:
                continue

            s = iou3d(p["aabb"], g["aabb"])
            if s > best:
                best = s
                best_i = i

        if best_i is not None and best >= thr:
            tp += 1
            used.add(best_i)

    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(perm, signs):
    totals = {
        0.25: [0, 0, 0],
        0.50: [0, 0, 0],
    }

    for scene_id in ids():
        gt = load_gt(scene_id)
        pred = load_votenet(scene_id, perm, signs)

        for thr in totals:
            tp, fp, fn = match(pred, gt, thr)
            totals[thr][0] += tp
            totals[thr][1] += fp
            totals[thr][2] += fn

    row = {
        "perm": "".join(str(x) for x in perm),
        "signs": "".join("+" if s > 0 else "-" for s in signs),
    }

    for thr, vals in totals.items():
        p, r, f = prf(*vals)
        key = str(thr).replace(".", "")
        row[f"precision@{key}"] = p
        row[f"recall@{key}"] = r
        row[f"f1@{key}"] = f
        row[f"tp@{key}"] = vals[0]
        row[f"fp@{key}"] = vals[1]
        row[f"fn@{key}"] = vals[2]

    return row


def main():
    rows = []

    for perm in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((1, -1), repeat=3):
            rows.append(evaluate(perm, signs))

    rows = sorted(rows, key=lambda r: (r["f1@025"], r["f1@05"]), reverse=True)

    out_csv = OUT_DIR / "votenet_coordinate_transform_audit.csv"
    keys = list(rows[0].keys())
    out_csv.write_text(
        ",".join(keys) + "\n" +
        "\n".join(",".join(str(r[k]) for k in keys) for r in rows) + "\n",
        encoding="utf-8",
    )

    out_md = OUT_DIR / "votenet_coordinate_transform_audit.md"
    top = rows[:12]

    md = "# VoteNet Coordinate Transform Audit\n\n"
    md += "This audit tests axis permutations and sign flips for VoteNet detections before IoU matching against ARKitScenes GT boxes.\n\n"
    md += "| rank | perm | signs | f1@0.25 | f1@0.50 | tp@0.25 | fp@0.25 | fn@0.25 |\n"
    md += "|---:|---|---|---:|---:|---:|---:|---:|\n"

    for i, r in enumerate(top, 1):
        md += (
            f"| {i} | {r['perm']} | {r['signs']} | "
            f"{r['f1@025']:.4f} | {r['f1@05']:.4f} | "
            f"{r['tp@025']} | {r['fp@025']} | {r['fn@025']} |\n"
        )

    out_md.write_text(md, encoding="utf-8")

    print("wrote", out_csv)
    print("wrote", out_md)
    print(md)


if __name__ == "__main__":
    main()
