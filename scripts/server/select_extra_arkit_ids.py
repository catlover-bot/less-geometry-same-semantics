#!/usr/bin/env python
from __future__ import annotations

import csv
import os
from pathlib import Path

root = Path(os.environ["ARKITSCENES_ROOT"])
csv_path = root / "_official_arkitscenes/threedod/3dod_train_val_splits.csv"

target_train = int(os.environ.get("TARGET_TRAIN", "50"))
target_val = int(os.environ.get("TARGET_VAL", "30"))

def downloaded(split: str) -> set[str]:
    d = root / "3dod" / split
    if not d.exists():
        return set()
    return {p.name for p in d.iterdir() if p.is_dir() and p.name.isdigit()}

existing_train = downloaded("Training")
existing_val = downloaded("Validation")

train_ids = []
val_ids = []

with csv_path.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        vid = str(row["video_id"]).strip()
        fold = str(row["fold"]).strip()
        if fold == "Training":
            train_ids.append(vid)
        elif fold == "Validation":
            val_ids.append(vid)

need_train = max(0, target_train - len(existing_train))
need_val = max(0, target_val - len(existing_val))

extra_train = [vid for vid in train_ids if vid not in existing_train][:need_train]
extra_val = [vid for vid in val_ids if vid not in existing_val][:need_val]

out_dir = Path("outputs/setup/arkitscenes_expand")
out_dir.mkdir(parents=True, exist_ok=True)

(out_dir / "extra_training_ids.txt").write_text("\n".join(extra_train) + ("\n" if extra_train else ""))
(out_dir / "extra_validation_ids.txt").write_text("\n".join(extra_val) + ("\n" if extra_val else ""))

print("csv:", csv_path)
print("all train ids:", len(train_ids))
print("all val ids:", len(val_ids))
print("existing train:", len(existing_train))
print("existing val:", len(existing_val))
print("target train:", target_train)
print("target val:", target_val)
print("need train:", need_train)
print("need val:", need_val)
print("extra train:", len(extra_train), extra_train[:30])
print("extra val:", len(extra_val), extra_val[:30])
