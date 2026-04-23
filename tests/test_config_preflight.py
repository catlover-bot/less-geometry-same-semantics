from __future__ import annotations

import json

import numpy as np

from less_geometry_same_semantics.data.preflight import check_dataset_config
from less_geometry_same_semantics.data.public_datasets import discover_arkitscenes_annotation_file
from less_geometry_same_semantics.utils.config import find_unresolved_env_vars, load_config


def test_config_expands_braced_env_vars_on_windows_style_paths(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "arkitscenes"
    monkeypatch.setenv("ARKITSCENES_ROOT", str(dataset_root))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  root: ${ARKITSCENES_ROOT}\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["data"]["root"] == str(dataset_root)
    assert find_unresolved_env_vars(config) == []


def test_unresolved_config_env_vars_are_reported(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ARKITSCENES_ROOT", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  root: ${ARKITSCENES_ROOT}\n", encoding="utf-8")

    config = load_config(config_path)

    assert config["data"]["root"] == "${ARKITSCENES_ROOT}"
    assert find_unresolved_env_vars(config) == ["ARKITSCENES_ROOT"]


def test_arkitscenes_discovery_handles_3dod_layout(tmp_path) -> None:
    root = tmp_path / "arkitscenes"
    scene = root / "3dod" / "Training" / "47333462"
    scene.mkdir(parents=True)
    annotation = scene / "47333462_3dod_annotation.json"
    annotation.write_text(json.dumps({"data": []}), encoding="utf-8")

    assert discover_arkitscenes_annotation_file(root, "47333462", "Training") == annotation


def test_dataset_preflight_passes_minimal_arkitscenes_layout(tmp_path) -> None:
    root = tmp_path / "arkitscenes"
    train = root / "3dod" / "Training" / "47333462"
    val = root / "3dod" / "Validation" / "47333463"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    np.save(train / "47333462.npy", np.random.randn(8, 3).astype("float32"))
    np.save(val / "47333463.npy", np.random.randn(8, 3).astype("float32"))
    annotation = {
        "data": [
            {
                "label": "chair",
                "centroid": [0.0, 0.0, 0.5],
                "axesLengths": [0.5, 0.5, 1.0],
            }
        ]
    }
    (train / "47333462_3dod_annotation.json").write_text(json.dumps(annotation), encoding="utf-8")
    (val / "47333463_3dod_annotation.json").write_text(json.dumps(annotation), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
data:
  dataset: arkitscenes
  root: {root}
  subset: 3dod
  train_split: Training
  val_split: Validation
""",
        encoding="utf-8",
    )

    report = check_dataset_config(config)

    assert report["status"] == "pass"
