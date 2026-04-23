from __future__ import annotations

import json

import numpy as np

from less_geometry_same_semantics.data.public_datasets import ARKitScenesDataset
from less_geometry_same_semantics.schemas.schema import is_valid_semantic_output


def test_arkitscenes_loader_normalizes_boxes_and_caches(tmp_path) -> None:
    root = tmp_path / "arkitscenes"
    scene = root / "3dod" / "Training" / "47333462"
    scene.mkdir(parents=True)
    np.save(scene / "47333462.npy", np.random.randn(16, 3).astype("float32"))
    (scene / "47333462_3dod_annotation.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "uid": "a",
                        "label": "chair",
                        "centroid": [0.0, 0.0, 0.5],
                        "axesLengths": [0.6, 0.6, 1.0],
                    },
                    {
                        "uid": "b",
                        "label": "table",
                        "centroid": [1.0, 0.0, 0.4],
                        "axesLengths": [1.2, 0.8, 0.2],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = ARKitScenesDataset(root=root, split="Training", cache_dir=tmp_path / "cache")
    sample = dataset[0]

    assert sample["points"].shape == (16, 3)
    assert dataset.preprocessing_summary["scenes_loaded"] == 1
    assert dataset.preprocessing_summary["scenes_skipped"] == 0
    assert is_valid_semantic_output(sample["target"])
    assert sample["target"]["object_counts"] == {"chair": 1, "table": 1}
    assert sample["target"]["relations"]


def test_arkitscenes_loader_uses_annotation_point_fallback(tmp_path) -> None:
    root = tmp_path / "arkitscenes"
    scene = root / "3dod" / "Validation" / "47333463"
    scene.mkdir(parents=True)
    (scene / "47333463_3dod_annotation.json").write_text(
        json.dumps(
            {
                "data": [
                    {
                        "label": "sofa",
                        "centroid": [0.0, 0.0, 0.5],
                        "axesLengths": [1.0, 0.8, 0.7],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = ARKitScenesDataset(root=root, split="Validation", cache_dir=tmp_path / "cache")
    sample = dataset[0]

    assert sample["points"].shape[1] == 3
    assert sample["metadata"]["graph_nodes"] == 1
    assert sample["target"]["scene_type"] == "living_room"
