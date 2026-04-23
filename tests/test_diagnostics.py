from __future__ import annotations

from less_geometry_same_semantics.data.diagnostics import dataset_diagnostics
from less_geometry_same_semantics.data.synthetic import SyntheticSceneDataset


def test_dataset_diagnostics_summarizes_split() -> None:
    dataset = SyntheticSceneDataset(num_samples=3, num_points=32, seed=5)

    summary = dataset_diagnostics(dataset)

    assert summary["split_size"] == 3
    assert summary["average_points_per_scene"] == 32
    assert summary["object_category_histogram"]
