from __future__ import annotations

import torch

from less_geometry_same_semantics.data.corruptions import (
    CorruptionFamilyConfig,
    CorruptionPipeline,
    PointCloudCorruptionConfig,
)


def test_corruption_pipeline_keeps_valid_point_cloud() -> None:
    points = torch.linspace(-1, 1, steps=300).reshape(100, 3)
    generator = torch.Generator().manual_seed(3)
    pipeline = CorruptionPipeline(
        PointCloudCorruptionConfig(
            seed=11,
            geometry_degradation=CorruptionFamilyConfig(enabled=True, severity="mild"),
            coordinate_perturbation=CorruptionFamilyConfig(enabled=True, severity="mild"),
            local_structural_corruption=CorruptionFamilyConfig(enabled=True, severity="mild"),
            token_point_compression=CorruptionFamilyConfig(enabled=True, severity="medium"),
        )
    )

    degraded = pipeline(points, generator=generator)

    assert degraded.ndim == 2
    assert degraded.shape[1] == 3
    assert 1 <= degraded.shape[0] <= points.shape[0]
    assert torch.isfinite(degraded).all()


def test_corruption_pipeline_is_deterministic_with_sample_seed() -> None:
    points = torch.randn(128, 3)
    pipeline = CorruptionPipeline(
        PointCloudCorruptionConfig(
            seed=5,
            coordinate_perturbation=CorruptionFamilyConfig(enabled=True, severity="severe"),
            token_point_compression=CorruptionFamilyConfig(enabled=True, severity="severe"),
        )
    )

    first = pipeline(points, sample_seed=10)
    second = pipeline(points, sample_seed=10)

    assert torch.equal(first, second)
