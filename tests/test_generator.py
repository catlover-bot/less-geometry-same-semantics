from __future__ import annotations

import torch

from less_geometry_same_semantics.data.generator import SyntheticBenchmarkConfig, SyntheticSceneGenerator
from less_geometry_same_semantics.schemas.schema import is_valid_semantic_output


def test_synthetic_generator_is_deterministic_and_count_aware() -> None:
    generator = SyntheticSceneGenerator(
        SyntheticBenchmarkConfig(
            num_points=64,
            min_instances=5,
            max_instances=5,
            allow_repeated_categories=True,
        )
    )

    first = generator.generate(index=2, seed=11)
    second = generator.generate(index=2, seed=11)

    assert torch.equal(first["points"], second["points"])
    assert first["target"] == second["target"]
    assert sum(first["target"]["object_counts"].values()) == 5
    assert is_valid_semantic_output(first["target"])
