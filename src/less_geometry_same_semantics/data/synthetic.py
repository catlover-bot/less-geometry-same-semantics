"""Synthetic point-cloud dataset wrapper for the controlled benchmark generator."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from less_geometry_same_semantics.data.corruptions import CorruptionPipeline
from less_geometry_same_semantics.data.generator import SyntheticBenchmarkConfig, SyntheticSceneGenerator


class SyntheticSceneDataset(Dataset[dict[str, Any]]):
    """Generate small labeled scenes with deterministic per-index randomness.

    The dataset is deliberately lightweight. It exists to exercise the research
    pipeline before real point-cloud datasets are connected.
    """

    def __init__(
        self,
        num_samples: int = 256,
        num_points: int = 512,
        corruption: CorruptionPipeline | None = None,
        seed: int = 0,
        return_clean: bool = False,
        generator_config: SyntheticBenchmarkConfig | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.num_points = num_points
        self.corruption = corruption
        self.seed = seed
        self.return_clean = return_clean
        self.generator = SyntheticSceneGenerator(generator_config or SyntheticBenchmarkConfig(num_points=num_points))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch_generator = torch.Generator().manual_seed(self.seed + index)
        generated = self.generator.generate(index=index, seed=self.seed)
        clean_points = generated["points"]
        target = generated["target"]

        points = clean_points
        if self.corruption is not None:
            points = self.corruption(clean_points, generator=torch_generator, sample_seed=index)

        metadata = {
            "index": index,
            "clean_num_points": int(clean_points.shape[0]),
            "degraded_num_points": int(points.shape[0]),
            "corruption": self.corruption.describe() if self.corruption is not None else None,
            "generator": generated["metadata"],
        }
        sample: dict[str, Any] = {
            "points": points,
            "target": target,
            "metadata": metadata,
        }
        if self.return_clean:
            sample["clean_points"] = clean_points
        return sample
