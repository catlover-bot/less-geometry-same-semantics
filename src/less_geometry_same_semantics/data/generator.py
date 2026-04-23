"""Controlled synthetic benchmark generator for paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from less_geometry_same_semantics.data.constants import (
    ATTRIBUTES,
    OBJECT_CATEGORIES,
    RELATION_PREDICATES,
    SCENE_TYPES,
)


@dataclass(frozen=True)
class SyntheticObjectSpec:
    """Prototype controlling one synthetic object category."""

    name: str
    scale: tuple[float, float, float]
    default_attributes: tuple[str, ...]


OBJECT_SPECS: dict[str, SyntheticObjectSpec] = {
    "chair": SyntheticObjectSpec("chair", (0.18, 0.18, 0.25), ("small",)),
    "table": SyntheticObjectSpec("table", (0.32, 0.22, 0.12), ("flat",)),
    "sofa": SyntheticObjectSpec("sofa", (0.42, 0.22, 0.18), ("large",)),
    "lamp": SyntheticObjectSpec("lamp", (0.08, 0.08, 0.40), ("tall",)),
    "plant": SyntheticObjectSpec("plant", (0.14, 0.14, 0.30), ("round",)),
    "cabinet": SyntheticObjectSpec("cabinet", (0.28, 0.14, 0.34), ("large", "tall")),
}


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    """Config for deterministic synthetic semantic scene generation."""

    num_points: int = 512
    min_instances: int = 2
    max_instances: int = 4
    allow_repeated_categories: bool = True
    scene_extent: float = 1.0
    z_min: float = 0.0
    z_max: float = 0.35
    left_right_threshold: float = 0.35
    front_back_threshold: float = 0.35
    near_threshold: float = 0.75
    max_relations: int = 8

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None, num_points: int) -> "SyntheticBenchmarkConfig":
        mapping = mapping or {}
        return cls(
            num_points=int(mapping.get("num_points", num_points)),
            min_instances=int(mapping.get("min_instances", 2)),
            max_instances=int(mapping.get("max_instances", 4)),
            allow_repeated_categories=bool(mapping.get("allow_repeated_categories", True)),
            scene_extent=float(mapping.get("scene_extent", 1.0)),
            z_min=float(mapping.get("z_min", 0.0)),
            z_max=float(mapping.get("z_max", 0.35)),
            left_right_threshold=float(mapping.get("left_right_threshold", 0.35)),
            front_back_threshold=float(mapping.get("front_back_threshold", 0.35)),
            near_threshold=float(mapping.get("near_threshold", 0.75)),
            max_relations=int(mapping.get("max_relations", 8)),
        )


class SyntheticSceneGenerator:
    """Generate deterministic point clouds with ground-truth structured semantics."""

    def __init__(self, config: SyntheticBenchmarkConfig | None = None) -> None:
        self.config = config or SyntheticBenchmarkConfig()

    def generate(self, index: int, seed: int) -> dict[str, Any]:
        """Generate one deterministic scene for a dataset index and base seed."""

        rng = np.random.default_rng(seed + index)
        scene_type = str(rng.choice(SCENE_TYPES))
        instance_count = int(rng.integers(self.config.min_instances, self.config.max_instances + 1))
        categories = self._sample_categories(rng, instance_count)
        centers = self._sample_centers(rng, instance_count)

        points_per_instance = self._split_points(self.config.num_points, instance_count)
        clusters = []
        for category, center, count in zip(categories, centers, points_per_instance, strict=True):
            spec = OBJECT_SPECS[category]
            cluster = rng.normal(loc=0.0, scale=np.asarray(spec.scale), size=(count, 3))
            clusters.append((cluster + center).astype(np.float32))

        clean_points = torch.from_numpy(np.concatenate(clusters, axis=0)).float()
        target = self._target(scene_type, categories, centers)
        return {
            "points": clean_points,
            "target": target,
            "metadata": {
                "synthetic_seed": seed + index,
                "instance_count": instance_count,
                "object_counts": target["object_counts"],
            },
        }

    def _sample_categories(self, rng: np.random.Generator, instance_count: int) -> list[str]:
        if self.config.allow_repeated_categories:
            categories = list(rng.choice(OBJECT_CATEGORIES, size=instance_count, replace=True))
            if len(set(categories)) == 1 and len(OBJECT_CATEGORIES) > 1:
                alternatives = [category for category in OBJECT_CATEGORIES if category != categories[0]]
                categories[-1] = str(rng.choice(alternatives))
            return [str(category) for category in categories]
        return [str(category) for category in rng.choice(OBJECT_CATEGORIES, size=instance_count, replace=False)]

    def _sample_centers(self, rng: np.random.Generator, instance_count: int) -> np.ndarray:
        centers = rng.uniform(
            low=-self.config.scene_extent,
            high=self.config.scene_extent,
            size=(instance_count, 3),
        ).astype(np.float32)
        centers[:, 2] = rng.uniform(low=self.config.z_min, high=self.config.z_max, size=instance_count)
        return centers

    @staticmethod
    def _split_points(total_points: int, instance_count: int) -> list[int]:
        base = total_points // instance_count
        counts = [base for _ in range(instance_count)]
        for i in range(total_points - base * instance_count):
            counts[i] += 1
        return counts

    def _target(self, scene_type: str, categories: list[str], centers: np.ndarray) -> dict[str, Any]:
        counts: dict[str, int] = {}
        category_attributes: dict[str, set[str]] = {}
        for category in categories:
            counts[category] = counts.get(category, 0) + 1
            category_attributes.setdefault(category, set()).update(OBJECT_SPECS[category].default_attributes)

        object_records = [
            {
                "category": category,
                "count": counts[category],
                "attributes": sorted(category_attributes[category]),
            }
            for category in OBJECT_CATEGORIES
            if category in counts
        ]
        attributes = sorted(
            {
                attribute
                for attrs in category_attributes.values()
                for attribute in attrs
                if attribute in ATTRIBUTES
            }
        )
        return {
            "objects": object_records,
            "object_counts": {category: counts[category] for category in sorted(counts)},
            "attributes": attributes,
            "relations": self._relations(categories, centers),
            "scene_type": scene_type,
            "caption": self._caption(scene_type, counts),
        }

    def _relations(self, categories: list[str], centers: np.ndarray) -> list[dict[str, str]]:
        relation_set: set[tuple[str, str, str]] = set()
        for i, subject in enumerate(categories):
            for j, obj in enumerate(categories):
                if i == j or subject == obj:
                    continue
                delta = centers[j] - centers[i]
                predicate: str | None = None
                if abs(delta[0]) > self.config.left_right_threshold and abs(delta[0]) >= abs(delta[1]):
                    predicate = "left_of" if delta[0] > 0 else "right_of"
                elif abs(delta[1]) > self.config.front_back_threshold:
                    predicate = "behind" if delta[1] > 0 else "in_front_of"
                elif np.linalg.norm(delta[:2]) < self.config.near_threshold:
                    predicate = "near"
                if predicate in RELATION_PREDICATES:
                    relation_set.add((str(subject), predicate, str(obj)))

        return [
            {"subject": subject, "predicate": predicate, "object": obj}
            for subject, predicate, obj in sorted(relation_set)[: self.config.max_relations]
        ]

    @staticmethod
    def _caption(scene_type: str, counts: dict[str, int]) -> str:
        parts = []
        for category in sorted(counts):
            count = counts[category]
            label = category if count == 1 else f"{category}s"
            parts.append(f"{count} {label}")
        object_text = ", ".join(parts[:-1]) + f" and {parts[-1]}" if len(parts) > 1 else parts[0]
        return f"A {scene_type} with {object_text}."
