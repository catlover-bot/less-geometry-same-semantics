"""Research-grade point-cloud degradation operators.

Corruptions are grouped into benchmark categories:

- geometry degradation: global pose/scale changes
- coordinate perturbation: noise and quantization
- local structural corruption: holes and local point removal
- token/point compression: global point dropping and density reduction

Every family has a severity level and can be seeded deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

Severity = Literal["none", "mild", "medium", "severe", "extreme"]

SEVERITY_VALUES: dict[str, float] = {
    "none": 0.0,
    "mild": 0.25,
    "medium": 0.50,
    "severe": 0.75,
    "extreme": 1.0,
}

FAMILY_SEED_OFFSETS = {
    "geometry_degradation": 1_001,
    "coordinate_perturbation": 2_003,
    "local_structural_corruption": 3_007,
    "token_point_compression": 4_009,
}


@dataclass(frozen=True)
class CorruptionFamilyConfig:
    """Severity and parameter overrides for one corruption family."""

    enabled: bool = False
    severity: Severity = "none"
    seed: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "CorruptionFamilyConfig":
        mapping = mapping or {}
        severity = str(mapping.get("severity", "none"))
        if severity not in SEVERITY_VALUES:
            raise ValueError(f"Unknown corruption severity: {severity}")
        reserved = {"enabled", "severity", "seed", "params"}
        params = dict(mapping.get("params", {}))
        for key, value in mapping.items():
            if key not in reserved:
                params[key] = value
        return cls(
            enabled=bool(mapping.get("enabled", severity != "none")),
            severity=severity,  # type: ignore[arg-type]
            seed=mapping.get("seed"),
            params=params,
        )

    @property
    def intensity(self) -> float:
        if not self.enabled:
            return 0.0
        return SEVERITY_VALUES[self.severity]


@dataclass(frozen=True)
class PointCloudCorruptionConfig:
    """Composable corruption configuration with benchmark families."""

    preset: str = "custom"
    seed: int = 0
    geometry_degradation: CorruptionFamilyConfig = field(default_factory=CorruptionFamilyConfig)
    coordinate_perturbation: CorruptionFamilyConfig = field(default_factory=CorruptionFamilyConfig)
    local_structural_corruption: CorruptionFamilyConfig = field(default_factory=CorruptionFamilyConfig)
    token_point_compression: CorruptionFamilyConfig = field(default_factory=CorruptionFamilyConfig)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "PointCloudCorruptionConfig":
        """Build a family-based config from YAML or legacy flat fields."""

        mapping = mapping or {}
        legacy_keys = {
            "random_dropout_prob",
            "gaussian_noise_std",
            "rotate_z",
            "scale_min",
            "scale_max",
            "density_fraction",
            "quantization_bins",
        }
        if legacy_keys & set(mapping):
            return cls.from_legacy_mapping(mapping)

        return cls(
            preset=str(mapping.get("preset", "custom")),
            seed=int(mapping.get("seed", 0)),
            geometry_degradation=CorruptionFamilyConfig.from_mapping(mapping.get("geometry_degradation")),
            coordinate_perturbation=CorruptionFamilyConfig.from_mapping(mapping.get("coordinate_perturbation")),
            local_structural_corruption=CorruptionFamilyConfig.from_mapping(mapping.get("local_structural_corruption")),
            token_point_compression=CorruptionFamilyConfig.from_mapping(mapping.get("token_point_compression")),
        )

    @classmethod
    def from_legacy_mapping(cls, mapping: dict[str, Any]) -> "PointCloudCorruptionConfig":
        """Map the initial flat corruption fields onto benchmark families."""

        geometry_enabled = bool(mapping.get("rotate_z", False)) or mapping.get("scale_min", 1.0) != 1.0
        geometry_enabled = geometry_enabled or mapping.get("scale_max", 1.0) != 1.0
        coordinate_enabled = float(mapping.get("gaussian_noise_std", 0.0)) > 0.0 or mapping.get("quantization_bins") is not None
        compression_enabled = float(mapping.get("random_dropout_prob", 0.0)) > 0.0
        compression_enabled = compression_enabled or float(mapping.get("density_fraction", 1.0)) < 1.0

        return cls(
            preset=str(mapping.get("preset", "legacy")),
            seed=int(mapping.get("seed", 0)),
            geometry_degradation=CorruptionFamilyConfig(
                enabled=geometry_enabled,
                severity="medium" if geometry_enabled else "none",
                params={
                    "rotate_z": bool(mapping.get("rotate_z", False)),
                    "scale_min": float(mapping.get("scale_min", 1.0)),
                    "scale_max": float(mapping.get("scale_max", 1.0)),
                },
            ),
            coordinate_perturbation=CorruptionFamilyConfig(
                enabled=coordinate_enabled,
                severity="medium" if coordinate_enabled else "none",
                params={
                    "gaussian_noise_std": float(mapping.get("gaussian_noise_std", 0.0)),
                    "quantization_bins": mapping.get("quantization_bins"),
                },
            ),
            token_point_compression=CorruptionFamilyConfig(
                enabled=compression_enabled,
                severity="medium" if compression_enabled else "none",
                params={
                    "random_dropout_prob": float(mapping.get("random_dropout_prob", 0.0)),
                    "density_fraction": float(mapping.get("density_fraction", 1.0)),
                },
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description for experiment logs."""

        def family_to_dict(family: CorruptionFamilyConfig) -> dict[str, Any]:
            return {
                "enabled": family.enabled,
                "severity": family.severity,
                "seed": family.seed,
                "params": family.params,
            }

        return {
            "preset": self.preset,
            "seed": self.seed,
            "geometry_degradation": family_to_dict(self.geometry_degradation),
            "coordinate_perturbation": family_to_dict(self.coordinate_perturbation),
            "local_structural_corruption": family_to_dict(self.local_structural_corruption),
            "token_point_compression": family_to_dict(self.token_point_compression),
        }


def _ensure_points(points: torch.Tensor) -> torch.Tensor:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected point tensor with shape [N, 3], got {tuple(points.shape)}")
    if points.shape[0] == 0:
        raise ValueError("Point cloud must contain at least one point.")
    return points


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device) if device.type != "cpu" else torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _param(family: CorruptionFamilyConfig, name: str, default: Any) -> Any:
    return family.params.get(name, default)


def _quantization_bins(intensity: float) -> int | None:
    if intensity <= 0.0:
        return None
    if intensity <= 0.25:
        return 128
    if intensity <= 0.50:
        return 64
    if intensity <= 0.75:
        return 32
    return 16


def random_point_dropout(
    points: torch.Tensor,
    dropout_prob: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly remove points while keeping at least one point."""

    points = _ensure_points(points)
    if dropout_prob <= 0.0:
        return points
    if dropout_prob >= 1.0:
        keep_idx = torch.randint(points.shape[0], (1,), generator=generator, device=points.device)
        return points[keep_idx]

    keep = torch.rand(points.shape[0], generator=generator, device=points.device) > dropout_prob
    if not bool(keep.any()):
        keep[torch.randint(points.shape[0], (1,), generator=generator, device=points.device)] = True
    return points[keep]


def gaussian_noise(
    points: torch.Tensor,
    std: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add isotropic Gaussian coordinate noise."""

    points = _ensure_points(points)
    if std <= 0.0:
        return points
    noise = torch.randn(points.shape, generator=generator, device=points.device, dtype=points.dtype)
    return points + noise * std


def rotate_around_z(
    points: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply a random yaw rotation around the z-axis."""

    points = _ensure_points(points)
    angle = torch.rand((), generator=generator, device=points.device, dtype=points.dtype) * (2.0 * torch.pi)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    rotation = torch.stack(
        [
            torch.stack([cos_a, -sin_a, torch.zeros_like(cos_a)]),
            torch.stack([sin_a, cos_a, torch.zeros_like(cos_a)]),
            torch.stack([torch.zeros_like(cos_a), torch.zeros_like(cos_a), torch.ones_like(cos_a)]),
        ]
    )
    return points @ rotation.T


def random_scaling(
    points: torch.Tensor,
    scale_min: float,
    scale_max: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Uniformly scale all coordinates."""

    points = _ensure_points(points)
    if scale_min == 1.0 and scale_max == 1.0:
        return points
    if scale_min <= 0.0 or scale_max <= 0.0 or scale_min > scale_max:
        raise ValueError("Scale range must be positive and ordered.")
    alpha = torch.rand((), generator=generator, device=points.device, dtype=points.dtype)
    scale = scale_min + (scale_max - scale_min) * alpha
    return points * scale


def random_translation(
    points: torch.Tensor,
    std: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply a global random translation."""

    points = _ensure_points(points)
    if std <= 0.0:
        return points
    offset = torch.randn((1, 3), generator=generator, device=points.device, dtype=points.dtype) * std
    return points + offset


def density_reduction(
    points: torch.Tensor,
    fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Keep an exact random fraction of points, up to rounding."""

    points = _ensure_points(points)
    if fraction >= 1.0:
        return points
    if fraction <= 0.0:
        keep = 1
    else:
        keep = max(1, int(round(points.shape[0] * fraction)))
    permutation = torch.randperm(points.shape[0], generator=generator, device=points.device)
    return points[permutation[:keep]]


def coordinate_quantization(points: torch.Tensor, bins: int | None) -> torch.Tensor:
    """Quantize coordinates into a fixed number of bins per sample bounding box."""

    points = _ensure_points(points)
    if bins is None:
        return points
    if bins < 2:
        raise ValueError("quantization_bins must be at least 2.")

    min_xyz = points.min(dim=0).values
    max_xyz = points.max(dim=0).values
    span = torch.clamp(max_xyz - min_xyz, min=1e-6)
    normalized = (points - min_xyz) / span
    quantized = torch.round(normalized * (bins - 1)) / (bins - 1)
    return quantized * span + min_xyz


def local_structural_dropout(
    points: torch.Tensor,
    hole_count: int,
    radius: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Remove points around randomly selected centers to simulate local holes."""

    points = _ensure_points(points)
    if hole_count <= 0 or radius <= 0.0:
        return points

    keep = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    available_indices = torch.arange(points.shape[0], device=points.device)
    for _ in range(hole_count):
        if not bool(keep.any()):
            break
        kept_indices = available_indices[keep]
        center_idx = kept_indices[
            torch.randint(kept_indices.numel(), (1,), generator=generator, device=points.device)
        ]
        center = points[center_idx]
        distances = torch.linalg.norm(points - center, dim=1)
        keep = keep & (distances > radius)

    if not bool(keep.any()):
        fallback_idx = torch.randint(points.shape[0], (1,), generator=generator, device=points.device)
        keep[fallback_idx] = True
    return points[keep]


def occlusion_removal(
    points: torch.Tensor,
    removal_fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Remove a contiguous half-space slab to mimic viewpoint occlusion."""

    points = _ensure_points(points)
    if removal_fraction <= 0.0:
        return points
    direction = torch.randn(3, generator=generator, device=points.device, dtype=points.dtype)
    direction = direction / torch.linalg.norm(direction).clamp_min(1e-6)
    projection = points @ direction
    threshold = torch.quantile(projection, max(0.0, min(0.95, removal_fraction)))
    kept = points[projection >= threshold]
    if kept.shape[0] == 0:
        return points[:1]
    return kept


class CorruptionPipeline:
    """Applies family-based corruptions in a fixed benchmark order."""

    def __init__(self, config: PointCloudCorruptionConfig | None = None) -> None:
        self.config = config or PointCloudCorruptionConfig()

    def describe(self) -> dict[str, Any]:
        """Return a JSON-serializable corruption description."""

        return self.config.to_dict()

    def _family_generator(
        self,
        family_name: str,
        family: CorruptionFamilyConfig,
        device: torch.device,
        sample_seed: int | None,
        generator: torch.Generator | None,
    ) -> torch.Generator | None:
        if generator is not None and sample_seed is None and family.seed is None:
            return generator
        seed = self.config.seed + FAMILY_SEED_OFFSETS[family_name]
        if sample_seed is not None:
            seed += int(sample_seed) * 10_000
        if family.seed is not None:
            seed += int(family.seed)
        return _make_generator(device, seed)

    def __call__(
        self,
        points: torch.Tensor,
        generator: torch.Generator | None = None,
        sample_seed: int | None = None,
    ) -> torch.Tensor:
        degraded = _ensure_points(points)

        family = self.config.geometry_degradation
        intensity = family.intensity
        if intensity > 0.0:
            family_generator = self._family_generator("geometry_degradation", family, points.device, sample_seed, generator)
            if bool(_param(family, "rotate_z", True)):
                degraded = rotate_around_z(degraded, family_generator)
            scale_jitter = float(_param(family, "scale_jitter", 0.05 + 0.30 * intensity))
            scale_min = float(_param(family, "scale_min", 1.0 - scale_jitter))
            scale_max = float(_param(family, "scale_max", 1.0 + scale_jitter))
            degraded = random_scaling(degraded, scale_min, scale_max, family_generator)
            degraded = random_translation(
                degraded,
                float(_param(family, "translation_std", 0.02 * intensity)),
                family_generator,
            )

        family = self.config.coordinate_perturbation
        intensity = family.intensity
        if intensity > 0.0:
            family_generator = self._family_generator("coordinate_perturbation", family, points.device, sample_seed, generator)
            noise_std = float(_param(family, "gaussian_noise_std", 0.005 + 0.035 * intensity))
            degraded = gaussian_noise(degraded, noise_std, family_generator)
            bins = _param(family, "quantization_bins", _quantization_bins(intensity))
            degraded = coordinate_quantization(degraded, int(bins) if bins is not None else None)

        family = self.config.local_structural_corruption
        intensity = family.intensity
        if intensity > 0.0:
            family_generator = self._family_generator(
                "local_structural_corruption", family, points.device, sample_seed, generator
            )
            mode = str(_param(family, "mode", "scene"))
            multiplier = 2 if mode == "object" else 1
            hole_count = int(_param(family, "hole_count", max(1, round((1 + 3 * intensity) * multiplier))))
            radius = float(_param(family, "hole_radius", 0.08 + 0.24 * intensity if mode == "object" else 0.10 + 0.28 * intensity))
            degraded = local_structural_dropout(degraded, hole_count, radius, family_generator)
            if bool(_param(family, "occlusion", False)):
                degraded = occlusion_removal(
                    degraded,
                    float(_param(family, "occlusion_fraction", 0.10 + 0.35 * intensity)),
                    family_generator,
                )

        family = self.config.token_point_compression
        intensity = family.intensity
        if intensity > 0.0:
            family_generator = self._family_generator("token_point_compression", family, points.device, sample_seed, generator)
            dropout_prob = float(_param(family, "random_dropout_prob", min(0.85, 0.05 + 0.35 * intensity)))
            density_fraction = float(_param(family, "density_fraction", max(0.05, 1.0 - 0.85 * intensity)))
            degraded = random_point_dropout(degraded, dropout_prob, family_generator)
            degraded = density_reduction(degraded, density_fraction, family_generator)
            target_point_budget = _param(family, "target_point_budget", None)
            if target_point_budget is not None:
                budget_fraction = min(1.0, max(1.0, float(target_point_budget)) / max(1.0, float(degraded.shape[0])))
                degraded = density_reduction(degraded, budget_fraction, family_generator)

        return degraded.contiguous()
