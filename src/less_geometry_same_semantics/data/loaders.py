"""DataLoader helpers."""

from __future__ import annotations

from typing import Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from less_geometry_same_semantics.data.corruptions import (
    CorruptionPipeline,
    PointCloudCorruptionConfig,
)
from less_geometry_same_semantics.data.presets import DEFAULT_CORRUPTION_PRESETS
from less_geometry_same_semantics.data.generator import SyntheticBenchmarkConfig
from less_geometry_same_semantics.data.public_datasets import ARKitScenesDataset
from less_geometry_same_semantics.data.synthetic import SyntheticSceneDataset
from less_geometry_same_semantics.utils.config import expand_env_vars, find_unresolved_env_vars


def collate_point_cloud_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length point clouds and preserve JSON targets."""

    max_points = max(sample["points"].shape[0] for sample in samples)
    batch_size = len(samples)
    points = torch.zeros(batch_size, max_points, 3, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_points, dtype=torch.bool)

    for i, sample in enumerate(samples):
        point_cloud = sample["points"].float()
        count = point_cloud.shape[0]
        points[i, :count] = point_cloud
        mask[i, :count] = True

    return {
        "points": points,
        "mask": mask,
        "targets": [sample["target"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
    }


def _recursive_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _recursive_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_corruption_mapping(
    config: dict[str, Any] | None,
    presets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(config or {})
    preset_name = config.get("preset")
    preset_registry = _recursive_update(DEFAULT_CORRUPTION_PRESETS, presets or {})
    if not preset_name:
        return config
    if preset_name not in preset_registry:
        family_keys = {
            "geometry_degradation",
            "coordinate_perturbation",
            "local_structural_corruption",
            "token_point_compression",
        }
        if family_keys & set(config):
            return config
        raise KeyError(f"Unknown corruption preset '{preset_name}'. Available: {sorted(preset_registry)}")

    preset_entry = preset_registry[preset_name]
    preset_corruption = dict(preset_entry.get("corruption", preset_entry))
    overrides = {key: value for key, value in config.items() if key != "preset"}
    resolved = _recursive_update(preset_corruption, overrides)
    resolved["preset"] = str(preset_name)
    return resolved


def corruption_from_config(
    config: dict[str, Any] | None,
    presets: dict[str, Any] | None = None,
) -> CorruptionPipeline:
    """Build a corruption pipeline from a plain config dictionary."""

    resolved = _resolve_corruption_mapping(config, presets)
    corruption_config = PointCloudCorruptionConfig.from_mapping(resolved)
    return CorruptionPipeline(corruption_config)


def build_synthetic_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Create train/validation dataloaders for the synthetic baseline."""

    data_cfg = config.get("data", {})
    benchmark_cfg = config.get("benchmark", {})
    corruption = corruption_from_config(data_cfg.get("corruption"), benchmark_cfg.get("presets"))
    num_points = int(data_cfg.get("num_points", 512))
    generator_config = SyntheticBenchmarkConfig.from_mapping(data_cfg.get("synthetic"), num_points=num_points)
    batch_size = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 0))
    seed = int(config.get("seed", 0))

    train_dataset = SyntheticSceneDataset(
        num_samples=int(data_cfg.get("train_samples", 256)),
        num_points=num_points,
        corruption=corruption,
        seed=seed,
        generator_config=generator_config,
    )
    val_dataset = SyntheticSceneDataset(
        num_samples=int(data_cfg.get("val_samples", 64)),
        num_points=num_points,
        corruption=corruption,
        seed=seed + 10_000,
        generator_config=generator_config,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_point_cloud_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_point_cloud_samples,
    )
    return train_loader, val_loader


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders for synthetic, ARKitScenes, or legacy public configs."""

    data_cfg = config.get("data", {})
    dataset_name = str(data_cfg.get("dataset", "synthetic")).lower()
    if dataset_name == "synthetic":
        return build_synthetic_dataloaders(config)

    benchmark_cfg = config.get("benchmark", {})
    corruption = corruption_from_config(data_cfg.get("corruption"), benchmark_cfg.get("presets"))
    batch_size = int(data_cfg.get("batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 0))
    seed = int(config.get("seed", 0))
    root = data_cfg.get("root")
    if root is None:
        raise ValueError(
            "Public dataset configs require data.root. Set ARKITSCENES_ROOT in PowerShell "
            "or edit data.root in the YAML config."
        )
    unresolved = find_unresolved_env_vars(str(root))
    if unresolved:
        raise ValueError(
            f"Unresolved environment variable(s) in data.root: {', '.join(unresolved)}. "
            "PowerShell example: $env:ARKITSCENES_ROOT='C:\\datasets\\ARKitScenes'."
        )
    root_path = Path(str(expand_env_vars(str(root)))).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(
            f"Configured dataset root does not exist: {root_path}. "
            "Fix data.root in the YAML config or set the expected environment variable."
        )

    dataset_cls = {
        "arkitscenes": ARKitScenesDataset,
        "arkit_scenes": ARKitScenesDataset,
    }.get(dataset_name)
    if dataset_cls is None:
        raise ValueError(
            f"Unknown active dataset '{dataset_name}'. Expected synthetic or arkitscenes. "
            "Deprecated public-dataset configs are archived under configs/legacy/."
        )

    common_kwargs = {
        "root": root_path,
        "corruption": corruption,
        "cache_dir": data_cfg.get("cache_dir"),
        "max_points": data_cfg.get("max_points", data_cfg.get("num_points")),
        "seed": seed,
    }
    if dataset_cls is ARKitScenesDataset:
        common_kwargs["subset"] = data_cfg.get("subset", "3dod")
        common_kwargs["allow_annotation_point_fallback"] = bool(data_cfg.get("allow_annotation_point_fallback", True))

    train_dataset = dataset_cls(
        split=str(data_cfg.get("train_split", "train")),
        limit=data_cfg.get("train_samples"),
        **common_kwargs,
    )
    val_dataset = dataset_cls(
        split=str(data_cfg.get("val_split", "validation")),
        limit=data_cfg.get("val_samples"),
        **common_kwargs,
    )
    if len(train_dataset) == 0:
        raise ValueError(
            f"No train samples loaded for dataset '{dataset_name}' from {root_path}. "
            "Run scripts/check_dataset_setup.py for split and annotation diagnostics."
        )
    if len(val_dataset) == 0:
        raise ValueError(
            f"No validation samples loaded for dataset '{dataset_name}' from {root_path}. "
            "Check data.val_split, annotation split names, and malformed-sample warnings."
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_point_cloud_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_point_cloud_samples,
    )
    return train_loader, val_loader
