"""Small point-cloud IO helpers for public dataset adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_point_cloud(path: str | Path, max_points: int | None = None) -> torch.Tensor:
    """Load xyz points from npy, npz, txt/csv, or ply files."""

    point_path = Path(path)
    suffix = point_path.suffix.lower()
    if suffix == ".npy":
        array = np.load(point_path)
    elif suffix == ".npz":
        data = np.load(point_path)
        key = "points" if "points" in data else data.files[0]
        array = data[key]
    elif suffix in {".txt", ".csv", ".pts"}:
        array = np.loadtxt(point_path, delimiter="," if suffix == ".csv" else None)
    elif suffix == ".ply":
        array = _load_ply_xyz(point_path)
    else:
        raise ValueError(f"Unsupported point-cloud file: {point_path}")

    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError(f"Expected point array with at least 3 columns, got {array.shape} from {point_path}")
    points = torch.as_tensor(array[:, :3], dtype=torch.float32)
    if max_points is not None and points.shape[0] > max_points:
        indices = torch.linspace(0, points.shape[0] - 1, steps=max_points).long()
        points = points[indices]
    return points.contiguous()


def _load_ply_xyz(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise ImportError("Install plyfile to read PLY point clouds: pip install plyfile") from exc

    ply = PlyData.read(path)
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
