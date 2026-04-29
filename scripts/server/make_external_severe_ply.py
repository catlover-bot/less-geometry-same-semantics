#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


def make_severe_xyz(xyz: np.ndarray, seed: int, noise_std: float = 0.03, bins: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = xyz.astype(np.float32).copy()

    # coordinate perturbation
    pts = pts + rng.normal(0.0, noise_std, size=pts.shape).astype(np.float32)

    # quantization per scene bounding box
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    q = np.round((pts - lo) / scale * (bins - 1))
    pts = lo + q / (bins - 1) * scale
    return pts.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-ply", required=True)
    ap.add_argument("--output-ply", required=True)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--noise-std", type=float, default=0.03)
    ap.add_argument("--bins", type=int, default=32)
    args = ap.parse_args()

    inp = Path(args.input_ply)
    out = Path(args.output_ply)
    out.parent.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(str(inp))
    v = ply["vertex"].data
    names = v.dtype.names

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    severe_xyz = make_severe_xyz(xyz, seed=args.seed, noise_std=args.noise_std, bins=args.bins)

    dtype = []
    arrays = []

    for name in names:
        dtype.append((name, v.dtype[name]))
        if name == "x":
            arrays.append(severe_xyz[:, 0].astype(v.dtype[name]))
        elif name == "y":
            arrays.append(severe_xyz[:, 1].astype(v.dtype[name]))
        elif name == "z":
            arrays.append(severe_xyz[:, 2].astype(v.dtype[name]))
        else:
            arrays.append(v[name])

    out_arr = np.empty(len(v), dtype=dtype)
    for name, arr in zip(names, arrays):
        out_arr[name] = arr

    PlyData([PlyElement.describe(out_arr, "vertex")], text=False).write(str(out))
    print(f"wrote {out} vertices={len(out_arr)}")


if __name__ == "__main__":
    main()
