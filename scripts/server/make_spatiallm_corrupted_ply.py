#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


def corrupt_xyz(xyz: np.ndarray, seed: int, noise: float, quant: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = xyz.astype(np.float32).copy()

    center = out.mean(axis=0, keepdims=True)
    scale = float(np.max(np.ptp(out, axis=0)))
    if scale <= 0:
        scale = 1.0

    # Coordinate perturbation.
    out = out + rng.normal(0.0, noise * scale, size=out.shape).astype(np.float32)

    # Coarse quantization.
    step = quant * scale
    if step > 0:
        out = np.round((out - center) / step) * step + center

    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--noise", type=float, default=0.035)
    ap.add_argument("--quant", type=float, default=0.025)
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(str(inp))
    vertex = ply["vertex"].data.copy()

    for key in ["x", "y", "z"]:
        if key not in vertex.dtype.names:
            raise ValueError(f"missing vertex field: {key}")

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    corrupted = corrupt_xyz(xyz, args.seed, args.noise, args.quant)

    vertex["x"] = corrupted[:, 0]
    vertex["y"] = corrupted[:, 1]
    vertex["z"] = corrupted[:, 2]

    elements = []
    for element in ply.elements:
        if element.name == "vertex":
            elements.append(PlyElement.describe(vertex, "vertex"))
        else:
            elements.append(element)

    PlyData(elements, text=ply.text, byte_order=ply.byte_order).write(str(outp))
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
