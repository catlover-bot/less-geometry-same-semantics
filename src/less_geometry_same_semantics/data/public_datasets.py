"""ARKitScenes public dataset adapter.

The active public benchmark target is ARKitScenes 3DOD. Legacy 3RScan/3DSSG
and ScanNet adapters are kept in ``legacy_public_datasets.py`` for explicit
reproducibility, but they are not part of the default workflow.
"""

from __future__ import annotations

import csv
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from less_geometry_same_semantics.data.cache import load_or_build_cached_example
from less_geometry_same_semantics.data.corruptions import CorruptionPipeline
from less_geometry_same_semantics.data.graph_conversion import target_to_scene_graph
from less_geometry_same_semantics.data.label_mapping import map_object_category, normalize_label
from less_geometry_same_semantics.data.point_io import load_point_cloud
from less_geometry_same_semantics.data.semantic import ObjectRecord, RelationRecord, SceneGraph
from less_geometry_same_semantics.utils.config import expand_env_vars, find_unresolved_env_vars


class PublicSceneGraphDataset(Dataset[dict[str, Any]]):
    """Base class for public point-cloud/scene-graph datasets."""

    dataset_name = "public"

    def __init__(
        self,
        root: str | Path,
        split: str = "Training",
        corruption: CorruptionPipeline | None = None,
        cache_dir: str | Path | None = None,
        max_points: int | None = 80_000,
        seed: int = 0,
        limit: int | None = None,
        skip_malformed: bool = True,
    ) -> None:
        unresolved = find_unresolved_env_vars(str(root))
        if unresolved:
            raise ValueError(
                f"Unresolved environment variable(s) in dataset root: {', '.join(unresolved)}. "
                "Set them before running, e.g. PowerShell: "
                "$env:ARKITSCENES_ROOT='C:\\datasets\\ARKitScenes'."
            )
        self.root = Path(str(expand_env_vars(str(root)))).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {self.root}. "
                "Set ARKITSCENES_ROOT or edit data.root in configs/arkitscenes.yaml."
            )
        self.split = split
        self.corruption = corruption
        self.cache_dir = cache_dir
        self.max_points = max_points
        self.seed = seed
        self.skip_malformed = skip_malformed
        discovered_scene_ids = self._discover_scene_ids()
        self.skipped_scenes: list[dict[str, str]] = []
        self.scene_ids = self._filter_scene_ids(discovered_scene_ids) if skip_malformed else discovered_scene_ids
        if limit is not None:
            self.scene_ids = self.scene_ids[:limit]
        self.preprocessing_summary = self._build_preprocessing_summary(discovered_scene_ids)

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_id = self.scene_ids[index]

        def build() -> dict[str, Any]:
            points = self._load_points(scene_id)
            graph = self._load_graph(scene_id)
            target = graph.to_target()
            return {
                "points": points,
                "target": target,
                "graph": graph,
                "metadata": {
                    "dataset": self.dataset_name,
                    "scene_id": scene_id,
                    "index": index,
                    "clean_num_points": int(points.shape[0]),
                    "degraded_num_points": int(points.shape[0]),
                    "corruption": None,
                    "graph_nodes": len(graph.nodes),
                    "graph_edges": len(graph.edges),
                    **graph.metadata,
                },
            }

        sample = load_or_build_cached_example(
            self.cache_dir,
            {
                "dataset": self.dataset_name,
                "root": str(self.root),
                "split": self.split,
                "scene_id": scene_id,
                "max_points": self.max_points,
                "version": 3,
            },
            build,
        )
        clean_points = sample["points"]
        points = clean_points
        if self.corruption is not None:
            generator = torch.Generator().manual_seed(self.seed + index)
            points = self.corruption(clean_points, generator=generator, sample_seed=index)
        metadata = dict(sample["metadata"])
        metadata["degraded_num_points"] = int(points.shape[0])
        metadata["corruption"] = self.corruption.describe() if self.corruption is not None else None
        return {
            "points": points,
            "target": sample["target"],
            "graph": sample.get("graph", target_to_scene_graph(sample["target"], metadata)),
            "metadata": metadata,
        }

    def _discover_scene_ids(self) -> list[str]:
        raise NotImplementedError

    def _load_points(self, scene_id: str) -> torch.Tensor:
        raise NotImplementedError

    def _load_graph(self, scene_id: str) -> SceneGraph:
        raise NotImplementedError

    def _point_candidates(self, scene_id: str) -> list[str]:
        return [f"points/{scene_id}.npy", f"points/{scene_id}.npz", f"{scene_id}.npy", f"{scene_id}.npz"]

    def _find_point_file(self, scene_id: str, candidates: Iterable[str]) -> Path:
        return find_point_file(self.root, scene_id, candidates)

    def _filter_scene_ids(self, scene_ids: list[str]) -> list[str]:
        valid_scene_ids = []
        for scene_id in scene_ids:
            try:
                self._find_point_file(scene_id, self._point_candidates(scene_id))
                graph = self._load_graph(scene_id)
                if not graph.nodes:
                    raise ValueError("no mapped object nodes")
                valid_scene_ids.append(scene_id)
            except Exception as exc:
                reason = str(exc)
                self.skipped_scenes.append({"scene_id": scene_id, "reason": reason})
                warnings.warn(f"Skipping malformed {self.dataset_name} scene '{scene_id}': {reason}", stacklevel=2)
        return valid_scene_ids

    def _build_preprocessing_summary(self, discovered_scene_ids: list[str]) -> dict[str, Any]:
        object_counts = []
        relation_counts = []
        object_hist: Counter[str] = Counter()
        relation_hist: Counter[str] = Counter()
        for scene_id in self.scene_ids:
            try:
                graph = self._load_graph(scene_id)
            except Exception:
                continue
            object_counts.append(len(graph.nodes))
            relation_counts.append(len(graph.edges))
            object_hist.update(node.category for node in graph.nodes)
            relation_hist.update(edge.predicate for edge in graph.edges)
        return {
            "dataset": self.dataset_name,
            "split": self.split,
            "scenes_discovered": len(discovered_scene_ids),
            "scenes_loaded": len(self.scene_ids),
            "scenes_skipped": len(self.skipped_scenes),
            "skipped": self.skipped_scenes,
            "object_count_stats": _summary_stats(object_counts),
            "relation_count_stats": _summary_stats(relation_counts),
            "object_category_histogram": dict(sorted(object_hist.items())),
            "relation_category_histogram": dict(sorted(relation_hist.items())),
            "label_coverage_statistics": self._label_coverage_stats(object_hist, relation_hist),
        }

    def _label_coverage_stats(self, object_hist: Counter[str], relation_hist: Counter[str]) -> dict[str, Any]:
        return {
            "mapped_object_labels": int(sum(object_hist.values())),
            "mapped_relation_labels": int(sum(relation_hist.values())),
            "unique_object_labels": len(object_hist),
            "unique_relation_labels": len(relation_hist),
        }


class ARKitScenesDataset(PublicSceneGraphDataset):
    """Adapter for ARKitScenes 3DOD scans with derived coarse scene graphs."""

    dataset_name = "arkitscenes"

    def __init__(
        self,
        root: str | Path,
        split: str = "Training",
        subset: str = "3dod",
        allow_annotation_point_fallback: bool = True,
        **kwargs: Any,
    ) -> None:
        self.subset = normalize_arkitscenes_subset(subset)
        self.allow_annotation_point_fallback = allow_annotation_point_fallback
        super().__init__(root=root, split=normalize_arkitscenes_split(split), **kwargs)

    def _discover_scene_ids(self) -> list[str]:
        scene_ids = discover_arkitscenes_scene_ids(self.root, self.split, self.subset)
        if not scene_ids:
            raise FileNotFoundError(
                f"No ARKitScenes scenes found for subset='{self.subset}' split='{self.split}' under {self.root}. "
                "Expected downloaded scenes under 3dod/Training or 3dod/Validation, "
                "or an official split CSV such as threedod/3dod_train_val_splits.csv. "
                "Run scripts/check_arkitscenes_setup.py for a focused setup report."
            )
        return scene_ids

    def _filter_scene_ids(self, scene_ids: list[str]) -> list[str]:
        valid_scene_ids = []
        for scene_id in scene_ids:
            try:
                graph = self._load_graph(scene_id)
                if not graph.nodes:
                    raise ValueError("no mapped object nodes")
                point_file = discover_arkitscenes_point_file(self.root, scene_id, self.split, self.subset)
                if point_file is None and not self.allow_annotation_point_fallback:
                    raise FileNotFoundError("missing point cloud/mesh and annotation fallback is disabled")
                valid_scene_ids.append(scene_id)
            except Exception as exc:
                reason = str(exc)
                self.skipped_scenes.append({"scene_id": scene_id, "reason": reason})
                warnings.warn(f"Skipping malformed {self.dataset_name} scene '{scene_id}': {reason}", stacklevel=2)
        return valid_scene_ids

    def _load_points(self, scene_id: str) -> torch.Tensor:
        point_file = discover_arkitscenes_point_file(self.root, scene_id, self.split, self.subset)
        if point_file is not None:
            return load_point_cloud(point_file, self.max_points)
        if not self.allow_annotation_point_fallback:
            raise FileNotFoundError(
                f"Missing ARKitScenes point cloud/mesh for scene '{scene_id}'. "
                "Expected <scene_id>_3dod_mesh.ply, prepared *_pc.npy files, or raw mesh/point-cloud PLY assets."
            )
        boxes = _parse_arkitscenes_annotation_boxes(self._annotation_path(scene_id), require_mapped=False)
        points = _points_from_arkitscenes_boxes(boxes)
        if self.max_points is not None and points.shape[0] > self.max_points:
            indices = torch.linspace(0, points.shape[0] - 1, steps=self.max_points).long()
            points = points[indices]
        return points.contiguous()

    def _load_graph(self, scene_id: str) -> SceneGraph:
        annotation_path = self._annotation_path(scene_id)
        boxes = _parse_arkitscenes_annotation_boxes(annotation_path, require_mapped=True)
        if not boxes:
            raise ValueError(f"No mappable object boxes in ARKitScenes annotation: {annotation_path}")
        category_counts: dict[str, int] = {}
        category_attributes: dict[str, set[str]] = {}
        for box in boxes:
            category_counts[box["category"]] = category_counts.get(box["category"], 0) + 1
            category_attributes.setdefault(box["category"], set()).update(box["attributes"])
        nodes = [
            ObjectRecord(category=category, count=count, attributes=tuple(sorted(category_attributes.get(category, set()))))
            for category, count in sorted(category_counts.items())
        ]
        point_file = discover_arkitscenes_point_file(self.root, scene_id, self.split, self.subset)
        return SceneGraph(
            nodes=tuple(nodes),
            edges=tuple(derive_relations_from_boxes(boxes)),
            scene_type=infer_arkitscenes_scene_type(boxes),
            caption=f"An ARKitScenes scene with {len(nodes)} coarse object categories.",
            metadata={
                "dataset": self.dataset_name,
                "scene_id": scene_id,
                "subset": self.subset,
                "split": self.split,
                "annotation_path": str(annotation_path),
                "point_path": str(point_file) if point_file else None,
                "relation_source": "bbox_heuristic",
            },
        )

    def _annotation_path(self, scene_id: str) -> Path:
        path = discover_arkitscenes_annotation_file(self.root, scene_id, self.split, self.subset)
        if path is None:
            raise FileNotFoundError(
                f"Missing ARKitScenes 3DOD annotation for scene '{scene_id}'. "
                "Expected <scene_id>_3dod_annotation.json inside the scene directory."
            )
        return path

    def _point_candidates(self, scene_id: str) -> list[str]:
        return [str(path.relative_to(self.root)) for path in candidate_arkitscenes_point_paths(self.root, scene_id, self.split, self.subset)]


def find_point_file(root: Path, scene_id: str, candidates: Iterable[str]) -> Path:
    checked = []
    for candidate in candidates:
        path = root / candidate
        checked.append(path)
        if path.exists():
            return path
    matches = list(root.rglob(f"{scene_id}*.npy")) + list(root.rglob(f"{scene_id}*.npz")) + list(root.rglob(f"{scene_id}*.ply"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find point cloud for scene '{scene_id}' under {root}. "
        f"Checked candidates: {[str(path) for path in checked[:12]]}"
    )


def normalize_arkitscenes_subset(subset: str) -> str:
    normalized = str(subset).strip().lower()
    aliases = {"threedod": "3dod", "depth_upsampling": "upsampling"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"3dod", "raw", "upsampling"}:
        raise ValueError("ARKitScenes subset must be one of: 3dod, raw, upsampling.")
    return normalized


def normalize_arkitscenes_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized in {"train", "training"}:
        return "Training"
    if normalized in {"val", "valid", "validation"}:
        return "Validation"
    raise ValueError("ARKitScenes split must be Training or Validation.")


def arkitscenes_csv_candidates(root: Path, subset: str) -> list[Path]:
    subset = normalize_arkitscenes_subset(subset)
    if subset == "3dod":
        return [root / "threedod" / "3dod_train_val_splits.csv", root / "3dod_train_val_splits.csv", root / "3dod" / "3dod_train_val_splits.csv"]
    if subset == "upsampling":
        return [root / "depth_upsampling" / "upsampling_train_val_splits.csv", root / "upsampling_train_val_splits.csv", root / "upsampling" / "upsampling_train_val_splits.csv"]
    return [root / "raw" / "raw_train_val_splits.csv", root / "raw_train_val_splits.csv"]


def discover_arkitscenes_scene_ids(root: Path, split: str, subset: str = "3dod") -> list[str]:
    split = normalize_arkitscenes_split(split)
    subset = normalize_arkitscenes_subset(subset)
    scene_ids: set[str] = set()
    csv_path = _first_existing(arkitscenes_csv_candidates(root, subset))
    if csv_path is not None:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                fold = str(row.get("fold") or row.get("split") or "").strip()
                if fold and normalize_arkitscenes_split(fold) != split:
                    continue
                video_id = row.get("video_id") or row.get("videoId") or row.get("scene_id")
                if video_id:
                    scene_ids.add(str(video_id).strip())
    for directory in arkitscenes_split_directories(root, split, subset):
        if directory.exists():
            scene_ids.update(path.name for path in directory.iterdir() if path.is_dir())
    return sorted(scene_ids)


def arkitscenes_split_directories(root: Path, split: str, subset: str = "3dod") -> list[Path]:
    split = normalize_arkitscenes_split(split)
    subset = normalize_arkitscenes_subset(subset)
    subset_dirs = [subset]
    if subset == "3dod":
        subset_dirs.append("threedod")
    elif subset == "upsampling":
        subset_dirs.append("depth_upsampling")
    return [*(root / name / split for name in subset_dirs), root / split]


def arkitscenes_scene_directories(root: Path, scene_id: str, split: str, subset: str = "3dod") -> list[Path]:
    return [directory / scene_id for directory in arkitscenes_split_directories(root, split, subset)] + [root / scene_id]


def candidate_arkitscenes_annotation_paths(root: Path, scene_id: str, split: str, subset: str = "3dod") -> list[Path]:
    return [
        *(scene_dir / f"{scene_id}_3dod_annotation.json" for scene_dir in arkitscenes_scene_directories(root, scene_id, split, subset)),
        root / "annotations" / f"{scene_id}_3dod_annotation.json",
    ]


def discover_arkitscenes_annotation_file(root: Path, scene_id: str, split: str, subset: str = "3dod") -> Path | None:
    path = _first_existing(candidate_arkitscenes_annotation_paths(root, scene_id, split, subset))
    if path is not None:
        return path
    for scene_dir in arkitscenes_scene_directories(root, scene_id, split, subset):
        if scene_dir.exists():
            matches = sorted(scene_dir.rglob("*annotation*.json"))
            if matches:
                return matches[0]
    return None


def candidate_arkitscenes_point_paths(root: Path, scene_id: str, split: str, subset: str = "3dod") -> list[Path]:
    candidates: list[Path] = []
    for scene_dir in arkitscenes_scene_directories(root, scene_id, split, subset):
        candidates.extend(
            [
                scene_dir / f"{scene_id}_offline_prepared_data" / f"{scene_id}_data" / f"{scene_id}_pc.npy",
                scene_dir / f"{scene_id}_3dod_mesh.ply",
                scene_dir / f"{scene_id}_mesh.ply",
                scene_dir / "mesh.ply",
                scene_dir / f"{scene_id}.ply",
                scene_dir / f"{scene_id}.npy",
                scene_dir / f"{scene_id}.npz",
            ]
        )
    candidates.extend([root / "points" / f"{scene_id}.npy", root / "points" / f"{scene_id}.npz"])
    return candidates


def discover_arkitscenes_point_file(root: Path, scene_id: str, split: str, subset: str = "3dod") -> Path | None:
    path = _first_existing(candidate_arkitscenes_point_paths(root, scene_id, split, subset))
    if path is not None:
        return path
    for scene_dir in arkitscenes_scene_directories(root, scene_id, split, subset):
        if scene_dir.exists():
            prepared = sorted(scene_dir.rglob("*_pc.npy"))
            if prepared:
                return prepared[0]
            ply_matches = sorted(scene_dir.rglob("*.ply"))
            if ply_matches:
                return ply_matches[0]
    return None


def derive_relations_from_boxes(boxes: list[dict[str, Any]], limit: int = 16) -> list[RelationRecord]:
    relations: set[tuple[str, str, str]] = set()
    mapped_boxes = [box for box in boxes if box["category"] != "unknown"]
    for i, left in enumerate(mapped_boxes):
        for j, right in enumerate(mapped_boxes):
            if i == j or left["category"] == right["category"]:
                continue
            predicate = _coarse_relation_from_bboxes(left, right)
            if predicate is not None:
                relations.add((left["category"], predicate, right["category"]))
            if len(relations) >= limit:
                break
        if len(relations) >= limit:
            break
    return [RelationRecord(subject=s, predicate=p, object=o) for s, p, o in sorted(relations)]


def infer_arkitscenes_scene_type(boxes: list[dict[str, Any]]) -> str:
    labels = {box["raw_label"] for box in boxes}
    if labels & {"sofa", "tv", "television", "fireplace"}:
        return "living_room"
    if labels & {"desk", "office chair"}:
        return "office"
    if labels and labels <= {"storage", "storage cabinet", "shelf", "shelves", "cabinet"}:
        return "storage"
    return "room"


def _parse_arkitscenes_annotation_boxes(path: Path, require_mapped: bool) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_boxes = payload.get("data", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    boxes: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_boxes):
        if not isinstance(raw, dict):
            continue
        label = raw.get("label") or raw.get("category") or raw.get("type")
        category = map_object_category(label)
        if category is None and require_mapped:
            continue
        box_source = _arkitscenes_box_source(raw)
        center = _vector3(
            raw.get("centroid")
            or raw.get("center")
            or raw.get("translation")
            or box_source.get("centroid")
            or box_source.get("center")
        )
        dims = _vector3(
            raw.get("axesLengths")
            or raw.get("dimensions")
            or raw.get("size")
            or box_source.get("axesLengths")
            or box_source.get("dimensions"),
            default=1.0,
        )
        if center is None or dims is None:
            continue
        dims = torch.clamp(torch.abs(dims), min=1e-3)
        half = dims / 2.0
        boxes.append(
            {
                "raw_label": normalize_label(label),
                "category": category or "unknown",
                "instance_id": str(raw.get("uid") or raw.get("id") or index),
                "center": center,
                "dims": dims,
                "min": center - half,
                "max": center + half,
                "attributes": _attributes_from_box_dims(dims),
            }
        )
    return boxes


def _arkitscenes_box_source(raw: dict[str, Any]) -> dict[str, Any]:
    segments = raw.get("segments")
    if not isinstance(segments, dict):
        return {}
    aligned = segments.get("obbAligned")
    if isinstance(aligned, dict):
        return aligned
    obb = segments.get("obb")
    return obb if isinstance(obb, dict) else {}


def _vector3(value: Any, default: float | None = None) -> torch.Tensor | None:
    if value is None:
        if default is None:
            return None
        return torch.full((3,), float(default), dtype=torch.float32)
    if isinstance(value, dict):
        ordered = [value.get(key) for key in ("x", "y", "z")]
    else:
        ordered = list(value) if isinstance(value, (list, tuple)) else []
    if len(ordered) < 3 or any(item is None for item in ordered[:3]):
        return None
    return torch.tensor([float(ordered[0]), float(ordered[1]), float(ordered[2])], dtype=torch.float32)


def _attributes_from_box_dims(dims: torch.Tensor) -> tuple[str, ...]:
    x, y, z = (float(v) for v in dims)
    volume = x * y * z
    attrs = []
    if volume < 0.25:
        attrs.append("small")
    if volume > 2.0:
        attrs.append("large")
    if z > max(x, y) * 1.3:
        attrs.append("tall")
    if z < max(x, y) * 0.35:
        attrs.append("flat")
    return tuple(sorted(set(attrs)))


def _points_from_arkitscenes_boxes(boxes: list[dict[str, Any]]) -> torch.Tensor:
    points = []
    for box in boxes:
        center = box["center"]
        half = box["dims"] / 2.0
        points.append(center)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    points.append(center + half * torch.tensor([sx, sy, sz], dtype=torch.float32))
    if not points:
        return torch.zeros(1, 3, dtype=torch.float32)
    return torch.stack(points).float()


def _coarse_relation_from_bboxes(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> str | None:
    delta = right["center"] - left["center"]
    xy_overlap = _axis_overlap(left, right, axis=0) and _axis_overlap(left, right, axis=1)
    z_overlap = _axis_overlap(left, right, axis=2)
    if xy_overlap and z_overlap:
        return "overlapping"
    if xy_overlap and abs(float(delta[2])) > 0.20:
        return "below" if float(delta[2]) > 0 else "above"
    if abs(float(delta[0])) > 0.35 and abs(float(delta[0])) >= abs(float(delta[1])):
        return "left_of" if float(delta[0]) > 0 else "right_of"
    if abs(float(delta[1])) > 0.35:
        return "behind" if float(delta[1]) > 0 else "in_front_of"
    if float(torch.linalg.norm(delta[:2])) < 0.75:
        return "near"
    return None


def _axis_overlap(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], axis: int) -> bool:
    return bool(left["min"][axis] <= right["max"][axis] and right["min"][axis] <= left["max"][axis])


def _summary_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": float(min(values)), "max": float(max(values)), "mean": float(sum(values) / len(values))}


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
