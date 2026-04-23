"""Public dataset adapters.

These loaders do not download data. They expect users to obtain datasets under
the official terms and point YAML configs at local roots. Each adapter converts
dataset-specific annotations into the shared semantic target and scene-graph
format used by the benchmark.

ARKitScenes is the active primary public benchmark. 3RScan/3DSSG and ScanNet
classes are retained below as legacy adapters for reproducibility only.
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
from less_geometry_same_semantics.data.label_mapping import (
    map_attributes,
    map_object_category,
    map_relation_predicate,
    normalize_label,
    map_scene_type,
)
from less_geometry_same_semantics.data.point_io import load_point_cloud
from less_geometry_same_semantics.data.semantic import ObjectRecord, RelationRecord, SceneGraph
from less_geometry_same_semantics.utils.config import expand_env_vars, find_unresolved_env_vars


class PublicSceneGraphDataset(Dataset[dict[str, Any]]):
    """Base class for public point-cloud/scene-graph datasets."""

    dataset_name = "public"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
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
                "Set the matching environment variable or edit data.root in the YAML config."
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
                "version": 2,
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
        return [
            f"points/{scene_id}.npy",
            f"points/{scene_id}.npz",
            f"{scene_id}.npy",
            f"{scene_id}.npz",
        ]

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

    def _label_coverage_stats(
        self,
        object_hist: Counter[str],
        relation_hist: Counter[str],
    ) -> dict[str, Any]:
        return {
            "mapped_object_labels": int(sum(object_hist.values())),
            "mapped_relation_labels": int(sum(relation_hist.values())),
            "unique_object_labels": len(object_hist),
            "unique_relation_labels": len(relation_hist),
        }


class ARKitScenesDataset(PublicSceneGraphDataset):
    """Adapter for ARKitScenes 3DOD scans with derived coarse scene graphs.

    ARKitScenes 3DOD provides object-oriented 3D bounding boxes, not explicit
    scene-graph relation labels. This adapter keeps the graph bottleneck by
    converting object boxes into category counts and coarse spatial relations
    using deterministic centroid/bbox heuristics.
    """

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
            category = box["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
            category_attributes.setdefault(category, set()).update(box["attributes"])
        nodes = [
            ObjectRecord(category=category, count=count, attributes=tuple(sorted(category_attributes.get(category, set()))))
            for category, count in sorted(category_counts.items())
        ]
        edges = derive_relations_from_boxes(boxes)
        scene_type = infer_arkitscenes_scene_type(boxes)
        point_file = discover_arkitscenes_point_file(self.root, scene_id, self.split, self.subset)
        return SceneGraph(
            nodes=tuple(nodes),
            edges=tuple(edges),
            scene_type=scene_type,
            caption=f"An ARKitScenes {scene_type} scene with {len(nodes)} coarse object categories.",
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


class ThreeRScan3DSSGDataset(PublicSceneGraphDataset):
    """Adapter for 3RScan scans with 3DSSG scene-graph supervision."""

    dataset_name = "3rscan_3dssg"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        annotation_file: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self.annotation_file = Path(str(expand_env_vars(str(annotation_file)))).expanduser() if annotation_file else None
        self._annotations: dict[str, Any] | None = None
        super().__init__(root=root, split=split, **kwargs)

    def _discover_scene_ids(self) -> list[str]:
        annotations = self._load_annotations()
        scene_ids = sorted(annotations)
        if not scene_ids:
            checked = [str(path) for path in candidate_3dssg_annotation_paths(self.root, self.annotation_file, self.split)]
            raise FileNotFoundError(
                f"No 3DSSG scenes found for split '{self.split}' under {self.root}. "
                "Set data.annotation_file to a 3DSSG relationships JSON file, or set "
                "THREERSCAN_ROOT to the directory that contains 3DSSG/ or 3DSSG_subset/. "
                f"Checked annotation candidates: {checked[:12]}"
            )
        return scene_ids

    def _load_points(self, scene_id: str) -> torch.Tensor:
        point_file = self._find_point_file(scene_id, self._point_candidates(scene_id))
        return load_point_cloud(point_file, self.max_points)

    def _point_candidates(self, scene_id: str) -> list[str]:
        return three_rscan_point_candidates(scene_id)

    def _load_graph(self, scene_id: str) -> SceneGraph:
        entry = self._load_annotations()[scene_id]
        nodes = _parse_3dssg_objects(entry)
        edges = _parse_3dssg_relations(entry, nodes)
        scene_type = map_scene_type(entry.get("scene_type") or entry.get("scene") or entry.get("type"))
        caption = f"A {scene_type} scene with {len(nodes)} coarse object categories."
        return SceneGraph(
            nodes=tuple(nodes.values()),
            edges=tuple(edges),
            scene_type=scene_type,
            caption=caption,
            metadata={"dataset": self.dataset_name, "scene_id": scene_id},
        )

    def _load_annotations(self) -> dict[str, Any]:
        if self._annotations is not None:
            return self._annotations
        path = discover_3dssg_annotation_file(self.root, self.annotation_file, self.split)
        if path is None:
            self._annotations = {}
            return self._annotations

        payload = json.loads(path.read_text(encoding="utf-8"))
        self._annotations = index_3dssg_payload(payload, split=self.split)
        return self._annotations

    def _label_coverage_stats(
        self,
        object_hist: Counter[str],
        relation_hist: Counter[str],
    ) -> dict[str, Any]:
        raw_object_labels = 0
        mapped_object_labels = 0
        raw_relation_labels = 0
        mapped_relation_labels = 0
        for entry in self._load_annotations().values():
            for label in _iter_3dssg_object_labels(entry):
                raw_object_labels += 1
                mapped_object_labels += int(map_object_category(label) is not None)
            for label in _iter_3dssg_relation_labels(entry):
                raw_relation_labels += 1
                mapped_relation_labels += int(map_relation_predicate(label) is not None)
        return {
            "raw_object_labels": raw_object_labels,
            "mapped_object_labels": mapped_object_labels,
            "object_label_coverage": mapped_object_labels / raw_object_labels if raw_object_labels else 0.0,
            "raw_relation_labels": raw_relation_labels,
            "mapped_relation_labels": mapped_relation_labels,
            "relation_label_coverage": mapped_relation_labels / raw_relation_labels if raw_relation_labels else 0.0,
            "unique_object_labels": len(object_hist),
            "unique_relation_labels": len(relation_hist),
        }


class ScanNetSceneDataset(PublicSceneGraphDataset):
    """Adapter for ScanNet instance annotations with derived coarse relations."""

    dataset_name = "scannet"

    def _discover_scene_ids(self) -> list[str]:
        scene_ids = discover_scannet_scene_ids(self.root, self.split)
        if not scene_ids:
            raise FileNotFoundError(
                f"No ScanNet scenes found for split '{self.split}' under {self.root}. "
                "Expected splits/scannetv2_<split>.txt, scans/scene*/ directories, "
                "or scene*.npy files. Set SCANNET_ROOT or edit data.root in configs/scannet.yaml."
            )
        return scene_ids

    def _load_points(self, scene_id: str) -> torch.Tensor:
        point_file = self._find_point_file(scene_id, self._point_candidates(scene_id))
        return load_point_cloud(point_file, self.max_points)

    def _point_candidates(self, scene_id: str) -> list[str]:
        return scannet_point_candidates(scene_id)

    def _load_graph(self, scene_id: str) -> SceneGraph:
        aggregation_path = _first_existing(
            scannet_aggregation_candidates(self.root, scene_id)
        )
        if aggregation_path is None:
            raise FileNotFoundError(
                f"Missing ScanNet aggregation JSON for {scene_id}. "
                "Expected <scene_id>.aggregation.json under scans/<scene_id>/, "
                f"{scene_id}/, or annotations/. Checked: "
                f"{[str(path) for path in scannet_aggregation_candidates(self.root, scene_id)]}"
            )
        aggregation = json.loads(aggregation_path.read_text(encoding="utf-8"))
        points = self._load_points(scene_id)
        segment_indices = _load_scannet_segments(self.root, scene_id)
        instances = []
        for group in aggregation.get("segGroups", []):
            category = map_object_category(group.get("label"))
            if category is None:
                continue
            group_points = _points_for_segments(points, segment_indices, group.get("segments", []))
            instances.append(
                {
                    "category": category,
                    "instance_id": str(group.get("id", len(instances))),
                    "points": group_points,
                    "bbox": _bbox(group_points),
                }
            )
        category_counts: dict[str, int] = {}
        for instance in instances:
            category = instance["category"]
            category_counts[category] = category_counts.get(category, 0) + 1
        merged_nodes = [
            ObjectRecord(category=category, count=count)
            for category, count in sorted(category_counts.items())
        ]
        edges = _derive_scannet_relations(instances)
        scene_type = map_scene_type(_scene_type_from_id(scene_id))
        return SceneGraph(
            nodes=tuple(merged_nodes),
            edges=tuple(edges),
            scene_type=scene_type,
            caption=f"A {scene_type} ScanNet scene with {len(merged_nodes)} coarse object categories.",
            metadata={"dataset": self.dataset_name, "scene_id": scene_id},
        )


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_arkitscenes_subset(subset: str) -> str:
    """Normalize supported ARKitScenes subset names."""

    normalized = str(subset).strip().lower()
    aliases = {"threedod": "3dod", "depth_upsampling": "upsampling"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"3dod", "raw", "upsampling"}:
        raise ValueError("ARKitScenes subset must be one of: 3dod, raw, upsampling.")
    return normalized


def normalize_arkitscenes_split(split: str) -> str:
    """Normalize ARKitScenes fold names to official capitalization."""

    normalized = str(split).strip().lower()
    if normalized in {"train", "training"}:
        return "Training"
    if normalized in {"val", "valid", "validation"}:
        return "Validation"
    raise ValueError("ARKitScenes split must be Training or Validation.")


def arkitscenes_csv_candidates(root: Path, subset: str) -> list[Path]:
    """Return official split CSV candidates for an ARKitScenes subset."""

    subset = normalize_arkitscenes_subset(subset)
    if subset == "3dod":
        return [
            root / "threedod" / "3dod_train_val_splits.csv",
            root / "3dod_train_val_splits.csv",
            root / "3dod" / "3dod_train_val_splits.csv",
        ]
    if subset == "upsampling":
        return [
            root / "depth_upsampling" / "upsampling_train_val_splits.csv",
            root / "upsampling_train_val_splits.csv",
            root / "upsampling" / "upsampling_train_val_splits.csv",
        ]
    return [
        root / "raw" / "raw_train_val_splits.csv",
        root / "raw_train_val_splits.csv",
    ]


def discover_arkitscenes_scene_ids(root: Path, split: str, subset: str = "3dod") -> list[str]:
    """Discover ARKitScenes video ids from official CSVs or downloaded folders."""

    split = normalize_arkitscenes_split(split)
    subset = normalize_arkitscenes_subset(subset)
    csv_path = _first_existing(arkitscenes_csv_candidates(root, subset))
    scene_ids: set[str] = set()
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
    """Common ARKitScenes split directories."""

    split = normalize_arkitscenes_split(split)
    subset = normalize_arkitscenes_subset(subset)
    subset_dirs = [subset]
    if subset == "3dod":
        subset_dirs.append("threedod")
    elif subset == "upsampling":
        subset_dirs.append("depth_upsampling")
    return [
        *(root / name / split for name in subset_dirs),
        root / split,
    ]


def arkitscenes_scene_directories(root: Path, scene_id: str, split: str, subset: str = "3dod") -> list[Path]:
    """Common ARKitScenes scene directory candidates."""

    return [directory / scene_id for directory in arkitscenes_split_directories(root, split, subset)] + [root / scene_id]


def candidate_arkitscenes_annotation_paths(root: Path, scene_id: str, split: str, subset: str = "3dod") -> list[Path]:
    """Common ARKitScenes annotation JSON candidates."""

    return [
        *(scene_dir / f"{scene_id}_3dod_annotation.json" for scene_dir in arkitscenes_scene_directories(root, scene_id, split, subset)),
        root / "annotations" / f"{scene_id}_3dod_annotation.json",
    ]


def discover_arkitscenes_annotation_file(root: Path, scene_id: str, split: str, subset: str = "3dod") -> Path | None:
    """Find the object annotation JSON for one ARKitScenes scene."""

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
    """Common mesh/point-cloud candidates for one ARKitScenes scene."""

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
    """Find an ARKitScenes point cloud, prepared NPY, or mesh PLY."""

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
        center = _vector3(raw.get("centroid") or raw.get("center") or raw.get("translation"))
        dims = _vector3(raw.get("axesLengths") or raw.get("dimensions") or raw.get("size"), default=1.0)
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


def derive_relations_from_boxes(boxes: list[dict[str, Any]], limit: int = 16) -> list[RelationRecord]:
    """Derive coarse category-level spatial relations from ARKitScenes boxes."""

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
    """Infer a coarse scene type from ARKitScenes object labels."""

    labels = {box["raw_label"] for box in boxes}
    if labels & {"sofa", "tv", "television", "fireplace"}:
        return "living_room"
    if labels & {"desk", "office chair"}:
        return "office"
    if labels and labels <= {"storage", "storage cabinet", "shelf", "shelves", "cabinet"}:
        return "storage"
    return "room"


def find_point_file(root: Path, scene_id: str, candidates: Iterable[str]) -> Path:
    """Find a point cloud for a scene and raise an actionable error if missing."""

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
        "Expected .npy/.npz/.txt/.ply files or official dataset scan paths. "
        f"Checked candidates: {[str(path) for path in checked[:12]]}"
    )


def candidate_3dssg_annotation_paths(root: Path, annotation_file: Path | None = None, split: str | None = None) -> list[Path]:
    """Return common 3DSSG annotation locations without touching the filesystem."""

    if annotation_file is not None:
        return [annotation_file]
    names = _candidate_3dssg_annotation_names(split)
    dirs = [
        root,
        root / "3DSSG",
        root / "3DSSG_subset",
        root / "3DSSG" / "3DSSG_subset",
        root / "annotations",
        root / "data",
    ]
    return [directory / name for directory in dirs for name in names]


def discover_3dssg_annotation_file(root: Path, annotation_file: Path | None = None, split: str | None = None) -> Path | None:
    """Find a 3DSSG relationships JSON file across common public layouts."""

    if annotation_file is not None:
        candidates = [annotation_file] if annotation_file.is_absolute() else [annotation_file, root / annotation_file]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Configured 3DSSG annotation file does not exist: {annotation_file}. "
            "Fix data.annotation_file in configs/3rscan_3dssg.yaml."
        )
    path = _first_existing(candidate_3dssg_annotation_paths(root, split=split))
    if path is not None:
        return path
    recursive_matches: list[Path] = []
    for name in _candidate_3dssg_annotation_names(split):
        recursive_matches.extend(root.rglob(name))
    recursive_matches = sorted(recursive_matches)
    return recursive_matches[0] if recursive_matches else None


def _candidate_3dssg_annotation_names(split: str | None) -> list[str]:
    normalized = (split or "").lower()
    preferred: list[str]
    if normalized in {"val", "validation", "valid"}:
        preferred = ["relationships_validation.json", "relationships_val.json"]
    elif normalized == "test":
        preferred = ["relationships_test.json"]
    else:
        preferred = ["relationships_train.json"]
    fallback = [
        "relationships.json",
        "SceneGraphAnnotation.json",
        "relationships_train.json",
        "relationships_validation.json",
        "relationships_val.json",
        "relationships_test.json",
    ]
    return [*preferred, *[name for name in fallback if name not in preferred]]


def index_3dssg_payload(payload: Any, split: str) -> dict[str, Any]:
    """Index a 3DSSG payload by scene id for a split."""

    return _index_3dssg_payload(payload, split)


def three_rscan_point_candidates(scene_id: str) -> list[str]:
    """Common 3RScan point-cloud candidates for one scene id."""

    return [
        f"3RScan/{scene_id}/labels.instances.align.annotated.v2.ply",
        f"3RScan/{scene_id}/mesh.refined.v2.ply",
        f"{scene_id}/labels.instances.align.annotated.v2.ply",
        f"{scene_id}/mesh.refined.v2.ply",
        f"scans/{scene_id}/labels.instances.align.annotated.v2.ply",
        f"scans/{scene_id}/mesh.refined.v2.ply",
        f"points/{scene_id}.npy",
        f"points/{scene_id}.npz",
        f"{scene_id}.npy",
        f"{scene_id}.npz",
    ]


def discover_scannet_scene_ids(root: Path, split: str) -> list[str]:
    """Discover ScanNet scene ids across split-file and directory layouts."""

    split_candidates = [
        root / "splits" / f"scannetv2_{split}.txt",
        root / f"scannetv2_{split}.txt",
        root / "Tasks" / "Benchmark" / f"scannetv2_{split}.txt",
    ]
    split_file = _first_existing(split_candidates)
    if split_file is not None:
        return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    scans_root = root / "scans"
    if scans_root.exists():
        return sorted(path.name for path in scans_root.iterdir() if path.is_dir() and path.name.startswith("scene"))
    return sorted(path.stem for path in root.glob("scene*.npy"))


def scannet_point_candidates(scene_id: str) -> list[str]:
    """Common ScanNet point-cloud candidates for one scene id."""

    return [
        f"scans/{scene_id}/{scene_id}_vh_clean_2.ply",
        f"scans/{scene_id}/{scene_id}_vh_clean.ply",
        f"{scene_id}/{scene_id}_vh_clean_2.ply",
        f"{scene_id}/{scene_id}_vh_clean.ply",
        f"points/{scene_id}.npy",
        f"points/{scene_id}.npz",
        f"{scene_id}.npy",
        f"{scene_id}.npz",
    ]


def scannet_aggregation_candidates(root: Path, scene_id: str) -> list[Path]:
    """Common ScanNet aggregation JSON candidates."""

    return [
        root / "scans" / scene_id / f"{scene_id}.aggregation.json",
        root / scene_id / f"{scene_id}.aggregation.json",
        root / "annotations" / f"{scene_id}.aggregation.json",
    ]


def _summary_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def _index_3dssg_payload(payload: Any, split: str) -> dict[str, Any]:
    if isinstance(payload, dict) and "scans" in payload:
        entries = payload["scans"]
    elif isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        return {str(key): value for key, value in payload.items()}
    else:
        return {}

    indexed: dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_split = entry.get("split")
        if entry_split and str(entry_split).lower() not in {split.lower(), "train" if split == "training" else split.lower()}:
            continue
        scene_id = entry.get("scan") or entry.get("scan_id") or entry.get("scene_id") or entry.get("reference")
        if scene_id is None:
            continue
        indexed[str(scene_id)] = entry
    return indexed


def _parse_3dssg_objects(entry: dict[str, Any]) -> dict[str, ObjectRecord]:
    raw_objects = entry.get("objects") or entry.get("nodes") or {}
    nodes: dict[str, ObjectRecord] = {}
    if isinstance(raw_objects, dict):
        iterable = raw_objects.items()
    else:
        iterable = ((obj.get("id", idx), obj) for idx, obj in enumerate(raw_objects) if isinstance(obj, dict))

    category_counts: dict[str, int] = {}
    for obj_id, raw in iterable:
        label = raw.get("label") if isinstance(raw, dict) else raw
        if label is None and isinstance(raw, dict):
            label = raw.get("class") or raw.get("category") or raw.get("name")
        category = map_object_category(label)
        if category is None:
            continue
        category_counts[category] = category_counts.get(category, 0) + 1
        nodes[str(obj_id)] = ObjectRecord(
            category=category,
            count=1,
            attributes=map_attributes(raw.get("attributes") if isinstance(raw, dict) else None),
            instance_id=str(obj_id),
        )

    by_category: dict[str, ObjectRecord] = {}
    for node in nodes.values():
        attrs = set(by_category.get(node.category, ObjectRecord(node.category)).attributes) | set(node.attributes)
        by_category[node.category] = ObjectRecord(
            category=node.category,
            count=category_counts[node.category],
            attributes=tuple(sorted(attrs)),
        )
    return by_category


def _parse_3dssg_relations(entry: dict[str, Any], nodes_by_category: dict[str, ObjectRecord]) -> list[RelationRecord]:
    raw_relations = entry.get("relationships") or entry.get("relations") or entry.get("edges") or []
    known_categories = set(nodes_by_category)
    id_to_category = _object_id_to_category(entry)
    relations: set[tuple[str, str, str]] = set()
    for rel in raw_relations:
        if isinstance(rel, (list, tuple)) and len(rel) >= 3:
            subject_label, object_label = rel[0], rel[1]
            predicate_label = rel[3] if len(rel) >= 4 and isinstance(rel[3], str) else rel[2]
        elif isinstance(rel, dict):
            subject_label = rel.get("subject") or rel.get("source") or rel.get("from") or rel.get("subject_id")
            object_label = rel.get("object") or rel.get("target") or rel.get("to") or rel.get("object_id")
            predicate_label = rel.get("predicate") or rel.get("relation") or rel.get("label") or rel.get("name")
        else:
            continue
        subject = map_object_category(subject_label) or id_to_category.get(str(subject_label))
        obj = map_object_category(object_label) or id_to_category.get(str(object_label))
        predicate = map_relation_predicate(predicate_label)
        if subject in known_categories and obj in known_categories and predicate is not None and subject != obj:
            relations.add((subject, predicate, obj))
    return [RelationRecord(subject=s, predicate=p, object=o) for s, p, o in sorted(relations)]


def _object_id_to_category(entry: dict[str, Any]) -> dict[str, str]:
    raw_objects = entry.get("objects") or entry.get("nodes") or {}
    if isinstance(raw_objects, dict):
        iterable = raw_objects.items()
    else:
        iterable = ((obj.get("id", idx), obj) for idx, obj in enumerate(raw_objects) if isinstance(obj, dict))
    mapping = {}
    for obj_id, raw in iterable:
        label = raw.get("label") if isinstance(raw, dict) else raw
        if label is None and isinstance(raw, dict):
            label = raw.get("class") or raw.get("category") or raw.get("name")
        category = map_object_category(label)
        if category is not None:
            mapping[str(obj_id)] = category
    return mapping


def _iter_3dssg_object_labels(entry: dict[str, Any]) -> list[Any]:
    raw_objects = entry.get("objects") or entry.get("nodes") or {}
    labels = []
    values = raw_objects.values() if isinstance(raw_objects, dict) else raw_objects
    for raw in values:
        if isinstance(raw, dict):
            labels.append(raw.get("label") or raw.get("class") or raw.get("category") or raw.get("name"))
        else:
            labels.append(raw)
    return [label for label in labels if label is not None]


def _iter_3dssg_relation_labels(entry: dict[str, Any]) -> list[Any]:
    raw_relations = entry.get("relationships") or entry.get("relations") or entry.get("edges") or []
    labels = []
    for rel in raw_relations:
        if isinstance(rel, (list, tuple)) and len(rel) >= 3:
            labels.append(rel[3] if len(rel) >= 4 and isinstance(rel[3], str) else rel[2])
        elif isinstance(rel, dict):
            labels.append(rel.get("predicate") or rel.get("relation") or rel.get("label") or rel.get("name"))
    return [label for label in labels if label is not None]


def _derive_category_relations_from_scan_nodes(nodes: list[ObjectRecord]) -> list[RelationRecord]:
    if len(nodes) < 2:
        return []
    relations = []
    categories = [node.category for node in nodes]
    for left, right in zip(categories[:-1], categories[1:], strict=False):
        if left != right:
            relations.append(RelationRecord(subject=left, predicate="near", object=right))
    return relations[:8]


def _load_scannet_segments(root: Path, scene_id: str) -> list[int] | None:
    path = _first_existing(
        [
            root / "scans" / scene_id / f"{scene_id}_vh_clean_2.0.010000.segs.json",
            root / "scans" / scene_id / f"{scene_id}_vh_clean.segs.json",
            root / scene_id / f"{scene_id}_vh_clean_2.0.010000.segs.json",
            root / "annotations" / f"{scene_id}.segs.json",
        ]
    )
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = payload.get("segIndices")
    return [int(value) for value in indices] if isinstance(indices, list) else None


def _points_for_segments(points: torch.Tensor, segment_indices: list[int] | None, segments: list[int]) -> torch.Tensor:
    if not segments or segment_indices is None:
        return points
    usable = min(points.shape[0], len(segment_indices))
    segment_set = {int(segment) for segment in segments}
    keep = torch.tensor(
        [segment_indices[i] in segment_set for i in range(usable)],
        dtype=torch.bool,
        device=points.device,
    )
    if not bool(keep.any()):
        return points
    return points[:usable][keep]


def _bbox(points: torch.Tensor) -> dict[str, torch.Tensor]:
    if points.numel() == 0:
        points = torch.zeros(1, 3)
    return {
        "min": points.min(dim=0).values,
        "max": points.max(dim=0).values,
        "center": points.mean(dim=0),
    }


def _derive_scannet_relations(instances: list[dict[str, Any]]) -> list[RelationRecord]:
    relations: set[tuple[str, str, str]] = set()
    for i, left in enumerate(instances):
        for j, right in enumerate(instances):
            if i == j or left["category"] == right["category"]:
                continue
            predicate = _coarse_relation_from_bboxes(left["bbox"], right["bbox"])
            if predicate is not None:
                relations.add((left["category"], predicate, right["category"]))
    return [RelationRecord(subject=s, predicate=p, object=o) for s, p, o in sorted(relations)[:12]]


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


def _scene_type_from_id(scene_id: str) -> str:
    if "office" in scene_id.lower():
        return "office"
    return "room"
