"""Preflight checks for local public-dataset execution."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

from less_geometry_same_semantics.data.public_datasets import (
    arkitscenes_csv_candidates,
    candidate_arkitscenes_annotation_paths,
    discover_arkitscenes_annotation_file,
    discover_arkitscenes_point_file,
    discover_arkitscenes_scene_ids,
)
from less_geometry_same_semantics.utils.config import find_unresolved_env_vars, load_config


REQUIRED_IMPORTS = {
    "torch": "torch",
    "numpy": "numpy",
    "yaml": "PyYAML",
    "jsonschema": "jsonschema",
    "matplotlib": "matplotlib",
    "plyfile": "plyfile",
    "psutil": "psutil",
}


def build_setup_report(config_paths: list[str | Path], max_scenes: int = 3) -> dict[str, Any]:
    """Run environment and dataset checks for one or more configs."""

    environment = check_environment()
    datasets = [check_dataset_config(path, max_scenes=max_scenes) for path in config_paths]
    report = {
        "status": _combined_status([environment, *datasets]),
        "environment": environment,
        "datasets": datasets,
    }
    return report


def check_environment() -> dict[str, Any]:
    """Check Python and required package availability."""

    checks: list[dict[str, Any]] = []
    python_ok = sys.version_info >= (3, 10)
    checks.append(
        _check(
            "python_version",
            python_ok,
            f"Python {platform.python_version()} on {platform.platform()}",
            "Install Python 3.10 or newer. On Windows, prefer Python 3.11 or 3.12 for broad PyTorch wheels.",
        )
    )
    for import_name, package_name in REQUIRED_IMPORTS.items():
        checks.append(
            _check(
                f"import_{import_name}",
                importlib.util.find_spec(import_name) is not None,
                f"Import available: {import_name}",
                f"Install missing dependency: pip install {package_name}",
            )
        )

    torch_detail: dict[str, Any] = {}
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch_detail = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    return {"status": _status_from_checks(checks), "checks": checks, "torch": torch_detail}


def check_dataset_config(config_path: str | Path, max_scenes: int = 3) -> dict[str, Any]:
    """Validate dataset paths, annotations, split discovery, and sample files."""

    path = Path(config_path)
    checks: list[dict[str, Any]] = []
    try:
        config = load_config(path)
    except Exception as exc:
        return {
            "status": "fail",
            "config": str(path),
            "dataset": "unknown",
            "checks": [_error("config_load", f"Could not load config: {exc}")],
        }

    data_cfg = config.get("data", {})
    dataset = str(data_cfg.get("dataset", "synthetic")).lower()
    if dataset == "synthetic":
        checks.append(_check("dataset_kind", True, "Synthetic dataset config does not require public data.", ""))
        return {"status": "pass", "config": str(path), "dataset": dataset, "checks": checks}

    root_value = data_cfg.get("root")
    if root_value is None:
        checks.append(
            _error(
                "data_root",
                "Missing data.root. Set ARKITSCENES_ROOT in PowerShell or edit the YAML config.",
            )
        )
        return {"status": "fail", "config": str(path), "dataset": dataset, "checks": checks}

    unresolved = find_unresolved_env_vars(root_value)
    if unresolved:
        checks.append(
            _error(
                "env_vars",
                f"Unresolved environment variable(s): {', '.join(unresolved)}. "
                "PowerShell example: $env:ARKITSCENES_ROOT='C:\\datasets\\ARKitScenes'.",
            )
        )
        return {"status": "fail", "config": str(path), "dataset": dataset, "checks": checks}

    root = Path(str(root_value)).expanduser()
    checks.append(
        _check(
            "data_root_exists",
            root.exists(),
            f"Dataset root exists: {root}",
            f"Dataset root does not exist: {root}. Set the environment variable or edit data.root.",
        )
    )
    if not root.exists():
        return {"status": "fail", "config": str(path), "dataset": dataset, "checks": checks}

    if dataset in {"arkitscenes", "arkit_scenes"}:
        checks.extend(_check_arkitscenes(root, data_cfg, max_scenes=max_scenes))
    else:
        checks.append(_error("dataset_name", f"Unsupported active dataset '{dataset}'. The public preflight path is ARKitScenes-first. Legacy configs are archived under configs/legacy/."))

    return {"status": _status_from_checks(checks), "config": str(path), "dataset": dataset, "root": str(root), "checks": checks}


def save_setup_report(report: dict[str, Any], output_dir: str | Path) -> None:
    """Save preflight report as JSON and Markdown."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_setup_check.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (out / "dataset_setup_check.md").write_text(format_setup_report_markdown(report), encoding="utf-8")


def format_setup_report_markdown(report: dict[str, Any]) -> str:
    """Render a setup report for humans."""

    lines = ["# Dataset Setup Check", "", f"Status: **{report['status']}**", ""]
    lines.extend(["## Environment", ""])
    for check in report["environment"]["checks"]:
        lines.append(_format_check(check))
    if report["environment"].get("torch"):
        torch = report["environment"]["torch"]
        lines.extend(
            [
                "",
                f"- torch version: `{torch.get('torch_version')}`",
                f"- CUDA available: `{torch.get('cuda_available')}`",
                f"- CUDA devices: `{torch.get('cuda_device_count')}`",
            ]
        )
    for dataset in report.get("datasets", []):
        lines.extend(["", f"## {dataset.get('dataset')} ({dataset.get('config')})", ""])
        for check in dataset.get("checks", []):
            lines.append(_format_check(check))
    return "\n".join(lines) + "\n"


def _check_arkitscenes(root: Path, data_cfg: dict[str, Any], max_scenes: int) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    subset = str(data_cfg.get("subset", "3dod"))
    cache_dir = Path(str(data_cfg.get("cache_dir", "outputs/cache/arkitscenes")))
    checks.append(
        _check(
            "cache_parent_writable",
            _is_writable(cache_dir.parent),
            f"Cache parent is writable: {cache_dir.parent}",
            f"Cache parent is not writable: {cache_dir.parent}",
        )
    )
    csv_candidates = arkitscenes_csv_candidates(root, subset)
    split_csv = next((path for path in csv_candidates if path.exists()), None)
    checks.append(
        _check(
            "split_csv_or_directories",
            split_csv is not None or any(directory.exists() for split in ["Training", "Validation"] for directory in _arkit_split_dirs_for_check(root, split, subset)),
            f"ARKitScenes split source found: {split_csv or 'downloaded split directories'}",
            f"No split CSV or downloaded split directories found. Checked CSVs: {[str(path) for path in csv_candidates]}",
        )
    )
    for split_key in ["train_split", "val_split"]:
        split = str(data_cfg.get(split_key, "Training" if split_key == "train_split" else "Validation"))
        try:
            scene_ids = discover_arkitscenes_scene_ids(root, split, subset)
        except Exception as exc:
            checks.append(_error(f"{split_key}_scene_count", str(exc)))
            continue
        checks.append(
            _check(
                f"{split_key}_scene_count",
                bool(scene_ids),
                f"{len(scene_ids)} ARKitScenes scenes discovered for split '{split}'.",
                f"No ARKitScenes scenes discovered for split '{split}'. Run scripts/setup_arkitscenes.py or check ARKITSCENES_ROOT.",
            )
        )
        for scene_id in scene_ids[:max_scenes]:
            annotation = discover_arkitscenes_annotation_file(root, scene_id, split, subset)
            checks.append(
                _check(
                    f"{split_key}_annotation_{scene_id}",
                    annotation is not None,
                    f"Annotation found: {annotation}",
                    f"Missing annotation for {scene_id}. Checked: {[str(path) for path in candidate_arkitscenes_annotation_paths(root, scene_id, split, subset)]}",
                )
            )
            point_file = discover_arkitscenes_point_file(root, scene_id, split, subset)
            allow_fallback = bool(data_cfg.get("allow_annotation_point_fallback", True))
            checks.append(
                _check(
                    f"{split_key}_points_{scene_id}",
                    point_file is not None or (allow_fallback and annotation is not None),
                    f"Point source found: {point_file or 'annotation-box fallback'}",
                    f"Missing point cloud/mesh for {scene_id}. Expected *_3dod_mesh.ply, prepared *_pc.npy, or enable annotation fallback.",
                )
            )
    return checks


def _arkit_split_dirs_for_check(root: Path, split: str, subset: str) -> list[Path]:
    subset = {"threedod": "3dod", "depth_upsampling": "upsampling"}.get(subset, subset)
    dirs = [root / subset / split, root / split]
    if subset == "3dod":
        dirs.append(root / "threedod" / split)
    if subset == "upsampling":
        dirs.append(root / "depth_upsampling" / split)
    return dirs


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _check(name: str, ok: bool, success: str, failure: str) -> dict[str, Any]:
    return {"name": name, "status": "pass" if ok else "error", "message": success if ok else failure}


def _error(name: str, message: str) -> dict[str, Any]:
    return {"name": name, "status": "error", "message": message}


def _status_from_checks(checks: list[dict[str, Any]]) -> str:
    return "fail" if any(check["status"] == "error" for check in checks) else "pass"


def _combined_status(sections: list[dict[str, Any]]) -> str:
    return "fail" if any(section.get("status") == "fail" for section in sections) else "pass"


def _format_check(check: dict[str, Any]) -> str:
    marker = "PASS" if check["status"] == "pass" else "ERROR"
    return f"- **{marker}** `{check['name']}`: {check['message']}"
