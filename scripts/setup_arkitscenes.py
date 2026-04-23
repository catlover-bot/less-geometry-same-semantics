"""Prepare or document an ARKitScenes download using the official script."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
import zipfile
from urllib.error import HTTPError, URLError
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_BASE = "https://raw.githubusercontent.com/apple/ARKitScenes/main"
DOWNLOAD_SCRIPT_URL = f"{OFFICIAL_BASE}/download_data.py"
SPLIT_CSV_URLS = {
    "3dod": f"{OFFICIAL_BASE}/threedod/3dod_train_val_splits.csv",
    "raw": f"{OFFICIAL_BASE}/raw/raw_train_val_splits.csv",
    "upsampling": f"{OFFICIAL_BASE}/depth_upsampling/upsampling_train_val_splits.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", required=True, help="Local ARKitScenes root/download directory.")
    parser.add_argument("--subset", choices=["3dod", "raw", "upsampling"], default="3dod")
    parser.add_argument("--split", choices=["Training", "Validation"], default=None)
    parser.add_argument("--video-id", nargs="*", default=None)
    parser.add_argument("--video-id-csv", default=None, help="Existing CSV of video ids to pass to official download script.")
    parser.add_argument("--raw-assets", nargs="*", default=None, help="Raw dataset assets when --subset raw is used.")
    parser.add_argument("--download-laser-scanner-point-cloud", action="store_true")
    parser.add_argument("--fetch-official-script", action="store_true", help="Download official script and split CSVs.")
    parser.add_argument("--run-download", action="store_true", help="Invoke official download_data.py after preparing it.")
    parser.add_argument("--resumable-3dod-download", action="store_true", help="Use a Windows-friendly resumable downloader for explicit 3dod video ids.")
    parser.add_argument("--output-dir", default="outputs/setup/arkitscenes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_dir = Path(args.download_dir).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)
    official_dir = download_dir / "_official_arkitscenes"
    script_path = official_dir / "download_data.py"
    csv_path = Path(args.video_id_csv).expanduser() if args.video_id_csv else _default_csv_path(official_dir, args.subset)

    actions: list[str] = []
    errors: list[str] = []
    if args.fetch_official_script or args.run_download:
        official_dir.mkdir(parents=True, exist_ok=True)
        try:
            _download(DOWNLOAD_SCRIPT_URL, script_path)
            actions.append(f"Downloaded official script to {script_path}")
            if not args.video_id_csv:
                _download(SPLIT_CSV_URLS[args.subset], csv_path)
                actions.append(f"Downloaded split CSV to {csv_path}")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            errors.append(
                "Failed to fetch the official ARKitScenes helper from GitHub. "
                f"Blocking issue: {type(exc).__name__}: {exc}. "
                "Manual step: download https://github.com/apple/ARKitScenes/blob/main/download_data.py "
                f"to {script_path} and rerun with --run-download."
            )

    command = _build_official_command(args, script_path, csv_path, download_dir)
    if errors:
        actions.append("Skipped official download command because setup prerequisites failed.")
    elif args.run_download and args.resumable_3dod_download:
        if args.subset != "3dod" or not args.video_id or not args.split:
            errors.append("--resumable-3dod-download requires --subset 3dod, --split, and explicit --video-id values.")
        else:
            try:
                _download_3dod_resumable(args.video_id, args.split, download_dir)
                actions.append("Downloaded explicit 3DOD scenes with the resumable ARKitScenes downloader.")
            except subprocess.CalledProcessError as exc:
                errors.append(
                    "Resumable ARKitScenes download failed. "
                    f"Exit code: {exc.returncode}. This is a network/download issue from Apple's asset host."
                )
            except (OSError, zipfile.BadZipFile) as exc:
                errors.append(f"Resumable ARKitScenes download or extraction failed: {type(exc).__name__}: {exc}")
    elif args.run_download:
        if not script_path.exists():
            errors.append("Official download_data.py is missing. Rerun with --fetch-official-script.")
        elif not args.video_id and not csv_path.exists():
            errors.append("No video ids or CSV are available. Provide --video-id, --video-id-csv, or --fetch-official-script.")
        elif importlib.util.find_spec("pandas") is None:
            errors.append(
                "The official ARKitScenes download script requires pandas. "
                "Install repo requirements first: pip install -r requirements.txt"
            )
        else:
            try:
                subprocess.run(command, cwd=download_dir, check=True)
                actions.append("Official ARKitScenes download command completed.")
                missing = _missing_requested_scenes(args, download_dir)
                if missing:
                    errors.append(
                        "Official script finished, but requested scene assets are not usable yet. "
                        f"Missing or incomplete scenes: {missing}. "
                        "This usually means the upstream download failed or was interrupted. "
                        "Remove any *.tmp files for those scenes and rerun the same command."
                    )
            except subprocess.CalledProcessError as exc:
                errors.append(
                    "The official ARKitScenes download script exited with an error. "
                    f"Exit code: {exc.returncode}. This is usually a network/download issue, "
                    "an unavailable video id, or an upstream script requirement. "
                    "Inspect the command above, then rerun it directly for the full official-script output."
                )
            except OSError as exc:
                errors.append(
                    "Could not launch the official ARKitScenes download script. "
                    f"Blocking issue: {type(exc).__name__}: {exc}."
                )
    else:
        actions.append("Dry run only. Add --fetch-official-script to download Apple's helper, then add --run-download to invoke it.")

    report = {
        "status": "fail" if errors else "ready" if args.run_download else "dry_run",
        "download_dir": str(download_dir),
        "subset": args.subset,
        "split": args.split,
        "video_id": args.video_id,
        "video_id_csv": str(csv_path) if csv_path else None,
        "official_script": str(script_path),
        "command": command,
        "actions": actions,
        "errors": errors,
        "notes": [
            "The full ARKitScenes 3DOD subset is large. Prefer a small CSV or explicit --video-id list for debug runs.",
            "This helper uses Apple's official download_data.py workflow; it does not rehost or bypass dataset terms.",
            "Set PowerShell env var after setup: $env:ARKITSCENES_ROOT='<download-dir>'.",
        ],
    }
    _save_report(report, output_dir)
    print(_format_report(report))
    if errors:
        raise SystemExit(1)


def _default_csv_path(official_dir: Path, subset: str) -> Path:
    if subset == "3dod":
        return official_dir / "threedod" / "3dod_train_val_splits.csv"
    if subset == "raw":
        return official_dir / "raw" / "raw_train_val_splits.csv"
    return official_dir / "depth_upsampling" / "upsampling_train_val_splits.csv"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())


def _build_official_command(args: argparse.Namespace, script_path: Path, csv_path: Path, download_dir: Path) -> list[str]:
    command = [sys.executable, str(script_path), args.subset]
    if args.video_id:
        if args.split:
            command.extend(["--split", args.split])
        command.extend(["--video_id", *args.video_id])
    else:
        command.extend(["--video_id_csv", str(csv_path)])
    command.extend(["--download_dir", str(download_dir)])
    if args.subset == "raw" and args.raw_assets:
        command.extend(["--raw_dataset_assets", *args.raw_assets])
    if args.download_laser_scanner_point_cloud:
        command.append("--download_laser_scanner_point_cloud")
    return command


def _missing_requested_scenes(args: argparse.Namespace, download_dir: Path) -> list[str]:
    if not args.video_id:
        return []
    split = args.split
    if split is None:
        return []
    missing = []
    for video_id in args.video_id:
        scene_dir = download_dir / args.subset / split / str(video_id)
        annotation = scene_dir / f"{video_id}_3dod_annotation.json"
        mesh = scene_dir / f"{video_id}_3dod_mesh.ply"
        zip_file = download_dir / args.subset / split / f"{video_id}.zip"
        tmp_file = download_dir / args.subset / split / f"{video_id}.zip.tmp"
        if args.subset == "3dod" and not ((annotation.exists() and mesh.exists()) or (scene_dir.exists() and any(scene_dir.iterdir()))):
            detail = str(video_id)
            if tmp_file.exists() and not zip_file.exists():
                detail += " (partial .tmp download remains)"
            missing.append(detail)
    return missing


def _download_3dod_resumable(video_ids: list[str], split: str, download_dir: Path) -> None:
    target_dir = download_dir / "3dod" / split
    target_dir.mkdir(parents=True, exist_ok=True)
    for video_id in video_ids:
        scene_dir = target_dir / str(video_id)
        annotation = scene_dir / f"{video_id}_3dod_annotation.json"
        mesh = scene_dir / f"{video_id}_3dod_mesh.ply"
        if annotation.exists() and mesh.exists():
            continue
        zip_path = target_dir / f"{video_id}.zip"
        tmp_path = target_dir / f"{video_id}.zip.tmp"
        if tmp_path.exists() and not zip_path.exists():
            tmp_path.replace(zip_path)
        url = f"https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/threedod/{split}/{video_id}.zip"
        command = [
            "curl.exe",
            "--location",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "--retry-all-errors",
            "--continue-at",
            "-",
            url,
            "--output",
            str(zip_path),
        ]
        subprocess.run(command, check=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(target_dir)


def _save_report(report: dict, output_dir: Path) -> None:
    (output_dir / "arkitscenes_setup_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "arkitscenes_setup_report.md").write_text(_format_report(report), encoding="utf-8")


def _format_report(report: dict) -> str:
    lines = [
        "# ARKitScenes Setup Report",
        "",
        f"Status: **{report['status']}**",
        "",
        "Command:",
        "",
        "```powershell",
        " ".join(str(part) for part in report["command"]),
        "```",
        "",
        "Actions:",
    ]
    lines.extend(f"- {item}" for item in report["actions"])
    if report["errors"]:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {item}" for item in report["errors"])
    lines.extend(["", "Notes:"])
    lines.extend(f"- {item}" for item in report["notes"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user.")
