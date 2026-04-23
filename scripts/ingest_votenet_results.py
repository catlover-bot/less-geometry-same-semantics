"""Convenience wrapper for ingesting VoteNet exports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ingest_external_baseline.py"),
            "--baseline-id",
            "votenet_import",
            *sys.argv[1:],
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
