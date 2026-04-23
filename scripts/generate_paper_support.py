"""Generate the paper-writing scaffold into an artifact directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.reporting.paper_support import generate_paper_support_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--template-dir", default="docs/paper_support")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_paper_support_package(args.output_dir, args.template_dir)


if __name__ == "__main__":
    main()
