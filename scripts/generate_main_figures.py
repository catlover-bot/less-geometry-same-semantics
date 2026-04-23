"""Generate the exact main paper figures from saved outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.reporting.main_figures import save_main_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-record", required=True)
    parser.add_argument("--severity-record", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_record = json.loads(Path(args.main_record).read_text(encoding="utf-8"))
    severity_record = json.loads(Path(args.severity_record).read_text(encoding="utf-8")) if args.severity_record else None
    save_main_figures(main_record=main_record, severity_record=severity_record, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
