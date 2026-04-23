"""Generate main paper tables from saved outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from less_geometry_same_semantics.comparisons.reporting import save_comparison_tables
from less_geometry_same_semantics.reporting.main_tables import save_main_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-record", required=True)
    parser.add_argument("--ablation-record", default=None)
    parser.add_argument("--comparison-record", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_record = json.loads(Path(args.main_record).read_text(encoding="utf-8"))
    ablation_record = json.loads(Path(args.ablation_record).read_text(encoding="utf-8")) if args.ablation_record else None
    save_main_tables(main_record=main_record, ablation_record=ablation_record, output_dir=args.output_dir)
    if args.comparison_record:
        comparison_record = json.loads(Path(args.comparison_record).read_text(encoding="utf-8"))
        save_comparison_tables(comparison_record, args.output_dir)


if __name__ == "__main__":
    main()
