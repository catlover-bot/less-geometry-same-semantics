"""Generate claim-tight paper analysis from completed ARKitScenes outputs."""

from __future__ import annotations

import argparse

from less_geometry_same_semantics.reporting.claim_tight_analysis import load_json, save_claim_tight_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate claim-tight lightweight robustness analysis.")
    parser.add_argument("--main-record", default="outputs/paper_package/primary_main/results.json")
    parser.add_argument("--severity-record", default="outputs/paper_package/primary_severity/results.json")
    parser.add_argument("--ablation-record", default="outputs/paper_package/primary_ablations/results.json")
    parser.add_argument("--frozen-claims", default="outputs/paper_package/claims/frozen_claims.json")
    parser.add_argument("--diagnostics-dir", default="outputs/diagnostics/arkitscenes")
    parser.add_argument("--output-dir", default="outputs/paper_package/claim_tight_analysis")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    main_record = load_json(args.main_record)
    severity_record = load_json(args.severity_record)
    ablation_record = load_json(args.ablation_record) if args.ablation_record else None
    frozen_claims = load_json(args.frozen_claims) if args.frozen_claims else None
    summary = save_claim_tight_analysis(
        main_record=main_record,
        severity_record=severity_record,
        ablation_record=ablation_record,
        frozen_claims=frozen_claims,
        diagnostics_dir=args.diagnostics_dir,
        output_dir=args.output_dir,
    )
    print(f"Saved claim-tight analysis to {summary['output_dir']}")


if __name__ == "__main__":
    main()
