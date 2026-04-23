"""Freeze only claims supported by completed run records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from less_geometry_same_semantics.reporting.claims import main_matrix_table
from less_geometry_same_semantics.reporting.tables import severity_semantic_metrics_table


def freeze_supported_claims(
    main_record: dict[str, Any],
    severity_record: dict[str, Any] | None = None,
    *,
    min_delta: float = 0.01,
    severe_retention_threshold: float = 0.75,
) -> dict[str, Any]:
    """Return supported and unsupported claim statements with numeric evidence.

    The function is deliberately conservative: if a needed comparison is absent
    or below the configured threshold, the claim is not listed as supported.
    """

    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    main_rows = main_matrix_table(main_record)

    clean = _best(main_rows, corruption="clean")
    severe = _best(main_rows, corruption="severe_corruption")
    if clean and severe and clean["semantic_macro_f1"] > 0:
        retention = severe["semantic_macro_f1"] / clean["semantic_macro_f1"]
        claim = {
            "claim": "aggressive degradation preserves a substantial fraction of coarse semantic performance",
            "evidence": {
                "clean_semantic_macro_f1": clean["semantic_macro_f1"],
                "severe_semantic_macro_f1": severe["semantic_macro_f1"],
                "relative_retention": retention,
                "criterion": f"relative_retention >= {severe_retention_threshold}",
            },
        }
        (supported if retention >= severe_retention_threshold else unsupported).append(claim)
    else:
        unsupported.append({"claim": "aggressive degradation semantic retention", "reason": "missing clean/severe comparison"})

    graph = _best(main_rows, corruption="severe_corruption", graph="simple_graph")
    no_graph = _best(main_rows, corruption="severe_corruption", graph="no_graph")
    if graph and no_graph:
        delta = graph["semantic_macro_f1"] - no_graph["semantic_macro_f1"]
        claim = {
            "claim": "graph bottlenecks improve robustness under corruption",
            "evidence": {
                "graph_semantic_macro_f1": graph["semantic_macro_f1"],
                "no_graph_semantic_macro_f1": no_graph["semantic_macro_f1"],
                "delta": delta,
                "criterion": f"delta >= {min_delta}",
            },
        }
        (supported if delta >= min_delta else unsupported).append(claim)
    else:
        unsupported.append({"claim": "graph bottleneck robustness", "reason": "missing graph/no_graph severe comparison"})

    constrained = _best(main_rows, constrained=True)
    unconstrained = _best(main_rows, constrained=False)
    if constrained and unconstrained:
        validity_delta = constrained["json_validity"] - unconstrained["json_validity"]
        stability_delta = constrained["semantic_macro_f1"] - unconstrained["semantic_macro_f1"]
        claim = {
            "claim": "constrained structured decoding improves validity or semantic stability",
            "evidence": {
                "constrained_json_validity": constrained["json_validity"],
                "unconstrained_json_validity": unconstrained["json_validity"],
                "json_validity_delta": validity_delta,
                "semantic_macro_f1_delta": stability_delta,
                "criterion": f"json_validity_delta >= {min_delta} or semantic_macro_f1_delta >= {min_delta}",
            },
        }
        (supported if validity_delta >= min_delta or stability_delta >= min_delta else unsupported).append(claim)
    else:
        unsupported.append({"claim": "constrained decoding validity/stability", "reason": "missing constrained comparison"})

    raw = _best(main_rows, point_budget="raw")
    compressed = _best(main_rows, point_budget="compressed")
    if raw and compressed:
        object_drop = raw["object_f1"] - compressed["object_f1"]
        relation_drop = raw["relation_f1"] - compressed["relation_f1"]
        claim = {
            "claim": "compression preserves coarse object semantics more than fine relations",
            "evidence": {
                "raw_object_f1": raw["object_f1"],
                "compressed_object_f1": compressed["object_f1"],
                "raw_relation_f1": raw["relation_f1"],
                "compressed_relation_f1": compressed["relation_f1"],
                "object_f1_drop": object_drop,
                "relation_f1_drop": relation_drop,
                "criterion": "relation_f1_drop > object_f1_drop",
            },
        }
        (supported if relation_drop > object_drop else unsupported).append(claim)
    else:
        unsupported.append({"claim": "compression object-vs-relation preservation", "reason": "missing raw/compressed comparison"})

    if severity_record is not None:
        severity_rows = severity_semantic_metrics_table(severity_record)
        if severity_rows:
            supported.append(
                {
                    "claim": "severity sweep completed",
                    "evidence": {
                        "presets": [row["preset"] for row in severity_rows],
                        "object_f1": {row["preset"]: row["object_f1"] for row in severity_rows},
                        "relation_f1": {row["preset"]: row["relation_f1"] for row in severity_rows},
                        "json_validity": {row["preset"]: row["json_validity"] for row in severity_rows},
                    },
                }
            )

    return {"supported_claims": supported, "unsupported_or_unfrozen_claims": unsupported}


def save_frozen_claims(report: dict[str, Any], output_dir: str | Path) -> None:
    """Save frozen claims as JSON and Markdown."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "frozen_claims.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Frozen Main Claims", ""]
    lines.append("## Supported")
    if report["supported_claims"]:
        for claim in report["supported_claims"]:
            lines.append(f"- {claim['claim']}")
            lines.append(f"  - Evidence: `{json.dumps(claim.get('evidence', {}), sort_keys=True)}`")
    else:
        lines.append("- No claims met the freeze criteria.")
    lines.extend(["", "## Not Frozen"])
    for claim in report["unsupported_or_unfrozen_claims"]:
        lines.append(f"- {claim['claim']}")
        if "reason" in claim:
            lines.append(f"  - Reason: {claim['reason']}")
        elif "evidence" in claim:
            lines.append(f"  - Evidence: `{json.dumps(claim['evidence'], sort_keys=True)}`")
    (out / "frozen_claims.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _best(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any] | None:
    candidates = [row for row in rows if all(row.get(key) == value for key, value in filters.items())]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["semantic_macro_f1"])
