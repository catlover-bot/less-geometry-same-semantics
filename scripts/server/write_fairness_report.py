#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

BASE = Path("outputs/paper_package_50_30")
OUT = BASE / "fair_eval/evaluation_fairness_report.md"

main = (BASE / "comparisons/main_comparisons.md").read_text()
native = (BASE / "fair_eval/external_native_metrics.md").read_text()
audit = (BASE / "audit/input_leakage_audit.md").read_text()

OUT.write_text(
f"""# Evaluation Fairness Report

## 1. Input leakage audit

{audit}

## 2. Current shared-schema comparison

This is the current task-specialized comparison. It is useful for evaluating coarse semantic retention, but it is favorable to LGS because LGS natively predicts the shared schema.

{main}

## 3. External-native diagnostics

This table gives SpatialLM and VoteNet a more favorable diagnostic by evaluating their raw box-like outputs with box matching and output-collapse metrics.

{native}

## 4. Recommended interpretation

The main claim should not be:

> LGS is a better general 3D detector or general 3D-language model.

The safer claim is:

> LGS is a task-specialized lightweight structured predictor that preserves coarse object-level semantics under severe geometric corruption. External 3D baselines collapse under the same shared-schema conversion, and external-native diagnostics should be reported as complementary evidence.

## 5. Metrics to emphasize

Primary:
- coarse object-level F1 under shared schema
- retention from clean to severe
- latency and parameter count
- empty-output rate of external baselines

Auxiliary:
- relation F1
- scene accuracy
- box-IoU diagnostics for external box-producing baselines

Limitations:
- relation targets are bbox-derived heuristics
- current object F1 is not 3D detection AP
- SpatialLM was evaluated through available `all/arch/object` layout interfaces, not native relation triples
""",
encoding="utf-8",
)

print("wrote", OUT)
