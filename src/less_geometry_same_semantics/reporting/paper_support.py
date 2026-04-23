"""Generate a lightweight paper-writing support package."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


SUPPORT_FILES = [
    "problem_statement.md",
    "main_claims_template.md",
    "experiment_setup_summary.md",
    "dataset_summary.md",
    "evaluation_metrics_summary.md",
    "main_results_summary_placeholders.md",
    "limitations.md",
    "future_work.md",
]


def generate_paper_support_package(
    output_dir: str | Path,
    template_dir: str | Path = "docs/paper_support",
    context: dict[str, Any] | None = None,
) -> None:
    """Copy paper-support scaffolds and add an index file."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    templates = Path(template_dir)
    for filename in SUPPORT_FILES:
        src = templates / filename
        if src.exists():
            shutil.copyfile(src, out / filename)
    _write_index(out, context or {})


def _write_index(out: Path, context: dict[str, Any]) -> None:
    lines = [
        "# Paper Support Package",
        "",
        "This directory is a writing scaffold generated from the experiment workflow.",
        "",
        "Recommended first files:",
        "",
        "- `problem_statement.md`",
        "- `main_claims_template.md`",
        "- `main_results_summary_placeholders.md`",
        "- `limitations.md`",
        "",
        "Linked artifact directories:",
    ]
    for key, value in sorted(context.items()):
        lines.append(f"- `{key}`: `{value}`")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
