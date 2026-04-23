"""Paper experiment plan loading and expected-condition helpers."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from less_geometry_same_semantics.utils.config import load_config


def load_experiment_plan(path: str | Path = "configs/paper_plan.yaml") -> dict[str, Any]:
    """Load the paper experiment plan YAML."""

    return load_config(path)


def plan_path(plan: dict[str, Any], *parts: str) -> Path:
    """Resolve a path inside the paper package output directory."""

    root = Path(plan.get("paper", {}).get("output_dir", "outputs/paper_package"))
    return root.joinpath(*parts)


def expected_main_case_names(plan: dict[str, Any]) -> list[str]:
    """Return expected main-matrix case names from the plan."""

    expected = plan.get("main_benchmark", {}).get("expected_conditions", {})
    corruptions = expected.get("corruption", ["clean", "severe_corruption"])
    point_budgets = expected.get("point_budget", ["raw", "compressed"])
    graphs = expected.get("graph", ["no_graph", "simple_graph"])
    constrained_values = expected.get("constrained", [False, True])
    adaptations = expected.get("adaptation", ["none", "input_normalization"])
    names = []
    for corruption, point_budget, graph, constrained, adaptation in product(
        corruptions,
        point_budgets,
        graphs,
        constrained_values,
        adaptations,
    ):
        names.append(
            f"{corruption}__{point_budget}__{graph}__{'constrained' if constrained else 'unconstrained'}__{adaptation}"
        )
    return names
