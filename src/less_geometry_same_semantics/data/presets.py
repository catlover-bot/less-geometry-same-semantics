"""Default corruption presets used when YAML does not override them."""

from __future__ import annotations

from typing import Any

DEFAULT_CORRUPTION_PRESETS: dict[str, dict[str, Any]] = {
    "clean": {
        "corruption": {
            "preset": "clean",
            "geometry_degradation": {"enabled": False, "severity": "none"},
            "coordinate_perturbation": {"enabled": False, "severity": "none"},
            "local_structural_corruption": {"enabled": False, "severity": "none"},
            "token_point_compression": {"enabled": False, "severity": "none"},
        }
    },
    "mild_corruption": {
        "corruption": {
            "preset": "mild_corruption",
            "geometry_degradation": {"enabled": True, "severity": "mild"},
            "coordinate_perturbation": {"enabled": True, "severity": "mild"},
            "local_structural_corruption": {"enabled": False, "severity": "none"},
            "token_point_compression": {"enabled": True, "severity": "mild"},
        }
    },
    "medium_corruption": {
        "corruption": {
            "preset": "medium_corruption",
            "geometry_degradation": {"enabled": True, "severity": "medium"},
            "coordinate_perturbation": {"enabled": True, "severity": "medium"},
            "local_structural_corruption": {"enabled": True, "severity": "medium"},
            "token_point_compression": {"enabled": True, "severity": "medium"},
        }
    },
    "severe_corruption": {
        "corruption": {
            "preset": "severe_corruption",
            "geometry_degradation": {"enabled": True, "severity": "severe"},
            "coordinate_perturbation": {"enabled": True, "severity": "severe"},
            "local_structural_corruption": {"enabled": True, "severity": "severe"},
            "token_point_compression": {"enabled": True, "severity": "severe"},
        }
    },
    "extreme_compression": {
        "corruption": {
            "preset": "extreme_compression",
            "geometry_degradation": {"enabled": True, "severity": "medium"},
            "coordinate_perturbation": {"enabled": True, "severity": "severe"},
            "local_structural_corruption": {"enabled": True, "severity": "severe"},
            "token_point_compression": {
                "enabled": True,
                "severity": "extreme",
                "density_fraction": 0.08,
                "random_dropout_prob": 0.55,
            },
        }
    },
    "target_8192_points": {
        "corruption": {
            "preset": "target_8192_points",
            "token_point_compression": {
                "enabled": True,
                "severity": "medium",
                "target_point_budget": 8192,
                "random_dropout_prob": 0.0
            }
        }
    },
    "target_2048_points": {
        "corruption": {
            "preset": "target_2048_points",
            "token_point_compression": {
                "enabled": True,
                "severity": "severe",
                "target_point_budget": 2048,
                "random_dropout_prob": 0.0
            }
        }
    },
}
