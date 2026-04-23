"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re

import yaml

ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)|%([A-Za-z_][A-Za-z0-9_]*)%")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return _expand_env_vars(data)


def recursive_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a config dictionary and return a new dictionary."""

    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = recursive_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def expand_env_vars(value: Any) -> Any:
    """Expand environment variables in strings, lists, and dictionaries."""

    return _expand_env_vars(value)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return _expand_env_string(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def _expand_env_string(value: str) -> str:
    """Expand Unix and Windows env placeholders consistently on all platforms.

    Python's ``os.path.expandvars`` is platform-sensitive. The configs use
    ``${ARKITSCENES_ROOT}``, so this helper expands
    ``${VAR}``, ``$VAR``, and ``%VAR%`` even when running on Windows.
    Missing variables are left untouched so preflight checks can report them.
    """

    def replace(match: re.Match[str]) -> str:
        name = next(group for group in match.groups() if group is not None)
        return os.environ.get(name, match.group(0))

    return os.path.expanduser(ENV_VAR_PATTERN.sub(replace, value))


def find_unresolved_env_vars(value: Any) -> list[str]:
    """Return unresolved environment variable names in a nested config value."""

    missing: set[str] = set()
    _collect_unresolved_env_vars(value, missing)
    return sorted(missing)


def _collect_unresolved_env_vars(value: Any, missing: set[str]) -> None:
    if isinstance(value, str):
        for match in ENV_VAR_PATTERN.finditer(value):
            name = next(group for group in match.groups() if group is not None)
            if name not in os.environ:
                missing.add(name)
    elif isinstance(value, list):
        for item in value:
            _collect_unresolved_env_vars(item, missing)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_unresolved_env_vars(item, missing)
