"""Shared naming rules for collision-forked experiment directories."""

from __future__ import annotations

import re


def numbered_experiment_fork_name(base_name: str, suffix: int) -> str:
    """Return the sibling name used for a collision-forked experiment."""
    return f"{base_name}_{suffix}"


def numbered_experiment_fork_pattern(base_name: str) -> re.Pattern[str]:
    """Match numbered collision forks of one canonical experiment name."""
    return re.compile(rf"{re.escape(base_name)}_(\d+)$")
