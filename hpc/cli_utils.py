from __future__ import annotations

import argparse
from typing import Any


def parse_bool_flag(value: Any) -> bool:
    """Best-effort boolean parser for CLI arguments."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got '{value}'")


def coerce_str_bool_none(
    args_dict: dict[str, Any],
    literal_none_keys: set[str],
    bool_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Normalize string CLI values representing booleans or None tokens."""

    bool_keys = set(bool_keys or [])

    for key, value in list(args_dict.items()):
        if not isinstance(value, str):
            continue
        lowered = value.strip().lower()
        if key in bool_keys and lowered in {"true", "false", "1", "0", "yes", "no", "y", "n", "on", "off"}:
            args_dict[key] = parse_bool_flag(lowered)
        elif lowered == "none":
            args_dict[key] = lowered if key in literal_none_keys else None
    return args_dict


def coerce_numeric_cli_values(args_dict: dict[str, Any]) -> dict[str, Any]:
    """Cast well-known numeric CLI arguments when passed as strings."""

    numeric_fields = {
        "adam_beta1": float,
        "adam_beta2": float,
        "learning_rate": float,
        "warmup_ratio": float,
        "weight_decay": float,
        "max_grad_norm": float,
        "num_train_epochs": float,
        "max_steps": int,
        "chunk_size": int,
    }
    for key, caster in numeric_fields.items():
        if key not in args_dict or args_dict[key] is None:
            continue
        value = args_dict[key]
        if isinstance(value, (int, float)):
            continue
        try:
            args_dict[key] = caster(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected {key} to be {caster.__name__}-like, got {value!r}") from exc
    return args_dict


__all__ = [
    "parse_bool_flag",
    "coerce_str_bool_none",
    "coerce_numeric_cli_values",
]
