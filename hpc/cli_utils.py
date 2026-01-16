from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Re-export from harbor_utils for backwards compatibility
from hpc.harbor_utils import run_harbor_cli


# =============================================================================
# Path Resolution Utilities
# =============================================================================


def looks_like_file_path(value: str) -> bool:
    """Check if a string looks like a file path (not an HF repo ID).

    Uses is_hf_dataset_path() from hf_utils to detect HF repos, and adds
    additional heuristics for file path detection.

    Args:
        value: String to check.

    Returns:
        True if value looks like a file path, False otherwise.
    """
    from hpc.hf_utils import is_hf_dataset_path

    if not isinstance(value, str) or not value:
        return False

    # If it matches HF dataset pattern, it's not a file path
    if is_hf_dataset_path(value):
        return False

    # Absolute paths or home-relative paths
    if value.startswith("/") or value.startswith("~"):
        return True

    # Relative paths with ./ or ../
    if value.startswith("./") or value.startswith("../"):
        return True

    # Has a file extension (common config/template/data files)
    path_extensions = {
        ".yaml", ".yml", ".json", ".jsonl", ".txt", ".md",
        ".py", ".sh", ".jinja", ".jinja2", ".j2",
        ".parquet", ".csv", ".tsv", ".arrow",
        ".safetensors", ".bin", ".pt", ".pth", ".ckpt",
        ".toml", ".ini", ".cfg", ".conf",
    }
    lower = value.lower()
    if any(lower.endswith(ext) for ext in path_extensions):
        return True

    # Multiple slashes indicate nested path (more than org/repo)
    if value.count("/") > 1:
        return True

    return False


def resolve_paths_in_dict(
    config: Dict[str, Any],
    base_dir: Optional[Path] = None,
    skip_keys: Optional[Set[str]] = None,
    _prefix: str = "",
) -> Dict[str, Any]:
    """Recursively resolve file paths in a config dictionary.

    Walks through the dict, identifies values that look like file paths
    (not HF repo IDs), and resolves them to absolute paths using PROJECT_ROOT.

    Args:
        config: Configuration dictionary to process.
        base_dir: Base directory for resolving relative paths. If None,
            uses resolve_repo_path() which defaults to PROJECT_ROOT.
        skip_keys: Set of dotted key names to skip (e.g., {"data.train_data"}).
        _prefix: Internal - tracks current key path for skip_keys matching.

    Returns:
        New dictionary with paths resolved (original is not modified).
    """
    from hpc.launch_utils import resolve_repo_path

    skip_keys = skip_keys or set()
    result = {}

    for key, value in config.items():
        full_key = f"{_prefix}.{key}" if _prefix else key

        # Skip explicitly excluded keys
        if full_key in skip_keys:
            result[key] = value
            continue

        if isinstance(value, dict):
            # Recurse into nested dicts
            result[key] = resolve_paths_in_dict(value, base_dir, skip_keys, full_key)
        elif isinstance(value, list):
            # Process list items
            resolved_list = []
            for item in value:
                if isinstance(item, str) and looks_like_file_path(item):
                    if base_dir:
                        resolved = (base_dir / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()
                        resolved_list.append(str(resolved))
                    else:
                        resolved_list.append(str(resolve_repo_path(item)))
                elif isinstance(item, dict):
                    resolved_list.append(resolve_paths_in_dict(item, base_dir, skip_keys, full_key))
                else:
                    resolved_list.append(item)
            result[key] = resolved_list
        elif isinstance(value, str) and looks_like_file_path(value):
            # Resolve path-like strings
            if base_dir:
                resolved = (base_dir / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
                result[key] = str(resolved)
            else:
                result[key] = str(resolve_repo_path(value))
        else:
            result[key] = value

    return result


def parse_comma_separated(value: str) -> List[str]:
    """Parse comma-separated string into list of stripped values.

    Args:
        value: Comma-separated string (e.g., "a, b, c")

    Returns:
        List of stripped non-empty values
    """
    return [v.strip() for v in value.split(",") if v.strip()]


def normalize_cli_args(raw_args: Any) -> List[str]:
    """Normalize extra CLI args to a list of strings.

    Handles various input formats:
    - None/empty: returns []
    - String: splits on whitespace
    - List/tuple: converts each element to string

    Args:
        raw_args: CLI arguments in various formats

    Returns:
        List of string arguments
    """
    if not raw_args:
        return []
    if isinstance(raw_args, str):
        return raw_args.split()
    if isinstance(raw_args, (list, tuple)):
        return [str(arg) for arg in raw_args]
    return []


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


def is_nullish(value: Any) -> bool:
    """Check if a value is nullish (not explicitly set by user).

    Returns True for values that indicate "not set":
    - None
    - Empty string ""
    - Whitespace-only strings

    Useful for determining whether to apply auto-derived defaults.

    Args:
        value: Any value to check.

    Returns:
        True if the value is nullish, False otherwise.

    Examples:
        >>> is_nullish(None)
        True
        >>> is_nullish("")
        True
        >>> is_nullish("   ")
        True
        >>> is_nullish(False)  # Explicit False is NOT nullish
        False
        >>> is_nullish(0)  # Explicit 0 is NOT nullish
        False
        >>> is_nullish("value")
        False
    """
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def normalize_job_type(exp_args: dict) -> str | None:
    """Normalize job_type string without applying a default.

    Args:
        exp_args: Experiment arguments dict

    Returns:
        Normalized job_type string (lowercase, stripped) or None if not set
    """
    raw_value = exp_args.get("job_type")
    if is_nullish(raw_value):
        return None
    return str(raw_value).strip().lower()


__all__ = [
    "looks_like_file_path",
    "resolve_paths_in_dict",
    "parse_comma_separated",
    "normalize_cli_args",
    "normalize_job_type",
    "is_nullish",
    "parse_bool_flag",
    "coerce_str_bool_none",
    "coerce_numeric_cli_values",
    "run_harbor_cli",
]
