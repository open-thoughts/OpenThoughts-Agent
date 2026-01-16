"""
Utility helpers shared across HPC launch entry points.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

from hpc.hpc import detect_hpc

# Re-export HuggingFace utilities for backwards compatibility
from hpc.hf_utils import sanitize_hf_repo_id

from .job_name_ignore_list import JOB_NAME_IGNORE_KEYS
from .arguments import JobType
from .cli_utils import normalize_job_type

# =============================================================================
# Type Aliases
# =============================================================================

PathInput = Union[str, PathLike[str], Path, None]
"""Flexible path input type for utility functions."""

_VALID_TRACE_BACKENDS = {"vllm", "ray", "vllm_local", "none"}
"""Valid backend options for trace generation."""

_HOSTED_VLLM_PREFIX = "hosted_vllm/"
"""Provider prefix expected by LiteLLM when routing to managed vLLM endpoints."""

# Cloud/SkyPilot job name length limit (DNS label constraint)
CLOUD_JOB_NAME_MAX_LENGTH = 63
"""Maximum job name length for cloud runs (SkyPilot/Kubernetes DNS label limit)."""


def truncate_for_cloud(job_name: str) -> str:
    """Truncate a job name to be compatible with cloud/SkyPilot runs.

    SkyPilot cluster names must be valid DNS labels, which are limited to 63 chars.
    This function is used by cloud launchers; HPC/SLURM launchers don't need it.

    Args:
        job_name: The job name to truncate.

    Returns:
        Job name truncated to 63 characters.
    """
    return job_name[:CLOUD_JOB_NAME_MAX_LENGTH]


# =============================================================================
# Memory Scaling Utilities
# =============================================================================


def parse_memory_string(mem_str: str) -> int:
    """Parse a memory string (e.g., '710GB', '192G', '188130M') to megabytes.

    Args:
        mem_str: Memory string with unit suffix (G/GB/M/MB/T/TB).

    Returns:
        Memory in megabytes.

    Raises:
        ValueError: If the memory string cannot be parsed.
    """
    if not mem_str:
        return 0

    mem_str = mem_str.strip().upper()
    match = re.match(r"^(\d+(?:\.\d+)?)\s*(TB?|GB?|MB?|KB?)?$", mem_str)
    if not match:
        raise ValueError(f"Cannot parse memory string: {mem_str}")

    value = float(match.group(1))
    unit = match.group(2) or "M"  # Default to MB if no unit

    # Convert to MB
    if unit.startswith("T"):
        return int(value * 1024 * 1024)
    elif unit.startswith("G"):
        return int(value * 1024)
    elif unit.startswith("K"):
        return int(value / 1024)
    else:  # M or MB
        return int(value)


def format_memory_mb(mem_mb: int) -> str:
    """Format memory in MB to a human-readable string.

    Args:
        mem_mb: Memory in megabytes.

    Returns:
        Formatted memory string (e.g., '188G', '512M').
    """
    if mem_mb >= 1024:
        return f"{mem_mb // 1024}G"
    return f"{mem_mb}M"


def scale_memory_for_partial_gpus(
    mem_str: str,
    requested_gpus: int,
    total_gpus: int,
) -> str:
    """Scale memory request proportionally to GPU allocation.

    When requesting fewer GPUs than available on a node, some schedulers
    (e.g., ZIH Capella) require memory to be scaled proportionally.

    Args:
        mem_str: Full node memory string (e.g., '710GB').
        requested_gpus: Number of GPUs being requested.
        total_gpus: Total GPUs available per node.

    Returns:
        Scaled memory string (e.g., '177G' for 1/4 of 710GB).
    """
    if not mem_str or requested_gpus <= 0 or total_gpus <= 0:
        return mem_str

    # If requesting all GPUs, no scaling needed
    if requested_gpus >= total_gpus:
        return mem_str

    total_mb = parse_memory_string(mem_str)
    scaled_mb = (total_mb * requested_gpus) // total_gpus

    return format_memory_mb(scaled_mb)


def generate_served_model_id() -> str:
    """Return a unique identifier for a hosted vLLM model."""
    return str(int(time.time() * 1_000_000))


def hosted_vllm_alias(served_id: str) -> str:
    """Build the hosted_vllm/<id> alias LiteLLM expects."""
    if not served_id:
        raise ValueError("served_id must be a non-empty string")
    return f"{_HOSTED_VLLM_PREFIX}{served_id}"


def is_hosted_vllm_alias(model_name: Optional[str]) -> bool:
    """Check whether ``model_name`` already has the hosted_vllm prefix."""
    return bool(model_name) and str(model_name).startswith(_HOSTED_VLLM_PREFIX)


def strip_hosted_vllm_alias(model_name: Optional[str]) -> str:
    """Remove the hosted_vllm prefix so vLLM can load the underlying HF model."""
    if not model_name:
        return ""
    if is_hosted_vllm_alias(model_name):
        return str(model_name)[len(_HOSTED_VLLM_PREFIX) :]
    return str(model_name)


# =============================================================================
# Endpoint File Utilities
# =============================================================================

def cleanup_endpoint_file(path_like: PathInput, *, descriptor: str = "endpoint file") -> None:
    """Remove a stale endpoint JSON if it exists."""

    if not path_like:
        return
    try:
        candidate = Path(path_like).expanduser()
    except Exception:
        return
    if not candidate.exists():
        return
    try:
        candidate.unlink()
        print(f"Removed {descriptor}: {candidate}")
    except OSError as exc:
        print(f"Warning: failed to remove {descriptor} {candidate}: {exc}")


def validate_trace_backend(
    backend_value: Optional[str],
    *,
    allow_vllm: bool,
    job_type: str,
) -> str:
    """Normalize and validate the requested trace backend."""

    backend = (backend_value or "vllm").strip().lower()
    if backend not in _VALID_TRACE_BACKENDS:
        raise ValueError(
            f"Unsupported trace backend '{backend_value}'. "
            f"Valid options: {sorted(_VALID_TRACE_BACKENDS)}"
        )
    if backend == "vllm" and not allow_vllm:
        raise RuntimeError(
            f"trace_backend=vllm is not supported for {job_type} jobs. "
            "Use a Ray-backed backend or disable trace generation."
        )
    return backend


# =============================================================================
# CLI Argument Normalization
# =============================================================================

def normalize_cli_args(args_spec: Any) -> list[str]:
    """Normalize a YAML-provided CLI arg spec into a flat list of strings.

    Supports multiple input formats:
    - String: split using shlex (e.g., "--foo bar --baz")
    - Dict: convert to --key value pairs (booleans become flags)
    - List/Tuple: convert items to strings

    Args:
        args_spec: CLI arguments in any supported format.

    Returns:
        Flat list of CLI argument strings.

    Raises:
        TypeError: If args_spec is not a supported type.
    """
    if args_spec in (None, "", [], (), {}):
        return []

    if isinstance(args_spec, str):
        return shlex.split(args_spec)

    if isinstance(args_spec, dict):
        normalized: list[str] = []
        for key, value in args_spec.items():
            flag = key if str(key).startswith("--") else f"--{key}"
            if isinstance(value, bool):
                if value:
                    normalized.append(flag)
                continue
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    if item is None:
                        continue
                    if isinstance(item, bool):
                        if item:
                            normalized.append(flag)
                        continue
                    normalized.extend([flag, str(item)])
            else:
                normalized.extend([flag, str(value)])
        return normalized

    if isinstance(args_spec, (list, tuple)):
        return [str(item) for item in args_spec if item is not None]

    raise TypeError(
        f"Unsupported CLI args specification of type {type(args_spec).__name__}; "
        "expected string, list/tuple, or mapping."
    )


# =============================================================================
# Dict Utilities
# =============================================================================

def set_or_pop(d: dict, key: str, value) -> None:
    """Set key in dict if value is not None, otherwise remove it.

    Useful for conditionally populating config dicts where None values
    should result in the key being absent rather than present with None.

    Args:
        d: Dictionary to modify in place.
        key: Key to set or remove.
        value: Value to set, or None to remove the key.
    """
    if value is not None:
        d[key] = value
    else:
        d.pop(key, None)


# =============================================================================
# Global Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""Root directory of the OpenThoughts-Agent project."""


# =============================================================================
# Path Resolution Utilities
# =============================================================================

def resolve_repo_path(path_like: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute.

    Args:
        path_like: A path string that may be relative or absolute.

    Returns:
        Resolved absolute Path object.
    """
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_workspace_path(path_like: str) -> Path:
    """Resolve a workspace path, keeping absolute paths as-is.

    Args:
        path_like: A path string that may be relative or absolute.

    Returns:
        Resolved Path object (absolute paths kept as-is, relative resolved to PROJECT_ROOT).
    """
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@dataclass
class ExperimentsPaths:
    """Paths for experiment artifacts."""
    root: Path
    sbatch: Path
    configs: Path
    logs: Path


def setup_experiments_dir(
    exp_args: dict,
    *,
    job_name: Optional[str] = None,
    create_dirs: bool = True,
    sbatch_subdir: str = "sbatch",
) -> ExperimentsPaths:
    """Resolve experiments directory and create standard subdirectories.

    This consolidates the common pattern of setting up experiment paths
    used across datagen, eval, and other job launchers.

    Args:
        exp_args: Experiment arguments dict containing optional "experiments_dir".
        job_name: Optional job name. When experiments_dir is not explicitly set,
                  the default becomes "experiments/<job_name>" instead of just
                  "experiments". This keeps each job's outputs isolated.
        create_dirs: Whether to create directories (default True).
        sbatch_subdir: Name of sbatch subdirectory (default "sbatch",
                       use "sbatch_scripts" for backwards compat where needed).

    Returns:
        ExperimentsPaths with root, sbatch, configs, and logs paths.
    """
    explicit_experiments_dir = exp_args.get("experiments_dir")
    if explicit_experiments_dir:
        # User explicitly set experiments_dir - use it as-is
        experiments_subdir = explicit_experiments_dir
    elif job_name:
        # Default to experiments/<job_name> when job_name is available
        experiments_subdir = f"experiments/{job_name}"
    else:
        # Fallback to just "experiments" (legacy behavior)
        experiments_subdir = "experiments"
    experiments_abs = resolve_workspace_path(experiments_subdir)

    paths = ExperimentsPaths(
        root=experiments_abs,
        sbatch=experiments_abs / sbatch_subdir,
        configs=experiments_abs / "configs",
        logs=experiments_abs / "logs",
    )

    if create_dirs:
        paths.sbatch.mkdir(parents=True, exist_ok=True)
        paths.configs.mkdir(parents=True, exist_ok=True)
        paths.logs.mkdir(parents=True, exist_ok=True)

    return paths


@dataclass
class JobSetupResult:
    """Result of resolve_job_and_paths()."""
    job_name: str
    paths: ExperimentsPaths


def resolve_job_and_paths(
    exp_args: dict,
    *,
    job_type_label: str = "job",
    derive_job_name_fn: Optional[Callable[[dict], str]] = None,
    sbatch_subdir: str = "sbatch",
) -> JobSetupResult:
    """Resolve job_name and experiments paths in one step.

    This consolidates the common pattern across launchers:
    1. Get job_name from exp_args (or derive it)
    2. Setup experiments directory using job_name

    Args:
        exp_args: Experiment arguments dict.
        job_type_label: Label for error messages (e.g., "Eval", "SFT", "Datagen").
        derive_job_name_fn: Optional function to derive job_name if not in exp_args.
                           If None and job_name is missing, raises ValueError.
        sbatch_subdir: Name of sbatch subdirectory (default "sbatch").

    Returns:
        JobSetupResult with job_name and paths.

    Raises:
        ValueError: If job_name is missing and no derive_job_name_fn provided.
    """
    job_name = exp_args.get("job_name")

    if not job_name:
        if derive_job_name_fn:
            job_name = derive_job_name_fn(exp_args)
        else:
            raise ValueError(f"{job_type_label} jobs require a --job_name.")

    paths = setup_experiments_dir(exp_args, job_name=job_name, sbatch_subdir=sbatch_subdir)

    return JobSetupResult(job_name=job_name, paths=paths)


def repo_relative(path_str: str, repo_root: Optional[Path] = None) -> str:
    """Convert an absolute path to repo-relative path.

    This is the inverse of resolve_repo_path() - it takes an absolute path
    and returns a POSIX-style path relative to the repo root.

    Args:
        path_str: Path to convert (absolute or relative)
        repo_root: Repository root (defaults to PROJECT_ROOT)

    Returns:
        POSIX-style path relative to repo root

    Raises:
        ValueError: If path is not inside the repo
    """
    if repo_root is None:
        repo_root = PROJECT_ROOT

    abs_path = Path(path_str).expanduser().resolve()
    try:
        relative = abs_path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path '{abs_path}' must live inside the repo ({repo_root})") from exc
    return relative.as_posix()


def resolve_config_path(
    raw_value: str,
    default_dir: Path | str,
    config_type: str = "config",
) -> Path:
    """Resolve a config path with fallback to a default directory.

    Tries paths in order:
    1. raw_value as-is (if exists)
    2. default_dir / raw_value
    3. default_dir / basename(raw_value)

    Args:
        raw_value: User-provided path string.
        default_dir: Default directory to check for configs.
        config_type: Description for error messages (e.g., "datagen", "harbor").

    Returns:
        Resolved absolute Path.

    Raises:
        FileNotFoundError: If config not found in any location.
    """
    candidate = Path(raw_value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    default_dir = Path(default_dir)
    default_candidate = default_dir / candidate
    if default_candidate.exists():
        return default_candidate.resolve()

    fallback_candidate = default_dir / candidate.name
    if fallback_candidate.exists():
        return fallback_candidate.resolve()

    raise FileNotFoundError(
        f"{config_type.capitalize()} config not found: {raw_value}. "
        f"Tried {candidate}, {default_candidate}, and {fallback_candidate}."
    )


def coerce_positive_int(value: Any, default: int) -> int:
    """Coerce a value to a positive integer, returning default if invalid.

    Args:
        value: Value to coerce (string, int, etc.)
        default: Default value if coercion fails or result is non-positive.

    Returns:
        Positive integer, or default.
    """
    try:
        parsed = int(str(value))
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def build_sbatch_directives(
    hpc,
    exp_args: dict,
    *,
    partition: str | None = None,
    account: str | None = None,
    qos: str | None = None,
    gpus: int | None = None,
    gpu_type: str | None = None,
    mem: str | None = None,
) -> list[str]:
    """Build list of SBATCH directives for job submission.

    Args:
        hpc: HPC configuration object.
        exp_args: Experiment arguments dict (used for fallback values).
        partition: Override partition (falls back to exp_args then hpc).
        account: Override account (falls back to exp_args then hpc).
        qos: Override QoS (falls back to exp_args).
        gpus: Override GPU count (falls back to exp_args then hpc).
        gpu_type: Override GPU type (e.g., "h200", "l40s"). Falls back to exp_args then hpc default.
        mem: Override memory (falls back to hpc.mem_per_node).

    Returns:
        List of SBATCH directive strings (e.g., ["#SBATCH -p gpu", ...]).
    """
    # Resolve values with fallbacks
    partition = partition or exp_args.get("partition") or hpc.partition
    account = account or exp_args.get("account") or hpc.account
    qos = qos or exp_args.get("qos") or ""
    gpus_requested = int(gpus if gpus is not None else (exp_args.get("gpus_per_node") or hpc.gpus_per_node or 0))
    gpu_type_resolved = gpu_type or exp_args.get("gpu_type") or None  # Let hpc.get_gpu_directive use its default

    directives = []
    if partition:
        directives.append(f"#SBATCH -p {partition}")
    if account:
        directives.append(f"#SBATCH --account {account}")
    if qos:
        directives.append(f"#SBATCH --qos {qos}")
    # Add GPU directive if the cluster uses one
    gpu_directive = hpc.get_gpu_directive(gpus_requested, gpu_type_resolved)
    if gpu_directive:
        directives.append(gpu_directive)

    # Scale memory proportionally when requesting partial GPUs
    # (required by some schedulers like ZIH Capella for fair sharing)
    total_gpus = hpc.gpus_per_node or 1
    if mem is None and gpus_requested < total_gpus and hpc.mem_per_node:
        scaled_mem = scale_memory_for_partial_gpus(
            hpc.mem_per_node,
            gpus_requested,
            total_gpus,
        )
        mem_directive = hpc.get_mem_directive(scaled_mem)
    else:
        mem_directive = hpc.get_mem_directive(mem)

    if mem_directive:
        directives.append(mem_directive)
    if hpc.node_exclusion_list:
        directives.append(f"#SBATCH --exclude={hpc.node_exclusion_list}")
    # Add constraint directive based on GPU type (e.g., Perlmutter A100 variants)
    gpu_type_constraints = getattr(hpc, "gpu_type_constraints", {})
    if gpu_type_constraints:
        constraint_key = gpu_type_resolved if gpu_type_resolved else "_default"
        constraint = gpu_type_constraints.get(constraint_key)
        if constraint:
            directives.append(f"#SBATCH --constraint {constraint}")
    # Add any extra cluster-specific directives (e.g., licenses)
    for directive in getattr(hpc, "extra_sbatch_directives", []):
        directives.append(directive)

    return directives


# =============================================================================
# JSON/Config Parsing Utilities
# =============================================================================

def coerce_agent_kwargs(value: Any) -> Dict[str, Any]:
    """Parse agent kwargs from various input formats.

    Args:
        value: None, empty string, dict, or JSON string.

    Returns:
        Dictionary of agent kwargs.

    Raises:
        ValueError: If the value cannot be parsed as a dict.
    """
    if value in (None, "", {}):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse agent kwargs JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Agent kwargs must decode to an object/dict.")
        return parsed
    raise ValueError("Agent kwargs must be provided as JSON string or dict.")


# =============================================================================
# vLLM Endpoint Utilities
# =============================================================================

def default_vllm_endpoint_path(
    experiments_dir: str | os.PathLike[str],
    *,
    trace: bool = False,
    chunk_index: int | None = None,
) -> str:
    """Compute a canonical vLLM endpoint JSON path under experiments_dir.

    Args:
        experiments_dir: Base experiments directory.
        trace: Whether the path is for trace collection (adds trace-specific suffix).
        chunk_index: Optional chunk index for sharded trace jobs.

    Returns:
        String path to the endpoint JSON file.
    """
    base = Path(experiments_dir).expanduser()

    if trace:
        if chunk_index is not None:
            filename = f"vllm_endpoint_trace_{chunk_index:03d}.json"
        else:
            filename = "vllm_endpoint_trace.json"
    else:
        filename = "vllm_endpoint.json"

    return str(base / filename)


# =============================================================================
# Local Execution Utilities
# =============================================================================

def is_local_mode(hpc) -> bool:
    """Check if HPC config indicates local (non-SLURM) execution."""
    return bool(getattr(hpc, "local_mode", False))


def run_local_script(script_path: str) -> str:
    """Execute a script locally via bash.

    Args:
        script_path: Path to the bash script to execute.

    Returns:
        A fake job ID string for consistency.

    Raises:
        RuntimeError: If the script exits with non-zero status.
    """
    print(f"Running locally: bash {script_path}")
    result = subprocess.run(["bash", script_path], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Local execution failed (exit {result.returncode}) for {script_path}")
    return f"local_{Path(script_path).stem}"


def submit_script(
    script_path: str,
    *,
    dependency: str | None = None,
    array: str | None = None,
    hpc=None,
) -> str:
    """Submit a script via sbatch or run locally based on HPC config.

    Args:
        script_path: Path to the sbatch script.
        dependency: Optional SLURM dependency string.
        array: Optional SLURM array specification.
        hpc: HPC configuration object.

    Returns:
        Job ID string.
    """
    if is_local_mode(hpc):
        if dependency:
            print(f"Warning: ignoring job dependency '{dependency}' for local execution.")
        if array:
            raise RuntimeError("Job arrays are not supported for local execution.")
        return run_local_script(script_path)
    return launch_sbatch(script_path, dependency=dependency, array=array)


def sanitize_repo_for_job(repo_id: str, keep_periods: bool = True) -> str:
    """Return a filesystem-safe representation of a repo identifier.

    Args:
        repo_id: The identifier to sanitize.
        keep_periods: If True, periods are kept. If False, periods are replaced
                      with hyphens (useful for job names where .yaml etc. is unwanted).

    Returns:
        Sanitized string safe for filesystem and job names.
    """
    if keep_periods:
        safe = re.sub(r"[^A-Za-z0-9._\-]+", "-", repo_id.strip())
    else:
        safe = re.sub(r"[^A-Za-z0-9_\-]+", "-", repo_id.strip())
    safe = re.sub(r"-+", "-", safe)  # collapse multiple hyphens
    safe = safe.strip("-_")
    return safe or "job"


def sanitize_repo_component(value: Optional[str]) -> Optional[str]:
    """Extract the meaningful suffix from trace repositories (traces-<slug>)."""

    if not value:
        return None
    match = re.search(r"traces-([A-Za-z0-9._\-]+)", value)
    return match.group(1) if match else None


def derive_datagen_job_name(cli_args: Mapping[str, Any]) -> str:
    """Construct a fallback job name for datagen/trace launches."""

    def _sanitize_component(value: str) -> str:
        value = value.strip().rstrip("/")
        if "/" in value:
            value = value.split("/")[-1]
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_") or "repo"

    def _extract_harbor_config_label(config_path: str) -> str:
        """Extract a short label from harbor config path (e.g., '16concurrency_eval')."""
        filename = Path(config_path).stem  # Remove .yaml extension
        # Remove common prefixes like 'trace_'
        for prefix in ("trace_", "harbor_"):
            if filename.startswith(prefix):
                filename = filename[len(prefix):]
        # Truncate if too long (keep first 30 chars)
        if len(filename) > 30:
            filename = filename[:30].rstrip("-_")
        return filename

    job_type_hint = str(cli_args.get("job_type") or "").lower()
    prefix = "eval" if job_type_hint == JobType.EVAL.value else "datagen"
    parts: list[str] = [prefix]

    repo_candidate = cli_args.get("datagen_target_repo") or cli_args.get("trace_target_repo")
    model_candidate = cli_args.get("datagen_model") or cli_args.get("trace_model")
    if model_candidate:
        parts.append(_sanitize_component(str(model_candidate)))
    elif repo_candidate:
        parts.append(_sanitize_component(str(repo_candidate)))

    # Include harbor config label for better job identification
    harbor_config = cli_args.get("trace_harbor_config")
    if harbor_config:
        config_label = _extract_harbor_config_label(str(harbor_config))
        if config_label:
            parts.append(config_label)

    dataset_component = None
    dataset_slug = cli_args.get("harbor_dataset")
    dataset_path = cli_args.get("tasks_input_path") or cli_args.get("eval_dataset_path")
    if dataset_slug:
        dataset_component = _sanitize_component(str(dataset_slug))
    elif dataset_path:
        dataset_component = _sanitize_component(str(dataset_path))
    if dataset_component:
        parts.append(dataset_component)

    job_name = "_".join(filter(None, parts))
    if job_type_hint == JobType.EVAL.value:
        if job_name.startswith("eval_"):
            job_name = "eval-" + job_name[len("eval_"):]
        elif job_name == "eval":
            job_name = "eval-run"
        elif not job_name.startswith("eval-"):
            job_name = f"eval-{job_name}"
    return job_name or "datagen_job"


def derive_consolidate_job_name(cli_args: Mapping[str, Any]) -> str:
    """Construct a consolidate-specific job name with a fixed suffix."""

    identifier_raw = (
        cli_args.get("consolidate_input")
        or cli_args.get("consolidate_output_repo")
        or cli_args.get("consolidate_base_repo")
        or "consolidate"
    )
    identifier = sanitize_repo_for_job(str(identifier_raw))
    suffix = "_consolidate"
    max_prefix_len = max(1, 96 - len(suffix))
    if len(identifier) > max_prefix_len:
        identifier = identifier[:max_prefix_len]
    return f"{identifier}{suffix}"


def derive_default_job_name(cli_args: Mapping[str, Any]) -> str:
    """Construct job names for non-datagen, non-consolidate workloads."""

    job_name_components: list[str] = []
    job_name_suffix: Optional[str] = None

    for key, value in cli_args.items():
        if not isinstance(value, (str, int, float)):
            continue
        if value == "None" or key in JOB_NAME_IGNORE_KEYS:
            continue

        if key == "seed":
            try:
                if float(value) == 42:
                    continue
            except (TypeError, ValueError):
                pass

        if key not in {"dataset", "model_name_or_path"}:
            job_name_components.append(str(key).replace("_", "-"))

        value_str = str(value)
        if value_str == "Qwen/Qwen2.5-32B-Instruct":
            job_name_suffix = "_32B"
        elif value_str == "Qwen/Qwen2.5-14B-Instruct":
            job_name_suffix = "_14B"
        elif value_str == "Qwen/Qwen2.5-3B-Instruct":
            job_name_suffix = "_3B"
        elif value_str == "Qwen/Qwen2.5-1.5B-Instruct":
            job_name_suffix = "_1.5B"
        else:
            job_name_components.append(value_str.split("/")[-1])

    job_name = "_".join(job_name_components)
    job_name = sanitize_repo_for_job(job_name, keep_periods=False)
    if job_name_suffix:
        job_name += job_name_suffix

    if len(job_name) > 96:
        print("Truncating job name to less than HF limit of 96 characters...")
        job_name = "_".join(
            "-".join(segment[:4] for segment in chunk.split("-"))
            for chunk in job_name.split("_")
        )
        if len(job_name) > 96:
            raise ValueError(
                f"Job name {job_name} is still too long (96 characters) after truncation. "
                "Try renaming the dataset or providing a shorter YAML config."
            )

    return job_name or "ot_agent_job"


def get_job_name(cli_args: Mapping[str, Any]) -> str:
    """Derive a stable job name from user-provided CLI arguments."""

    job_type = str(cli_args.get("job_type", JobType.default_value()) or JobType.default_value()).lower()
    if job_type == JobType.CONSOLIDATE.value:
        return derive_consolidate_job_name(cli_args)
    if job_type in (JobType.DATAGEN.value, JobType.EVAL.value):
        return derive_datagen_job_name(cli_args)

    # SFT, RL, and other job types use derive_default_job_name which includes
    # all CLI args (learning_rate, batch_size, etc.) except those in JOB_NAME_IGNORE_KEYS
    base_name = derive_default_job_name(cli_args)

    # Add job type prefix
    if job_type == JobType.RL.value:
        return f"rl_{base_name}"
    if job_type == JobType.SFT.value:
        return f"sft_{base_name}"
    if job_type == JobType.SFT_MCA.value:
        return f"sft_mca_{base_name}"
    return base_name

def _parse_optional_int(value: Any, label: Optional[str] = None) -> Optional[int]:
    """Parse a value as int, returning None if empty/missing.

    Args:
        value: Value to parse (int, float, str, or None)
        label: If provided, raises ValueError with this label on invalid input.
               If None, returns None on invalid input (permissive mode).

    Returns:
        Parsed integer or None

    Raises:
        ValueError: If label is provided and value is invalid (strict mode)
    """
    if value in (None, "", "None"):
        return None
    if isinstance(value, bool):
        if label:
            raise ValueError(f"{label} must be an integer, got boolean {value!r}")
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        if label:
            raise ValueError(f"{label} must be an integer, got {value!r}") from exc
        return None


def parse_bool_with_default(value: Any, default: bool) -> bool:
    """Parse a value as boolean with a default for None/missing values.

    Handles CLI argument quirks where booleans may arrive as strings like "false".

    Args:
        value: Value to parse (bool, str, int, or None)
        default: Default value to return if value is None

    Returns:
        Parsed boolean value

    Examples:
        >>> parse_bool_with_default(None, True)
        True
        >>> parse_bool_with_default("false", True)
        False
        >>> parse_bool_with_default(False, True)
        False
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    # Handle string values like "false", "true", "0", "1"
    return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}


def maybe_int(value: Any) -> Optional[int]:
    """Parse a value as int, returning None if not possible.

    This is a permissive alias for _parse_optional_int(value, label=None).
    """
    return _parse_optional_int(value, label=None)


def apply_env_overrides(
    exp_args: dict,
    cli_args_filtered: dict,
    hpc,
    *,
    apply_mca_template_fn: Optional[Any] = None,
    apply_cluster_overrides_fn: Optional[Any] = None,
    prepare_datagen_fn: Optional[Any] = None,
    prepare_eval_fn: Optional[Any] = None,
) -> tuple[dict, str, Optional[Any]]:
    """Normalize resource overrides, defaults, and job-type specific toggles.

    This function preprocesses experiment arguments before job dispatch:
    - Normalizes GPU/CPU counts
    - Validates cluster-specific constraints (e.g., Perlmutter 4 GPUs)
    - Sets default time limits from HPC config
    - Validates job_type and job_creator
    - Applies job-type specific configurations via callbacks

    Args:
        exp_args: Experiment arguments dict
        cli_args_filtered: CLI arguments dict (without internal keys)
        hpc: HPC cluster configuration object
        apply_mca_template_fn: Callback for MCA template (SFT_MCA jobs)
        apply_cluster_overrides_fn: Callback for cluster-specific overrides
        prepare_datagen_fn: Callback for datagen configuration
        prepare_eval_fn: Callback for eval configuration

    Returns:
        Tuple of (updated exp_args, job_type string, datagen_runtime or None)

    Raises:
        ValueError: If job_type is missing or invalid, or constraint violations
    """
    hpc_name = str(getattr(hpc, "name", "") or "").lower()

    # Perlmutter requires exactly 4 GPUs per node
    if hpc_name == "perlmutter":
        requested = cli_args_filtered.get("gpus_per_node") or exp_args.get("gpus_per_node")
        if requested not in (None, "", "None"):
            requested_int = _parse_optional_int(requested, "--gpus_per_node")
            if requested_int is not None and requested_int != 4:
                raise ValueError("Perlmutter requires 4 GPUs per node.")
        exp_args = update_exp_args(exp_args, {"gpus_per_node": 4})

    # Normalize GPU count
    gpus_per_node_norm = _parse_optional_int(exp_args.get("gpus_per_node"), "--gpus_per_node") or 0
    exp_args = update_exp_args(exp_args, {"gpus_per_node": gpus_per_node_norm})

    # Normalize CPU counts
    cpus_per_node_norm = _parse_optional_int(exp_args.get("cpus_per_node"), "--cpus_per_node")
    if cpus_per_node_norm is not None:
        exp_args = update_exp_args(exp_args, {"cpus_per_node": cpus_per_node_norm})

    cpus_per_gpu_norm = _parse_optional_int(exp_args.get("cpus_per_gpu"), "--cpus_per_gpu")
    if cpus_per_gpu_norm is not None:
        exp_args = update_exp_args(exp_args, {"cpus_per_gpu": cpus_per_gpu_norm})

    cpus_per_node_cli_norm = _parse_optional_int(cli_args_filtered.get("cpus_per_node"), "--cpus_per_node")
    cpus_per_gpu_cli_norm = _parse_optional_int(cli_args_filtered.get("cpus_per_gpu"), "--cpus_per_gpu")

    # Handle cpus_per_gpu -> cpus_per_node derivation
    if cpus_per_gpu_cli_norm is not None:
        if cpus_per_node_cli_norm is not None:
            raise ValueError("Provide only one of --cpus_per_node or --cpus_per_gpu, not both.")
        if gpus_per_node_norm <= 0:
            raise ValueError("--cpus_per_gpu requires --gpus_per_node to be greater than zero.")
        cpus_per_node_norm = cpus_per_gpu_cli_norm * gpus_per_node_norm
        exp_args = update_exp_args(
            exp_args,
            {
                "cpus_per_gpu": cpus_per_gpu_cli_norm,
                "cpus_per_node": cpus_per_node_norm,
            },
        )
        cpus_per_gpu_norm = cpus_per_gpu_cli_norm
    else:
        if cpus_per_node_norm is None and cpus_per_gpu_norm is not None and gpus_per_node_norm > 0:
            cpus_per_node_norm = cpus_per_gpu_norm * gpus_per_node_norm
            exp_args = update_exp_args(exp_args, {"cpus_per_node": cpus_per_node_norm})
        elif (
            cpus_per_node_norm is not None
            and (cpus_per_gpu_norm is None or cpus_per_gpu_norm == 0)
            and gpus_per_node_norm > 0
        ):
            derived_cpus_per_gpu = max(1, math.ceil(cpus_per_node_norm / gpus_per_node_norm))
            exp_args = update_exp_args(exp_args, {"cpus_per_gpu": derived_cpus_per_gpu})

    # Validate job_creator
    job_creator = str(exp_args.get("job_creator", "mlfoundations-dev") or "mlfoundations-dev").strip()
    if not job_creator:
        raise ValueError("--job_creator must be a non-empty string.")
    if len(job_creator) > 96:
        raise ValueError("--job_creator must be 96 characters or fewer.")
    exp_args = update_exp_args(exp_args, {"job_creator": job_creator})

    # Set default time_limit from HPC config
    if exp_args.get("time_limit") in (None, ""):
        default_time = getattr(hpc, "default_time_limit", "24:00:00")
        exp_args = update_exp_args(exp_args, {"time_limit": default_time})
        print(f"Using default time_limit: {default_time}")

    # Normalize and validate job_type
    job_type = normalize_job_type(exp_args)
    if job_type is None:
        raise ValueError(
            f"--job_type is required. Valid options: {', '.join(jt.value for jt in JobType)}"
        )
    exp_args = update_exp_args(exp_args, {"job_type": job_type})

    # Handle MCA upgrade for SFT jobs
    if exp_args.get("use_mca") and job_type == JobType.SFT.value:
        job_type = JobType.SFT_MCA.value
        exp_args = update_exp_args(exp_args, {"job_type": job_type})

    if job_type == JobType.SFT_MCA.value:
        exp_args = update_exp_args(exp_args, {"use_mca": True})
        if apply_mca_template_fn is not None:
            exp_args = apply_mca_template_fn(
                exp_args,
                hpc,
                update_exp_args_fn=update_exp_args,
            )

    # Apply cluster-specific overrides
    if apply_cluster_overrides_fn is not None:
        exp_args = apply_cluster_overrides_fn(exp_args, hpc)

    # Prepare job-type specific configurations
    datagen_runtime = None
    if job_type == JobType.DATAGEN.value or exp_args.get("datagen_script"):
        if prepare_datagen_fn is not None:
            datagen_runtime = prepare_datagen_fn(exp_args)
    elif job_type == JobType.EVAL.value:
        if prepare_datagen_fn is not None:
            datagen_runtime = prepare_datagen_fn(exp_args)
        if prepare_eval_fn is not None:
            exp_args = prepare_eval_fn(exp_args)

    return exp_args, job_type, datagen_runtime


def _merge_dependencies(*deps: Optional[str]) -> Optional[str]:
    merged: list[str] = []
    for dep in deps:
        if not dep:
            continue
        dep_str = str(dep).strip()
        if not dep_str:
            continue
        merged.append(dep_str)
    if not merged:
        return None
    return ",".join(merged)


def launch_sbatch(sbatch_script_path, dependency=None, array: str | None = None) -> str:
    extra_args: list[str] = []
    if dependency is not None:
        extra_args.append(f"--dependency={dependency}")
    if array:
        extra_args.append(f"--array={array}")
    extra_flags = " ".join(extra_args)
    sbatch_cmd = f"sbatch {extra_flags} {sbatch_script_path}".strip()

    result = subprocess.run(
        sbatch_cmd,
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stdout.strip()
        err = result.stderr.strip()
        combined = "\n".join(filter(None, [msg, err]))
        raise RuntimeError(
            f"sbatch command failed (code {result.returncode}): {sbatch_cmd}\n{combined}"
        )

    raw_output = (result.stdout or "").strip()
    job_id = raw_output.split()[::-1][0]
    print(
        f"Job {job_id} submitted"
        f"{f' with dependency {dependency}' if dependency else ''}"
        f"{f' and array {array}' if array else ''}."
    )
    return job_id


def update_exp_args(exp_args, args, *, explicit_keys: Optional[set[str]] = None):
    explicit_keys = set(explicit_keys or [])
    for key, value in args.items():
        if key.startswith("_"):
            continue

        has_existing = key in exp_args
        existing_value = exp_args.get(key)
        is_explicit = not explicit_keys or key in explicit_keys

        if value is None:
            if has_existing and is_explicit:
                del exp_args[key]
                print(f"Removed {key} from experiment arguments")
            continue

        if has_existing:
            if not is_explicit and value != existing_value:
                continue
            if value != existing_value:
                print(f"Overwrote {key} from {existing_value} to {value}")
        exp_args[key] = value
    return exp_args


def check_exists(local_path: str | os.PathLike[str]) -> bool:
    """Return True when ``local_path`` exists."""

    return os.path.exists(local_path)


def extract_template_keys(file_path: str) -> list[str]:
    with open(file_path, "r") as f:
        file = f.read()
    return re.findall(r"(?<!\$)\{([^{}]*)\}", file)


def fill_template(file_path: str, exp_args: dict, new_file_path: str) -> None:
    with open(file_path, "r") as f:
        file = f.read()

    file = re.sub(r"(?<!\$)\{([^{}]*)\}", lambda m: exp_args[m.group(1)], file)

    with open(new_file_path, "w") as f:
        f.write(file)


def substitute_template(template_text: str, substitutions: Dict[str, Any]) -> str:
    """Substitute {key} placeholders in template text with values from substitutions dict.

    This is a simpler alternative to str.format() that only replaces keys present
    in the substitutions dict, leaving other {placeholders} untouched.

    Args:
        template_text: Template string with {key} placeholders.
        substitutions: Dict mapping placeholder names to replacement values.
            Values are converted to strings.

    Returns:
        Template text with placeholders replaced.

    Example:
        >>> substitute_template("Job: {job_name}, Time: {time_limit}", {
        ...     "job_name": "my_job",
        ...     "time_limit": "24:00:00",
        ... })
        'Job: my_job, Time: 24:00:00'
    """
    result = template_text
    for key, value in substitutions.items():
        result = result.replace("{" + key + "}", str(value))
    return result


# =============================================================================
# Benchmark Derivation Utilities
# =============================================================================


def derive_benchmark_repo(
    harbor_dataset: Optional[str] = None,
    dataset_path: Optional[PathInput] = None,
    explicit_repo: Optional[str] = None,
) -> str:
    """Derive benchmark repository identifier from dataset info.

    Single source of truth for deriving eval_benchmark_repo. Used by both
    HPC eval jobs and local eval runner.

    Args:
        harbor_dataset: Harbor dataset slug (e.g., "terminal-bench@2.0").
        dataset_path: Path to a local dataset directory.
        explicit_repo: Explicitly provided benchmark repo (takes precedence).

    Returns:
        Normalized benchmark repository identifier string (filesystem-safe).
    """
    raw: Optional[str] = None
    if explicit_repo:
        raw = explicit_repo
    elif harbor_dataset:
        raw = harbor_dataset
    elif dataset_path:
        raw = Path(dataset_path).name

    if not raw:
        return "unknown-benchmark"

    # Normalize using existing sanitize utility
    return sanitize_repo_for_job(raw)


def derive_benchmark_from_job_dir(job_dir: PathInput) -> str:
    """Derive benchmark name from a completed job's config.json.

    This is the single source of truth for reconstructing benchmark info
    from a job directory. Used by manual upload scripts and unified_db upload.

    Priority:
    1. Harbor registry config (datasets[0].name + version)
    2. HF cache path parsing (datasets[0].path)
    3. Job directory name parsing (fallback)

    Args:
        job_dir: Path to the completed Harbor job directory.

    Returns:
        Benchmark name string (e.g., "terminal-bench@2.0").
    """
    import json
    job_path = Path(job_dir)

    # Method 1: Try to read from config.json
    config_path = job_path / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            datasets = config.get("datasets", [])
            if datasets:
                first_dataset = datasets[0] if isinstance(datasets, list) else {}

                # Harbor registry style: {"name": "terminal-bench", "version": "2.0"}
                registry_name = first_dataset.get("name")
                registry_version = first_dataset.get("version")
                if registry_name:
                    if registry_version:
                        return f"{registry_name}@{registry_version}"
                    return registry_name

                # HF cache path style: {"path": "/cache/datasets--org--name/..."}
                dataset_path = first_dataset.get("path", "")
                if dataset_path:
                    path_parts = dataset_path.split("/")
                    for part in path_parts:
                        if "datasets--" in part:
                            return part.split("--")[-1]
        except (json.JSONDecodeError, OSError):
            pass  # Fall through to directory name parsing

    # Method 2: Parse from job directory name
    # e.g., eval-terminal-bench@2.0-gpt-5-nano-20260113_145348
    name = job_path.name

    if name.startswith("eval-"):
        name = name[5:]

    # Split by common model name patterns
    for sep in ["-gpt-", "-claude-", "-qwen", "-llama", "-gemini", "-o1-", "-o3-"]:
        if sep in name.lower():
            idx = name.lower().index(sep)
            return name[:idx]

    # Remove timestamp suffix
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].replace("_", "").isdigit():
        return parts[0]

    return name


# =============================================================================
# Upload Utilities
# =============================================================================


def _ensure_database_module_path() -> None:
    """Add the database module directory to sys.path if not already present."""
    import sys
    db_path = PROJECT_ROOT / "database"
    if db_path.exists():
        db_path_str = str(db_path)
        if db_path_str not in sys.path:
            sys.path.insert(0, db_path_str)


def upload_traces_to_hf(
    job_dir: PathInput,
    hf_repo_id: str,
    *,
    hf_private: bool = False,
    hf_token: Optional[str] = None,
    hf_episodes: str = "last",
    hf_success_filter: Optional[str] = None,
    hf_verbose: bool = False,
    hf_include_verifier_output: bool = True,
    dry_run: bool = False,
) -> Optional[str]:
    """Upload Harbor job traces to HuggingFace.

    This function handles uploading trace data from a Harbor job directory
    to a HuggingFace dataset repository. Used by both eval and datagen jobs.

    Args:
        job_dir: Path to the Harbor job directory containing traces.
        hf_repo_id: HuggingFace repository ID (e.g., 'org/dataset-name').
        hf_private: Whether to create a private HF repository.
        hf_token: HuggingFace API token. Falls back to HF_TOKEN env var.
        hf_episodes: Which episodes to export: "all" or "last".
        hf_success_filter: Filter by success status: "success", "failure", or None.
        hf_verbose: Enable verbose logging for HF upload.
        hf_include_verifier_output: Include verifier stdout/stderr in traces (default: True).
        dry_run: If True, skip actual upload and return None.

    Returns:
        HuggingFace dataset URL on success, or None if dry_run or upload fails.

    Raises:
        RuntimeError: If the database upload module is unavailable.
    """
    if dry_run:
        print(f"[upload] DRY RUN: Would upload traces from {job_dir} to {hf_repo_id}")
        return None

    if not job_dir:
        print("[upload] No job directory provided; skipping HF upload.")
        return None

    job_path = Path(job_dir)
    if not job_path.exists():
        print(f"[upload] Job directory {job_path} does not exist; skipping HF upload.")
        return None

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("[upload] No HF token provided (set HF_TOKEN env var); skipping HF upload.")
        return None

    _ensure_database_module_path()
    try:
        from unified_db.utils import upload_traces_to_hf as _hf_upload
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace upload helpers are unavailable. "
            "Ensure the database module is installed or on PYTHONPATH."
        ) from exc

    # Defensive sanitization - ensure repo ID complies with HF naming rules
    hf_repo_id = sanitize_hf_repo_id(hf_repo_id)

    print(f"[upload] Uploading traces from {job_path} to HuggingFace: {hf_repo_id}")
    try:
        hf_url = _hf_upload(
            job_dir=str(job_path),
            hf_repo_id=hf_repo_id,
            private=hf_private,
            token=token,
            episodes=hf_episodes,
            success_filter=hf_success_filter,
            verbose=hf_verbose,
            include_verifier_output=hf_include_verifier_output,
        )
        print(f"[upload] HuggingFace upload complete: {hf_url}")
        return hf_url
    except Exception as exc:
        print(f"[upload] HuggingFace upload failed: {exc}")
        raise


def sync_eval_to_database(
    job_dir: PathInput,
    *,
    username: Optional[str] = None,
    error_mode: str = "skip_on_error",
    agent_name: Optional[str] = None,
    model_name: Optional[str] = None,
    benchmark_name: Optional[str] = None,
    benchmark_version_hash: Optional[str] = None,
    dataset_path: Optional[str] = None,
    register_benchmark: bool = True,
    hf_repo_id: Optional[str] = None,
    hf_private: bool = False,
    hf_token: Optional[str] = None,
    hf_episodes: str = "last",
    forced_update: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Sync evaluation results to Supabase database (with optional HF upload).

    This function orchestrates the complete upload workflow for eval jobs:
    1. Upload traces to HuggingFace (if hf_repo_id provided)
    2. Upload job/trial records to Supabase database

    For datagen jobs that only need HF upload, use upload_traces_to_hf() directly.

    Args:
        job_dir: Path to the Harbor job directory.
        username: Username for job registration. Falls back to UPLOAD_USERNAME env or current user.
        error_mode: Error handling mode:
            - "rollback_on_error": Abort on any error (atomic).
            - "skip_on_error": Continue on individual trial failures.
        agent_name: Agent name (auto-detected if not provided).
        model_name: Model name (auto-detected if not provided).
        benchmark_name: Benchmark name (auto-detected if not provided).
        benchmark_version_hash: Version hash for the benchmark. If not provided:
            - Auto-detected from dataset_path if it contains HF cache-style "snapshots/" path
            - Otherwise, generated from benchmark_name using SHA-256
        dataset_path: Path to the dataset (used for auto-detecting benchmark_version_hash).
        register_benchmark: Auto-register benchmark if not found.
        hf_repo_id: HuggingFace repository ID for traces. If None, skips HF upload.
        hf_private: Whether to create a private HF repository.
        hf_token: HuggingFace API token. Falls back to HF_TOKEN env var.
        hf_episodes: Which episodes to export: "all" or "last".
        forced_update: Allow updating existing job records.
        dry_run: If True, skip actual upload and return empty result.

    Returns:
        Dict with upload summary including:
        - success: Whether the upload succeeded
        - job_id: UUID of the job in Supabase
        - n_trials_uploaded: Number of trials uploaded
        - hf_dataset_url: HuggingFace dataset URL (if uploaded)

    Raises:
        RuntimeError: If the database upload module is unavailable.
    """
    import getpass
    import hashlib

    if dry_run:
        print(f"[upload] DRY RUN: Would sync eval results from {job_dir} to database")
        return {"success": True, "dry_run": True, "job_id": None, "n_trials_uploaded": 0}

    if not job_dir:
        print("[upload] No job directory provided; skipping database sync.")
        return {"success": False, "error": "No job directory provided"}

    job_path = Path(job_dir)
    if not job_path.exists():
        print(f"[upload] Job directory {job_path} does not exist; skipping database sync.")
        return {"success": False, "error": f"Job directory does not exist: {job_path}"}

    resolved_username = username or os.environ.get("UPLOAD_USERNAME") or getpass.getuser()
    token = hf_token or os.environ.get("HF_TOKEN")

    # Warn if HF repo requested but no token
    if hf_repo_id and not token:
        print("[upload] HF repo requested but no token provided; skipping HF upload step.")
        hf_repo_id = None

    # Sanitize HF repo ID if provided (defensive - callers should also sanitize)
    if hf_repo_id:
        hf_repo_id = sanitize_hf_repo_id(hf_repo_id)

    # Generate benchmark_version_hash if not provided
    resolved_version_hash = benchmark_version_hash
    if not resolved_version_hash:
        # Try to extract from dataset_path if it's an HF cache path
        if dataset_path and "snapshots/" in dataset_path:
            snapshot_part = dataset_path.split("snapshots/")[1]
            raw_hash = snapshot_part.strip("/").split("/")[0]
            if len(raw_hash) == 40:
                # Convert git hash to SHA-256 for consistency
                resolved_version_hash = hashlib.sha256(raw_hash.encode()).hexdigest()
            else:
                resolved_version_hash = raw_hash
            print(f"[upload] Auto-detected benchmark_version_hash from path: {resolved_version_hash[:16]}...")
        elif benchmark_name:
            # Generate deterministic hash from benchmark name
            resolved_version_hash = hashlib.sha256(benchmark_name.encode()).hexdigest()
            print(f"[upload] Generated benchmark_version_hash from name: {resolved_version_hash[:16]}...")

    _ensure_database_module_path()
    try:
        from unified_db.utils import upload_eval_results
    except ImportError as exc:
        raise RuntimeError(
            "Database upload helpers are unavailable. "
            "Install the database extras or ensure unified_db is on PYTHONPATH."
        ) from exc

    print(f"[upload] Syncing eval results from {job_path} to database (user: {resolved_username})")
    result = upload_eval_results(
        job_dir=str(job_path),
        username=resolved_username,
        error_mode=error_mode,
        agent_name=agent_name,
        model_name=model_name,
        benchmark_name=benchmark_name,
        benchmark_version_hash=resolved_version_hash,
        register_benchmark=register_benchmark,
        hf_repo_id=hf_repo_id,
        hf_private=hf_private,
        hf_token=token,
        hf_episodes=hf_episodes,
        forced_update=forced_update,
    )

    uploaded = result.get("n_trials_uploaded", 0)
    job_id = result.get("job_id")
    hf_url = result.get("hf_dataset_url")
    print(f"[upload] Database sync complete (job_id={job_id}, trials={uploaded}, hf={hf_url or 'n/a'})")
    return result


__all__ = [
    # Constants
    "PROJECT_ROOT",
    # Path resolution
    "resolve_repo_path",
    "resolve_workspace_path",
    "repo_relative",
    "resolve_config_path",
    # Experiments directory setup
    "ExperimentsPaths",
    "setup_experiments_dir",
    "JobSetupResult",
    "resolve_job_and_paths",
    # Value coercion
    "coerce_positive_int",
    # Dict utilities
    "set_or_pop",
    # JSON/Config parsing
    "coerce_agent_kwargs",
    # Endpoint file utilities
    "cleanup_endpoint_file",
    "validate_trace_backend",
    # CLI argument normalization
    "normalize_cli_args",
    # vLLM utilities
    "default_vllm_endpoint_path",
    # Local execution
    "is_local_mode",
    "run_local_script",
    "submit_script",
    # Job naming
    "derive_datagen_job_name",
    "get_job_name",
    "sanitize_repo_for_job",
    "sanitize_repo_component",
    "sanitize_hf_repo_id",
    # Memory scaling utilities
    "parse_memory_string",
    "format_memory_mb",
    "scale_memory_for_partial_gpus",
    # SBATCH utilities
    "build_sbatch_directives",
    "_parse_optional_int",
    "maybe_int",
    "_merge_dependencies",
    "launch_sbatch",
    "update_exp_args",
    # File utilities
    "check_exists",
    "extract_template_keys",
    "fill_template",
    "substitute_template",
    # Benchmark derivation
    "derive_benchmark_repo",
    "derive_benchmark_from_job_dir",
    # Upload utilities
    "upload_traces_to_hf",
    "sync_eval_to_database",
]
