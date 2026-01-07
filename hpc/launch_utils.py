"""
Utility helpers shared across HPC launch entry points.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import socket
import shutil
import subprocess
import time
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from hpc.hpc import detect_hpc

from .job_name_ignore_list import JOB_NAME_IGNORE_KEYS
from .arguments import JobType
from .sft_launch_utils import build_accelerate_config_block

# =============================================================================
# Type Aliases
# =============================================================================

PathInput = Union[str, PathLike[str], Path, None]
"""Flexible path input type for utility functions."""

_VALID_TRACE_BACKENDS = {"vllm", "ray", "vllm_local", "none"}
"""Valid backend options for trace generation."""

_HOSTED_VLLM_PREFIX = "hosted_vllm/"
"""Provider prefix expected by LiteLLM when routing to managed vLLM endpoints."""


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


def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    """Sanitize a HuggingFace repo_id to comply with naming rules.

    Keeps org prefix (e.g. 'mlfoundations-dev/') and cleans up the rest.
    Used when deriving HF dataset repo names from job names or model paths.

    Args:
        repo_id: The repository ID to sanitize (e.g., 'org/some-name').
        max_length: Maximum allowed length for the full repo_id.

    Returns:
        Sanitized repo_id that complies with HuggingFace naming rules.
    """

    def collapse(value: str) -> str:
        prev = None
        while value != prev:
            prev = value
            value = value.replace("--", "-").replace("..", ".")
        return value

    org, name = repo_id.split("/", 1) if "/" in repo_id else (None, repo_id)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = collapse(name).strip("-.")
    if not name:
        name = "repo"
    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit > 8:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest))
        base = name[:keep].rstrip("-.") or "r"
        name = collapse(f"{base}{digest}").strip("-.")
    if name[0] in "-.":
        name = f"r{name[1:]}"
    if name[-1] in "-.":
        name = f"{name[:-1]}0"
    return f"{org}/{name}" if org else name


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
        directives.append(f"#SBATCH -q {qos}")
    # Add GPU directive if the cluster uses one
    gpu_directive = hpc.get_gpu_directive(gpus_requested, gpu_type_resolved)
    if gpu_directive:
        directives.append(gpu_directive)
    # Add memory directive if the cluster uses one
    mem_directive = hpc.get_mem_directive(mem)
    if mem_directive:
        directives.append(mem_directive)
    if hpc.node_exclusion_list:
        directives.append(f"#SBATCH --exclude={hpc.node_exclusion_list}")

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


def sanitize_repo_for_job(repo_id: str) -> str:
    """Return a filesystem-safe representation of a repo identifier."""

    safe = re.sub(r"[^A-Za-z0-9._\-]+", "-", repo_id.strip())
    safe = safe.strip("-_")
    return safe or "consolidate"


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

    job_type_hint = str(cli_args.get("job_type") or "").lower()
    prefix = "eval" if job_type_hint == JobType.EVAL.value else "datagen"
    parts: list[str] = [prefix]
    engine = cli_args.get("datagen_engine") or cli_args.get("trace_engine") or "engine"
    parts.append(str(engine or "engine"))

    repo_candidate = cli_args.get("datagen_target_repo") or cli_args.get("trace_target_repo")
    model_candidate = cli_args.get("datagen_model") or cli_args.get("trace_model")
    if model_candidate:
        parts.append(_sanitize_component(str(model_candidate)))
    elif repo_candidate:
        parts.append(_sanitize_component(str(repo_candidate)))

    dataset_component = None
    dataset_slug = cli_args.get("harbor_dataset")
    dataset_path = cli_args.get("trace_input_path") or cli_args.get("eval_dataset_path")
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
    job_name = (
        job_name.replace("/", "_")
        .replace("?", "")
        .replace("*", "")
        .replace("{", "")
        .replace("}", "")
        .replace(":", "")
        .replace('"', "")
        .replace(" ", "_")
    )
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
    return derive_default_job_name(cli_args)

def _parse_optional_int(value: Any, label: str) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got boolean {value!r}")
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}") from exc


def _inject_env_block(text: str, env_map: dict) -> str:
    exports = []
    for k, v in env_map.items():
        if v in (None, ""):
            continue
        quoted = shlex.quote(str(v))
        exports.append(f"export {k}={quoted}")
    if not exports:
        return text
    lines = text.splitlines(True)
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    while idx < len(lines) and (
        lines[idx].startswith("#SBATCH")
        or lines[idx].strip() == ""
        or lines[idx].startswith("#")
    ):
        idx += 1
    return "".join(lines[:idx] + ["\n".join(exports) + "\n"] + lines[idx:])


def _ensure_dependency_directive(text: str, dependency: Optional[str]) -> str:
    if not dependency:
        return text

    directive_prefix = "#SBATCH --dependency"
    lines = text.splitlines()
    for line in lines:
        if directive_prefix in line:
            return text

    insert_idx = 0
    for idx, line in enumerate(lines):
        if idx == 0 and line.startswith("#!"):
            insert_idx = 1
            continue
        stripped = line.strip()
        if stripped.startswith("#SBATCH"):
            insert_idx = idx + 1
            continue
        if not stripped:
            insert_idx = idx + 1
            continue
        break

    dependency_line = f"#SBATCH --dependency={dependency}"
    lines.insert(insert_idx, dependency_line)
    new_text = "\n".join(lines)
    if text.endswith("\n"):
        new_text += "\n"
    return new_text


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


def _escape_bash_variables(text: str) -> str:
    result: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text[i] == "$" and i + 1 < length and text[i + 1] == "{":
            start = i
            depth = 1
            j = i + 2
            while j < length and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            inner = text[i + 2 : j - 1]
            escaped_inner = _escape_bash_variables(inner)
            result.append("${{" + escaped_inner + "}}")
            i = j
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def construct_sbatch_script(exp_args: dict) -> str:
    base_script_path = exp_args["train_sbatch_path"]
    with open(base_script_path, "r") as f:
        base_script = f.read()

    kwargs = defaultdict(str, **exp_args)
    kwargs["accelerate_config_block"] = build_accelerate_config_block(exp_args)

    json_files_cat = re.findall(r"cat.*?<<EOT >.*?EOT", base_script, re.DOTALL)
    json_filenames = []
    for json_file in json_files_cat:
        json_file_name = re.match(
            r"cat.*?<<EOT >.*?(\S+).*?EOT", json_file, re.DOTALL
        ).group(1)
        json_filenames.append(json_file_name)

        base_script = re.sub(
            r"cat.*?<<EOT >.*?" + json_file_name.replace("$", "\\$") + r".*?EOT",
            f"cat {json_file_name}",
            base_script,
            count=1,
            flags=re.DOTALL,
        )

    base_script = _escape_bash_variables(base_script)

    time_limit = kwargs.get("time_limit")
    if time_limit is None:
        time_limit = "01:00:00"
        kwargs["time_limit"] = time_limit

    hpc = detect_hpc()
    hpc_name = hpc.name
    if hpc_name == "jureca" or hpc_name == "juwels":
        login_node = socket.gethostname().split(".")[0] + "i"
        if "{login_node}" in base_script:
            if kwargs.get("internet_node", False):
                if not shutil.which("proxychains4"):
                    raise RuntimeError("proxychains4 not found, please install it to use internet_node")
            base_script = base_script.replace("{login_node}", login_node)

    sbatch_script = base_script.format(**kwargs)
    sbatch_script = _ensure_dependency_directive(sbatch_script, exp_args.get("dependency"))

    env_block = {
        "DISABLE_VERSION_CHECK": "1",
    }
    stage_value = str(exp_args.get("stage") or "").lower()
    if exp_args.get("use_mca") and stage_value == "sft":
        env_block["USE_MCA"] = "1"
        os.environ.setdefault("USE_MCA", "1")

    sbatch_script = _inject_env_block(sbatch_script, env_block)

    for json_file, json_file_name in zip(json_files_cat, json_filenames):
        sbatch_script = sbatch_script.replace(f"cat {json_file_name}", json_file)

    sbatch_dir = os.path.join(kwargs["experiments_dir"], "sbatch_scripts")
    os.makedirs(sbatch_dir, exist_ok=True)
    sbatch_script_path = os.path.join(sbatch_dir, f"{kwargs['job_name']}.sbatch")
    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)
        print(f"Wrote sbatch script to {sbatch_script_path}")

    return sbatch_script_path


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
    "resolve_config_path",
    # Value coercion
    "coerce_positive_int",
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
    # SBATCH utilities
    "_parse_optional_int",
    "_inject_env_block",
    "_ensure_dependency_directive",
    "_merge_dependencies",
    "launch_sbatch",
    "update_exp_args",
    # File utilities
    "check_exists",
    "construct_sbatch_script",
    "extract_template_keys",
    "fill_template",
    # Benchmark derivation
    "derive_benchmark_repo",
    # Upload utilities
    "upload_traces_to_hf",
    "sync_eval_to_database",
]
