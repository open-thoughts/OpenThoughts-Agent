"""Harbor CLI utilities for HPC launchers.

This module provides utilities for interacting with the Harbor CLI:
- Config path resolution and loading
- Registry validation for dataset slugs
- Agent kwargs extraction and serialization
- Command building for harbor jobs start

These utilities are shared across all execution paths:
- Local runners (data/local/run_tracegen.py, eval/local/run_eval.py)
- Cloud launchers (data/cloud/launch_tracegen_cloud.py, eval/cloud/launch_eval_cloud.py)
- HPC SLURM launchers (hpc/launch.py --job_type datagen/eval)
"""
from __future__ import annotations

import copy
import errno
import json
import os
import pty
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml


# ---------------------------------------------------------------------------
# Harbor config paths and directories
# ---------------------------------------------------------------------------

# Directory containing Harbor YAML configs (relative to this file)
_DIRENV = os.path.dirname(__file__)
HARBOR_CONFIG_DIR = os.path.join(_DIRENV, "harbor_yaml")


def resolve_harbor_config_path(
    raw_value: str,
    config_dir: Optional[str] = None,
) -> Path:
    """Resolve a Harbor config path to an absolute path.

    Checks in order:
    1. raw_value as-is (if it exists)
    2. Fallback to config_dir / raw_value

    Args:
        raw_value: Path to harbor config (absolute or relative)
        config_dir: Directory to search if raw_value not found directly.
                   Defaults to HARBOR_CONFIG_DIR.

    Returns:
        Resolved absolute path to the config file

    Raises:
        FileNotFoundError: If config file doesn't exist in any location
    """
    if config_dir is None:
        config_dir = HARBOR_CONFIG_DIR

    # Try raw_value directly first
    path = Path(raw_value).expanduser()
    if path.exists():
        return path.resolve()

    # Try relative to config_dir
    fallback = Path(config_dir) / raw_value
    if fallback.exists():
        return fallback.resolve()

    # Not found - raise with helpful message
    raise FileNotFoundError(
        f"Harbor job config not found: {raw_value} "
        f"(also checked {config_dir})"
    )


def resolve_jobs_dir_path(
    jobs_dir_value: Optional[str],
    repo_root: Optional[Path] = None,
) -> Path:
    """Resolve jobs_dir from Harbor config to an absolute path.

    Args:
        jobs_dir_value: The jobs_dir value from Harbor config (or None).
                       Defaults to "jobs" if not provided.
        repo_root: Repository root path for resolving relative paths.
                  Defaults to PROJECT_ROOT from launch_utils.

    Returns:
        Absolute path to jobs directory
    """
    if repo_root is None:
        try:
            from hpc.launch_utils import PROJECT_ROOT
            repo_root = PROJECT_ROOT
        except ImportError:
            repo_root = Path(__file__).resolve().parent.parent

    raw_value = jobs_dir_value or "jobs"
    path = Path(raw_value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


# ---------------------------------------------------------------------------
# Harbor registry utilities
# ---------------------------------------------------------------------------

# Default locations to search for Harbor registry.json
def _get_default_registry_hints() -> List[Optional[Path]]:
    """Get default paths to check for Harbor registry."""
    # Import here to avoid circular imports
    try:
        from hpc.launch_utils import PROJECT_ROOT
        project_parent = PROJECT_ROOT.parent
    except ImportError:
        project_parent = Path(__file__).resolve().parent.parent.parent

    return [
        Path(os.environ.get("HARBOR_REGISTRY_PATH", "")).expanduser()
        if os.environ.get("HARBOR_REGISTRY_PATH")
        else None,
        project_parent / "harbor" / "registry.json",
    ]


def load_harbor_registry() -> Optional[Dict[str, Any]]:
    """Load the Harbor dataset registry from known locations.

    Searches DEFAULT_REGISTRY_HINTS in order and returns the first
    valid registry found.

    Returns:
        Parsed registry dict, or None if not found/invalid
    """
    for candidate in _get_default_registry_hints():
        if candidate and candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except Exception:
                return None
    return None


def build_dataset_slug_set(registry: Optional[Dict[str, Any]]) -> Set[str]:
    """Build a set of valid dataset slugs from a Harbor registry.

    Args:
        registry: Parsed registry dict (list of dataset entries)

    Returns:
        Set of valid slug strings (e.g., {"terminal-bench", "terminal-bench@2.0"})
    """
    if not registry:
        return set()

    entries: Set[str] = set()
    for item in registry:
        name = item.get("name")
        version = item.get("version")
        if not name:
            continue
        if version:
            entries.add(f"{name}@{version}")
        entries.add(name)
    return entries


def validate_harbor_dataset_slug(slug: str) -> None:
    """Validate that a dataset slug exists in the Harbor registry.

    Args:
        slug: Dataset slug to validate (e.g., "terminal-bench@2.0")

    Raises:
        ValueError: If slug is not found in registry (when registry exists)
    """
    registry = load_harbor_registry()
    if not registry:
        # No registry available - skip validation
        return

    valid = build_dataset_slug_set(registry)
    if slug not in valid:
        raise ValueError(
            f"Dataset '{slug}' is not in the local Harbor registry "
            f"(known datasets: {sorted(list(valid))[:8]} ...). "
            "Specify --eval-dataset-path instead or update the registry hint."
        )


# ---------------------------------------------------------------------------
# Harbor config loading
# ---------------------------------------------------------------------------


def load_harbor_config(harbor_config_path: str) -> Dict[str, Any]:
    """Load and parse Harbor config YAML.

    Args:
        harbor_config_path: Path to harbor config file

    Returns:
        Parsed harbor config dict (empty dict if file not found)
    """
    try:
        with open(harbor_config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


def get_harbor_env_from_config(
    harbor_config: Union[str, Dict[str, Any], None],
    default: str = "daytona",
) -> str:
    """Extract Harbor environment type from config.

    Reads the `environment.type` field from Harbor YAML config to determine
    the sandbox backend (daytona, docker, modal, apptainer).

    Args:
        harbor_config: Either a path to Harbor config YAML, a parsed config dict,
                      or None.
        default: Default environment type if not found in config (default: "daytona").

    Returns:
        Environment type string: "daytona", "docker", "modal", "apptainer", etc.

    Examples:
        >>> get_harbor_env_from_config("hpc/harbor_yaml/trace_docker_32concurrency_ctx131k.yaml")
        'docker'
        >>> get_harbor_env_from_config({"environment": {"type": "daytona"}})
        'daytona'
        >>> get_harbor_env_from_config(None)
        'daytona'
    """
    if harbor_config is None:
        return default

    # Load config if path provided
    if isinstance(harbor_config, str):
        config_dict = load_harbor_config(harbor_config)
    else:
        config_dict = harbor_config

    # Extract environment.type
    env_config = config_dict.get("environment") or {}
    env_type = env_config.get("type")

    if env_type and isinstance(env_type, str):
        return env_type.lower()

    return default


# ---------------------------------------------------------------------------
# Endpoint metadata utilities
# ---------------------------------------------------------------------------


def build_endpoint_meta(endpoint_url: str) -> Dict[str, str]:
    """Build endpoint metadata dict from a vLLM endpoint URL.

    Handles both formats:
    - With /v1 suffix: "http://host:port/v1" (from VLLMServer.endpoint)
    - Without suffix: "http://host:port" (from endpoint JSON)

    Args:
        endpoint_url: vLLM endpoint URL (with or without /v1 suffix)

    Returns:
        Dict with 'api_base' and 'metrics_endpoint' keys
    """
    url = endpoint_url.rstrip("/")

    # Determine base URL (without /v1)
    if url.endswith("/v1"):
        base_url = url[:-3].rstrip("/")
        api_base = url
    else:
        base_url = url
        api_base = f"{url}/v1"

    metrics_endpoint = f"{base_url}/metrics"

    return {
        "api_base": api_base,
        "metrics_endpoint": metrics_endpoint,
    }


def load_endpoint_metadata(endpoint_json: Path) -> Dict[str, Any]:
    """Load and parse vLLM endpoint metadata from JSON file.

    Reads the endpoint JSON written by vLLM and computes api_base and
    metrics_endpoint URLs from the endpoint_url field.

    Args:
        endpoint_json: Path to the endpoint JSON file

    Returns:
        Dict with all endpoint data plus computed api_base and metrics_endpoint
    """
    data = json.loads(endpoint_json.read_text())
    endpoint_url = data.get("endpoint_url") or ""

    if endpoint_url:
        meta = build_endpoint_meta(endpoint_url)
        data["api_base"] = meta["api_base"]
        data["metrics_endpoint"] = meta["metrics_endpoint"]
    else:
        data["api_base"] = ""
        data["metrics_endpoint"] = ""

    return data


# ---------------------------------------------------------------------------
# Agent kwargs utilities
# ---------------------------------------------------------------------------


def extract_agent_kwargs_from_config(harbor_config: dict, agent_name: Optional[str]) -> dict:
    """Extract kwargs for the specified agent from harbor config.

    The Harbor YAML is the ground truth for agent configuration. This function
    finds the agent by name and returns a copy of its kwargs dict.

    Args:
        harbor_config: Parsed harbor config dict (from YAML)
        agent_name: Name of the agent to find (e.g., "terminus-2"). If None, uses first agent.

    Returns:
        Copy of the agent's kwargs dict, or empty dict if not found
    """
    agents = harbor_config.get("agents", [])
    for agent in agents:
        if agent.get("name") == agent_name:
            return copy.deepcopy(agent.get("kwargs", {}))
    # Fallback: return first agent's kwargs if no match (backwards compat)
    if agents and isinstance(agents[0], dict):
        return copy.deepcopy(agents[0].get("kwargs", {}))
    return {}


def apply_nested_key(target: dict, dotted_key: str, value: Any) -> None:
    """Apply a value to a nested dict using dotted key notation.

    Args:
        target: Dict to modify in-place
        dotted_key: Key like "model_info.max_tokens" for nested access
        value: Value to set
    """
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_agent_kwarg_strings(entries: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    """Parse --agent-kwarg CLI entries into overrides and passthrough.

    Args:
        entries: List of "key=value" strings (or passthrough entries without =)

    Returns:
        Tuple of (overrides dict, passthrough list)
    """
    overrides: Dict[str, Any] = {}
    passthrough: List[str] = []
    for entry in entries:
        if "=" not in entry:
            passthrough.append(entry)
            continue
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            passthrough.append(entry)
            continue
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        overrides[key] = value
    return overrides, passthrough


def serialize_agent_kwargs(kwargs: dict) -> List[str]:
    """Serialize agent kwargs dict to CLI argument strings.

    Args:
        kwargs: Dict of agent kwargs

    Returns:
        List of "key=value" strings suitable for --agent-kwarg
    """
    serialized: List[str] = []
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            serialized.append(f"{key}={json.dumps(value)}")
        else:
            serialized.append(f"{key}={value}")
    return serialized


# ---------------------------------------------------------------------------
# Job naming utilities
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    """Generate a timestamp string for job names."""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def default_job_name(prefix: str, dataset_label: str, model_label: str) -> str:
    """Generate a default job name.

    Args:
        prefix: Job type prefix (e.g., "eval", "tracegen")
        dataset_label: Dataset/tasks identifier
        model_label: Model identifier

    Returns:
        Formatted job name like "eval-dataset-model-20240101_120000"
    """
    sanitized_dataset = Path(dataset_label).name.replace("/", "-").replace(" ", "_")
    sanitized_model = model_label.replace("/", "-").replace(" ", "_")
    return f"{prefix}-{sanitized_dataset}-{sanitized_model}-{_timestamp()}"


# ---------------------------------------------------------------------------
# Harbor command building
# ---------------------------------------------------------------------------


def collect_extra_agent_kwargs(
    datagen_extras: Optional[Dict[str, Any]] = None,
    cli_kwargs: Any = None,
) -> Dict[str, Any]:
    """Collect extra agent kwargs from datagen config and CLI.

    This is a pre-merge helper for prepare functions that run before sbatch.
    The result should be passed to build_harbor_command(extra_agent_kwargs=...)
    for final merging with Harbor YAML base kwargs via merge_agent_kwargs().

    Precedence (lowest to highest):
    1. datagen_extras (from datagen config's extra_agent_kwargs)
    2. cli_kwargs (from --trace-agent-kwargs CLI arg)

    Args:
        datagen_extras: Dict from datagen config's extra_agent_kwargs field
        cli_kwargs: CLI argument value - can be dict, JSON string, or None

    Returns:
        Merged dict of extra kwargs (NOT including Harbor YAML base)
    """
    result: Dict[str, Any] = dict(datagen_extras or {})

    if cli_kwargs:
        if isinstance(cli_kwargs, dict):
            result.update(cli_kwargs)
        elif isinstance(cli_kwargs, str):
            try:
                parsed = json.loads(cli_kwargs)
                if isinstance(parsed, dict):
                    result.update(parsed)
            except json.JSONDecodeError:
                pass

    return result


def merge_agent_kwargs(
    harbor_config_data: dict,
    agent_name: Optional[str],
    endpoint_meta: Optional[Dict[str, Any]] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Merge agent kwargs from multiple sources with proper precedence.

    This is the consolidated helper for agent kwargs handling. Precedence (lowest to highest):
    1. Base kwargs from Harbor YAML (agents[].kwargs)
    2. Endpoint-specific values (api_base, metrics_endpoint) for local vLLM
    3. Extra kwargs from datagen config (extra_agent_kwargs)
    4. CLI --agent-kwarg overrides (highest precedence, supports dotted keys)

    Args:
        harbor_config_data: Parsed Harbor config dict
        agent_name: Agent name to extract kwargs for. If None, uses first agent from config.
        endpoint_meta: Dict with api_base/metrics_endpoint from vLLM (None for API engines)
        extra_kwargs: Additional kwargs from datagen config or other sources
        cli_overrides: Raw --agent-kwarg strings from CLI (e.g., ["key=value", "nested.key=value"])

    Returns:
        Tuple of (merged_kwargs_dict, passthrough_strings)
        - merged_kwargs_dict: Final merged kwargs dict
        - passthrough_strings: CLI entries that couldn't be parsed as key=value
    """
    # 1. Start with base kwargs from Harbor YAML
    agent_kwargs = extract_agent_kwargs_from_config(harbor_config_data, agent_name)

    # 2. Apply endpoint-specific values (only for local vLLM)
    if endpoint_meta:
        if endpoint_meta.get("metrics_endpoint"):
            agent_kwargs["metrics_endpoint"] = endpoint_meta["metrics_endpoint"]
        if endpoint_meta.get("api_base"):
            agent_kwargs["api_base"] = endpoint_meta["api_base"]

    # 3. Apply extra kwargs from datagen config
    if extra_kwargs:
        for key, value in extra_kwargs.items():
            apply_nested_key(agent_kwargs, key, value)

    # 4. CLI --agent-kwarg flags take highest precedence (supports dotted keys)
    passthrough: List[str] = []
    if cli_overrides:
        override_kwargs, passthrough = parse_agent_kwarg_strings(cli_overrides)
        for dotted_key, override_value in override_kwargs.items():
            apply_nested_key(agent_kwargs, dotted_key, override_value)

    return agent_kwargs, passthrough


def build_harbor_command(
    harbor_binary: str,
    harbor_config_path: str,
    harbor_config_data: dict,
    job_name: str,
    agent_name: Optional[str],
    model_name: str,
    env_type: str,
    n_concurrent: int,
    n_attempts: int,
    endpoint_meta: Optional[dict],
    agent_kwarg_overrides: List[str],
    harbor_extra_args: List[str],
    dataset_slug: Optional[str] = None,
    dataset_path: Optional[str] = None,
    jobs_dir: Optional[str] = None,
    extra_agent_kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Build the harbor jobs start command.

    The Harbor YAML is the ground truth for agent configuration. This function
    uses merge_agent_kwargs() to combine kwargs from multiple sources:
    1. Base kwargs from Harbor YAML (agents[].kwargs)
    2. Endpoint-specific values (api_base, metrics_endpoint) for local vLLM
    3. Extra kwargs from datagen config (extra_agent_kwargs parameter)
    4. CLI --agent-kwarg overrides (highest precedence)

    Args:
        harbor_binary: Path to harbor CLI
        harbor_config_path: Path to harbor config YAML
        harbor_config_data: Parsed harbor config dict
        job_name: Name for this harbor job
        agent_name: Agent to run (e.g., "terminus-2"). If None, uses the agent from harbor config.
        model_name: Model identifier for --model flag
        env_type: Environment type for --env flag (daytona, docker, modal, apptainer)
        n_concurrent: Number of concurrent trials
        n_attempts: Number of attempts per task
        endpoint_meta: Dict with api_base and metrics_endpoint from vLLM (None for API engines)
        agent_kwarg_overrides: Raw --agent-kwarg strings from CLI
        harbor_extra_args: Additional args to pass through to harbor
        dataset_slug: Harbor dataset slug (mutually exclusive with dataset_path)
        dataset_path: Path to tasks directory (mutually exclusive with dataset_slug)
        jobs_dir: Override for --jobs-dir (where Harbor writes job outputs)
        extra_agent_kwargs: Additional kwargs from datagen config (merged before CLI overrides)

    Returns:
        Complete harbor command as list of strings
    """
    # Merge agent kwargs using consolidated helper
    agent_kwargs, passthrough = merge_agent_kwargs(
        harbor_config_data=harbor_config_data,
        agent_name=agent_name,
        endpoint_meta=endpoint_meta,
        extra_kwargs=extra_agent_kwargs,
        cli_overrides=agent_kwarg_overrides,
    )

    # Create a modified config with model_name and kwargs merged directly.
    # This avoids using --agent/--model CLI flags which would cause Harbor to
    # create a fresh AgentConfig and lose settings like override_setup_timeout_sec.
    modified_config = copy.deepcopy(harbor_config_data)

    # Update all agents in the config with model_name and merged kwargs
    agents = modified_config.get("agents", [])
    for agent in agents:
        # Set model_name directly in config
        agent["model_name"] = model_name
        # Merge kwargs into the agent's existing kwargs
        existing_kwargs = agent.get("kwargs", {})
        existing_kwargs.update(agent_kwargs)
        agent["kwargs"] = existing_kwargs

    # Write the modified config to the experiment directory (jobs_dir).
    # This keeps the merged config alongside the experiment outputs for reproducibility.
    if jobs_dir:
        config_dir = Path(jobs_dir) / job_name
        config_dir.mkdir(parents=True, exist_ok=True)
        merged_config_path = config_dir / "merged_harbor_config.yaml"
    else:
        # Fallback to current directory if no jobs_dir specified
        merged_config_path = Path(f"merged_harbor_config_{job_name}.yaml")

    with open(merged_config_path, "w") as f:
        yaml.safe_dump(modified_config, f)
    temp_config_path = str(merged_config_path)

    # Build base command using the temp config (no --model or --agent needed)
    cmd = [
        harbor_binary,
        "jobs",
        "start",
        "--config",
        temp_config_path,
        "--job-name",
        job_name,
        "--env",
        env_type,
        "--n-concurrent",
        str(n_concurrent),
        "--n-attempts",
        str(n_attempts),
    ]

    # Add dataset (slug or path)
    if dataset_slug:
        cmd.extend(["--dataset", dataset_slug])
    elif dataset_path:
        cmd.extend(["-p", dataset_path])

    # Add jobs_dir if specified
    if jobs_dir:
        cmd.extend(["--jobs-dir", jobs_dir])

    # Add passthrough kwargs that couldn't be parsed (e.g., complex nested structures)
    for passthrough_kw in passthrough:
        cmd.extend(["--agent-kwarg", passthrough_kw])

    # Process extra args with sensible defaults
    extra_args = list(harbor_extra_args or [])

    def _flag_present(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in extra_args)

    if not (_flag_present("--export-traces") or _flag_present("--no-export-traces")):
        extra_args.append("--export-traces")
    if not (_flag_present("--export-verifier-metadata") or _flag_present("--no-export-verifier-metadata")):
        extra_args.append("--export-verifier-metadata")
    if not _flag_present("--export-episodes"):
        extra_args.extend(["--export-episodes", "last"])

    for extra in extra_args:
        cmd.append(extra)

    return cmd


def run_harbor_cli(cmd: List[str], log_path: Optional[Path] = None) -> int:
    """Run Harbor CLI with proper TTY handling.

    Harbor CLI requires a pseudo-terminal (PTY) for proper output handling.
    Without it, Harbor may buffer output indefinitely or hang waiting for
    terminal interaction.

    Args:
        cmd: Command list to execute (e.g., ["harbor", "jobs", "start", ...])
        log_path: Optional path to write Harbor output to a file instead of stdout.

    Returns:
        Exit code from Harbor process.

    Raises:
        subprocess.CalledProcessError: If Harbor exits with non-zero status.
    """
    if log_path:
        # File-based output - no PTY needed (line-buffered for real-time tail access)
        with open(log_path, "w", encoding="utf-8", buffering=1) as harbor_log_file:
            print(f"Streaming Harbor output to {log_path}")
            result = subprocess.run(
                cmd,
                check=False,
                stdout=harbor_log_file,
                stderr=subprocess.STDOUT,
            )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result.returncode

    # PTY-based output for interactive-like behavior
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            text=False,
        )
        os.close(slave_fd)

        # Read and forward output in real-time
        while True:
            try:
                data = os.read(master_fd, 4096)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                break
            if not data:
                break
            os.write(sys.stdout.fileno(), data)
    finally:
        os.close(master_fd)

    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)
    return ret


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "HARBOR_CONFIG_DIR",
    # Config/path resolution
    "resolve_harbor_config_path",
    "resolve_jobs_dir_path",
    "load_harbor_config",
    "get_harbor_env_from_config",
    # Endpoint metadata
    "build_endpoint_meta",
    "load_endpoint_metadata",
    # Registry utilities
    "load_harbor_registry",
    "build_dataset_slug_set",
    "validate_harbor_dataset_slug",
    # Agent kwargs
    "extract_agent_kwargs_from_config",
    "apply_nested_key",
    "parse_agent_kwarg_strings",
    "serialize_agent_kwargs",
    "collect_extra_agent_kwargs",
    "merge_agent_kwargs",
    # Job naming
    "default_job_name",
    # Command building and execution
    "build_harbor_command",
    "run_harbor_cli",
]
