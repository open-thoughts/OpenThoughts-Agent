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


def derive_vllm_supports_tool_calling(vllm_cfg: Any) -> Optional[bool]:
    """Determine if vLLM tool calling is enabled based on config.

    Returns:
        True if tool_call_parser is set, False if explicitly absent, None if unknown.
    """
    if vllm_cfg is None:
        return None

    tool_call_parser = None

    if hasattr(vllm_cfg, "tool_call_parser"):
        tool_call_parser = getattr(vllm_cfg, "tool_call_parser", None)
    elif isinstance(vllm_cfg, dict):
        tool_call_parser = vllm_cfg.get("tool_call_parser")

    if tool_call_parser:
        return True

    extra_args = None
    if hasattr(vllm_cfg, "extra_args"):
        extra_args = getattr(vllm_cfg, "extra_args", None)
    elif isinstance(vllm_cfg, dict):
        extra_args = vllm_cfg.get("extra_args")

    if isinstance(extra_args, dict):
        if extra_args.get("tool_call_parser") or extra_args.get("tool-call-parser"):
            return True
    elif isinstance(extra_args, (list, tuple)):
        for entry in extra_args:
            if isinstance(entry, str) and "tool_call_parser" in entry:
                return True

    return False

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
        Formatted job name like "eval__dataset__model__20240101_120000"
    """
    from hpc.launch_utils import shorten_model_name, JOB_NAME_SEP

    sanitized_dataset = Path(dataset_label).name.replace("/", "-").replace(" ", "_")
    sanitized_model = shorten_model_name(model_label)
    return JOB_NAME_SEP.join([prefix, sanitized_dataset, sanitized_model, _timestamp()])


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


def merge_harbor_config(
    harbor_config_data: dict,
    *,
    agent_name: Optional[str],
    model_name: str,
    n_concurrent: int,
    endpoint_meta: Optional[dict],
    agent_kwarg_overrides: List[str],
    extra_agent_kwargs: Optional[Dict[str, Any]] = None,
) -> dict:
    """Materialize the merged Harbor config dict without writing files.

    This is the side-effect-free core of ``build_harbor_command``: it
    applies the same precedence rules (YAML base → endpoint values →
    extra kwargs → CLI overrides) and returns a fresh dict ready to be
    serialized as YAML or compared against a prior run's
    ``config.json`` / ``merged_harbor_config.yaml``.

    Used by ``build_harbor_command`` for the actual sbatch path and by
    ``hpc.resume_manager`` for the resume-policy materialization step.

    The result mirrors the legacy nested-orchestrator shape; callers that
    need the unified flat shape can read top-level fields directly (they
    are normalized by Harbor's own ``_migrate_orchestrator_config``
    validator on load).
    """
    agent_kwargs, _ = merge_agent_kwargs(
        harbor_config_data=harbor_config_data,
        agent_name=agent_name,
        endpoint_meta=endpoint_meta,
        extra_kwargs=extra_agent_kwargs,
        cli_overrides=agent_kwarg_overrides,
    )

    modified_config = copy.deepcopy(harbor_config_data)

    if "orchestrator" not in modified_config:
        modified_config["orchestrator"] = {}
    modified_config["orchestrator"]["n_concurrent_trials"] = n_concurrent

    agents = modified_config.get("agents", [])
    for agent in agents:
        if agent_name:
            agent["name"] = agent_name
        agent["model_name"] = model_name
        existing_kwargs = agent.get("kwargs", {})
        existing_kwargs.update(agent_kwargs)
        agent["kwargs"] = existing_kwargs

    return modified_config


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
    export_hf_repo: Optional[str] = None,
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
    # Compute passthrough for CLI flags; the merged-config path goes through
    # the side-effect-free merge_harbor_config helper.
    _, passthrough = merge_agent_kwargs(
        harbor_config_data=harbor_config_data,
        agent_name=agent_name,
        endpoint_meta=endpoint_meta,
        extra_kwargs=extra_agent_kwargs,
        cli_overrides=agent_kwarg_overrides,
    )

    modified_config = merge_harbor_config(
        harbor_config_data,
        agent_name=agent_name,
        model_name=model_name,
        n_concurrent=n_concurrent,
        endpoint_meta=endpoint_meta,
        agent_kwarg_overrides=agent_kwarg_overrides,
        extra_agent_kwargs=extra_agent_kwargs,
    )

    # Write the modified config locally for the Harbor subprocess.
    if jobs_dir:
        config_dir = Path(jobs_dir) / job_name
        config_dir.mkdir(parents=True, exist_ok=True)
        merged_config_path = config_dir / "merged_harbor_config.yaml"
    else:
        # Fallback to current directory if no jobs_dir specified
        merged_config_path = Path(f"merged_harbor_config_{job_name}.yaml")

    # Strip OT-Agent-only metrics (e.g. ``mean-drop-ei``, ``accuracy-drop-ei``)
    # that the pinned harbor (0.7.0) rejects with a Pydantic enum
    # ValidationError when it parses --config on its own. The same filter
    # already runs at scripts/harbor/job_config_utils.py load time, but the
    # merge-and-write path here takes the raw YAML dict and never goes
    # through that loader, so the unsupported types survive into the file
    # harbor reads back. Apply the filter here too — see commit 1ba6a4a7.
    try:
        from scripts.harbor.job_config_utils import _filter_supported_metrics
        modified_config = _filter_supported_metrics(modified_config)
    except ImportError:
        # _filter_supported_metrics lives under scripts.harbor; if we're
        # somehow invoked from an install layout where scripts/ isn't a
        # package (older path), skip silently — harbor will surface its
        # own ValidationError downstream and we'll see it in the log.
        pass

    def _write_merged_config() -> None:
        config_text = yaml.safe_dump(modified_config)
        merged_config_path.parent.mkdir(parents=True, exist_ok=True)
        merged_config_path.write_text(config_text, encoding="utf-8")

    _write_merged_config()
    temp_config_path = str(merged_config_path)

    # Build base command using the temp config (no --model or --agent needed)
    cmd = [
        harbor_binary,
        "jobs",
        "start",
        # Non-interactive: suppress harbor's host-env confirmation prompt
        # (`_confirm_host_env_access` console.input, jobs.py, added in PR #1195).
        # Verified configs declare host-env vars (OPENAI_API_KEY/JUDGE_MODEL for
        # the LLM judge) which trigger the prompt; under SLURM there is no TTY so
        # console.input() blocks forever → 12h TIMEOUT, zero output (datagen #22
        # 2026-06-10). The launcher is always non-interactive, so always confirm.
        "--yes",
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

    # Add dataset (slug or path).
    # CLI dataset flags take top priority — clear the YAML datasets so there is
    # no ambiguity between YAML placeholder paths and the actual dataset.
    if dataset_slug:
        # Clear YAML datasets so Harbor only sees the CLI --dataset flag.
        modified_config.pop("datasets", None)
        modified_config.pop("tasks", None)
        _write_merged_config()
        cmd.extend(["--dataset", dataset_slug])
    elif dataset_path:
        # ``dataset_path`` is overloaded: it can be a local FS path, a
        # HuggingFace dataset id (``org/name`` format), or a harbor
        # registry slug. Harbor 0.7.0 requires the source flag on the CLI
        # (``--path`` for local dirs, ``--dataset`` for slugs / HF refs)
        # — YAML datasets without a CLI flag now error out with
        # "Cannot specify --registry-url ... without also specifying
        # --dataset, --task, or --path" whenever any registry-* field is
        # populated by the merge.
        #
        # Heuristic: actually-existing FS path -> ``--path``; otherwise
        # treat as a remote reference and use ``--dataset`` (harbor's
        # ``--dataset`` accepts both slugs and HF org/name refs — see
        # harbor/src/harbor/cli/jobs.py:1226).
        modified_config.pop("datasets", None)
        modified_config.pop("tasks", None)
        _write_merged_config()
        # gs://, s3:// etc. are remote schemes; never local.
        looks_local = not dataset_path.startswith(("gs://", "s3://", "http://", "https://"))
        if looks_local and Path(dataset_path).expanduser().exists():
            cmd.extend(["--path", str(Path(dataset_path).expanduser().resolve())])
        else:
            cmd.extend(["--dataset", dataset_path])
    else:
        # Neither dataset_slug nor dataset_path provided.  Strip the YAML
        # placeholder (if any) so we fail fast with a clear message rather
        # than having Harbor attempt to load a bogus placeholder path.
        _placeholder = "/replace/with/tasks/path"
        yaml_datasets = modified_config.get("datasets") or []
        if yaml_datasets and any(
            d.get("path", "") == _placeholder for d in yaml_datasets if isinstance(d, dict)
        ):
            modified_config.pop("datasets", None)
            modified_config.pop("tasks", None)
            _write_merged_config()

        # Final safety check: merged config must have datasets, tasks, or a
        # --dataset CLI flag.  Without any of these Harbor will error with
        # "Either datasets or tasks must be provided."
        has_datasets = bool(modified_config.get("datasets"))
        has_tasks = bool(modified_config.get("tasks"))
        has_cli_dataset = "--dataset" in cmd
        if not (has_datasets or has_tasks or has_cli_dataset):
            raise ValueError(
                "[build_harbor_command] BUG: No datasets, tasks, or --dataset flag. "
                f"dataset_slug={dataset_slug!r}, dataset_path={dataset_path!r}. "
                "The merged config will cause Harbor to fail. "
                "Ensure --dataset_path or --dataset is provided."
            )

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

    # NOTE: --auto-resume was an older-harbor CLI flag (<=0.6.x) that retried
    # on transient Daytona errors. harbor 0.7.0 (penfever/otagent-latest)
    # removed the option entirely; retry behavior moved into the trial layer
    # and is now driven by --filter-error-type / job-config defaults.
    # Auto-injecting --auto-resume now causes `harbor jobs start` to exit 2
    # with "Error: No such option '--auto-resume'." Removed 2026-05-21.
    # If pinning to an older harbor again, re-add the auto-inject guarded
    # on harbor version.
    # NOTE: --filter-error-type was a <=0.6.x harbor CLI flag for
    # auto-resume's error-type allowlist. harbor 0.7.0 replaced this with
    # --retry-include / --retry-exclude exposed on `jobs start`, but the new
    # retry model is driven primarily by --max-retries + job-config-level
    # filter settings. Auto-injecting --filter-error-type now causes
    # `harbor jobs start` to exit 2 with "No such option '--filter-error-type'".
    # Dropped 2026-05-21 along with --auto-resume above; if pinning to an
    # older harbor, re-add behind a version gate.

    # Restored 2026-05-21 (initial removal was wrong). The --export-traces /
    # --export-verifier-metadata / --export-episodes flags ARE still consumed
    # by `harbor jobs start` in 0.7.0 (penfever/otagent-latest) — they're
    # declared in src/harbor/cli/jobs.py:902-953 with `hidden=True`, so
    # `--help` doesn't list them but click/typer still parses them.
    # OT-Agent needs both traces and verifier metadata; without these flags
    # the post-run pipeline that uploads to Supabase / HF gets no inputs.
    if not (_flag_present("--export-traces") or _flag_present("--no-export-traces")):
        extra_args.append("--export-traces")
    if not (_flag_present("--export-verifier-metadata") or _flag_present("--no-export-verifier-metadata")):
        extra_args.append("--export-verifier-metadata")
    if not _flag_present("--export-episodes"):
        extra_args.extend(["--export-episodes", "last"])

    # Push the exported traces to HF directly from harbor's (gs://-aware)
    # export path. Without --export-push/--export-repo, `harbor jobs start`
    # builds the dataset in memory and discards it — the historical reason
    # datagen jobs produced no HF repo on a clean completion. harbor reads the
    # token from HF_TOKEN/HUGGINGFACE_TOKEN (forwarded to the worker via
    # --secrets-env). This supersedes the OT-Agent post_harbor_hook upload,
    # which silently no-op'd on a gs:// jobs_dir.
    if export_hf_repo and not _flag_present("--export-repo"):
        if not _flag_present("--export-push"):
            extra_args.append("--export-push")
        extra_args.extend(["--export-repo", export_hf_repo])

    for extra in extra_args:
        cmd.append(extra)

    # Sync the on-disk Harbor config.json (if any) with the YAML+CLI
    # effective state, so Harbor's _maybe_init_existing_job equality check
    # passes on resume. The on-disk JSON gets written by Harbor on first
    # run (or rewritten by hpc.resume_manager), but runtime-derived fields
    # — agent kwargs.api_base/metrics_endpoint that depend on the live
    # vLLM IP, and orchestrator.retry.include_exceptions that depend on
    # the CLI's --filter-error-type defaults — would otherwise diverge
    # from the JobConfig Harbor builds at runtime. Without this sync, a
    # second-pass launch crashes at job.py:_maybe_init_existing_job with
    # FileExistsError despite the resume_manager having mutated the
    # static fields correctly.
    if jobs_dir:
        config_json_path = Path(jobs_dir) / job_name / "config.json"
        if not config_json_path.exists():
            print(
                f"[build_harbor_command] No config.json at {config_json_path}; skip sync",
                flush=True,
            )
            return cmd
        print(
            f"[build_harbor_command] Syncing runtime fields into {config_json_path}",
            flush=True,
        )
        try:
            _sync_runtime_fields_into_config_json(
                config_json_path=config_json_path,
                modified_config=modified_config,
                extra_args=extra_args,
                n_concurrent=n_concurrent,
                n_attempts=n_attempts,
            )
            # fsync to make sure the write is visible to the harbor
            # subprocess we are about to spawn (NFS-style filesystems
            # can return stale data on close-to-open).
            with open(config_json_path, "rb") as _f:
                os.fsync(_f.fileno())
            print(
                "[build_harbor_command] Sync complete, fsync'd",
                flush=True,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            # Best-effort: if this fails, Harbor's equality check will
            # surface the original drift with a clearer trace than
            # whatever exception we'd add here.
            print(
                f"[build_harbor_command] WARNING: could not sync runtime "
                f"fields into {config_json_path}: {exc}",
                file=sys.stderr,
                flush=True,
            )

    return cmd


def _sync_runtime_fields_into_config_json(
    *,
    config_json_path: "Path",
    modified_config: dict,
    extra_args: List[str],
    n_concurrent: int,
    n_attempts: int,
) -> None:
    """Patch on-disk config.json so it matches the YAML+CLI effective state.

    Only touches fields known to be runtime-derived (and therefore guaranteed
    to drift across launches against a static on-disk JSON):
      - agents[*].kwargs.{api_base, api_key, base_url, metrics_endpoint}
      - agents[*].model_name (defensive — should already be the deterministic
        ``hosted_vllm/<id>`` alias from a prior resume_manager mutation, but
        we keep the on-disk value in lockstep with the YAML to handle
        chain-restart edge cases)
      - orchestrator.n_concurrent_trials (from CLI ``--n-concurrent``)
      - n_attempts (from CLI ``--n-attempts``)

    Notably **NOT** touched:
      - orchestrator.retry.include_exceptions. The harbor CLI's
        ``--filter-error-type`` flag is the ``auto_resume_filter_error_types``
        option (separate semantics — error types to remove from prior trial
        results on resume), NOT a JobConfig override for retry.include_exceptions.
        The retry.include_exceptions field comes from the YAML only (unless
        ``--retry-include`` is passed, which our launcher does not pass).
        Earlier versions of this sync wrote the ``--filter-error-type`` values
        into retry.include_exceptions, which caused Harbor's auto-resume
        equality check to fail because the on-disk JSON had {3 items} while
        Harbor's self.config (from YAML + no override) had {2 items}.
    """
    cj = json.loads(config_json_path.read_text())

    yaml_agents = modified_config.get("agents") or []
    cj_agents = cj.get("agents") or []
    runtime_kwarg_keys = ("api_base", "api_key", "base_url", "metrics_endpoint")
    for i, yaml_agent in enumerate(yaml_agents):
        if i >= len(cj_agents):
            break
        if not isinstance(yaml_agent, dict) or not isinstance(cj_agents[i], dict):
            continue
        cj_agent = cj_agents[i]
        yaml_kw = yaml_agent.get("kwargs") or {}
        cj_kw = cj_agent.setdefault("kwargs", {})
        for key in runtime_kwarg_keys:
            if key in yaml_kw:
                cj_kw[key] = yaml_kw[key]
        if "model_name" in yaml_agent:
            cj_agent["model_name"] = yaml_agent["model_name"]

    orchestrator = cj.setdefault("orchestrator", {})
    orchestrator["n_concurrent_trials"] = n_concurrent
    cj["n_attempts"] = n_attempts

    # Mirror the YAML's retry block (or n_concurrent_trials if top-level)
    # into config.json. This is defensive: if the YAML was updated between
    # the prior resume_manager mutation and now (e.g. exception lists in
    # the harbor_config.yaml were tweaked), the on-disk JSON must follow
    # so Harbor's self.config (which is just YAML+CLI overrides) matches.
    yaml_orchestrator = modified_config.get("orchestrator") or {}
    yaml_retry = yaml_orchestrator.get("retry") if isinstance(yaml_orchestrator, dict) else None
    if isinstance(yaml_retry, dict):
        cj_retry = orchestrator.setdefault("retry", {})
        # Only sync the include/exclude exception lists — other retry fields
        # (max_retries, wait_multiplier, etc.) are not runtime-derived and
        # don't need re-syncing.
        for k in ("include_exceptions", "exclude_exceptions", "mask_exceptions", "passthrough_exceptions"):
            if k in yaml_retry:
                cj_retry[k] = yaml_retry[k]

    config_json_path.write_text(json.dumps(cj, indent=2))


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
    "merge_harbor_config",
    "run_harbor_cli",
]
