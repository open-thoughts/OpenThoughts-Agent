"""Consolidated datagen configuration parsing utilities.

This module provides a unified interface for parsing datagen configuration YAML files.
It consolidates the duplicate parsing logic from:
- hpc/datagen_launch_utils.py (_prepare_datagen_configuration)
- hpc/local_runner_utils.py (apply_datagen_defaults)

The main entry point is `parse_datagen_config()` which returns a `ParsedDatagenConfig`
dataclass with all extracted values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.generation.utils import (
    LoadedDatagenConfig,
    load_datagen_config,
    resolve_engine_runtime,
)
from hpc.cli_utils import normalize_cli_args
from hpc.launch_utils import maybe_int


# API-based engines that don't require local Ray/vLLM
API_ENGINES = frozenset({"openai", "anthropic", "azure", "together", "fireworks", "groq", "google_gemini", "gemini_openai"})


@dataclass
class ParsedDatagenConfig:
    """Result of parsing a datagen configuration file.

    This dataclass holds all values extracted from the datagen YAML config.
    Both HPC and local runner paths use this for consistent config handling.
    """

    # Path info
    config_path: Path
    """Resolved absolute path to the config file."""

    # Loaded config objects (for advanced use cases)
    loaded: LoadedDatagenConfig
    """The full LoadedDatagenConfig with raw and structured config."""

    # Engine settings
    engine_type: str
    """Engine type: 'vllm_local', 'openai', 'anthropic', etc."""

    model: Optional[str]
    """Model path/name from config."""

    max_output_tokens: Optional[int]
    """Maximum output tokens if specified."""

    healthcheck_interval: int
    """Healthcheck interval in seconds (default: 300)."""

    needs_local_vllm: bool
    """Whether this engine requires local Ray/vLLM server."""

    # Parallelism settings (from vllm_server or backend)
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int

    # Port settings
    ray_port: int
    api_port: int

    # Extra agent kwargs
    extra_agent_kwargs: Dict[str, Any]
    """Additional agent kwargs from datagen config."""

    # vLLM server config (if present)
    vllm_server_config: Optional[Any] = None
    """Raw VLLMServerConfig dataclass if vllm_server section exists."""

    vllm_extra_args: List[str] = field(default_factory=list)
    """Extra CLI args for vLLM from config."""

    # Backend settings
    endpoint_json_path: Optional[str] = None
    """Path to write endpoint JSON."""

    wait_for_endpoint: bool = False
    """Whether to wait for endpoint on startup."""

    # Health check settings
    health_max_attempts: Optional[int] = None
    health_retry_delay: Optional[int] = None

    # HPC-specific settings (only used by SLURM path)
    chunk_array_max: Optional[int] = None
    ray_cgraph_submit_timeout: Optional[str] = None
    ray_cgraph_get_timeout: Optional[str] = None
    ray_cgraph_max_inflight_executions: Optional[str] = None


def parse_datagen_config(
    config_path: str,
    model_override: Optional[str] = None,
    config_dir_fallback: Optional[str] = None,
) -> ParsedDatagenConfig:
    """Parse a datagen configuration file and extract all settings.

    This is the consolidated entry point for datagen config parsing, used by both:
    - HPC/SLURM path (datagen_launch_utils._prepare_datagen_configuration)
    - Local runner path (local_runner_utils.apply_datagen_defaults)

    Args:
        config_path: Path to datagen config YAML (absolute or relative)
        model_override: Optional model name to override config value
        config_dir_fallback: Optional directory to search if config_path not found directly

    Returns:
        ParsedDatagenConfig with all extracted values

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    # Resolve config path
    path = Path(config_path).expanduser()
    if not path.is_absolute() and config_dir_fallback:
        fallback_path = Path(config_dir_fallback) / config_path
        if fallback_path.exists():
            path = fallback_path
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Datagen config not found: {path}")

    # Load config using structured loader
    loaded = load_datagen_config(path)
    config = loaded.config

    # Apply model override if provided
    if model_override:
        config.engine.model = model_override
        if config.vllm_server:
            config.vllm_server.model_path = model_override

    # Resolve engine runtime for type detection
    runtime = resolve_engine_runtime(config)
    engine_type = runtime.type.lower()

    # Determine model (from runtime, override, or config)
    model = (
        runtime.engine_kwargs.get("model")
        or runtime.engine_kwargs.get("model_name")
        or model_override
        or config.engine.model
    )
    if config.vllm_server and not model:
        model = config.vllm_server.model_path

    # Determine if local vLLM is needed
    needs_local_vllm = engine_type not in API_ENGINES

    # Extract parallelism settings (prefer vllm_server, fallback to backend)
    vllm_cfg = config.vllm_server
    backend = config.backend

    tensor_parallel_size = (
        (vllm_cfg.tensor_parallel_size if vllm_cfg else None)
        or backend.tensor_parallel_size
        or 1
    )
    pipeline_parallel_size = (
        (vllm_cfg.pipeline_parallel_size if vllm_cfg else None)
        or backend.pipeline_parallel_size
        or 1
    )
    data_parallel_size = (
        (vllm_cfg.data_parallel_size if vllm_cfg else None)
        or backend.data_parallel_size
        or 1
    )

    # Port settings
    ray_port = backend.ray_port or 6379
    api_port = backend.api_port or 8000

    # Endpoint JSON path (prefer vllm_server, fallback to backend)
    endpoint_json_path = None
    if vllm_cfg and vllm_cfg.endpoint_json_path:
        endpoint_json_path = vllm_cfg.endpoint_json_path
    elif backend.endpoint_json_path:
        endpoint_json_path = backend.endpoint_json_path

    # Extra vLLM CLI args
    vllm_extra_args = normalize_cli_args(vllm_cfg.extra_args if vllm_cfg else None)

    # Health check settings
    health_max_attempts = maybe_int(backend.healthcheck_max_attempts)
    health_retry_delay = maybe_int(backend.healthcheck_retry_delay)

    # HPC-specific settings
    chunk_array_max = maybe_int(config.chunk_array_max)
    ray_cgraph_submit_timeout = (
        str(backend.ray_cgraph_submit_timeout)
        if backend.ray_cgraph_submit_timeout is not None
        else None
    )
    ray_cgraph_get_timeout = (
        str(backend.ray_cgraph_get_timeout)
        if backend.ray_cgraph_get_timeout is not None
        else None
    )
    ray_cgraph_max_inflight_executions = (
        str(backend.ray_cgraph_max_inflight_executions)
        if backend.ray_cgraph_max_inflight_executions is not None
        else None
    )

    return ParsedDatagenConfig(
        config_path=path,
        loaded=loaded,
        engine_type=engine_type,
        model=model,
        max_output_tokens=runtime.max_output_tokens,
        healthcheck_interval=runtime.healthcheck_interval or 300,
        needs_local_vllm=needs_local_vllm,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        ray_port=ray_port,
        api_port=api_port,
        extra_agent_kwargs=dict(config.extra_agent_kwargs or {}),
        vllm_server_config=vllm_cfg,
        vllm_extra_args=vllm_extra_args,
        endpoint_json_path=endpoint_json_path,
        wait_for_endpoint=backend.wait_for_endpoint,
        health_max_attempts=health_max_attempts,
        health_retry_delay=health_retry_delay,
        chunk_array_max=chunk_array_max,
        ray_cgraph_submit_timeout=ray_cgraph_submit_timeout,
        ray_cgraph_get_timeout=ray_cgraph_get_timeout,
        ray_cgraph_max_inflight_executions=ray_cgraph_max_inflight_executions,
    )


__all__ = [
    "ParsedDatagenConfig",
    "parse_datagen_config",
    "API_ENGINES",
]
