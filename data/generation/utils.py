"""Utility helpers for generator CLI wiring and engine configuration."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from .engines import InferenceEngine, create_inference_engine


@dataclass
class _ProviderAuthConfig:
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None


@dataclass
class OpenAIProviderConfig(_ProviderAuthConfig):
    organization: Optional[str] = None
    project: Optional[str] = None


@dataclass
class AnthropicProviderConfig(_ProviderAuthConfig):
    pass


@dataclass
class VLLMLocalProviderConfig(_ProviderAuthConfig):
    endpoint_json: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    healthcheck_interval: Optional[int] = None


@dataclass
class DatagenEngineConfig:
    type: str
    model: Optional[str] = None
    max_output_tokens: Optional[int] = None
    healthcheck_interval: Optional[int] = 300
    request_params: Dict[str, Any] = field(default_factory=dict)
    openai: Optional[OpenAIProviderConfig] = None
    anthropic: Optional[AnthropicProviderConfig] = None
    vllm_local: Optional[VLLMLocalProviderConfig] = None
    extra_args: Any = None


@dataclass
class DatagenBackendConfig:
    type: str = "vllm"
    wait_for_endpoint: bool = False
    endpoint_json_path: Optional[str] = None
    ray_port: int = 6379
    api_port: int = 8000
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    ray_cgraph_submit_timeout: Optional[Union[int, str]] = None
    ray_cgraph_get_timeout: Optional[Union[int, str]] = None
    ray_cgraph_max_inflight_executions: Optional[Union[int, str]] = None
    healthcheck_max_attempts: Optional[int] = None
    healthcheck_retry_delay: Optional[int] = None


@dataclass
class VLLMServerConfig:
    model_path: str
    num_replicas: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    custom_model_name: Optional[str] = None
    endpoint_json_path: Optional[str] = None
    time_limit: str = "48:00:00"
    hf_overrides: Optional[str] = None
    use_deep_gemm: bool = False
    max_num_seqs: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    enable_expert_parallel: bool = False
    swap_space: Optional[int] = None
    max_seq_len_to_capture: Optional[int] = None
    max_model_len: Optional[int] = None
    cpu_offload_gb: Optional[float] = None
    kv_offloading_size: Optional[float] = None
    kv_offloading_backend: Optional[str] = None
    trust_remote_code: bool = False
    disable_log_requests: bool = False
    enable_auto_tool_choice: bool = False
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    logging_level: Optional[str] = None
    extra_args: Any = None


@dataclass
class DataGenerationConfig:
    engine: DatagenEngineConfig
    backend: DatagenBackendConfig = field(default_factory=DatagenBackendConfig)
    extra_agent_kwargs: Dict[str, Any] = field(default_factory=dict)
    chunk_array_max: Optional[int] = None
    vllm_server: Optional[VLLMServerConfig] = None


@dataclass
class RuntimeEngineSettings:
    type: str
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)
    request_params: Dict[str, Any] = field(default_factory=dict)
    max_output_tokens: Optional[int] = None
    healthcheck_interval: Optional[int] = None


@dataclass
class LoadedDatagenConfig:
    path: Path
    config: DataGenerationConfig
    raw: DictConfig


def add_generation_args(
    parser: argparse.ArgumentParser,
    *,
    default_target_repo: Optional[str] = None,
    default_input_dir: Optional[str] = None,
    include_no_upload: bool = True,
    default_engine: Optional[str] = None,  # legacy; ignored but kept for backward compat signature
) -> argparse.ArgumentParser:
    """Augment ``parser`` with standard generation CLI flags."""

    general_group = parser.add_argument_group("General Generation Options")
    general_group.add_argument(
        "--input-dir",
        type=str,
        default=default_input_dir,
        help="Optional input directory for generation pipeline",
    )
    general_group.add_argument(
        "--target-repo",
        type=str,
        default=default_target_repo,
        help="Target HuggingFace repository for generated data",
    )
    general_group.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write generated artifacts",
    )
    general_group.add_argument(
        "--tasks-input",
        type=str,
        default=None,
        help="Path to an existing task dataset when running trace generation",
    )
    general_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to generate/process (default: unlimited)",
    )
    if include_no_upload:
        general_group.add_argument(
            "--no-upload",
            action="store_true",
            help="Skip uploading generated data",
        )

    engine_group = parser.add_argument_group("Inference Engine Options")
    engine_group.add_argument(
        "--engine-config",
        type=str,
        default=os.environ.get("DATAGEN_CONFIG_PATH"),
        help="Path to YAML file describing inference engine + backend configuration "
        "(defaults to DATAGEN_CONFIG_PATH env var if unset).",
    )

    trace_group = parser.add_argument_group("Trace Generation Options")
    trace_group.add_argument(
        "--trace-harbor-config",
        type=str,
        help="Path to Harbor job YAML defining trace execution parameters",
    )
    trace_group.add_argument(
        "--trace-model",
        type=str,
        help="Model name to use during trace generation",
    )
    trace_group.add_argument(
        "--trace-agent-name",
        type=str,
        help="Sandboxes agent name for trace generation",
    )
    trace_group.add_argument(
        "--trace-jobs-dir",
        type=str,
        help="Directory to store sandboxes jobs",
    )
    trace_group.add_argument(
        "--trace-n-concurrent",
        type=int,
        help="Number of concurrent trials during trace generation",
    )
    trace_group.add_argument(
        "--trace-agent-kwargs",
        type=str,
        help="JSON string of additional kwargs for the trace agent",
    )
    trace_group.add_argument(
        "--trace-env",
        type=str,
        help="Environment type for trace generation (e.g., docker, daytona)",
    )
    trace_group.add_argument(
        "--trace-episodes",
        type=str,
        choices=["all", "last"],
        help="Which episodes to export",
    )
    trace_group.add_argument(
        "--trace-export-filter",
        type=str,
        choices=["success", "failure", "none"],
        help="Filter for exported traces",
    )
    trace_group.add_argument(
        "--trace-dataset-type",
        type=str,
        help="Dataset type label when uploading traces (default: SFT)",
    )
    trace_group.add_argument(
        "--endpoint-json",
        dest="endpoint_json",
        type=str,
        help="Path to a running vLLM endpoint JSON (required for vllm_local engines).",
    )
    trace_group.add_argument(
        "--trace-export-subagents",
        dest="trace_export_subagents",
        action="store_true",
        help="Export subagent traces (e.g., context summarization) alongside main agent traces (default: enabled).",
    )
    trace_group.add_argument(
        "--trace-skip-subagents",
        dest="trace_export_subagents",
        action="store_false",
        help="Disable subagent trace export when exporting traces.",
    )
    parser.set_defaults(trace_export_subagents=True)
    trace_group.add_argument(
        "--trace-agent-timeout-sec",
        type=float,
        help="Override Harbor agent timeout_sec for each trial",
    )
    trace_group.add_argument(
        "--trace-verifier-timeout-sec",
        type=float,
        help="Override Harbor verifier timeout_sec for each trial",
    )
    trace_group.add_argument(
        "--disable-verification",
        action="store_true",
        dest="disable_verification",
        help="Disable Harbor verification when collecting traces",
    )

    sandbox_group = parser.add_argument_group("Task Environment Overrides")
    sandbox_group.add_argument(
        "--sandbox-cpu",
        "--sandbox_cpu",
        dest="sandbox_cpu",
        type=int,
        default=None,
        help="Override Daytona sandbox vCPU allocation when generating tasks.",
    )
    sandbox_group.add_argument(
        "--sandbox-memory-gb",
        "--sandbox_memory_gb",
        dest="sandbox_memory_gb",
        type=int,
        default=None,
        help="Override Daytona sandbox memory (GB) when generating tasks.",
    )
    sandbox_group.add_argument(
        "--sandbox-disk-gb",
        "--sandbox_disk_gb",
        dest="sandbox_disk_gb",
        type=int,
        default=None,
        help="Override Daytona sandbox disk (GB) when generating tasks.",
    )

    return parser


def _resolve_api_key(config: Optional[_ProviderAuthConfig]) -> Optional[str]:
    if not config:
        return None
    if config.api_key:
        return config.api_key
    if config.api_key_env:
        return os.environ.get(config.api_key_env)
    return None


def resolve_engine_runtime(config: DataGenerationConfig) -> RuntimeEngineSettings:
    """Normalize ``config`` into runtime-friendly engine settings."""

    engine_cfg = config.engine
    engine_type = engine_cfg.type.lower()

    if engine_type not in {"openai", "anthropic", "vllm_local", "gemini_openai", "google_gemini", "none"}:
        raise ValueError(f"Unsupported engine type: {engine_cfg.type}")

    if engine_type == "none":
        return RuntimeEngineSettings(type="none")

    engine_kwargs: Dict[str, Any] = {}

    if engine_cfg.max_output_tokens is not None:
        engine_kwargs["max_tokens"] = int(engine_cfg.max_output_tokens)

    if engine_type in {"openai", "anthropic", "gemini_openai", "google_gemini"}:
        provider = (
            engine_cfg.openai
            if engine_type in {"openai", "gemini_openai", "google_gemini"}
            else engine_cfg.anthropic
        )
        if engine_cfg.model:
            engine_kwargs["model"] = engine_cfg.model
        api_key = _resolve_api_key(provider)
        if api_key:
            engine_kwargs["api_key"] = api_key
    elif engine_type == "vllm_local":
        provider = engine_cfg.vllm_local
        if provider is None:
            provider = VLLMLocalProviderConfig()
            engine_cfg.vllm_local = provider

        if provider.endpoint_json:
            engine_kwargs["endpoint_json"] = provider.endpoint_json
        else:
            base_url = provider.base_url
            model_name = provider.model_name or engine_cfg.model
            if base_url:
                engine_kwargs["base_url"] = base_url.rstrip("/")
            if model_name:
                engine_kwargs["model_name"] = model_name

        api_key = _resolve_api_key(provider)
        if api_key:
            engine_kwargs["api_key"] = api_key

        effective_interval = provider.healthcheck_interval or engine_cfg.healthcheck_interval
        if effective_interval is not None:
            engine_kwargs["healthcheck_interval"] = int(effective_interval)
    else:  # pragma: no cover - defensive programming for future types
        raise ValueError(f"Unhandled engine type: {engine_type}")

    if engine_cfg.healthcheck_interval is not None and engine_type != "vllm_local":
        engine_kwargs["healthcheck_interval"] = int(engine_cfg.healthcheck_interval)

    extra_args = engine_cfg.extra_args
    if isinstance(extra_args, dict):
        engine_kwargs.update(extra_args)
    elif extra_args not in (None, "", [], ()):
        raise ValueError(
            f"engine.extra_args must be a mapping when provided (got {type(extra_args).__name__})"
        )

    return RuntimeEngineSettings(
        type=engine_type,
        engine_kwargs=engine_kwargs,
        request_params=dict(engine_cfg.request_params or {}),
        max_output_tokens=engine_cfg.max_output_tokens,
        healthcheck_interval=engine_kwargs.get("healthcheck_interval"),
    )


def load_datagen_config(config_path: str | os.PathLike[str]) -> LoadedDatagenConfig:
    """Load and validate a datagen engine configuration from YAML."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Datagen engine config not found: {path}")

    try:
        raw_cfg = OmegaConf.load(path)
    except OmegaConfBaseException as exc:
        raise ValueError(f"Failed to load datagen config at {path}: {exc}") from exc

    if not isinstance(raw_cfg, DictConfig):
        raise TypeError(f"Datagen config at {path} is not a mapping (got {type(raw_cfg).__name__})")

    template = OmegaConf.structured(DataGenerationConfig)
    try:
        merged: DictConfig = OmegaConf.merge(template, raw_cfg)
        OmegaConf.resolve(merged)
    except OmegaConfBaseException as exc:
        raise ValueError(f"Invalid datagen config at {path}: {exc}") from exc

    config_obj = OmegaConf.to_object(merged)
    if not isinstance(config_obj, DataGenerationConfig):
        raise TypeError("OmegaConf returned unexpected object when materializing datagen config.")

    return LoadedDatagenConfig(path=path, config=config_obj, raw=merged)


def create_engine_from_args(args: argparse.Namespace) -> Optional[InferenceEngine]:
    """Instantiate an inference engine using the resolved configuration."""

    runtime: Optional[RuntimeEngineSettings] = getattr(args, "_engine_runtime", None)
    if runtime is None:
        config_path = getattr(args, "engine_config", None) or os.environ.get("DATAGEN_CONFIG_PATH")
        if not config_path:
            return None
        loaded = load_datagen_config(config_path)
        runtime = resolve_engine_runtime(loaded.config)
        setattr(args, "_engine_runtime", runtime)
        setattr(args, "_datagen_config", loaded.config)
        setattr(args, "_datagen_config_raw", loaded.raw)
        setattr(args, "_datagen_config_path", str(loaded.path))

    if runtime.type == "none":
        return None

    return create_inference_engine(runtime.type, **runtime.engine_kwargs)


__all__ = [
    "add_generation_args",
    "create_engine_from_args",
    "load_datagen_config",
    "resolve_engine_runtime",
    "RuntimeEngineSettings",
    "DataGenerationConfig",
    "DatagenEngineConfig",
    "DatagenBackendConfig",
    "VLLMServerConfig",
    "LoadedDatagenConfig",
]
