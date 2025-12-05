"""Shared utilities for datagen-oriented HPC launches."""

from __future__ import annotations

import json
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from omegaconf import OmegaConf

from data.generation.utils import load_datagen_config, resolve_engine_runtime

DIRENV = os.path.dirname(__file__)
DATAGEN_CONFIG_DIR = os.path.join(DIRENV, "datagen_yaml")
HARBOR_CONFIG_DIR = os.path.join(DIRENV, "harbor_yaml")
DEFAULT_RAY_CGRAPH_TIMEOUT = os.environ.get("RAY_CGRAPH_TIMEOUT_DEFAULT", "86500")
DEFAULT_RAY_CGRAPH_MAX_INFLIGHT = os.environ.get("RAY_CGRAPH_MAX_INFLIGHT_DEFAULT", "")


def default_vllm_endpoint_path(
    experiments_dir: str | os.PathLike[str],
    *,
    trace: bool = False,
    chunk_index: int | None = None,
) -> str:
    """Compute a canonical vLLM endpoint JSON path under ``experiments_dir``.

    Args:
        experiments_dir: Base experiments directory.
        trace: Whether the path is for trace collection (adds trace-specific suffix).
        chunk_index: Optional chunk index for sharded trace jobs.
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


def resolve_datagen_config_path(raw_value: str) -> Path:
    """Resolve ``raw_value`` to an absolute datagen config path."""

    candidate = Path(raw_value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    default_candidate = Path(DATAGEN_CONFIG_DIR) / candidate
    if default_candidate.exists():
        return default_candidate.resolve()

    fallback_candidate = Path(DATAGEN_CONFIG_DIR) / candidate.name
    if fallback_candidate.exists():
        return fallback_candidate.resolve()

    raise FileNotFoundError(
        f"Datagen config not found: {raw_value}. "
        f"Tried {candidate}, {default_candidate}, and {fallback_candidate}."
    )


def resolve_harbor_config_path(raw_value: str) -> Path:
    """Resolve ``raw_value`` to an absolute Harbor job config path."""

    candidate = Path(raw_value).expanduser()
    if candidate.exists():
        return candidate.resolve()

    default_candidate = Path(HARBOR_CONFIG_DIR) / candidate
    if default_candidate.exists():
        return default_candidate.resolve()

    fallback_candidate = Path(HARBOR_CONFIG_DIR) / candidate.name
    if fallback_candidate.exists():
        return fallback_candidate.resolve()

    raise FileNotFoundError(
        f"Harbor job config not found: {raw_value}. "
        f"Tried {candidate}, {default_candidate}, and {fallback_candidate}."
    )


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value))
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _estimate_max_inflight(env: Dict[str, str]) -> int:
    pipeline = _coerce_positive_int(
        env.get("VLLM_PIPELINE_PARALLEL_SIZE")
        or os.environ.get("VLLM_PIPELINE_PARALLEL_SIZE"),
        default=1,
    )
    tensor = _coerce_positive_int(
        env.get("VLLM_TENSOR_PARALLEL_SIZE")
        or os.environ.get("VLLM_TENSOR_PARALLEL_SIZE"),
        default=1,
    )
    # Heuristic: leave room for two concurrent batches per PP stage,
    # and ensure tensor-parallel groups don't bottleneck tiny defaults.
    concurrency_hint = max(pipeline * 2, tensor * 2)
    return max(16, concurrency_hint)


def _maybe_set_ray_cgraph_env(env: Dict[str, str]) -> None:
    """Ensure Ray compiled-DAG knobs are exported when using Ray backends."""

    submit_timeout = os.environ.get("RAY_CGRAPH_submit_timeout", DEFAULT_RAY_CGRAPH_TIMEOUT)
    get_timeout = os.environ.get("RAY_CGRAPH_get_timeout", DEFAULT_RAY_CGRAPH_TIMEOUT)
    env.setdefault("RAY_CGRAPH_submit_timeout", str(submit_timeout))
    env.setdefault("RAY_CGRAPH_get_timeout", str(get_timeout))

    inflight_override = env.get("RAY_CGRAPH_max_inflight_executions") or os.environ.get(
        "RAY_CGRAPH_max_inflight_executions"
    )
    if not inflight_override:
        inflight_override = DEFAULT_RAY_CGRAPH_MAX_INFLIGHT or _estimate_max_inflight(env)
    env.setdefault("RAY_CGRAPH_max_inflight_executions", str(inflight_override))


def _normalize_cli_args(args_spec: Any) -> list[str]:
    """Normalize a YAML-provided CLI arg spec into a flat list of strings."""

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


def _prepare_datagen_configuration(exp_args: dict):
    """Load the YAML datagen configuration and derive launch metadata."""

    raw_config = exp_args.get("datagen_config") or os.environ.get("DATAGEN_CONFIG_PATH")
    if not raw_config:
        raise ValueError(
            "Data generation requires --datagen-config or DATAGEN_CONFIG_PATH to specify the engine YAML."
        )

    resolved_path = resolve_datagen_config_path(raw_config)
    loaded = load_datagen_config(resolved_path)

    trace_model_override = exp_args.get("trace_model")
    if trace_model_override:
        engine_cfg = loaded.config.engine
        engine_type = (engine_cfg.type or "").lower()
        if engine_type in {"openai", "anthropic"}:
            engine_cfg.model = trace_model_override
            try:
                loaded.raw.engine.model = trace_model_override
            except AttributeError:
                pass

    runtime = resolve_engine_runtime(loaded.config)

    exp_args["_datagen_config_original_path"] = str(resolved_path)
    exp_args["_datagen_config_raw"] = loaded.raw
    exp_args["_datagen_config_obj"] = loaded.config
    extra_agent_kwargs = dict(getattr(loaded.config, "extra_agent_kwargs", {}) or {})
    exp_args["_datagen_extra_agent_kwargs"] = extra_agent_kwargs
    chunk_array_max = getattr(loaded.config, "chunk_array_max", None)
    try:
        chunk_array_max = int(chunk_array_max) if chunk_array_max is not None else None
    except (TypeError, ValueError):
        chunk_array_max = None
    exp_args["_chunk_array_max"] = chunk_array_max
    exp_args["_datagen_engine_runtime"] = runtime
    exp_args["datagen_config_path"] = str(resolved_path)

    exp_args["datagen_engine"] = runtime.type
    exp_args["datagen_healthcheck_interval"] = runtime.healthcheck_interval or 300
    if runtime.engine_kwargs.get("model"):
        exp_args["datagen_model"] = runtime.engine_kwargs["model"]
    else:
        exp_args.pop("datagen_model", None)
    if runtime.max_output_tokens is not None:
        exp_args["datagen_max_tokens"] = runtime.max_output_tokens
    else:
        exp_args.pop("datagen_max_tokens", None)

    backend = loaded.config.backend
    exp_args["_datagen_backend_config"] = backend
    exp_args["datagen_backend"] = backend.type
    exp_args["datagen_wait_for_endpoint"] = backend.wait_for_endpoint
    exp_args["datagen_ray_port"] = backend.ray_port
    exp_args["datagen_api_port"] = backend.api_port
    if backend.endpoint_json_path:
        exp_args["vllm_endpoint_json_path"] = backend.endpoint_json_path
    if backend.ray_cgraph_submit_timeout is not None:
        exp_args["ray_cgraph_submit_timeout"] = str(backend.ray_cgraph_submit_timeout)
    else:
        exp_args.pop("ray_cgraph_submit_timeout", None)
    if backend.ray_cgraph_get_timeout is not None:
        exp_args["ray_cgraph_get_timeout"] = str(backend.ray_cgraph_get_timeout)
    else:
        exp_args.pop("ray_cgraph_get_timeout", None)
    if backend.ray_cgraph_max_inflight_executions is not None:
        exp_args["ray_cgraph_max_inflight_executions"] = str(
            backend.ray_cgraph_max_inflight_executions
        )
    else:
        exp_args.pop("ray_cgraph_max_inflight_executions", None)
    if backend.healthcheck_max_attempts is not None:
        exp_args["trace_health_max_attempts"] = int(backend.healthcheck_max_attempts)
    elif "trace_health_max_attempts" in exp_args:
        exp_args.pop("trace_health_max_attempts")
    if backend.healthcheck_retry_delay is not None:
        exp_args["trace_health_retry_delay"] = int(backend.healthcheck_retry_delay)
    elif "trace_health_retry_delay" in exp_args:
        exp_args.pop("trace_health_retry_delay")

    vllm_cfg = loaded.config.vllm_server
    exp_args["_datagen_vllm_server_config"] = vllm_cfg
    if vllm_cfg and vllm_cfg.endpoint_json_path:
        exp_args["vllm_endpoint_json_path"] = vllm_cfg.endpoint_json_path
    elif exp_args.get("vllm_endpoint_json_path") and not vllm_cfg:
        exp_args.pop("vllm_endpoint_json_path", None)
    if vllm_cfg:
        extra_cli_args = _normalize_cli_args(vllm_cfg.extra_args)
        if extra_cli_args:
            exp_args["_vllm_server_extra_args"] = extra_cli_args
        elif "_vllm_server_extra_args" in exp_args:
            exp_args.pop("_vllm_server_extra_args")
    elif "_vllm_server_extra_args" in exp_args:
        exp_args.pop("_vllm_server_extra_args")

    return runtime


def _snapshot_datagen_config(
    exp_args: dict,
    *,
    output_filename: str | None = None,
    update_exp_args: bool = True,
) -> str:
    """Persist the resolved datagen config into the experiment directory."""

    raw_cfg = exp_args.get("_datagen_config_raw")
    if raw_cfg is None:
        raise ValueError("Datagen configuration not initialized before snapshot.")

    endpoint_path = exp_args.get("vllm_endpoint_json_path")
    cfg_to_save = raw_cfg
    if endpoint_path:
        cfg_to_save = OmegaConf.create(OmegaConf.to_container(raw_cfg, resolve=False))
        try:
            cfg_to_save.engine.vllm_local.endpoint_json = endpoint_path  # type: ignore[attr-defined]
        except AttributeError:
            pass
        try:
            cfg_to_save.backend.endpoint_json_path = endpoint_path  # type: ignore[attr-defined]
        except AttributeError:
            pass
        try:
            cfg_to_save.vllm_server.endpoint_json_path = endpoint_path  # type: ignore[attr-defined]
        except AttributeError:
            pass

    experiments_dir = exp_args.get("experiments_dir")
    if not experiments_dir:
        path_candidate = exp_args.get("datagen_config_path") or exp_args.get("_datagen_config_original_path")
        if path_candidate:
            return path_candidate
        raise ValueError("Unable to determine datagen config path to use without experiments_dir.")

    configs_dir = Path(experiments_dir) / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    filename = output_filename or "datagen_config.resolved.yaml"
    snapshot_path = configs_dir / filename
    OmegaConf.save(cfg_to_save, snapshot_path)
    if update_exp_args:
        exp_args["datagen_config_path"] = str(snapshot_path)
    return str(snapshot_path)


def _build_vllm_env_vars(exp_args: dict) -> Tuple[Dict[str, str], dict]:
    """Return environment variables used to configure vLLM processes."""

    env: Dict[str, str] = {}
    cfg = exp_args.get("_datagen_vllm_server_config")
    if not cfg:
        return env, exp_args

    env["VLLM_MODEL_PATH"] = cfg.model_path
    env["VLLM_NUM_REPLICAS"] = str(cfg.num_replicas or 1)
    env["VLLM_TENSOR_PARALLEL_SIZE"] = str(cfg.tensor_parallel_size or 1)
    env["VLLM_PIPELINE_PARALLEL_SIZE"] = str(cfg.pipeline_parallel_size or 1)
    env["VLLM_DATA_PARALLEL_SIZE"] = str(cfg.data_parallel_size or 1)

    if cfg.hf_overrides:
        env["VLLM_HF_OVERRIDES"] = cfg.hf_overrides
    if cfg.use_deep_gemm:
        env["VLLM_USE_DEEP_GEMM"] = "1"
    if cfg.max_num_seqs is not None:
        env["VLLM_MAX_NUM_SEQS"] = str(cfg.max_num_seqs)
    if cfg.gpu_memory_utilization is not None:
        env["VLLM_GPU_MEMORY_UTILIZATION"] = str(cfg.gpu_memory_utilization)
    if cfg.cpu_offload_gb is not None:
        env["VLLM_CPU_OFFLOAD_GB"] = str(cfg.cpu_offload_gb)
    if cfg.kv_offloading_size is not None:
        env["VLLM_KV_OFFLOADING_SIZE"] = str(cfg.kv_offloading_size)
    if cfg.kv_offloading_backend:
        env["VLLM_KV_OFFLOADING_BACKEND"] = cfg.kv_offloading_backend
    if cfg.enable_expert_parallel:
        env["VLLM_ENABLE_EXPERT_PARALLEL"] = "1"
    if cfg.swap_space is not None:
        env["VLLM_SWAP_SPACE"] = str(cfg.swap_space)
    if cfg.max_seq_len_to_capture is not None:
        env["VLLM_MAX_SEQ_LEN_TO_CAPTURE"] = str(cfg.max_seq_len_to_capture)
    if cfg.max_model_len is not None:
        env["VLLM_MAX_MODEL_LEN"] = str(cfg.max_model_len)
    if cfg.trust_remote_code:
        env["VLLM_TRUST_REMOTE_CODE"] = "1"
    if cfg.disable_log_requests:
        env["VLLM_DISABLE_LOG_REQUESTS"] = "1"
    if cfg.custom_model_name:
        env["VLLM_CUSTOM_MODEL_NAME"] = cfg.custom_model_name
    if cfg.enable_auto_tool_choice:
        env["VLLM_ENABLE_AUTO_TOOL_CHOICE"] = "1"
    if cfg.tool_call_parser:
        env["VLLM_TOOL_CALL_PARSER"] = cfg.tool_call_parser
    if cfg.reasoning_parser:
        env["VLLM_REASONING_PARSER"] = cfg.reasoning_parser
    if cfg.logging_level:
        env["VLLM_LOGGING_LEVEL"] = str(cfg.logging_level)

    max_output_tokens = (
        exp_args.get("trace_max_tokens")
        if exp_args.get("trace_max_tokens") not in (None, "", "None")
        else exp_args.get("datagen_max_tokens")
    )

    if max_output_tokens not in (None, "", "None"):
        env["VLLM_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)

    endpoint_path = exp_args.get("vllm_endpoint_json_path")
    if not endpoint_path and cfg.endpoint_json_path:
        endpoint_path = cfg.endpoint_json_path
    if not endpoint_path:
        experiments_dir = exp_args.get("experiments_dir")
        if not experiments_dir:
            raise ValueError("experiments_dir is required to compute default vLLM endpoint path")
        endpoint_path = default_vllm_endpoint_path(experiments_dir)
        exp_args["vllm_endpoint_json_path"] = endpoint_path
    env["VLLM_ENDPOINT_JSON_PATH"] = endpoint_path

    extra_cli_args = exp_args.get("_vllm_server_extra_args")
    if extra_cli_args:
        env["VLLM_SERVER_EXTRA_ARGS_JSON"] = json.dumps(extra_cli_args)

    submit_timeout = exp_args.get("ray_cgraph_submit_timeout")
    get_timeout = exp_args.get("ray_cgraph_get_timeout")
    max_inflight = exp_args.get("ray_cgraph_max_inflight_executions")
    if submit_timeout:
        env["RAY_CGRAPH_submit_timeout"] = str(submit_timeout)
    if get_timeout:
        env["RAY_CGRAPH_get_timeout"] = str(get_timeout)
    if max_inflight:
        env["RAY_CGRAPH_max_inflight_executions"] = str(max_inflight)

    _maybe_set_ray_cgraph_env(env)

    return env, exp_args


@dataclass
class TraceChunkPlan:
    index: int
    tasks_path: Path
    output_dir: Path
    jobs_dir: Path
    target_repo: str
    task_names: list[str]


def _format_chunk_target_repo(base_repo: str, chunk_index: int) -> str:
    owner: Optional[str]
    name: str
    if "/" in base_repo:
        owner, name = base_repo.split("/", 1)
        return f"{owner}/{name}-chunk{chunk_index:03d}"
    return f"{base_repo}-chunk{chunk_index:03d}"


def _discover_task_entries(tasks_root: Path, *, create_if_missing: bool = False) -> list[Path]:
    if not tasks_root.exists():
        if not create_if_missing:
            raise FileNotFoundError(f"Trace tasks path does not exist: {tasks_root}")
        tasks_root.mkdir(parents=True, exist_ok=True)
        return []

    candidates = [
        child
        for child in tasks_root.iterdir()
        if not child.name.startswith(".")
    ]
    directories = sorted([c for c in candidates if c.is_dir()], key=lambda p: p.name)
    if directories:
        return directories
    files = sorted([c for c in candidates if c.is_file()], key=lambda p: p.name)
    return files


def _prepare_trace_chunk_plans(
    *,
    tasks_root: Path,
    task_entries: list[Path],
    chunk_size: int,
    trace_jobs_dir: str,
    trace_output_dir: str,
    trace_target_repo: str,
    dry_run: bool,
) -> tuple[list[TraceChunkPlan], Optional[Path]]:
    if chunk_size <= 0:
        return [], None

    total_tasks = len(task_entries)
    if total_tasks <= chunk_size:
        return [], None

    chunk_count = (total_tasks + chunk_size - 1) // chunk_size
    print(
        f"Chunking trace tasks: {total_tasks} tasks into {chunk_count} jobs "
        f"(chunk size: {chunk_size})"
    )

    trace_jobs_base = Path(trace_jobs_dir)
    tasks_chunk_root = trace_jobs_base / "task_chunks"
    chunk_map: dict[str, int] = {}
    chunk_plans: list[TraceChunkPlan] = []

    reuse_chunks = False
    if not dry_run and tasks_chunk_root.exists():
        expected_names = {f"chunk_{i:03d}" for i in range(chunk_count)}
        existing_dirs = {
            child.name
            for child in tasks_chunk_root.iterdir()
            if child.is_dir() and child.name.startswith("chunk_")
        }
        if existing_dirs == expected_names:
            reuse_chunks = True
            print(
                f"[chunking] Reusing existing chunk directories under {tasks_chunk_root}"
            )
        else:
            shutil.rmtree(tasks_chunk_root)
            tasks_chunk_root.mkdir(parents=True, exist_ok=True)
    elif not dry_run:
        tasks_chunk_root.mkdir(parents=True, exist_ok=True)
    else:
        print(f"DRY RUN: Would create chunk root at {tasks_chunk_root}")

    common_files = [
        child
        for child in tasks_root.iterdir()
        if child.is_file() and not child.name.startswith(".")
    ]

    output_base = Path(trace_output_dir)
    if not dry_run:
        output_base.mkdir(parents=True, exist_ok=True)
        trace_jobs_base.mkdir(parents=True, exist_ok=True)

    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, total_tasks)
        chunk_entries = task_entries[start:end]
        chunk_dir = tasks_chunk_root / f"chunk_{chunk_index:03d}"

        if not dry_run:
            if reuse_chunks:
                if not chunk_dir.exists():
                    raise FileNotFoundError(
                        f"Expected chunk directory {chunk_dir} to exist for reuse."
                    )
            else:
                chunk_dir.mkdir(parents=True, exist_ok=True)
                for common_file in common_files:
                    shutil.copy2(common_file, chunk_dir / common_file.name)
        else:
            print(f"DRY RUN: Would prepare chunk directory {chunk_dir}")

        chunk_task_names: list[str] = []
        for entry in chunk_entries:
            chunk_task_names.append(entry.name)
            chunk_map[entry.name] = chunk_index
            if dry_run or reuse_chunks:
                continue
            destination = chunk_dir / entry.name
            if entry.is_dir():
                shutil.copytree(entry, destination)
            else:
                shutil.copy2(entry, destination)

        chunk_output_dir = output_base / f"chunk_{chunk_index:03d}"
        chunk_jobs_dir = trace_jobs_base / f"chunk_{chunk_index:03d}"
        if not dry_run:
            chunk_output_dir.mkdir(parents=True, exist_ok=True)
            chunk_jobs_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(
                "DRY RUN: Would assign output/job dirs "
                f"{chunk_output_dir} and {chunk_jobs_dir}"
            )

        chunk_plans.append(
            TraceChunkPlan(
                index=chunk_index,
                tasks_path=chunk_dir,
                output_dir=chunk_output_dir,
                jobs_dir=chunk_jobs_dir,
                target_repo=_format_chunk_target_repo(trace_target_repo, chunk_index),
                task_names=chunk_task_names,
            )
        )

    map_path = tasks_chunk_root / "task_chunk_map.json"
    if not dry_run:
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(chunk_map, f, indent=2, sort_keys=True)
        print(f"Wrote task chunk map: {map_path}")
    else:
        print(
            "DRY RUN: Would write task chunk map to "
            f"{map_path} with entries: {json.dumps(chunk_map, indent=2, sort_keys=True)}"
        )

    return chunk_plans, map_path


__all__ = [
    "DATAGEN_CONFIG_DIR",
    "HARBOR_CONFIG_DIR",
    "DEFAULT_RAY_CGRAPH_TIMEOUT",
    "DEFAULT_RAY_CGRAPH_MAX_INFLIGHT",
    "default_vllm_endpoint_path",
    "_maybe_set_ray_cgraph_env",
    "_normalize_cli_args",
    "_prepare_datagen_configuration",
    "_snapshot_datagen_config",
    "_build_vllm_env_vars",
    "resolve_datagen_config_path",
    "resolve_harbor_config_path",
    "TraceChunkPlan",
    "_format_chunk_target_repo",
    "_discover_task_entries",
    "_prepare_trace_chunk_plans",
]
