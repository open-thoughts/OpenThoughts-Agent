"""Shared utilities for datagen-oriented HPC launches."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Mapping, Union

from omegaconf import OmegaConf

from data.generation import BaseDataGenerator
from data.generation.utils import load_datagen_config, resolve_engine_runtime
from harbor.models.environment_type import EnvironmentType
from hpc.launch_utils import (
    _inject_env_block,
    _merge_dependencies,
    launch_sbatch,
    update_exp_args,
)
from scripts.harbor.job_config_utils import (
    dump_job_config,
    load_job_config,
    overwrite_agent_fields,
    set_job_metadata,
    set_local_dataset,
    update_agent_kwargs,
)

DIRENV = os.path.dirname(__file__)
DATAGEN_CONFIG_DIR = os.path.join(DIRENV, "datagen_yaml")
HARBOR_CONFIG_DIR = os.path.join(DIRENV, "harbor_yaml")
DEFAULT_RAY_CGRAPH_TIMEOUT = os.environ.get("RAY_CGRAPH_TIMEOUT_DEFAULT", "86500")
DEFAULT_RAY_CGRAPH_MAX_INFLIGHT = os.environ.get("RAY_CGRAPH_MAX_INFLIGHT_DEFAULT", "")
HARBOR_MODEL_PLACEHOLDER = "placeholder/override-at-runtime"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _is_local_mode(hpc) -> bool:
    return bool(getattr(hpc, "local_mode", False))


def _run_local_script(script_path: str) -> str:
    print(f"Running locally: bash {script_path}")
    result = subprocess.run(["bash", script_path], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Local execution failed (exit {result.returncode}) for {script_path}")
    return f"local_{Path(script_path).stem}"


def _submit_script(script_path: str, *, dependency: str | None = None, array: str | None = None, hpc=None) -> str:
    if _is_local_mode(hpc):
        if dependency:
            print(f"Warning: ignoring job dependency '{dependency}' for local execution.")
        if array:
            raise RuntimeError("Job arrays are not supported for local execution.")
        return _run_local_script(script_path)
    return launch_sbatch(script_path, dependency=dependency, array=array)


def derive_datagen_job_name(cli_args: Mapping[str, Any]) -> str:
    """Construct a fallback job name for datagen/trace launches."""

    def _sanitize_component(value: str) -> str:
        value = value.strip().rstrip("/")
        if "/" in value:
            value = value.split("/")[-1]
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_") or "repo"

    parts: list[str] = ["datagen"]
    engine = cli_args.get("datagen_engine") or cli_args.get("trace_engine") or "engine"
    parts.append(str(engine or "engine"))

    repo_candidate = cli_args.get("datagen_target_repo") or cli_args.get("trace_target_repo")
    model_candidate = cli_args.get("datagen_model") or cli_args.get("trace_model")
    if model_candidate:
        parts.append(_sanitize_component(str(model_candidate)))
    elif repo_candidate:
        parts.append(_sanitize_component(str(repo_candidate)))

    job_name = "_".join(filter(None, parts))
    return job_name or "datagen_job"


def _detect_gpu_required(datagen_script: str) -> bool:
    """Best-effort detection of GPU requirement for a datagen script."""

    try:
        script_path = os.path.abspath(datagen_script)
        if not os.path.exists(script_path):
            return False

        spec = importlib.util.spec_from_file_location("datagen_module", script_path)
        if spec is None or spec.loader is None:
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]

        generator_cls = None
        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseDataGenerator)
                and obj is not BaseDataGenerator
            ):
                generator_cls = obj
                break

        if not generator_cls:
            return False

        generator = generator_cls()
        run_fn = getattr(generator, "run_task_generation", None)
        return bool(getattr(run_fn, "_gpu_required", False))
    except Exception:
        return False


def _validate_sbatch_templates(hpc_obj) -> None:
    if getattr(hpc_obj, "local_mode", False):
        print(f"Local execution detected for {hpc_obj.name}; skipping sbatch template validation.")
        return
    req_path = Path(__file__).parent / "sbatch_data_requirements.json"
    if not req_path.exists():
        return

    data = json.loads(req_path.read_text())
    entries = data.get(hpc_obj.name.lower())
    if not entries:
        raise ValueError(
            f"No sbatch templates registered for cluster '{hpc_obj.name}'. "
            "Please add entries to hpc/sbatch_data_requirements.json."
        )

    missing = [entry["path"] for entry in entries if not Path(entry["path"]).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing sbatch templates for datagen: " + ", ".join(missing)
        )


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


def _cleanup_stale_vllm_endpoint(exp_args: Mapping[str, Any]) -> None:
    """Remove a leftover vllm_endpoint.json in the experiments directory."""

    experiments_dir = exp_args.get("experiments_dir")
    if not experiments_dir:
        return

    try:
        base_dir = Path(experiments_dir).expanduser()
    except Exception:
        return

    endpoint_path = base_dir / "vllm_endpoint.json"
    if endpoint_path.exists():
        try:
            endpoint_path.unlink()
            print(f"Removed stale vLLM endpoint file: {endpoint_path}")
        except OSError as exc:
            print(f"Warning: failed to remove stale vLLM endpoint file {endpoint_path}: {exc}")


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


def _build_vllm_env_vars(
    exp_args: dict,
    *,
    include_pinggy: bool = False,
) -> Tuple[Dict[str, str], dict]:
    """Return environment variables used to configure vLLM processes."""

    env: Dict[str, str] = {}
    cfg = exp_args.get("_datagen_vllm_server_config")
    if not cfg:
        return env, exp_args

    env["VLLM_MODEL_PATH"] = cfg.model_path
    env["VLLM_NUM_REPLICAS"] = str(cfg.num_replicas or 1)
    env["VLLM_TENSOR_PARALLEL_SIZE"] = str(cfg.tensor_parallel_size or 1)
    env["VLLM_PIPELINE_PARALLEL_SIZE"] = str(cfg.pipeline_parallel_size or 1)
    env["VLLM_DATA_PARALLEL_SIZE"] = str(getattr(cfg, "data_parallel_size", None) or 1)

    if cfg.hf_overrides:
        env["VLLM_HF_OVERRIDES"] = cfg.hf_overrides
    if cfg.use_deep_gemm:
        env["VLLM_USE_DEEP_GEMM"] = "1"
    if cfg.max_num_seqs is not None:
        env["VLLM_MAX_NUM_SEQS"] = str(cfg.max_num_seqs)
    if cfg.gpu_memory_utilization is not None:
        env["VLLM_GPU_MEMORY_UTILIZATION"] = str(cfg.gpu_memory_utilization)
    if getattr(cfg, "cpu_offload_gb", None) is not None:
        env["VLLM_CPU_OFFLOAD_GB"] = str(cfg.cpu_offload_gb)
    if getattr(cfg, "kv_offloading_size", None) is not None:
        env["VLLM_KV_OFFLOADING_SIZE"] = str(cfg.kv_offloading_size)
    if getattr(cfg, "kv_offloading_backend", None):
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
    if getattr(cfg, "logging_level", None) is not None:
        env["VLLM_LOGGING_LEVEL"] = str(cfg.logging_level)

    if include_pinggy:
        explicit_cli_keys = set(exp_args.get("_explicit_cli_keys", []) or [])
        pinggy_fields = (
            ("VLLM_PINGGY_PERSISTENT_URL", "pinggy_persistent_url", "PINGGY_PERSISTENT_URL"),
            ("VLLM_PINGGY_SSH_COMMAND", "pinggy_ssh_command", "PINGGY_SSH_COMMAND"),
            ("VLLM_PINGGY_DEBUGGER_URL", "pinggy_debugger_url", "PINGGY_DEBUGGER_URL"),
        )
        for env_key, arg_key, fallback_env in pinggy_fields:
            candidate = exp_args.get(arg_key)
            explicit = arg_key in explicit_cli_keys
            if isinstance(candidate, str):
                candidate = candidate.strip()
            fallback_allowed = not explicit
            if candidate in (None, "", "None") and fallback_allowed:
                fallback = os.environ.get(fallback_env)
                if isinstance(fallback, str):
                    fallback = fallback.strip()
                candidate = fallback
            if candidate in (None, "", "None"):
                continue
            candidate_str = str(candidate)
            env[env_key] = candidate_str
            if exp_args.get(arg_key) != candidate_str:
                exp_args[arg_key] = candidate_str

    max_output_tokens = exp_args.get("datagen_max_tokens")
    if max_output_tokens not in (None, "", "None"):
        env["VLLM_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)

    endpoint_path = exp_args.get("vllm_endpoint_json_path")
    if not endpoint_path and getattr(cfg, "endpoint_json_path", None):
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


def launch_datagen_job(exp_args: dict, hpc) -> None:
    """Handle datagen/trace launch orchestration."""

    print("\n=== DATA GENERATION MODE ===")
    _cleanup_stale_vllm_endpoint(exp_args)

    hpc_name = str(getattr(hpc, "name", "")).lower()
    if hpc_name == "nyutorch":
        warning = (
            "Datagen jobs are not supported on the NYU Torch cluster. "
            "Use the dedicated sbatch helper OpenThoughts-Agent/hpc/scripts/glm46-tracegen-torch.sbatch instead."
        )
        print(f"WARNING: {warning}")
        raise RuntimeError(warning)

    task_enabled = bool(exp_args.get("enable_task_gen", True))
    trace_enabled = bool(exp_args.get("enable_trace_gen", False))

    # Keep datagen/trace engine/backend selections in sync when only one is provided
    datagen_engine_value = exp_args.get("datagen_engine")
    trace_engine_value = exp_args.get("trace_engine")
    if datagen_engine_value is None and trace_engine_value is not None:
        exp_args = update_exp_args(exp_args, {"datagen_engine": trace_engine_value})
        datagen_engine_value = trace_engine_value
    elif trace_engine_value is None and datagen_engine_value is not None:
        exp_args = update_exp_args(exp_args, {"trace_engine": datagen_engine_value})
        trace_engine_value = datagen_engine_value

    datagen_backend_value = exp_args.get("datagen_backend")
    trace_backend_value = exp_args.get("trace_backend")
    if datagen_backend_value is None and trace_backend_value is not None:
        exp_args = update_exp_args(exp_args, {"datagen_backend": trace_backend_value})
        datagen_backend_value = trace_backend_value
    elif trace_backend_value is None and datagen_backend_value is not None:
        exp_args = update_exp_args(exp_args, {"trace_backend": datagen_backend_value})
        trace_backend_value = datagen_backend_value

    if not task_enabled and not trace_enabled:
        raise ValueError("Enable at least one of task or trace generation")

    if task_enabled and not exp_args.get("datagen_script"):
        raise ValueError("--datagen-script is required for task generation")

    datagen_extra_args_value = exp_args.get("datagen_extra_args")
    no_upload_requested = False
    if isinstance(datagen_extra_args_value, str):
        lowered = datagen_extra_args_value.replace("_", "-").lower()
        no_upload_requested = "--no-upload" in lowered
    elif isinstance(datagen_extra_args_value, (list, tuple)):
        normalized = [
            str(item).replace("_", "-").lower() for item in datagen_extra_args_value
        ]
        no_upload_requested = any("--no-upload" in token for token in normalized)

    if task_enabled and not exp_args.get("datagen_target_repo") and not no_upload_requested:
        raise ValueError("--datagen-target-repo is required for task generation (omit only when --no-upload is set)")

    task_job_id = None
    vllm_job_id = None

    tasks_output_dir = exp_args.get("datagen_output_dir")

    if task_enabled:
        engine = str(exp_args.get("datagen_engine") or "openai").lower()
        backend = str(exp_args.get("datagen_backend") or "vllm").lower()
        vllm_cfg = exp_args.get("_datagen_vllm_server_config")
        if vllm_cfg and engine == "vllm_local":
            if backend == "vllm":
                print("Mode: Local VLLM inference for task generation (standalone server)")
                vllm_job_id = launch_vllm_server(exp_args, hpc)
                if vllm_job_id and vllm_job_id != "dry_run_vllm_job_id":
                    exp_args = update_exp_args(exp_args, {"vllm_job_id": vllm_job_id})
            elif backend == "ray":
                print("Mode: Ray-backed local VLLM inference for task generation")
                _, exp_args = _build_vllm_env_vars(exp_args, include_pinggy=False)
            else:
                raise ValueError(f"Unsupported datagen backend: {backend}")
        else:
            print(f"Mode: {engine} task generation")

        task_job_id = launch_task_job(exp_args, hpc, vllm_job_id)
        print("\n=== Task Generation Submitted ===")
        print(f"Task Job ID: {task_job_id}")
        if vllm_job_id:
            print(f"VLLM Server Job ID: {vllm_job_id}")
        tasks_output_dir = exp_args.get("datagen_output_dir")
        if tasks_output_dir:
            print(f"Task output directory: {tasks_output_dir}")

    trace_job_id = None
    trace_requires_vllm_server = False

    if trace_enabled:
        trace_script = exp_args.get("trace_script") or exp_args.get("datagen_script")
        if not trace_script:
            trace_script = "data/nemo_prism_math/generate_abstract.py"
            exp_args = update_exp_args(exp_args, {"trace_script": trace_script})
            print(f"No trace script provided; defaulting to {trace_script}")
        if not trace_script:
            raise ValueError("Trace generation requires --trace-script (or reuse --datagen-script)")

        trace_input_path = exp_args.get("trace_input_path")
        if not trace_input_path:
            trace_input_path = tasks_output_dir
        if trace_input_path:
            exp_args = update_exp_args(exp_args, {"trace_input_path": trace_input_path})
        if not trace_input_path:
            raise ValueError("Trace generation requires --trace-input-path when task generation is disabled")

        if not task_enabled and not os.path.exists(trace_input_path):
            raise FileNotFoundError(f"Trace input path not found: {trace_input_path}")

        if exp_args.get("trace_use_gpu") and getattr(hpc, "gpus_per_node", 0) == 0:
            raise ValueError("trace_use_gpu requested but HPC configuration has no GPUs available")

        trace_backend = str(exp_args.get("trace_backend") or exp_args.get("datagen_backend") or "vllm").lower()
        trace_engine = str(exp_args.get("trace_engine") or exp_args.get("datagen_engine") or "").lower()

        if trace_backend == "vllm":
            warning = (
                "trace_backend=vllm is currently unsupported on all clusters. "
                "Please switch to a Ray-backed trace backend or disable trace generation."
            )
            print(f"WARNING: {warning}")
            raise RuntimeError(warning)

        trace_requires_vllm_server = trace_engine == "vllm_local" and trace_backend == "vllm"

        vllm_cfg = exp_args.get("_datagen_vllm_server_config")
        if trace_requires_vllm_server and not vllm_job_id and vllm_cfg:
            print("Trace stage requires a standalone VLLM server – launching now")
            vllm_job_id = launch_vllm_server(exp_args, hpc)
            if vllm_job_id and vllm_job_id != "dry_run_vllm_job_id":
                exp_args = update_exp_args(exp_args, {"vllm_job_id": vllm_job_id})

        dependency = None
        if task_enabled and task_job_id and task_job_id != "dry_run_task_job_id":
            dependency = f"afterany:{task_job_id}"

        trace_job_id = launch_trace_job(exp_args, hpc, trace_input_path, dependency=dependency)
        print("\n=== Trace Generation Submitted ===")
        if isinstance(trace_job_id, list):
            print(f"Trace Job IDs: {', '.join(trace_job_id)}")
        else:
            print(f"Trace Job ID: {trace_job_id}")
        print(f"Trace input path: {trace_input_path}")

    return


def launch_vllm_server(exp_args: dict, hpc) -> str:
    """
    Launch VLLM server job with Pinggy tunnel.

    Returns:
        Job ID of the VLLM server job
    """
    print("\n=== Launching VLLM Server ===")

    # Prepare environment variables for VLLM server
    vllm_cfg = exp_args.get("_datagen_vllm_server_config")
    if not vllm_cfg:
        raise ValueError("vLLM server launch requested but no vllm_server configuration provided.")

    vllm_env_vars, exp_args = _build_vllm_env_vars(exp_args, include_pinggy=True)

    model_path = vllm_cfg.model_path

    num_replicas = vllm_cfg.num_replicas or 1
    tensor_parallel = vllm_cfg.tensor_parallel_size or 1
    pipeline_parallel = vllm_cfg.pipeline_parallel_size or 1
    server_time_limit = vllm_cfg.time_limit or "48:00:00"

    # Prepare sbatch script path
    sbatch_dir = os.path.join(os.path.dirname(__file__), "sbatch_data")
    vllm_sbatch_template = os.path.join(sbatch_dir, f"{hpc.name}_vllm_server.sbatch")

    if not os.path.exists(vllm_sbatch_template):
        raise FileNotFoundError(
            f"No vLLM server sbatch template found for cluster '{hpc.name}'. "
            f"Expected at {vllm_sbatch_template}. Please add {hpc.name}_vllm_server.sbatch under hpc/sbatch_data."
        )

    # Create output sbatch script
    vllm_job_name = f"{exp_args['job_name']}_vllm_server"
    vllm_sbatch_output = os.path.join(
        exp_args["experiments_dir"],
        "sbatch",
        f"{vllm_job_name}.sbatch"
    )
    os.makedirs(os.path.dirname(vllm_sbatch_output), exist_ok=True)
    os.makedirs(os.path.join(exp_args["experiments_dir"], "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_args["experiments_dir"], "logs"), exist_ok=True)

    # Read template and substitute
    with open(vllm_sbatch_template) as f:
        sbatch_content = f.read()

    # Calculate GPUs needed (must account for TP * PP per replica)
    num_gpus = int(num_replicas or 1) * int(tensor_parallel or 1) * int(pipeline_parallel or 1)

    num_nodes = exp_args.get("num_nodes") or getattr(hpc, "num_nodes", 1) or 1
    gpus_per_node = exp_args.get("gpus_per_node") or getattr(hpc, "gpus_per_node", num_gpus)

    cpus_per_node_effective = exp_args.get("cpus_per_node")
    if cpus_per_node_effective in (None, "", "None"):
        cpus_per_node_effective = getattr(hpc, "cpus_per_node", 1)
    cpus_per_node_effective = int(cpus_per_node_effective)

    substitutions = {
        "{partition}": hpc.partition,
        "{time_limit}": server_time_limit,
        "{num_nodes}": str(num_nodes),
        "{cpus_per_node}": str(cpus_per_node_effective),
        "{num_gpus}": str(num_gpus),
        "{gpus_per_node}": str(gpus_per_node),
        "{account}": hpc.account,
        "{experiments_dir}": exp_args["experiments_dir"],
        "{job_name}": vllm_job_name,
    }

    for key, value in substitutions.items():
        sbatch_content = sbatch_content.replace(key, value)

    if not gpus_per_node:
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node {gpus_per_node}\n", "")
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node 0\n", "")
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node=0\n", "")

    sbatch_content = _inject_env_block(sbatch_content, vllm_env_vars)

    # Write output sbatch
    with open(vllm_sbatch_output, 'w') as f:
        f.write(sbatch_content)

    print(f"VLLM server sbatch: {vllm_sbatch_output}")
    print(f"Endpoint JSON will be saved to: {exp_args['vllm_endpoint_json_path']}")

    # Submit job
    if not exp_args.get("dry_run"):
        job_id = _submit_script(vllm_sbatch_output, hpc=hpc)
        print(f"✓ VLLM server job submitted: {job_id}")
        return job_id
    else:
        print("DRY RUN: Would submit VLLM server job")
        return "dry_run_vllm_job_id"


def launch_task_job(exp_args: dict, hpc, vllm_job_id: str = None) -> str:
    """
    Launch data generation job.

    Args:
        exp_args: Experiment arguments
        hpc: HPC configuration
        vllm_job_id: Optional VLLM server job ID for dependency

    Returns:
        Job ID of the datagen job
    """
    print("\n=== Launching Task Generation Job ===")

    runtime = exp_args.get("_datagen_engine_runtime")
    if runtime is None:
        runtime = _prepare_datagen_configuration(exp_args)

    # Prepare environment variables for datagen
    datagen_engine_value = exp_args.get("datagen_engine", runtime.type if runtime else "openai")
    engine = str(datagen_engine_value or "openai").lower()
    datagen_env_vars = {
        "DATAGEN_SCRIPT": exp_args["datagen_script"],
        "DATAGEN_HEALTHCHECK_INTERVAL": str(exp_args.get("datagen_healthcheck_interval", 300)),
        "DATAGEN_ENGINE": engine,
        "DATAGEN_STAGE": "tasks",
        "DATAGEN_BACKEND": str(exp_args.get("datagen_backend", "vllm")),
        "PYTHON_EXECUTABLE": sys.executable,
    }
    task_type_value = exp_args.get("task_type")
    if task_type_value:
        datagen_env_vars["DATAGEN_TASK_TYPE"] = str(task_type_value)
        print(f"Task generation task type: {task_type_value}")
    if exp_args.get("datagen_target_repo"):
        datagen_env_vars["DATAGEN_TARGET_REPO"] = exp_args["datagen_target_repo"]

    backend = str(datagen_env_vars["DATAGEN_BACKEND"]).lower()
    if backend != datagen_env_vars["DATAGEN_BACKEND"]:
        datagen_env_vars["DATAGEN_BACKEND"] = backend
    is_ray_backend = backend == "ray"

    is_cpu_only = engine == "none" and backend in ("vllm", "none")
    if is_cpu_only:
        backend = "none"
        datagen_env_vars["DATAGEN_BACKEND"] = backend
        exp_args = update_exp_args(exp_args, {"datagen_backend": backend})
    vllm_env_vars, exp_args = _build_vllm_env_vars(exp_args, include_pinggy=False)
    datagen_env_vars.update(vllm_env_vars)

    if is_ray_backend:
        if exp_args.get("ray_cgraph_submit_timeout"):
            datagen_env_vars["RAY_CGRAPH_submit_timeout"] = str(exp_args["ray_cgraph_submit_timeout"])
        if exp_args.get("ray_cgraph_get_timeout"):
            datagen_env_vars["RAY_CGRAPH_get_timeout"] = str(exp_args["ray_cgraph_get_timeout"])
        if exp_args.get("ray_cgraph_max_inflight_executions"):
            datagen_env_vars["RAY_CGRAPH_max_inflight_executions"] = str(
                exp_args["ray_cgraph_max_inflight_executions"]
            )
        _maybe_set_ray_cgraph_env(datagen_env_vars)

    script_gpu_required = False
    script_path = exp_args.get("datagen_script")
    if script_path:
        script_gpu_required = _detect_gpu_required(script_path)

    requested_gpus = 0
    try:
        requested_gpus = int(exp_args.get("gpus_per_node") or 0)
    except (TypeError, ValueError):
        requested_gpus = 0

    if script_gpu_required and requested_gpus == 0:
        requested_gpus = int(getattr(hpc, "gpus_per_node", 0) or 0)

    datagen_gpu_required = script_gpu_required or requested_gpus > 0

    datagen_env_vars["DATAGEN_GPU_REQUIRED"] = "1" if requested_gpus > 0 else "0"
    gpus_per_node = exp_args.get("gpus_per_node") or getattr(hpc, "gpus_per_node", requested_gpus)
    datagen_env_vars["DATAGEN_GPUS_PER_NODE"] = str(gpus_per_node)
    datagen_env_vars["DATAGEN_RAY_PORT"] = str(exp_args.get("datagen_ray_port", 6379))
    datagen_env_vars["DATAGEN_API_PORT"] = str(exp_args.get("datagen_api_port", 8000))
    tensor_parallel_size = None
    pipeline_parallel_size = None
    backend_cfg = exp_args.get("_datagen_backend_config")
    vllm_cfg = exp_args.get("_datagen_vllm_server_config")
    data_parallel_size = None
    if vllm_cfg:
        tensor_parallel_size = getattr(vllm_cfg, "tensor_parallel_size", None)
        pipeline_parallel_size = getattr(vllm_cfg, "pipeline_parallel_size", None)
        data_parallel_size = getattr(vllm_cfg, "data_parallel_size", None)
    if tensor_parallel_size is None and backend_cfg:
        tensor_parallel_size = getattr(backend_cfg, "tensor_parallel_size", None)
    if pipeline_parallel_size is None and backend_cfg:
        pipeline_parallel_size = getattr(backend_cfg, "pipeline_parallel_size", None)
    if data_parallel_size is None and backend_cfg:
        data_parallel_size = getattr(backend_cfg, "data_parallel_size", None)
    if tensor_parallel_size is None:
        tensor_parallel_size = 1
    if pipeline_parallel_size is None:
        pipeline_parallel_size = 1
    if data_parallel_size is None:
        data_parallel_size = 1
    datagen_env_vars["DATAGEN_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
    datagen_env_vars["DATAGEN_PIPELINE_PARALLEL_SIZE"] = str(pipeline_parallel_size)
    datagen_env_vars["DATAGEN_DATA_PARALLEL_SIZE"] = str(data_parallel_size)
    datagen_env_vars["DATAGEN_NUM_NODES"] = str(exp_args.get("num_nodes") or getattr(hpc, "num_nodes", 1) or 1)
    datagen_env_vars["PROJECT_ROOT"] = str(PROJECT_ROOT)
    gcs_credentials_path = os.environ.get("GCS_CREDENTIALS_PATH")

    is_gemini_engine = str(engine).lower() in {"gemini_openai", "google_gemini", "gemini"}
    if is_gemini_engine:
        default_gcs_path = os.path.join(PROJECT_ROOT, "..", ".gcs_datacomp.json")
        candidate_path = Path(default_gcs_path).expanduser().resolve()
        if not candidate_path.exists():
            print(
                "Gemini datagen requires GCS credentials at ../.gcs_datacomp.json "
                "but the file was not found. Aborting."
            )
            raise SystemExit(1)
        gcs_credentials_path = str(candidate_path)
        os.environ["GCS_CREDENTIALS_PATH"] = gcs_credentials_path

    if gcs_credentials_path:
        datagen_env_vars["GCS_CREDENTIALS_PATH"] = gcs_credentials_path
    if exp_args.get("datagen_engine") != engine:
        exp_args = update_exp_args(exp_args, {"datagen_engine": engine})

    if exp_args.get("datagen_input_dir"):
        datagen_env_vars["DATAGEN_INPUT_DIR"] = exp_args["datagen_input_dir"]

    output_dir = exp_args.get("datagen_output_dir")
    if not output_dir:
        output_dir = os.path.join(exp_args["experiments_dir"], "outputs", "tasks")
        exp_args = update_exp_args(exp_args, {"datagen_output_dir": output_dir})
    datagen_env_vars["DATAGEN_OUTPUT_DIR"] = output_dir

    datagen_extra_args = exp_args.get("datagen_extra_args") or ""
    extra_tokens: list[str]
    if datagen_extra_args:
        try:
            extra_tokens = shlex.split(datagen_extra_args)
        except ValueError:
            extra_tokens = datagen_extra_args.split()
    else:
        extra_tokens = []

    if exp_args.get("disable_verification"):
        disable_flags = {"--disable-verification", "--disable_verification"}
        if not any(token in disable_flags for token in extra_tokens):
            extra_tokens.append("--disable-verification")

    if extra_tokens:
        datagen_extra_args = " ".join(extra_tokens)
        datagen_env_vars["DATAGEN_EXTRA_ARGS"] = datagen_extra_args
        exp_args = update_exp_args(exp_args, {"datagen_extra_args": datagen_extra_args})

    datagen_env_vars["VLLM_JOB_ID"] = (str(vllm_job_id) if vllm_job_id and vllm_job_id != "dry_run_vllm_job_id" else "")

    sandbox_overrides = {
        "SANDBOX_CPU": exp_args.get("sandbox_cpu"),
        "SANDBOX_MEMORY_GB": exp_args.get("sandbox_memory_gb"),
        "SANDBOX_DISK_GB": exp_args.get("sandbox_disk_gb"),
        "SANDBOX_GPU": exp_args.get("sandbox_gpu"),
    }
    for env_key, value in sandbox_overrides.items():
        if value is not None:
            datagen_env_vars[env_key] = str(value)

    # Set endpoint JSON path
    endpoint_path = exp_args.get("vllm_endpoint_json_path")
    requires_endpoint = engine == "vllm_local" and (script_gpu_required or is_ray_backend)
    if requires_endpoint and not endpoint_path:
        endpoint_path = os.path.join(exp_args["experiments_dir"], "vllm_endpoint.json")
        exp_args["vllm_endpoint_json_path"] = endpoint_path

    datagen_env_vars["DATAGEN_ENDPOINT_JSON"] = endpoint_path or ""
    datagen_env_vars["DATAGEN_REQUIRE_ENDPOINT"] = "1" if requires_endpoint else "0"
    datagen_wait_flag = exp_args.get("datagen_wait_for_endpoint")
    if datagen_wait_flag is None:
        datagen_wait_flag = "1" if requires_endpoint and backend != "ray" else "0"
    else:
        datagen_wait_flag = "1" if str(datagen_wait_flag).lower() in ("1", "true", "yes") else "0"
    datagen_env_vars["DATAGEN_WAIT_FOR_ENDPOINT"] = datagen_wait_flag

    config_path = _snapshot_datagen_config(exp_args)
    datagen_env_vars["DATAGEN_CONFIG_PATH"] = config_path

    # Prepare sbatch script
    sbatch_dir = os.path.join(os.path.dirname(__file__), "sbatch_data")
    hpc_name = getattr(hpc, "name", "") or ""
    uses_vllm_cluster = is_ray_backend or requires_endpoint
    datagen_sbatch_template = os.path.join(sbatch_dir, f"{hpc_name}_datagen.sbatch")

    if hpc_name.lower() == "vista":
        datagen_sbatch_template = os.path.join(
            sbatch_dir,
            "vista_datagen.sbatch" if uses_vllm_cluster else "vista_datagen_cpu.sbatch",
        )

    if not os.path.exists(datagen_sbatch_template):
        # Fallback to vista template
        datagen_sbatch_template = os.path.join(sbatch_dir, "vista_datagen.sbatch")

    # Create output sbatch script
    datagen_job_name = f"{exp_args['job_name']}_datagen"
    datagen_sbatch_output = os.path.join(
        exp_args["experiments_dir"],
        "sbatch",
        f"{datagen_job_name}.sbatch"
    )

    # Read template and substitute
    with open(datagen_sbatch_template) as f:
        sbatch_content = f.read()

    num_nodes = exp_args.get("num_nodes") or getattr(hpc, "num_nodes", 1) or 1
    if is_cpu_only:
        num_nodes = 1
    if hpc_name.lower() == "vista" and not uses_vllm_cluster and not exp_args.get("num_nodes"):
        num_nodes = 1
    gpus_per_node_directive = gpus_per_node
    if getattr(hpc, "name", "").lower() == "vista":
        gpus_per_node_directive = None

    cpus_per_node_effective = exp_args.get("cpus_per_node")
    if cpus_per_node_effective in (None, "", "None"):
        cpus_per_node_effective = getattr(hpc, "cpus_per_node", 1)
    cpus_per_node_effective = int(cpus_per_node_effective)

    qos_value = str(exp_args.get("qos") or getattr(hpc, "qos", "") or "").strip()

    substitutions = {
        "{partition}": hpc.partition,
        "{time_limit}": exp_args["time_limit"],
        "{num_nodes}": str(num_nodes),
        "{cpus_per_node}": str(cpus_per_node_effective),
        "{account}": hpc.account,
        "{experiments_dir}": exp_args["experiments_dir"],
        "{job_name}": datagen_job_name,
        "{num_gpus}": str(requested_gpus),
        "{gpus_per_node}": "" if gpus_per_node_directive is None else str(gpus_per_node_directive),
        "{qos}": qos_value,
    }

    for key, value in substitutions.items():
        sbatch_content = sbatch_content.replace(key, value)

    if datagen_env_vars.get("DATAGEN_NUM_NODES") != str(num_nodes):
        datagen_env_vars["DATAGEN_NUM_NODES"] = str(num_nodes)

    if requested_gpus == 0:
        sbatch_content = sbatch_content.replace("#SBATCH --gres=gpu:0\n", "")
    if not gpus_per_node_directive:
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node {gpus_per_node}\n", "")
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node 0\n", "")
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node=0\n", "")
    if not qos_value:
        sbatch_content = re.sub(r"#SBATCH\s+-q[^\n]*\n", "", sbatch_content)

    sbatch_content = _inject_env_block(sbatch_content, datagen_env_vars)

    # Ensure output directory exists and write output sbatch
    os.makedirs(os.path.dirname(datagen_sbatch_output), exist_ok=True)
    os.makedirs(os.path.join(exp_args["experiments_dir"], "logs"), exist_ok=True)
    with open(datagen_sbatch_output, 'w') as f:
        f.write(sbatch_content)

    print(f"Task generation sbatch: {datagen_sbatch_output}")
    print(f"Generation script: {exp_args['datagen_script']}")
    target_repo_display = exp_args.get("datagen_target_repo") or "<none>"
    print(f"Target repo: {target_repo_display}")
    print(f"Datagen model: {exp_args.get('datagen_model')}")

    # Submit job with dependency if VLLM job exists
    if not exp_args.get("dry_run"):
        dependency = None
        if vllm_job_id and vllm_job_id != "dry_run_vllm_job_id":
            # Wait until the VLLM job begins running so the endpoint is ready
            dependency = f"after:{vllm_job_id}"
            print(f"Setting dependency on VLLM job start: {vllm_job_id}")

        job_id = _submit_script(datagen_sbatch_output, dependency=dependency, hpc=hpc)
        print(f"✓ Task generation job submitted: {job_id}")
        return job_id

    print("DRY RUN: Would submit task generation job")
    if vllm_job_id:
        print(f"  with dependency on VLLM job: {vllm_job_id}")
    return "dry_run_task_job_id"


def launch_trace_job(
    exp_args: dict,
    hpc,
    tasks_input_path: str,
    dependency: Optional[str] = None,
) -> Union[str, list[str]]:
    """Launch trace generation job."""

    print("\n=== Launching Trace Generation Job ===")

    trace_script = exp_args.get("trace_script") or exp_args.get("datagen_script")
    if not trace_script:
        raise ValueError("Trace generation requires --trace-script or --datagen-script")

    trace_target_repo = exp_args.get("trace_target_repo")
    if not trace_target_repo:
        raise ValueError("--trace-target-repo is required when enabling trace generation")

    trace_output_dir = exp_args.get("trace_output_dir")
    if not trace_output_dir:
        trace_output_dir = os.path.join(exp_args["experiments_dir"], "outputs", "traces")
        exp_args = update_exp_args(exp_args, {"trace_output_dir": trace_output_dir})

    dry_run_flag = bool(exp_args.get("dry_run"))

    chunk_size_raw = exp_args.get("chunk_size")
    chunk_size_value: Optional[int]
    if chunk_size_raw in (None, "", "None"):
        chunk_size_value = None
    elif isinstance(chunk_size_raw, int):
        chunk_size_value = chunk_size_raw
    else:
        try:
            chunk_size_value = int(chunk_size_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("--chunk_size must be an integer value") from exc
        exp_args = update_exp_args(exp_args, {"chunk_size": chunk_size_value})

    tasks_root = Path(tasks_input_path)
    if chunk_size_value and not tasks_root.exists():
        raise ValueError(
            "Chunked trace generation is not yet supported when task generation runs in the same job. "
            f"Tasks directory not found at {tasks_root}. Generate/download tasks first (without --chunk_size) "
            "and re-run traces with chunking enabled."
        )

    task_entries = _discover_task_entries(
        tasks_root,
        create_if_missing=bool(dependency) and not dry_run_flag,
    )
    total_tasks = len(task_entries)
    if total_tasks == 0:
        # Do not fail early; the downstream trace runner waits for tasks to appear.
        # This allows submitting a trace job even when tasks are not yet materialized.
        if dependency:
            print(
                "Trace tasks directory is empty; dependency on task job will ensure "
                "tasks populate before traces start."
            )
        else:
            print(
                f"No tasks currently found under {tasks_root}; submitting trace job that will wait for tasks."
            )
    else:
        print(f"Discovered {total_tasks} tasks for trace generation at {tasks_root}")

    trace_config_spec = exp_args.get("trace_harbor_config")
    if not trace_config_spec:
        raise ValueError("Trace generation requires --trace-harbor-config.")
    trace_config_path = resolve_harbor_config_path(trace_config_spec)
    base_trace_config = load_job_config(trace_config_path)

    trace_configs_dir = Path(exp_args["experiments_dir"]) / "configs" / "trace"
    trace_configs_dir.mkdir(parents=True, exist_ok=True)

    if "trace_jobs_dir" in exp_args and exp_args["trace_jobs_dir"]:
        trace_jobs_dir_path = Path(str(exp_args["trace_jobs_dir"])).expanduser().resolve()
    else:
        experiments_dir = Path(str(exp_args["experiments_dir"])).expanduser().resolve()
        trace_jobs_dir_path = experiments_dir / "trace_jobs"
        exp_args = update_exp_args(exp_args, {"trace_jobs_dir": str(trace_jobs_dir_path)})
    trace_jobs_dir_path.mkdir(parents=True, exist_ok=True)
    trace_jobs_dir = str(trace_jobs_dir_path)

    chunk_plans: list[TraceChunkPlan] = []
    chunk_map_path: Optional[Path] = None
    if chunk_size_value:
        chunk_plans, chunk_map_path = _prepare_trace_chunk_plans(
            tasks_root=tasks_root,
            task_entries=task_entries,
            chunk_size=chunk_size_value,
            trace_jobs_dir=trace_jobs_dir,
            trace_output_dir=trace_output_dir,
            trace_target_repo=trace_target_repo,
            dry_run=dry_run_flag,
        )
        if chunk_plans:
            print(f"Prepared chunk plan for {len(chunk_plans)} trace jobs.")
        else:
            print(
                f"Chunk size {chunk_size_value} does not require splitting "
                f"(total tasks: {total_tasks})."
            )

    cfg_agent = base_trace_config.agents[0] if base_trace_config.agents else None

    trace_agent_timeout_override = exp_args.get("trace_agent_timeout_sec")
    base_agent_kwargs = exp_args.get("_datagen_extra_agent_kwargs") or {}
    if not isinstance(base_agent_kwargs, dict):
        base_agent_kwargs = dict(base_agent_kwargs)
    trace_agent_kwargs: dict[str, Any] = dict(base_agent_kwargs)
    trace_agent_kwargs_raw = exp_args.get("trace_agent_kwargs")
    cli_agent_kwargs: dict[str, Any] | None = None
    if isinstance(trace_agent_kwargs_raw, dict):
        cli_agent_kwargs = dict(trace_agent_kwargs_raw)
    elif trace_agent_kwargs_raw not in (None, "", "None"):
        try:
            parsed_kwargs = json.loads(str(trace_agent_kwargs_raw))
        except json.JSONDecodeError as exc:
            raise ValueError("--trace_agent_kwargs must be valid JSON") from exc
        if not isinstance(parsed_kwargs, dict):
            raise ValueError("--trace_agent_kwargs must be a JSON object")
        cli_agent_kwargs = parsed_kwargs
    if cli_agent_kwargs is not None:
        trace_agent_kwargs.update(cli_agent_kwargs)

    trace_n_concurrent_override = exp_args.get("trace_n_concurrent")
    trace_env_override = exp_args.get("trace_env")
    trace_model_override = exp_args.get("trace_model")
    trace_agent_name_override = exp_args.get("trace_agent_name")

    trace_model = trace_model_override or (
        cfg_agent.model_name if cfg_agent and cfg_agent.model_name else None
    )
    if trace_model and str(trace_model).strip() == HARBOR_MODEL_PLACEHOLDER:
        trace_model = ""
    raw_trace_model = trace_model

    datagen_runtime = exp_args.get("_datagen_engine_runtime")
    runtime_engine_type = ""
    runtime_model_name: str | None = None
    if datagen_runtime is not None:
        runtime_engine_type = str(getattr(datagen_runtime, "type", "") or "").lower()
        runtime_model_name = (
            datagen_runtime.engine_kwargs.get("model")
            or datagen_runtime.engine_kwargs.get("model_name")
        )

    if (
        trace_model_override is None
        and runtime_model_name
        and runtime_engine_type not in ("vllm_local", "none")
    ):
        trace_model = runtime_model_name

    if not trace_model:
        vllm_cfg = exp_args.get("_datagen_vllm_server_config")
        if vllm_cfg:
            trace_model = vllm_cfg.model_path
    trace_model = trace_model or ""

    provider_prefix: str | None = None
    if runtime_engine_type == "openai":
        provider_prefix = "openai"
    elif runtime_engine_type == "anthropic":
        provider_prefix = "anthropic"
    elif runtime_engine_type in {"gemini_openai", "google_gemini"}:
        provider_prefix = "gemini"

    def _normalize_model_for_provider(model_value: str, provider: str | None) -> str:
        if not model_value:
            return ""
        candidate = str(model_value).strip()
        if provider and candidate.lower().startswith(f"{provider.lower()}/"):
            return candidate
        for prefix in (
            "openai/",
            "anthropic/",
            "gemini/",
            "google_gemini/",
        ):
            if candidate.lower().startswith(prefix):
                candidate = candidate[len(prefix):]
                break
        if candidate.lower().startswith("models/"):
            candidate = candidate[len("models/"):]
        candidate = candidate.lstrip("/")
        if provider:
            if not candidate:
                return ""
            return f"{provider}/{candidate}"
        return candidate

    trace_model_dispatch = _normalize_model_for_provider(trace_model, provider_prefix)

    if not trace_model_dispatch and runtime_model_name:
        trace_model_dispatch = _normalize_model_for_provider(runtime_model_name, provider_prefix)

    if not trace_model_dispatch:
        raise ValueError(
            "Unable to determine trace model. Provide --trace_model or specify engine.model in the datagen config."
        )

    if trace_model_dispatch == HARBOR_MODEL_PLACEHOLDER:
        raise ValueError(
            "Trace Harbor config still references placeholder/override-at-runtime. "
            "Ensure the datagen config supplies a concrete model."
        )

    raw_model_display = raw_trace_model or runtime_model_name or trace_model_dispatch
    if trace_model_dispatch != (raw_trace_model or runtime_model_name or ""):
        print(f"Trace model selected: {trace_model_dispatch} (raw: {raw_model_display})")
    else:
        print(f"Trace model selected: {trace_model_dispatch}")

    exp_args = update_exp_args(exp_args, {"trace_model": trace_model_dispatch})

    trace_agent_name = (
        trace_agent_name_override
        or (cfg_agent.name if cfg_agent and cfg_agent.name else None)
        or "terminus-2"
    )

    trace_episodes = exp_args.get("trace_episodes") or "last"
    trace_export_filter = exp_args.get("trace_export_filter") or "none"
    trace_dataset_type = exp_args.get("trace_dataset_type") or "SFT"
    disable_verification_flag = bool(exp_args.get("disable_verification"))
    trace_export_subagents_value = exp_args.get("trace_export_subagents")
    if trace_export_subagents_value in (None, "", "None"):
        trace_export_subagents = True
    elif isinstance(trace_export_subagents_value, str):
        trace_export_subagents = trace_export_subagents_value.strip().lower() not in {"0", "false", "no"}
    else:
        trace_export_subagents = bool(trace_export_subagents_value)

    trace_engine_value = exp_args.get("trace_engine") or exp_args.get("datagen_engine") or ""
    trace_engine = str(trace_engine_value).lower()

    trace_backend_value = exp_args.get("trace_backend") or exp_args.get("datagen_backend") or "vllm"
    trace_backend = str(trace_backend_value).lower()
    cpu_only_trace = trace_backend == "vllm_local"

    if trace_backend != trace_backend_value:
        exp_args = update_exp_args(exp_args, {"trace_backend": trace_backend})

    if trace_engine == "none" and trace_backend in ("vllm", "none"):
        trace_backend = "none"
        exp_args = update_exp_args(exp_args, {"trace_backend": trace_backend})

    requires_endpoint = trace_engine == "vllm_local" and trace_backend in ("vllm", "ray")

    trace_endpoint_json: str = ""
    if requires_endpoint:
        base_endpoint_path = exp_args.get("trace_endpoint_json") or exp_args.get("vllm_endpoint_json_path")
        if base_endpoint_path:
            trace_endpoint_json = base_endpoint_path
        else:
            experiments_dir = exp_args.get("experiments_dir")
            if not experiments_dir:
                raise ValueError("experiments_dir is required to compute trace endpoint path")
            trace_endpoint_json = default_vllm_endpoint_path(
                experiments_dir,
                trace=bool(chunk_plans),
            )
    else:
        trace_endpoint_json = exp_args.get("trace_endpoint_json") or exp_args.get("vllm_endpoint_json_path") or ""

    trace_use_gpu_flag = bool(exp_args.get("trace_use_gpu"))
    if cpu_only_trace and trace_use_gpu_flag:
        print("TRACE backend 'vllm_local' is CPU-only; ignoring --trace_use_gpu request.")
        trace_use_gpu_flag = False
        exp_args = update_exp_args(exp_args, {"trace_use_gpu": False})
    if requires_endpoint and not trace_use_gpu_flag:
        trace_use_gpu_flag = True
        exp_args = update_exp_args(exp_args, {"trace_use_gpu": True})

    num_gpus = hpc.gpus_per_node if trace_use_gpu_flag else 0
    trace_memory_limit_gb = 32 if cpu_only_trace else None

    trace_n_concurrent_effective = (
        trace_n_concurrent_override
        if trace_n_concurrent_override is not None
        else base_trace_config.orchestrator.n_concurrent_trials
    )
    if trace_n_concurrent_effective is None:
        trace_n_concurrent_effective = 8
    print(f"Trace concurrency selected: {trace_n_concurrent_effective}")
    if trace_env_override:
        print(f"Trace environment override: {trace_env_override}")

    task_type_value = exp_args.get("task_type")

    trace_env_vars = {
        "TRACE_SCRIPT": trace_script,
        "TRACE_STAGE": "traces",
        "TRACE_TASKS_PATH": tasks_input_path,
        "TRACE_TARGET_REPO": trace_target_repo,
        "TRACE_MODEL": trace_model_dispatch,
        "TRACE_ENGINE": trace_engine,
        "TRACE_OUTPUT_DIR": trace_output_dir,
        "TRACE_USE_GPU": "1" if trace_use_gpu_flag else "0",
        "TRACE_EPISODES": trace_episodes,
        "TRACE_EXPORT_FILTER": trace_export_filter,
        "TRACE_DATASET_TYPE": trace_dataset_type,
        "TRACE_EXPORT_SUBAGENTS": "1" if trace_export_subagents else "0",
        "TRACE_JOBS_DIR": trace_jobs_dir,
        "TRACE_ENDPOINT_JSON": trace_endpoint_json or "",
        "TRACE_REQUIRE_ENDPOINT": "1" if requires_endpoint else "0",
        "TRACE_WAIT_FOR_ENDPOINT": "1" if requires_endpoint else "0",
        "TRACE_HEALTH_MAX_ATTEMPTS": str(
            exp_args.get("trace_health_max_attempts", getattr(BaseDataGenerator, "HEALTHCHECK_MAX_ATTEMPTS", 20))
        ),
        "TRACE_HEALTH_RETRY_DELAY": str(
            exp_args.get("trace_health_retry_delay", getattr(BaseDataGenerator, "HEALTHCHECK_RETRY_DELAY", 30))
        ),
        "TRACE_DISABLE_VERIFICATION": "1" if disable_verification_flag else "0",
        "TRACE_EVAL_ONLY": "1" if exp_args.get("trace_eval_only") else "0",
        "VLLM_JOB_ID": str(exp_args.get("vllm_job_id", "")),
        "TRACE_NUM_GPUS": str(num_gpus),
        "TRACE_BACKEND": trace_backend,
        "TRACE_AGENT_TIMEOUT_SEC": str(trace_agent_timeout_override or ""),
        "TRACE_VERIFIER_TIMEOUT_SEC": str(exp_args.get("trace_verifier_timeout_sec") or ""),
        "TRACE_TOTAL_TASKS": str(total_tasks),
        "TRACE_CHUNK_SIZE": str(chunk_size_value or ""),
        "TRACE_CHUNK_COUNT": str(len(chunk_plans) if chunk_plans else 1),
        "TRACE_CHUNK_MAP_PATH": str(chunk_map_path) if chunk_map_path else "",
        "TRACE_JOB_INDEX": "0",
        "TRACE_TARGET_REPO_BASE": trace_target_repo,
        "TRACE_HARBOR_CONFIG": "",
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "PYTHON_EXECUTABLE": sys.executable,
    }
    if task_type_value:
        trace_env_vars["TRACE_TASK_TYPE"] = str(task_type_value)
    def _materialize_trace_config(
        dataset_path: Path,
        job_suffix: str,
        jobs_dir: Path | str,
        *,
        agent_name_override: str | None,
        agent_model_override: str | None,
        agent_timeout_override: float | None,
        agent_kwargs_override: dict[str, Any] | None,
        n_concurrent_override: int | None,
        env_override: str | None,
        disable_verification: bool,
    ) -> Path:
        config = set_local_dataset(base_trace_config, Path(dataset_path))
        config = set_job_metadata(
            config,
            job_name=f"{exp_args['job_name']}{job_suffix}",
            jobs_dir=jobs_dir,
        )
        config = overwrite_agent_fields(
            config,
            name=agent_name_override,
            model_name=agent_model_override or None,
            override_timeout_sec=agent_timeout_override,
        )
        if (
            config.agents
            and getattr(config.agents[0], "model_name", None) == HARBOR_MODEL_PLACEHOLDER
        ):
            raise ValueError(
                "Trace Harbor config still references placeholder/override-at-runtime after resolution."
            )
        config = update_agent_kwargs(config, agent_kwargs_override)
        if n_concurrent_override is not None:
            config.orchestrator.n_concurrent_trials = int(n_concurrent_override)
        if env_override:
            env_str = str(env_override).strip()
            try:
                env_enum = EnvironmentType(env_str.lower())
            except ValueError:
                try:
                    env_enum = EnvironmentType[env_str.upper()]
                except KeyError as exc:
                    raise ValueError(f"Invalid trace environment override: {env_override}") from exc
            config.environment.type = env_enum
        if disable_verification:
            config.verifier.disable = True
        output_path = trace_configs_dir / f"{config.job_name}.yaml"
        if not dry_run_flag:
            dump_job_config(config, output_path)
        return output_path

    # Overwrite trace health max_attempts and delay if environment variable is specified
    if "TRACE_HEALTH_MAX_ATTEMPTS" in os.environ:
        trace_env_vars["TRACE_HEALTH_MAX_ATTEMPTS"] = os.environ["TRACE_HEALTH_MAX_ATTEMPTS"]

    if "TRACE_HEALTH_RETRY_DELAY" in os.environ:
        trace_env_vars["TRACE_HEALTH_RETRY_DELAY"] = os.environ["TRACE_HEALTH_RETRY_DELAY"]

    backend = str(trace_env_vars["TRACE_BACKEND"]).lower()
    is_trace_ray_backend = backend == "ray"
    vllm_trace_env, exp_args = _build_vllm_env_vars(exp_args, include_pinggy=False)
    vllm_cfg = exp_args.get("_datagen_vllm_server_config")
    backend_cfg = exp_args.get("_datagen_backend_config")
    trace_env_vars.update(vllm_trace_env)

    if dry_run_flag:
        trace_engine_config_path = exp_args.get("datagen_config_path") or exp_args.get(
            "_datagen_config_original_path"
        )
    else:
        trace_engine_config_path = _snapshot_datagen_config(exp_args)
    if trace_engine_config_path:
        trace_env_vars["TRACE_ENGINE_CONFIG_PATH"] = str(trace_engine_config_path)
    if is_trace_ray_backend:
        if exp_args.get("ray_cgraph_submit_timeout"):
            trace_env_vars["RAY_CGRAPH_submit_timeout"] = str(exp_args["ray_cgraph_submit_timeout"])
        if exp_args.get("ray_cgraph_get_timeout"):
            trace_env_vars["RAY_CGRAPH_get_timeout"] = str(exp_args["ray_cgraph_get_timeout"])
        if exp_args.get("ray_cgraph_max_inflight_executions"):
            trace_env_vars["RAY_CGRAPH_max_inflight_executions"] = str(
                exp_args["ray_cgraph_max_inflight_executions"]
            )
        _maybe_set_ray_cgraph_env(trace_env_vars)
    trace_env_vars["TRACE_RAY_PORT"] = str(exp_args.get("datagen_ray_port", 6379))
    trace_env_vars["TRACE_API_PORT"] = str(exp_args.get("datagen_api_port", 8000))
    tensor_parallel_size = None
    pipeline_parallel_size = None
    data_parallel_size = None
    if vllm_cfg:
        tensor_parallel_size = getattr(vllm_cfg, "tensor_parallel_size", None)
        pipeline_parallel_size = getattr(vllm_cfg, "pipeline_parallel_size", None)
        data_parallel_size = getattr(vllm_cfg, "data_parallel_size", None)
    if tensor_parallel_size is None and backend_cfg:
        tensor_parallel_size = getattr(backend_cfg, "tensor_parallel_size", None)
    if pipeline_parallel_size is None and backend_cfg:
        pipeline_parallel_size = getattr(backend_cfg, "pipeline_parallel_size", None)
    if data_parallel_size is None and backend_cfg:
        data_parallel_size = getattr(backend_cfg, "data_parallel_size", None)
    if tensor_parallel_size is None:
        tensor_parallel_size = 1
    if pipeline_parallel_size is None:
        pipeline_parallel_size = 1
    if data_parallel_size is None:
        data_parallel_size = 1
    trace_env_vars["TRACE_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
    trace_env_vars["TRACE_PIPELINE_PARALLEL_SIZE"] = str(pipeline_parallel_size)
    trace_env_vars["TRACE_DATA_PARALLEL_SIZE"] = str(data_parallel_size)
    trace_env_vars["TRACE_NUM_NODES"] = str(exp_args.get("num_nodes") or getattr(hpc, "num_nodes", 1) or 1)
    trace_env_vars["TRACE_GPUS_PER_NODE"] = str(exp_args.get("gpus_per_node") or getattr(hpc, "gpus_per_node", num_gpus))

    sandbox_overrides = {
        "SANDBOX_CPU": exp_args.get("sandbox_cpu"),
        "SANDBOX_MEMORY_GB": exp_args.get("sandbox_memory_gb"),
        "SANDBOX_DISK_GB": exp_args.get("sandbox_disk_gb"),
        "SANDBOX_GPU": exp_args.get("sandbox_gpu"),
    }
    for env_key, value in sandbox_overrides.items():
        if value is not None:
            trace_env_vars[env_key] = str(value)

    daytona_api_key = os.environ.get("DAYTONA_API_KEY")
    if daytona_api_key:
        trace_env_vars["DAYTONA_API_KEY"] = daytona_api_key
    else:
        print(
            "Warning: DAYTONA_API_KEY not found in environment; trace jobs may fail to start Daytona sandboxes."
        )

    sbatch_dir = os.path.join(os.path.dirname(__file__), "sbatch_data")
    trace_sbatch_template = os.path.join(sbatch_dir, f"{hpc.name}_datatrace.sbatch")
    if not os.path.exists(trace_sbatch_template):
        trace_sbatch_template = os.path.join(sbatch_dir, "vista_datatrace.sbatch")

    with open(trace_sbatch_template) as f:
        trace_sbatch_template_text = f.read()

    partition = getattr(hpc, "partition", "") or ""
    account = getattr(hpc, "account", "") or ""
    qos_value = str(exp_args.get("qos") or getattr(hpc, "qos", "") or "").strip()
    cpus_per_task_raw = exp_args.get("cpus_per_node") or getattr(hpc, "cpus_per_node", 1)
    cpus_per_task = int(cpus_per_task_raw)
    if cpu_only_trace and cpus_per_task > 32:
        print("TRACE backend 'vllm_local' is CPU-only; limiting CPUs per node to 32.")
        cpus_per_task = 32
        exp_args = update_exp_args(exp_args, {"cpus_per_node": cpus_per_task})
    trace_env_vars["TRACE_CPUS_PER_NODE"] = str(cpus_per_task)
    num_nodes = exp_args.get("num_nodes") or getattr(hpc, "num_nodes", 1) or 1
    gpus_per_node_trace = exp_args.get("gpus_per_node") or getattr(hpc, "gpus_per_node", num_gpus)
    if cpu_only_trace:
        gpus_per_node_trace = 0
    gpus_per_node_trace_directive = gpus_per_node_trace
    if getattr(hpc, "name", "").lower() == "vista":
        gpus_per_node_trace_directive = None

    base_substitutions = {
        "{partition}": partition,
        "{time_limit}": exp_args["time_limit"],
        "{num_nodes}": str(num_nodes),
        "{cpus_per_node}": str(cpus_per_task),
        "{num_gpus}": str(num_gpus),
        "{gpus_per_node}": "" if gpus_per_node_trace_directive is None else str(gpus_per_node_trace_directive),
        "{account}": account,
        "{experiments_dir}": exp_args["experiments_dir"],
        "{qos}": qos_value,
    }

    def _render_trace_sbatch_content(
        job_name: str,
        env_vars: dict[str, str],
        *,
        memory_limit_gb: int | None = None,
        array_spec: str | None = None,
    ) -> str:
        content = trace_sbatch_template_text
        for key, value in base_substitutions.items():
            content = content.replace(key, value)
        content = content.replace("{job_name}", job_name)
        if array_spec:
            job_line = f"#SBATCH --job-name {job_name}\n"
            array_line = f"#SBATCH --array={array_spec}\n"
            if job_line in content:
                content = content.replace(job_line, job_line + array_line, 1)
            else:
                if content.startswith("#!"):
                    newline_idx = content.find("\n")
                    if newline_idx == -1:
                        newline_idx = len(content)
                    content = (
                        content[: newline_idx + 1]
                        + array_line
                        + content[newline_idx + 1 :]
                    )
                else:
                    content = array_line + content
        if num_gpus == 0:
            content = content.replace("#SBATCH --gres=gpu:0\n", "")
        if not gpus_per_node_trace_directive:
            content = content.replace("#SBATCH --gpus-per-node {gpus_per_node}\n", "")
            content = content.replace("#SBATCH --gpus-per-node 0\n", "")
            content = content.replace("#SBATCH --gpus-per-node=0\n", "")
        if not partition:
            content = content.replace("#SBATCH -p \n", "")
        if not account:
            content = content.replace("#SBATCH --account \n", "")
        if not qos_value:
            content = re.sub(r"#SBATCH\s+-q[^\n]*\n", "", content)
        if memory_limit_gb is not None:
            memory_override = f"{memory_limit_gb}GB"
            existing_mem = re.search(
                r"^#SBATCH --mem=([0-9]+)([A-Za-z]+)",
                content,
                flags=re.MULTILINE,
            )
            if existing_mem:
                unit = existing_mem.group(2)
                try:
                    current_mem_value = int(existing_mem.group(1))
                except ValueError:
                    current_mem_value = None
                if current_mem_value is not None:
                    target_mem_value = min(current_mem_value, memory_limit_gb)
                    memory_override = f"{target_mem_value}{unit}"
            if re.search(r"^#SBATCH --mem=\S+", content, flags=re.MULTILINE):
                content = re.sub(
                    r"^#SBATCH --mem=\S+",
                    f"#SBATCH --mem={memory_override}",
                    content,
                    count=1,
                    flags=re.MULTILINE,
                )
            else:
                cpu_line_pattern = rf"(#SBATCH --cpus-per-task={cpus_per_task}\s*\n)"
                if re.search(cpu_line_pattern, content):
                    content = re.sub(
                        cpu_line_pattern,
                        r"\1#SBATCH --mem=" + memory_override + "\n",
                        content,
                        count=1,
                    )
                else:
                    content = content.replace(
                        "#SBATCH --ntasks-per-node 1\n",
                        "#SBATCH --ntasks-per-node 1\n" + f"#SBATCH --mem={memory_override}\n",
                        1,
                    )
            if not re.search(r"^#SBATCH --mem=\S+", content, flags=re.MULTILINE):
                content = f"#SBATCH --mem={memory_override}\n" + content
        return _inject_env_block(content, env_vars)

    def _write_chunk_env_file(path: Path, env_map: dict[str, Any]) -> None:
        lines: list[str] = []
        for key, value in env_map.items():
            if value is None:
                continue
            lines.append(f"export {key}={shlex.quote(str(value))}")
        lines.append("")
        path.write_text("\n".join(lines))

    sbatch_output_dir = os.path.join(exp_args["experiments_dir"], "sbatch")
    logs_dir = os.path.join(exp_args["experiments_dir"], "logs")
    os.makedirs(sbatch_output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    if chunk_plans and _is_local_mode(hpc):
        raise RuntimeError("Trace chunking (--chunk_size) is not supported for local execution.")

    if chunk_plans:
        if chunk_map_path:
            print(f"Trace chunk map saved to: {chunk_map_path}")
        chunk_array_max = exp_args.get("_chunk_array_max")
        try:
            chunk_array_max = int(chunk_array_max) if chunk_array_max is not None else None
        except (TypeError, ValueError):
            chunk_array_max = None
        if not chunk_array_max or chunk_array_max <= 0:
            chunk_array_max = 3

        chunk_env_dir = trace_configs_dir / "chunk_envs"
        if dry_run_flag:
            print(f"DRY RUN: Would create chunk env directory at {chunk_env_dir}")
        else:
            if chunk_env_dir.exists():
                shutil.rmtree(chunk_env_dir)
            chunk_env_dir.mkdir(parents=True, exist_ok=True)

        original_endpoint_json_path = exp_args.get("vllm_endpoint_json_path")
        base_trace_engine_config_path = trace_env_vars.get("TRACE_ENGINE_CONFIG_PATH")

        for plan in chunk_plans:
            chunk_overrides: dict[str, Any] = {
                "TRACE_TASKS_PATH": str(plan.tasks_path),
                "TRACE_TARGET_REPO": plan.target_repo,
                "TRACE_OUTPUT_DIR": str(plan.output_dir),
                "TRACE_JOBS_DIR": str(plan.jobs_dir),
                "TRACE_JOB_INDEX": str(plan.index),
                "TRACE_CHUNK_COUNT": str(len(chunk_plans)),
                "JOB_SUBMIT_ORDER": str(plan.index + 1),
            }
            chunk_config_path = _materialize_trace_config(
                dataset_path=plan.tasks_path,
                job_suffix=f"_trace_{plan.index:03d}",
                jobs_dir=plan.jobs_dir,
                agent_name_override=trace_agent_name,
                agent_model_override=trace_model_dispatch or None,
                agent_timeout_override=trace_agent_timeout_override,
                agent_kwargs_override=trace_agent_kwargs,
                n_concurrent_override=trace_n_concurrent_effective,
                env_override=trace_env_override,
                disable_verification=disable_verification_flag,
            )
            chunk_overrides["TRACE_HARBOR_CONFIG"] = str(chunk_config_path)
            chunk_endpoint_json = (
                default_vllm_endpoint_path(
                    exp_args["experiments_dir"],
                    trace=True,
                    chunk_index=plan.index,
                )
                if requires_endpoint
                else ""
            )
            chunk_engine_config_path = base_trace_engine_config_path
            if chunk_endpoint_json:
                update_exp_args(
                    exp_args,
                    {"vllm_endpoint_json_path": chunk_endpoint_json},
                )
                chunk_engine_config_path = _snapshot_datagen_config(
                    exp_args,
                    output_filename=f"datagen_config_trace_{plan.index:03d}.resolved.yaml",
                    update_exp_args=False,
                )
                chunk_overrides["TRACE_ENDPOINT_JSON"] = chunk_endpoint_json
                chunk_overrides["VLLM_ENDPOINT_JSON_PATH"] = chunk_endpoint_json
            if chunk_engine_config_path:
                chunk_overrides["TRACE_ENGINE_CONFIG_PATH"] = str(chunk_engine_config_path)

            env_file = chunk_env_dir / f"chunk_{plan.index:03d}.env"
            if dry_run_flag:
                print(
                    f"DRY RUN: Would write chunk env {env_file} "
                    f"(tasks: {len(plan.task_names)}, repo: {plan.target_repo})"
                )
            else:
                _write_chunk_env_file(env_file, chunk_overrides)
            print(
                f"Prepared trace chunk {plan.index}: tasks={len(plan.task_names)} "
                f"repo={plan.target_repo} env_file={env_file}"
            )

        if original_endpoint_json_path is not None:
            update_exp_args(
                exp_args,
                {"vllm_endpoint_json_path": original_endpoint_json_path},
            )
        elif "vllm_endpoint_json_path" in exp_args:
            del exp_args["vllm_endpoint_json_path"]

        if dry_run_flag:
            return [f"dry_run_trace_job_id_{plan.index:03d}" for plan in chunk_plans]

        chunk_base_env = dict(trace_env_vars)
        chunk_base_env.update(
            {
                "TRACE_CHUNK_MODE": "1",
                "TRACE_CHUNK_ENV_DIR": str(chunk_env_dir),
                "TRACE_CHUNK_COUNT": str(len(chunk_plans)),
            }
        )

        array_limit = min(chunk_array_max, len(chunk_plans))
        array_spec = f"0-{len(chunk_plans) - 1}%{array_limit}"
        chunk_job_name = f"{exp_args['job_name']}_trace_chunks"
        chunk_sbatch_output = os.path.join(sbatch_output_dir, f"{chunk_job_name}.sbatch")
        sbatch_content = _render_trace_sbatch_content(
            chunk_job_name,
            chunk_base_env,
            memory_limit_gb=trace_memory_limit_gb,
            array_spec=array_spec,
        )
        with open(chunk_sbatch_output, "w") as f:
            f.write(sbatch_content)

        print(
            f"Trace chunk job array sbatch: {chunk_sbatch_output} "
            f"(chunks: {len(chunk_plans)}, array: {array_spec})"
        )

        job_id = _submit_script(chunk_sbatch_output, dependency=dependency, array=array_spec, hpc=hpc)
        print(f"✓ Trace chunk array job submitted: {job_id}")
        return [job_id]

    trace_config_output = _materialize_trace_config(
        dataset_path=tasks_root,
        job_suffix="_trace",
        jobs_dir=trace_jobs_dir_path,
        agent_name_override=trace_agent_name,
        agent_model_override=trace_model_dispatch or None,
        agent_timeout_override=trace_agent_timeout_override,
        agent_kwargs_override=trace_agent_kwargs,
        n_concurrent_override=trace_n_concurrent_effective,
        env_override=trace_env_override,
        disable_verification=disable_verification_flag,
    )
    trace_env_vars["TRACE_HARBOR_CONFIG"] = str(trace_config_output)

    trace_job_name = f"{exp_args['job_name']}_trace"
    trace_sbatch_output = os.path.join(sbatch_output_dir, f"{trace_job_name}.sbatch")
    sbatch_content = _render_trace_sbatch_content(
        trace_job_name,
        trace_env_vars,
        memory_limit_gb=trace_memory_limit_gb,
    )
    with open(trace_sbatch_output, "w") as f:
        f.write(sbatch_content)

    print(f"Trace generation sbatch: {trace_sbatch_output}")
    print(f"Trace script: {trace_script}")
    print(f"Trace target repo: {trace_target_repo}")

    if dry_run_flag:
        print("DRY RUN: Would submit trace generation job")
        return "dry_run_trace_job_id"

    job_id = _submit_script(trace_sbatch_output, dependency=dependency, hpc=hpc)
    print(f"✓ Trace generation job submitted: {job_id}")
    return job_id




__all__ = [
    "DATAGEN_CONFIG_DIR",
    "HARBOR_CONFIG_DIR",
    "DEFAULT_RAY_CGRAPH_TIMEOUT",
    "DEFAULT_RAY_CGRAPH_MAX_INFLIGHT",
    "derive_datagen_job_name",
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
    "launch_datagen_job",
    "launch_vllm_server",
    "launch_task_job",
    "launch_trace_job",
]
