"""Shared utilities for datagen-oriented HPC launches."""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from hpc.launch_utils import (
    default_vllm_endpoint_path,
    derive_datagen_job_name,  # Re-exported for backwards compatibility
    launch_sbatch,
    cleanup_endpoint_file,
    normalize_cli_args,
    resolve_config_path,
    build_sbatch_directives,
    generate_served_model_id,
    get_daytona_api_key_override,
    hosted_vllm_alias,
    strip_hosted_vllm_alias,
    set_or_pop,
    resolve_job_and_paths,
    substitute_template,
    resolve_conda_activate,
)
from hpc.harbor_utils import (
    get_harbor_env_from_config,
    HARBOR_CONFIG_DIR,
    load_harbor_config,
    resolve_harbor_config_path,
)
from hpc.hf_utils import resolve_dataset_path, derive_default_hf_repo_id, sanitize_hf_repo_id
from hpc.cli_utils import resolve_n_attempts, resolve_n_concurrent

# Backward compatibility aliases
_normalize_cli_args = normalize_cli_args

DIRENV = os.path.dirname(__file__)
DATAGEN_CONFIG_DIR = os.path.join(DIRENV, "datagen_yaml")
DEFAULT_RAY_CGRAPH_TIMEOUT = os.environ.get("RAY_CGRAPH_TIMEOUT_DEFAULT", "86500")
DEFAULT_RAY_CGRAPH_MAX_INFLIGHT = os.environ.get("RAY_CGRAPH_MAX_INFLIGHT_DEFAULT", "")
HARBOR_MODEL_PLACEHOLDER = "placeholder/override-at-runtime"


def _count_tasks_in_path(tasks_path: str) -> int:
    """Count the number of tasks in a dataset directory or file.

    Tries multiple loading strategies in order:
    1. HuggingFace ``load_from_disk`` (Arrow format saved with ``save_to_disk``)
    2. HuggingFace ``load_dataset`` (directory of Parquet/JSON/CSV files)
    3. Parquet row count via pyarrow (single file)
    4. JSONL line count (single file)

    Raises ``ValueError`` if the task count cannot be determined.
    """
    p = Path(tasks_path)
    if not p.exists():
        raise FileNotFoundError(f"Tasks path does not exist: {tasks_path}")

    # Strategy 1: HF load_from_disk (Arrow dataset_dict or dataset)
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(p))
        # DatasetDict → use the first split
        if hasattr(ds, "keys"):
            split = list(ds.keys())[0]
            count = len(ds[split])
        else:
            count = len(ds)
        print(f"[chunk] Counted {count} tasks via load_from_disk")
        return count
    except Exception:
        pass

    # Strategy 2: HF load_dataset (directory of data files)
    try:
        from datasets import load_dataset
        ds = load_dataset(str(p))
        if hasattr(ds, "keys"):
            split = list(ds.keys())[0]
            count = len(ds[split])
        else:
            count = len(ds)
        print(f"[chunk] Counted {count} tasks via load_dataset")
        return count
    except Exception:
        pass

    # Strategy 3: single Parquet file
    parquet_files = list(p.glob("*.parquet")) if p.is_dir() else ([p] if p.suffix == ".parquet" else [])
    if parquet_files:
        try:
            import pyarrow.parquet as pq
            count = sum(pq.read_metadata(str(f)).num_rows for f in parquet_files)
            print(f"[chunk] Counted {count} tasks via Parquet metadata")
            return count
        except Exception:
            pass

    # Strategy 4: JSONL line count
    jsonl_files = list(p.glob("*.jsonl")) if p.is_dir() else ([p] if p.suffix == ".jsonl" else [])
    if jsonl_files:
        count = 0
        for f in jsonl_files:
            with open(f) as fh:
                count += sum(1 for _ in fh)
        print(f"[chunk] Counted {count} tasks via JSONL line count")
        return count

    # Strategy 5: Pre-extracted task directories (each task has an instruction.md)
    if p.is_dir():
        task_marker = "instruction.md"
        count = sum(1 for d in p.rglob(task_marker) if d.is_file())
        if count > 0:
            print(f"[chunk] Counted {count} tasks via task folder marker ({task_marker})")
            return count

    raise ValueError(
        f"Cannot determine task count from {tasks_path}. "
        "Supported formats: HF datasets (Arrow), Parquet, JSONL, "
        "pre-extracted task directories (with instruction.md)."
    )


def _prepare_datagen_configuration(exp_args: dict):
    """Load the YAML datagen configuration and derive launch metadata.

    Uses the consolidated parse_datagen_config() for common parsing logic.
    """
    from hpc.datagen_config_utils import parse_datagen_config
    from data.generation.utils import resolve_engine_runtime

    raw_config = exp_args.get("datagen_config") or os.environ.get("DATAGEN_CONFIG_PATH")
    if not raw_config:
        raise ValueError(
            "Data generation requires --datagen-config or DATAGEN_CONFIG_PATH to specify the engine YAML."
        )

    # Resolve path and parse config
    resolved_path = resolve_config_path(raw_config, DATAGEN_CONFIG_DIR, "datagen")
    parsed = parse_datagen_config(
        config_path=str(resolved_path),
        model_override=exp_args.get("trace_model"),
    )

    # Engine runtime (for backwards compatibility)
    runtime = resolve_engine_runtime(parsed.loaded.config)
    backend = parsed.loaded.config.backend

    # Direct assignments (always set)
    exp_args.update({
        # Internal objects
        "_parsed_datagen_config": parsed,
        "_datagen_config_original_path": str(parsed.config_path),
        "_datagen_config_raw": parsed.loaded.raw,
        "_datagen_config_obj": parsed.loaded.config,
        "_datagen_engine_runtime": runtime,
        "_datagen_extra_agent_kwargs": parsed.extra_agent_kwargs,
        "_datagen_backend_config": backend,
        "_datagen_vllm_server_config": parsed.vllm_server_config,
        "_chunk_array_max": parsed.chunk_array_max,
        # Public settings
        "datagen_config_path": str(parsed.config_path),
        "datagen_engine": parsed.engine_type,
        "datagen_healthcheck_interval": parsed.healthcheck_interval,
        "datagen_backend": backend.type,
        "datagen_wait_for_endpoint": parsed.wait_for_endpoint,
        "datagen_ray_port": parsed.ray_port,
        "datagen_api_port": parsed.api_port,
    })

    # Conditional assignments (set if present, remove if None)
    set_or_pop(exp_args, "datagen_model", parsed.model)
    set_or_pop(exp_args, "datagen_max_tokens", parsed.max_output_tokens)
    set_or_pop(exp_args, "vllm_endpoint_json_path", parsed.endpoint_json_path)
    set_or_pop(exp_args, "ray_cgraph_submit_timeout", parsed.ray_cgraph_submit_timeout)
    set_or_pop(exp_args, "ray_cgraph_get_timeout", parsed.ray_cgraph_get_timeout)
    set_or_pop(exp_args, "ray_cgraph_max_inflight_executions", parsed.ray_cgraph_max_inflight_executions)
    set_or_pop(exp_args, "trace_health_max_attempts", parsed.health_max_attempts)
    set_or_pop(exp_args, "trace_health_retry_delay", parsed.health_retry_delay)
    set_or_pop(exp_args, "_vllm_server_extra_args", parsed.vllm_extra_args or None)

    return runtime


def launch_datagen_job_v2(exp_args: dict, hpc) -> None:
    """Launch datagen job using the new universal template system.

    1. Creating TaskgenJobConfig and/or TracegenJobConfig from exp_args
    2. Writing configs to JSON
    3. Using universal_taskgen.sbatch and universal_tracegen.sbatch templates
    4. Submitting the jobs
    """
    # asdict and launch_sbatch are imported at module level

    print("\n=== DATA GENERATION MODE (Universal Launcher) ===")

    explicit_cli_keys = set(exp_args.get("_explicit_cli_keys") or [])

    # Auto-configure when tasks_input_path is provided
    # (user is providing pre-existing tasks, so task gen is unnecessary but trace gen is implied)
    if exp_args.get("tasks_input_path"):
        if "enable_trace_gen" not in explicit_cli_keys:
            exp_args["enable_trace_gen"] = True
            print("[datagen] Auto-enabled trace generation (--tasks-input-path provided)")

    # Determine what to run
    task_enabled = str(exp_args.get("enable_task_gen", True)).lower() not in {"false", "0", "no", "none"}
    trace_enabled = str(exp_args.get("enable_trace_gen", False)).lower() not in {"false", "0", "no", "none"}

    if not task_enabled and not trace_enabled:
        raise ValueError("Enable at least one of task or trace generation")

    if task_enabled and not exp_args.get("datagen_script"):
        raise ValueError("--datagen-script is required for task generation")

    # Resolve job_name and paths (auto-derives job_name if not provided)
    job_setup = resolve_job_and_paths(
        exp_args,
        job_type_label="Datagen",
        derive_job_name_fn=derive_datagen_job_name,
    )
    job_name = job_setup.job_name
    exp_paths = job_setup.paths
    experiments_subdir = str(exp_paths.root)  # String form for config dicts

    # vLLM settings
    vllm_cfg = exp_args.get("_datagen_vllm_server_config")
    engine = str(exp_args.get("datagen_engine") or "openai").lower()
    requires_vllm = bool(vllm_cfg and engine == "vllm_local")

    # Pre-download model for no-internet clusters (JSC).
    # Must happen on the login node (which has internet) before sbatch submission.
    # Also avoids race conditions from multiple Ray workers downloading simultaneously.
    from hpc.checkpoint_utils import pre_download_model, is_huggingface_repo
    vllm_model = getattr(vllm_cfg, "model_path", None) if vllm_cfg else None
    datagen_model = exp_args.get("datagen_model") or exp_args.get("trace_model")
    model_to_download = vllm_model or datagen_model
    if model_to_download and is_huggingface_repo(model_to_download):
        print(f"Pre-downloading model: {model_to_download}")
        dl_result = pre_download_model(model_to_download)
        if vllm_cfg and hasattr(vllm_cfg, "model_path"):
            vllm_cfg.model_path = dl_result.local_path
        if exp_args.get("datagen_model"):
            exp_args["datagen_model"] = dl_result.local_path
        if exp_args.get("trace_model"):
            exp_args["trace_model"] = dl_result.local_path
        print(f"Model available at: {dl_result.local_path}")

    gpus_per_node = int(exp_args.get("gpus_per_node") or getattr(hpc, "gpus_per_node", 0) or 0)
    cpus_per_node = int(exp_args.get("cpus_per_node") or getattr(hpc, "cpus_per_node", 24) or 24)
    tensor_parallel_size = getattr(vllm_cfg, "tensor_parallel_size", None) or 1
    pipeline_parallel_size = getattr(vllm_cfg, "pipeline_parallel_size", None) or 1
    data_parallel_size = getattr(vllm_cfg, "data_parallel_size", None) or 1

    endpoint_json_path = None
    if requires_vllm:
        endpoint_json_path = exp_args.get("vllm_endpoint_json_path") or str(
            default_vllm_endpoint_path(experiments_subdir)
        )
        cleanup_endpoint_file(endpoint_json_path, descriptor="stale datagen endpoint file")

    # Determine cluster env file
    cluster_env_file = hpc.dotenv_filename if hasattr(hpc, "dotenv_filename") else f"{hpc.name.lower()}.env"

    task_job_id = None

    # === Task Generation ===
    if task_enabled:
        # Convert vllm_cfg dataclass to dict for pass-through
        vllm_server_config = asdict(vllm_cfg) if vllm_cfg else {}

        # Auto-derive datagen_target_repo if not set
        datagen_target_repo = exp_args.get("datagen_target_repo")
        if not datagen_target_repo:
            datagen_target_repo = sanitize_hf_repo_id(derive_default_hf_repo_id(f"{job_name}-tasks"))
            print(f"[datagen] Auto-derived --datagen-target-repo: {datagen_target_repo}")

        task_config = TaskgenJobConfig(
            job_name=f"{job_name}_tasks",
            datagen_script=exp_args.get("datagen_script") or "",
            experiments_dir=experiments_subdir,
            cluster_name=hpc.name,
            output_dir=exp_args.get("datagen_output_dir"),
            input_dir=exp_args.get("datagen_input_dir"),
            target_repo=datagen_target_repo,
            engine=engine,
            datagen_config_path=exp_args.get("datagen_config_path"),
            needs_vllm=requires_vllm,
            vllm_model_path=getattr(vllm_cfg, "model_path", None) if vllm_cfg else None,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            endpoint_json_path=endpoint_json_path,
            ray_port=int(exp_args.get("datagen_ray_port") or 6379),
            api_port=int(exp_args.get("datagen_api_port") or 8000),
            extra_args=exp_args.get("datagen_extra_args") or "",
            disable_verification=bool(exp_args.get("disable_verification")),
            num_nodes=int(exp_args.get("num_nodes") or 1),
            gpus_per_node=gpus_per_node,
            cpus_per_node=cpus_per_node,
            vllm_server_config=vllm_server_config,
            ray_object_store_gb=float(exp_args.get("ray_object_store_gb", 40.0)),
        )

        # Write task config JSON
        task_config_path = exp_paths.configs / f"{job_name}_taskgen_config.json"
        task_config_path.write_text(json.dumps(asdict(task_config), indent=2))

        # Load and populate taskgen template
        template_path = Path(__file__).parent / "sbatch_data" / "universal_taskgen.sbatch"
        if not template_path.exists():
            raise FileNotFoundError(f"Universal taskgen template not found: {template_path}")

        template_text = template_path.read_text()

        # Build SBATCH directives using shared utility
        sbatch_directives = build_sbatch_directives(hpc, exp_args)

        substitutions = {
            "time_limit": exp_args.get("time_limit") or "24:00:00",
            "num_nodes": str(exp_args.get("num_nodes") or 1),
            "cpus_per_node": str(exp_args.get("cpus_per_node") or hpc.cpus_per_node),
            "experiments_dir": experiments_subdir,
            "job_name": f"{job_name}_tasks",
            "sbatch_extra_directives": "\n".join(sbatch_directives),
            "module_commands": hpc.get_module_commands(),
            "conda_activate": resolve_conda_activate(hpc, exp_args),
            "cluster_env_file": cluster_env_file,
            "config_path": str(task_config_path),
            "email_address": os.environ.get("EMAIL_ADDRESS", ""),
            "env_exports": hpc.get_env_exports(),
            "ray_env_exports": hpc.get_ray_env_exports(experiments_subdir),
            "daytona_api_key_override": get_daytona_api_key_override(exp_args),
            "ssh_tunnel_setup": hpc.get_ssh_tunnel_setup(),
            "proxy_setup": hpc.get_proxy_setup(),
        }

        sbatch_text = substitute_template(template_text, substitutions)

        task_sbatch_output = exp_paths.sbatch / f"{job_name}_taskgen.sbatch"
        task_sbatch_output.write_text(sbatch_text)
        os.chmod(task_sbatch_output, 0o750)

        # Get CLI dependency for first job in pipeline
        cli_dependency = exp_args.get("dependency")

        if exp_args.get("dry_run"):
            print(f"DRY RUN: Taskgen sbatch script written to {task_sbatch_output}")
            if cli_dependency:
                print(f"  Would submit with dependency: {cli_dependency}")
            task_job_id = "dry_run_task_job_id"
        else:
            task_job_id = launch_sbatch(str(task_sbatch_output), dependency=cli_dependency)
            print(f"✓ Task generation job submitted: {task_job_id}")

    # === Trace Generation ===
    if trace_enabled:
        trace_script = exp_args.get("trace_script") or exp_args.get("datagen_script")
        trace_target_repo = exp_args.get("trace_target_repo")
        if not trace_target_repo:
            # Auto-derive from job_name: <org>/<job_name>-traces
            trace_target_repo = sanitize_hf_repo_id(derive_default_hf_repo_id(f"{job_name}-traces"))
            print(f"[datagen] Auto-derived --trace-target-repo: {trace_target_repo}")

        harbor_config = exp_args.get("trace_harbor_config")
        if not harbor_config:
            raise ValueError("--trace-harbor-config is required for trace generation")
        harbor_config_resolved = str(resolve_harbor_config_path(harbor_config))
        harbor_config_data = load_harbor_config(harbor_config_resolved)

        tasks_input_path = exp_args.get("tasks_input_path")
        if tasks_input_path:
            # Use shared utility to handle both HF repos and local paths
            tasks_input_path = resolve_dataset_path(tasks_input_path, verbose=True)
        elif task_enabled:
            # Fallback to generated output dir from task generation
            tasks_input_path = exp_args.get("datagen_output_dir") or str(
                exp_paths.root / "outputs" / "tasks"
            )

        trace_model = exp_args.get("trace_model") or exp_args.get("datagen_model") or ""
        if vllm_cfg and not trace_model:
            trace_model = getattr(vllm_cfg, "model_path", "") or ""

        vllm_model_path = getattr(vllm_cfg, "model_path", None) if vllm_cfg else (trace_model or None)
        served_model_id = None
        harbor_model_name = trace_model
        if requires_vllm:
            served_model_id = generate_served_model_id()
            harbor_model_name = hosted_vllm_alias(served_model_id)
            if not vllm_model_path:
                vllm_model_path = trace_model or ""

        # Collect extra agent kwargs using consolidated helper
        from hpc.harbor_utils import collect_extra_agent_kwargs, derive_vllm_supports_tool_calling
        agent_kwargs = collect_extra_agent_kwargs(
            datagen_extras=exp_args.get("_datagen_extra_agent_kwargs"),
            cli_kwargs=exp_args.get("trace_agent_kwargs"),
        )
        trace_agent_name = exp_args.get("trace_agent_name")
        if not trace_agent_name:
            agents = harbor_config_data.get("agents") or []
            if agents and isinstance(agents[0], dict):
                trace_agent_name = agents[0].get("name") or ""
        if (trace_agent_name or "") == "swe-agent":
            supports_tool_calling = derive_vllm_supports_tool_calling(vllm_cfg)
            if supports_tool_calling is not None:
                agent_kwargs.setdefault("supports_tool_calling", supports_tool_calling)

        # Convert vllm_cfg dataclass to dict for pass-through (if not already done)
        trace_vllm_server_config = asdict(vllm_cfg) if vllm_cfg else {}

        # --- Chunking logic ---
        chunk_size = int(exp_args.get("chunk_size") or 0)
        num_chunks = 1
        if chunk_size > 0 and tasks_input_path:
            total_tasks = _count_tasks_in_path(tasks_input_path)
            num_chunks = math.ceil(total_tasks / chunk_size)
            if num_chunks <= 1:
                print(f"[datagen] Task count ({total_tasks}) ≤ chunk_size ({chunk_size}), no chunking needed")
                chunk_size = 0
                num_chunks = 1
            else:
                print(f"[datagen] Splitting {total_tasks} tasks into {num_chunks} chunks of ≤{chunk_size}")

        # Build the list of (chunk_index | None) to iterate over.
        # None means "no chunking, single job".
        chunk_indices = list(range(num_chunks)) if chunk_size > 0 else [None]

        # Shared template + directives (same for all chunks)
        template_path = Path(__file__).parent / "sbatch_data" / "universal_tracegen.sbatch"
        if not template_path.exists():
            raise FileNotFoundError(f"Universal tracegen template not found: {template_path}")
        template_text = template_path.read_text()
        sbatch_directives = build_sbatch_directives(hpc, exp_args)
        harbor_env = exp_args.get("trace_env") or get_harbor_env_from_config(harbor_config_resolved)

        base_hf_repo_id = exp_args.get("upload_hf_repo") or trace_target_repo

        # Set dependency on task job if both are enabled, otherwise use CLI dependency
        if task_enabled and task_job_id and task_job_id != "dry_run_task_job_id":
            dependency = f"afterok:{task_job_id}"
        else:
            dependency = exp_args.get("dependency")

        for chunk_idx in chunk_indices:
            is_chunked = chunk_idx is not None
            suffix = f"_chunk{chunk_idx}" if is_chunked else ""
            chunk_job_name = f"{job_name}_traces{suffix}"
            chunk_hf_repo_id = sanitize_hf_repo_id(f"{base_hf_repo_id}{suffix}") if is_chunked else base_hf_repo_id
            # Each chunk gets its own experiments subdirectory to avoid output collisions
            chunk_experiments_dir = (
                str(Path(experiments_subdir).parent / f"{Path(experiments_subdir).name}{suffix}")
                if is_chunked else experiments_subdir
            )

            trace_config = TracegenJobConfig(
                job_name=chunk_job_name,
                harbor_config=harbor_config_resolved,
                trace_script=trace_script or "",
                experiments_dir=chunk_experiments_dir,
                cluster_name=hpc.name,
                tasks_input_path=tasks_input_path or "",
                output_dir=exp_args.get("trace_output_dir"),
                target_repo=trace_target_repo,
                engine=engine,
                datagen_config_path=exp_args.get("datagen_config_path"),
                needs_vllm=requires_vllm,
                vllm_model_path=vllm_model_path,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                data_parallel_size=data_parallel_size,
                endpoint_json_path=endpoint_json_path,
                ray_port=int(exp_args.get("datagen_ray_port") or 6379),
                api_port=int(exp_args.get("datagen_api_port") or 8000),
                model=harbor_model_name,
                served_model_id=served_model_id,
                agent=trace_agent_name or "",
                trace_env=exp_args.get("trace_env") or get_harbor_env_from_config(harbor_config_resolved),
                n_concurrent=resolve_n_concurrent(exp_args.get("trace_n_concurrent"), harbor_config_data),
                n_attempts=resolve_n_attempts(exp_args.get("trace_n_attempts"), harbor_config_data),
                agent_kwargs=agent_kwargs,
                num_nodes=int(exp_args.get("num_nodes") or 1),
                gpus_per_node=gpus_per_node,
                cpus_per_node=cpus_per_node,
                vllm_server_config=trace_vllm_server_config,
                hf_repo_id=chunk_hf_repo_id,
                hf_private=bool(exp_args.get("upload_hf_private")),
                hf_episodes=exp_args.get("upload_hf_episodes") or "last",
                pinggy_persistent_url=exp_args.get("pinggy_persistent_url"),
                pinggy_token=exp_args.get("pinggy_token"),
                # Chunking fields (None when not chunking)
                chunk_size=chunk_size if is_chunked else None,
                chunk_index=chunk_idx,
                num_chunks=num_chunks if is_chunked else None,
                ray_object_store_gb=float(exp_args.get("ray_object_store_gb", 40.0)),
            )

            trace_config_path = exp_paths.configs / f"{chunk_job_name}_tracegen_config.json"
            trace_config_path.write_text(json.dumps(asdict(trace_config), indent=2))

            substitutions = {
                "time_limit": exp_args.get("time_limit") or "24:00:00",
                "num_nodes": str(exp_args.get("num_nodes") or 1),
                "cpus_per_node": str(exp_args.get("cpus_per_node") or hpc.cpus_per_node),
                "experiments_dir": chunk_experiments_dir,
                "job_name": chunk_job_name,
                "sbatch_extra_directives": "\n".join(sbatch_directives),
                "module_commands": hpc.get_module_commands(),
                "conda_activate": resolve_conda_activate(hpc, exp_args),
                "cluster_env_file": cluster_env_file,
                "config_path": str(trace_config_path),
                "email_address": os.environ.get("EMAIL_ADDRESS", ""),
                "harbor_env": harbor_env,
                "env_exports": hpc.get_env_exports(),
                "ray_env_exports": hpc.get_ray_env_exports(chunk_experiments_dir),
                "daytona_api_key_override": get_daytona_api_key_override(exp_args),
                "ssh_tunnel_setup": hpc.get_ssh_tunnel_setup(),
                "proxy_setup": hpc.get_proxy_setup(),
            }

            sbatch_text = substitute_template(template_text, substitutions)

            trace_sbatch_output = exp_paths.sbatch / f"{chunk_job_name}_tracegen.sbatch"
            trace_sbatch_output.write_text(sbatch_text)
            os.chmod(trace_sbatch_output, 0o750)

            if exp_args.get("dry_run"):
                print(f"DRY RUN: Tracegen sbatch script written to {trace_sbatch_output}")
                if dependency:
                    print(f"  Would submit with dependency: {dependency}")
            else:
                job_id = launch_sbatch(str(trace_sbatch_output), dependency=dependency)
                print(f"✓ Trace generation job submitted: {job_id} ({chunk_job_name})")


# ==============================================================================
# Job Runner Classes for Universal SBATCH Scripts
# ==============================================================================
#
# These classes encapsulate the job logic. They are called from universal_taskgen.sbatch
# and universal_tracegen.sbatch templates.


@dataclass
class TaskgenJobConfig:
    """Configuration for a task generation job (serialized to JSON for sbatch)."""

    job_name: str
    datagen_script: str
    experiments_dir: str
    cluster_name: str = ""

    # Output settings
    output_dir: Optional[str] = None
    input_dir: Optional[str] = None
    target_repo: Optional[str] = None

    # Engine settings
    engine: str = "openai"
    datagen_config_path: Optional[str] = None

    # vLLM settings (if engine requires it)
    needs_vllm: bool = False
    vllm_model_path: Optional[str] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    endpoint_json_path: Optional[str] = None
    ray_port: int = 6379
    api_port: int = 8000
    vllm_server_config: Dict[str, Any] = field(default_factory=dict)  # Raw vllm_server config from YAML

    # Health check settings
    health_max_attempts: int = 120
    health_retry_delay: int = 15
    healthcheck_interval: int = 300

    # Extra args
    extra_args: str = ""
    disable_verification: bool = False

    # Resource allocation (from CLI overrides, None = use HPC cluster defaults)
    num_nodes: int = 1
    gpus_per_node: Optional[int] = None
    cpus_per_node: Optional[int] = None

    # Ray object store size in GB (default: 40)
    ray_object_store_gb: float = 40.0


class TaskgenJobRunner:
    """Runs task generation jobs with optional vLLM management.

    Usage (from sbatch):
        python -m hpc.datagen_launch_utils --mode taskgen --config /path/to/config.json
    """

    def __init__(self, config: TaskgenJobConfig):
        self.config = config
        self._hpc = None
        self._proxychains_binary = ""

    def _get_hpc(self):
        """Lazy-load HPC configuration."""
        if self._hpc is None:
            from hpc.hpc import detect_hpc, clusters
            if self.config.cluster_name:
                for c in clusters:
                    if c.name.lower() == self.config.cluster_name.lower():
                        self._hpc = c
                        break
                if self._hpc is None:
                    raise ValueError(f"Unknown cluster: {self.config.cluster_name}")
            else:
                self._hpc = detect_hpc()
            # Stash proxychains binary for wrapping commands on no-internet clusters
            self._proxychains_binary = getattr(self._hpc, "proxychains_binary", "") or ""
        return self._hpc

    def run(self) -> int:
        """Execute the task generation job.

        Returns:
            Exit code (0 for success)
        """
        print(f"=== TaskgenJobRunner: {self.config.job_name} ===")

        try:
            # Ensure HPC is loaded (sets _proxychains_binary for no-internet clusters)
            self._get_hpc()

            if self.config.needs_vllm:
                exit_code = self._run_with_vllm()
            else:
                exit_code = self._run_datagen(endpoint=None)

            if exit_code == 0:
                print(f"Task generation job '{self.config.job_name}' completed successfully")
            else:
                print(f"Task generation job '{self.config.job_name}' failed with code {exit_code}")

            return exit_code

        except Exception as e:
            print(f"Task generation job failed with exception: {e}", file=sys.stderr)
            raise

    def _run_with_vllm(self) -> int:
        """Run task generation with managed Ray cluster and vLLM server."""
        from hpc.ray_utils import (
            RayCluster,
            RayClusterConfig,
            compute_ray_memory_from_slurm,
            DEFAULT_OBJECT_STORE_MEMORY_BYTES,
        )
        from hpc.vllm_utils import VLLMServer, VLLMConfig
        from hpc.model_utils import is_gpt_oss_model, setup_gpt_oss_tiktoken

        hpc = self._get_hpc()
        # Stash proxychains binary for wrapping datagen script (needed on no-internet clusters like JSC)
        self._proxychains_binary = getattr(hpc, "proxychains_binary", "") or ""
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", self.config.num_nodes))

        # Use config values (from CLI overrides) instead of cluster defaults
        gpus_per_node = self.config.gpus_per_node or hpc.gpus_per_node
        cpus_per_node = self.config.cpus_per_node or hpc.cpus_per_node

        # Compute Ray memory limit from SLURM allocation (prevents OOM from over-detection)
        ray_memory = compute_ray_memory_from_slurm()
        if ray_memory:
            print(f"[TaskgenJobRunner] Ray memory limit: {ray_memory / (1024**3):.1f} GB", flush=True)

        ray_cfg = RayClusterConfig(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            cpus_per_node=cpus_per_node,
            ray_port=self.config.ray_port,
            srun_export_env=hpc.get_srun_export_env(),
            ray_env_vars=hpc.get_ray_env_vars(),
            memory_per_node=ray_memory,
            object_store_memory=int(self.config.ray_object_store_gb * 1024 * 1024 * 1024),
            disable_cpu_bind=getattr(hpc, "disable_cpu_bind", False),
            gpu_bind=getattr(hpc, "gpu_bind", "none"),
            proxychains_binary=self._proxychains_binary or None,
        )

        model_path = self.config.vllm_model_path or ""

        # Setup tiktoken encodings for GPT-OSS models
        extra_env_vars = {}
        if is_gpt_oss_model(model_path):
            _, tiktoken_env = setup_gpt_oss_tiktoken()
            extra_env_vars.update(tiktoken_env)

        vllm_cfg = VLLMConfig(
            model_path=model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            data_parallel_size=self.config.data_parallel_size,
            api_port=self.config.api_port,
            endpoint_json_path=self.config.endpoint_json_path,
            health_max_attempts=self.config.health_max_attempts,
            health_retry_delay=self.config.health_retry_delay,
            server_config=self.config.vllm_server_config,  # Pass through YAML config
        )

        log_dir = Path(self.config.experiments_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        vllm_log = log_dir / f"{self.config.job_name}_vllm.log"

        with RayCluster.from_slurm(ray_cfg) as ray_cluster:
            vllm_server = VLLMServer(
                config=vllm_cfg,
                ray_cluster=ray_cluster,
                log_path=vllm_log,
                extra_env_vars=extra_env_vars if extra_env_vars else None,
            )
            with vllm_server:
                return self._run_datagen(endpoint=vllm_server.endpoint)

    def _run_datagen(self, endpoint: Optional[str]) -> int:
        """Execute the data generation script."""
        script_path = Path(self.config.datagen_script)
        if not script_path.exists():
            print(f"Error: Datagen script not found: {script_path}", file=sys.stderr)
            return 1

        cmd = [
            sys.executable,
            str(script_path),
            "--stage", "tasks",
        ]

        if self.config.output_dir:
            cmd.extend(["--output-dir", self.config.output_dir])

        if self.config.input_dir:
            cmd.extend(["--input-dir", self.config.input_dir])

        if self.config.target_repo:
            cmd.extend(["--target-repo", self.config.target_repo])

        if self.config.datagen_config_path:
            cmd.extend(["--config", self.config.datagen_config_path])

        if endpoint:
            cmd.extend(["--endpoint", endpoint])

        if self.config.disable_verification:
            cmd.append("--disable-verification")

        # Add extra args
        if self.config.extra_args:
            extra_tokens = shlex.split(self.config.extra_args)
            cmd.extend(extra_tokens)

        # Wrap with proxychains on no-internet clusters (e.g., JSC)
        proxychains_binary = getattr(self, "_proxychains_binary", "")
        proxychains_conf = os.environ.get("PROXYCHAINS_CONF_FILE", "") if proxychains_binary else ""
        if proxychains_binary and proxychains_conf:
            print(f"[TaskgenJobRunner] Using proxychains: {proxychains_binary} -f {proxychains_conf}", flush=True)
            cmd = [proxychains_binary, "-f", proxychains_conf] + cmd
        elif proxychains_binary:
            print(f"[TaskgenJobRunner] Using proxychains: {proxychains_binary} (no conf file)", flush=True)
            cmd = [proxychains_binary] + cmd

        print(f"Running datagen command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode


@dataclass
class TracegenJobConfig:
    """Configuration for a trace generation job (serialized to JSON for sbatch)."""

    job_name: str
    harbor_config: str
    trace_script: str
    experiments_dir: str
    cluster_name: str = ""

    # Input/output settings
    tasks_input_path: str = ""
    output_dir: Optional[str] = None
    target_repo: str = ""

    # Engine settings
    engine: str = "vllm_local"
    datagen_config_path: Optional[str] = None

    # vLLM settings
    needs_vllm: bool = True
    vllm_model_path: Optional[str] = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    endpoint_json_path: Optional[str] = None
    ray_port: int = 6379
    api_port: int = 8000
    vllm_server_config: Dict[str, Any] = field(default_factory=dict)  # Raw vllm_server config from YAML

    # Health check settings
    health_max_attempts: int = 120
    health_retry_delay: int = 15

    # Harbor settings
    model: str = ""
    served_model_id: Optional[str] = None
    agent: str = ""
    trace_env: str = "daytona"
    n_concurrent: int = 64
    n_attempts: int = 1

    # Agent kwargs (serialized as JSON)
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Upload settings
    upload_username: str = ""
    hf_repo_id: Optional[str] = None
    hf_private: bool = False
    hf_episodes: str = "last"

    # Resource allocation (from CLI overrides, None = use HPC cluster defaults)
    num_nodes: int = 1
    gpus_per_node: Optional[int] = None
    cpus_per_node: Optional[int] = None

    # Pinggy tunnel settings (for cloud backends that can't reach local vLLM)
    pinggy_persistent_url: Optional[str] = None
    pinggy_token: Optional[str] = None

    # Ray object store size in GB (default: 40)
    ray_object_store_gb: float = 40.0

    # Chunking settings (for splitting large task sets across parallel jobs)
    chunk_size: Optional[int] = None
    chunk_index: Optional[int] = None
    num_chunks: Optional[int] = None


class TracegenJobRunner:
    """Runs trace generation jobs with optional vLLM management.

    This class encapsulates the trace generation logic that was previously
    spread across 600+ lines of sbatch scripts.

    Usage (from sbatch):
        python -m hpc.datagen_launch_utils --mode tracegen --config /path/to/config.json
    """

    def __init__(self, config: TracegenJobConfig):
        self.config = config
        self._hpc = None
        self._proxychains_binary = ""

    def _get_hpc(self):
        """Lazy-load HPC configuration."""
        if self._hpc is None:
            from hpc.hpc import detect_hpc, clusters
            if self.config.cluster_name:
                for c in clusters:
                    if c.name.lower() == self.config.cluster_name.lower():
                        self._hpc = c
                        break
                if self._hpc is None:
                    raise ValueError(f"Unknown cluster: {self.config.cluster_name}")
            else:
                self._hpc = detect_hpc()
            # Stash proxychains binary for wrapping commands on no-internet clusters
            self._proxychains_binary = getattr(self._hpc, "proxychains_binary", "") or ""
        return self._hpc

    @staticmethod
    def _is_task_folder_dir(path: Path) -> bool:
        """Check if path is a directory of pre-extracted task folders (with instruction.md)."""
        if not path.is_dir():
            return False
        # Quick check: any instruction.md under the directory?
        return any(path.rglob("instruction.md"))

    def _slice_task_folders(self, tasks_path: str, chunk_index: int, chunk_size: int) -> str:
        """Slice a directory of pre-extracted task folders for this chunk.

        Creates a chunk directory with symlinks to the selected task folders.
        Returns the path to the chunk directory.
        """
        p = Path(tasks_path)
        # Collect all task directories (containing instruction.md), sorted for determinism
        task_dirs = sorted(
            d.parent for d in p.rglob("instruction.md") if d.is_file()
        )
        total = len(task_dirs)
        start = chunk_index * chunk_size
        end = min(start + chunk_size, total)
        if start >= total:
            print(f"[chunk] Warning: chunk {chunk_index} start ({start}) >= total task folders ({total}), nothing to process")
            end = start

        chunk_dir = Path(self.config.experiments_dir) / "task_chunks" / f"chunk_{chunk_index}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        selected = task_dirs[start:end]
        for task_dir in selected:
            link = chunk_dir / task_dir.name
            if not link.exists():
                link.symlink_to(task_dir.resolve())

        print(f"[chunk] Chunk {chunk_index}/{self.config.num_chunks}: "
              f"task folders [{start}:{end}] ({len(selected)} of {total}) linked in {chunk_dir}")
        return str(chunk_dir)

    def _slice_hf_dataset(self, tasks_path: str, chunk_index: int, chunk_size: int) -> str:
        """Slice an HF dataset (Arrow/Parquet/JSONL) for this chunk.

        Saves the chunk to disk and returns the path.
        """
        from datasets import load_from_disk, load_dataset

        try:
            ds = load_from_disk(tasks_path)
            if hasattr(ds, "keys"):
                split = list(ds.keys())[0]
                ds = ds[split]
        except Exception:
            ds = load_dataset(tasks_path, split="train")

        total = len(ds)
        start = chunk_index * chunk_size
        end = min(start + chunk_size, total)
        if start >= total:
            print(f"[chunk] Warning: chunk {chunk_index} start ({start}) >= total tasks ({total}), nothing to process")
            end = start

        ds_chunk = ds.select(range(start, end))

        chunk_dir = Path(self.config.experiments_dir) / "task_chunks" / f"chunk_{chunk_index}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ds_chunk.save_to_disk(str(chunk_dir))

        print(f"[chunk] Chunk {chunk_index}/{self.config.num_chunks}: "
              f"tasks [{start}:{end}] ({len(ds_chunk)} of {total}) saved to {chunk_dir}")
        return str(chunk_dir)

    def _maybe_slice_tasks_for_chunk(self) -> None:
        """If this is a chunked job, slice tasks_input_path to this chunk's subset.

        Supports both HF datasets (Arrow/Parquet/JSONL) and pre-extracted task
        folder directories (containing instruction.md markers).
        """
        if self.config.chunk_index is None or not self.config.chunk_size:
            return

        chunk_index = self.config.chunk_index
        chunk_size = self.config.chunk_size
        tasks_path = self.config.tasks_input_path
        if not tasks_path:
            print(f"[chunk] Warning: chunk_index={chunk_index} set but no tasks_input_path, skipping slice")
            return

        if self._is_task_folder_dir(Path(tasks_path)):
            self.config.tasks_input_path = self._slice_task_folders(tasks_path, chunk_index, chunk_size)
        else:
            self.config.tasks_input_path = self._slice_hf_dataset(tasks_path, chunk_index, chunk_size)

    def run(self) -> int:
        """Execute the trace generation job.

        Returns:
            Exit code (0 for success)
        """
        print(f"=== TracegenJobRunner: {self.config.job_name} ===")

        try:
            # Ensure HPC is loaded (sets _proxychains_binary for no-internet clusters)
            self._get_hpc()

            # If this is a chunked job, slice tasks to this chunk's subset
            self._maybe_slice_tasks_for_chunk()

            if self.config.needs_vllm:
                exit_code = self._run_with_vllm()
            else:
                exit_code = self._run_harbor(endpoint=None)

            if exit_code == 0:
                print(f"Trace generation job '{self.config.job_name}' completed successfully")
                # Attempt HF upload after successful Harbor run
                self._maybe_upload_traces()
            else:
                print(f"Trace generation job '{self.config.job_name}' failed with code {exit_code}")

            return exit_code

        except Exception as e:
            print(f"Trace generation job failed with exception: {e}", file=sys.stderr)
            raise

    def _maybe_upload_traces(self) -> None:
        """Upload traces to HuggingFace after Harbor completes.

        On no-internet clusters (JSC), the upload runs as a subprocess wrapped
        with proxychains so that HuggingFace Hub API calls are routed through
        the SSH tunnel proxy.  On clusters with direct internet access the
        upload runs in-process for simplicity.
        """
        if not self.config.hf_repo_id:
            print("[upload] No HF repo configured; skipping upload.")
            return

        # Determine job directory
        jobs_dir = Path(self.config.experiments_dir) / "trace_jobs"
        job_dir = jobs_dir / self.config.job_name
        if not job_dir.exists():
            print(f"[upload] Job directory {job_dir} does not exist; skipping upload.")
            return

        # On no-internet clusters using proxychains binary mode (e.g., Jupiter ARM),
        # LD_PRELOAD is NOT inherited so in-process HTTP calls have no proxy.
        # Run the upload as a subprocess wrapped with proxychains instead.
        proxychains_binary = getattr(self, "_proxychains_binary", "")
        proxychains_conf = os.environ.get("PROXYCHAINS_CONF_FILE", "") if proxychains_binary else ""

        if proxychains_binary:
            self._upload_traces_via_subprocess(job_dir, proxychains_binary, proxychains_conf)
        else:
            self._upload_traces_in_process(job_dir)

    def _upload_traces_via_subprocess(
        self, job_dir: Path, proxychains_binary: str, proxychains_conf: str
    ) -> None:
        """Upload traces as a proxychains-wrapped subprocess (for no-internet clusters)."""
        cmd = [
            sys.executable, "-c",
            "import sys, os; "
            "sys.path.insert(0, os.environ.get('OT_AGENT', '.')); "
            "from hpc.launch_utils import upload_traces_to_hf; "
            f"upload_traces_to_hf("
            f"job_dir={str(job_dir)!r}, "
            f"hf_repo_id={self.config.hf_repo_id!r}, "
            f"hf_private={self.config.hf_private!r}, "
            f"hf_episodes={self.config.hf_episodes!r})"
        ]
        if proxychains_binary and proxychains_conf:
            print(f"[upload] Using proxychains: {proxychains_binary} -f {proxychains_conf}", flush=True)
            cmd = [proxychains_binary, "-f", proxychains_conf] + cmd
        elif proxychains_binary:
            print(f"[upload] Using proxychains: {proxychains_binary}", flush=True)
            cmd = [proxychains_binary] + cmd

        print(f"[upload] Uploading traces via subprocess to {self.config.hf_repo_id}", flush=True)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0:
                print(f"[upload] Subprocess upload failed (exit {result.returncode})", file=sys.stderr)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
            else:
                print(f"[upload] Subprocess upload completed successfully", flush=True)
        except subprocess.TimeoutExpired:
            print("[upload] Subprocess upload timed out after 600s", file=sys.stderr)
        except Exception as e:
            print(f"[upload] Subprocess upload error: {e}", file=sys.stderr)

    def _upload_traces_in_process(self, job_dir: Path) -> None:
        """Upload traces directly in-process (for clusters with internet access)."""
        from hpc.launch_utils import upload_traces_to_hf

        try:
            hf_url = upload_traces_to_hf(
                job_dir=job_dir,
                hf_repo_id=self.config.hf_repo_id,
                hf_private=self.config.hf_private,
                hf_episodes=self.config.hf_episodes,
            )
            if hf_url:
                print(f"[upload] HuggingFace upload successful: {hf_url}")
        except Exception as e:
            print(f"[upload] HuggingFace upload error: {e}", file=sys.stderr)

    def _run_with_vllm(self) -> int:
        """Run trace generation with managed Ray cluster and vLLM server."""
        from hpc.ray_utils import (
            RayCluster,
            RayClusterConfig,
            compute_ray_memory_from_slurm,
            DEFAULT_OBJECT_STORE_MEMORY_BYTES,
        )
        from hpc.vllm_utils import VLLMServer, VLLMConfig
        from hpc.model_utils import is_gpt_oss_model, setup_gpt_oss_tiktoken

        hpc = self._get_hpc()
        # Stash proxychains binary for wrapping Harbor CLI (needed on no-internet clusters like JSC)
        self._proxychains_binary = getattr(hpc, "proxychains_binary", "") or ""
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", self.config.num_nodes))

        # Use config values (from CLI overrides) instead of cluster defaults
        gpus_per_node = self.config.gpus_per_node or hpc.gpus_per_node
        cpus_per_node = self.config.cpus_per_node or hpc.cpus_per_node

        # Compute Ray memory limit from SLURM allocation (prevents OOM from over-detection)
        ray_memory = compute_ray_memory_from_slurm()
        if ray_memory:
            print(f"[TracegenJobRunner] Ray memory limit: {ray_memory / (1024**3):.1f} GB", flush=True)

        ray_cfg = RayClusterConfig(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            cpus_per_node=cpus_per_node,
            ray_port=self.config.ray_port,
            srun_export_env=hpc.get_srun_export_env(),
            ray_env_vars=hpc.get_ray_env_vars(),
            memory_per_node=ray_memory,
            object_store_memory=int(self.config.ray_object_store_gb * 1024 * 1024 * 1024),
            disable_cpu_bind=getattr(hpc, "disable_cpu_bind", False),
            gpu_bind=getattr(hpc, "gpu_bind", "none"),
            proxychains_binary=self._proxychains_binary or None,
        )

        raw_model_path = self.config.vllm_model_path or self.config.model
        model_path = strip_hosted_vllm_alias(raw_model_path) or raw_model_path

        # Setup tiktoken encodings for GPT-OSS models
        extra_env_vars = {}
        if is_gpt_oss_model(model_path):
            _, tiktoken_env = setup_gpt_oss_tiktoken()
            extra_env_vars.update(tiktoken_env)

        vllm_cfg = VLLMConfig(
            model_path=model_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            data_parallel_size=self.config.data_parallel_size,
            api_port=self.config.api_port,
            endpoint_json_path=self.config.endpoint_json_path,
            custom_model_name=self.config.served_model_id,
            health_max_attempts=self.config.health_max_attempts,
            health_retry_delay=self.config.health_retry_delay,
            server_config=self.config.vllm_server_config,  # Pass through YAML config
        )

        log_dir = Path(self.config.experiments_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        vllm_log = log_dir / f"{self.config.job_name}_vllm.log"

        with RayCluster.from_slurm(ray_cfg) as ray_cluster:
            # Enable distributed containers for multi-node local backend jobs
            # This allows Harbor to spread container workload across all Ray nodes
            local_backends = {"podman_hpc", "docker", "apptainer"}
            if ray_cluster.total_nodes > 1 and self.config.trace_env in local_backends:
                os.environ["HARBOR_DISTRIBUTED_CONTAINERS"] = "1"
                print(f"[TracegenJobRunner] Enabled distributed {self.config.trace_env} "
                      f"across {ray_cluster.total_nodes} nodes", flush=True)

            vllm_server = VLLMServer(
                config=vllm_cfg,
                ray_cluster=ray_cluster,
                log_path=vllm_log,
                extra_env_vars=extra_env_vars if extra_env_vars else None,
            )
            with vllm_server:
                # Check if we need Pinggy tunnel for cloud backends with installed agents
                from hpc.pinggy_utils import (
                    needs_pinggy_tunnel,
                    PinggyTunnel,
                    PinggyConfig,
                    parse_endpoint_host_port,
                )

                # Evaluate Pinggy conditions with diagnostic logging
                has_url = bool(self.config.pinggy_persistent_url)
                has_token = bool(self.config.pinggy_token)
                needs_tunnel = needs_pinggy_tunnel(self.config.agent, self.config.trace_env)
                use_pinggy = has_url and has_token and needs_tunnel

                print(f"[TracegenJobRunner] Pinggy check: url={has_url}, token={has_token}, "
                      f"needs_tunnel={needs_tunnel} (agent={self.config.agent}, env={self.config.trace_env})")
                print(f"[TracegenJobRunner] use_pinggy={use_pinggy}, vllm_endpoint={vllm_server.endpoint}")

                if use_pinggy:
                    # Parse the vLLM endpoint to get the actual host:port
                    # (vLLM may bind to a specific IP, not localhost)
                    local_host, local_port = parse_endpoint_host_port(vllm_server.endpoint)
                    print(f"[TracegenJobRunner] Starting Pinggy tunnel: {local_host}:{local_port} -> {self.config.pinggy_persistent_url}")
                    pinggy_cfg = PinggyConfig(
                        persistent_url=self.config.pinggy_persistent_url,
                        token=self.config.pinggy_token,
                        local_port=local_port,
                        local_host=local_host,
                    )
                    pinggy_log = log_dir / f"{self.config.job_name}_pinggy.log"
                    pinggy_tunnel = PinggyTunnel(pinggy_cfg, log_path=pinggy_log)

                    with pinggy_tunnel:
                        # Use Pinggy's public endpoint instead of local vLLM endpoint
                        public_endpoint = pinggy_tunnel.public_endpoint
                        print(f"[TracegenJobRunner] Using Pinggy endpoint for Harbor: {public_endpoint}")
                        return self._run_harbor(endpoint=public_endpoint)
                else:
                    # Use local vLLM endpoint directly
                    print(f"[TracegenJobRunner] Using local vLLM endpoint for Harbor: {vllm_server.endpoint}")
                    return self._run_harbor(endpoint=vllm_server.endpoint)

    def _run_harbor(self, endpoint: Optional[str]) -> int:
        """Execute the Harbor CLI for trace generation."""
        from hpc.harbor_utils import build_harbor_command, load_harbor_config, build_endpoint_meta

        # Build endpoint metadata for vLLM
        endpoint_meta = build_endpoint_meta(endpoint) if endpoint else None

        # Load harbor config data for agent kwargs extraction
        harbor_config_data = load_harbor_config(self.config.harbor_config)

        # Set jobs_dir inside experiments folder (not repo root)
        jobs_dir = str(Path(self.config.experiments_dir) / "trace_jobs")

        # Build command using shared utility
        # Pass config.agent_kwargs as extra_agent_kwargs (from datagen config + CLI overrides)
        cmd = build_harbor_command(
            harbor_binary="harbor",
            harbor_config_path=self.config.harbor_config,
            harbor_config_data=harbor_config_data,
            job_name=self.config.job_name,
            agent_name=self.config.agent,
            model_name=self.config.model,
            env_type=self.config.trace_env,
            n_concurrent=self.config.n_concurrent,
            n_attempts=self.config.n_attempts,
            endpoint_meta=endpoint_meta,
            agent_kwarg_overrides=[],  # CLI overrides already merged into config.agent_kwargs
            harbor_extra_args=[],
            dataset_path=self.config.tasks_input_path,
            jobs_dir=jobs_dir,
            extra_agent_kwargs=self.config.agent_kwargs or None,
        )

        # Wrap with proxychains on no-internet clusters (e.g., JSC)
        # This routes Daytona API calls through the SSH tunnel proxy
        proxychains_binary = getattr(self, "_proxychains_binary", "")
        proxychains_conf = os.environ.get("PROXYCHAINS_CONF_FILE", "") if proxychains_binary else ""
        if proxychains_binary and proxychains_conf:
            print(f"[TracegenJobRunner] Using proxychains: {proxychains_binary} -f {proxychains_conf}", flush=True)
            cmd = [proxychains_binary, "-f", proxychains_conf] + cmd
        elif proxychains_binary:
            print(f"[TracegenJobRunner] Using proxychains: {proxychains_binary} (no conf file)", flush=True)
            cmd = [proxychains_binary] + cmd

        print(f"Running Harbor command: {' '.join(cmd)}")
        sys.stdout.flush()

        # Use PTY-based runner for proper Harbor output handling
        from hpc.cli_utils import run_harbor_cli
        try:
            return run_harbor_cli(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Harbor exited with code {e.returncode}", file=sys.stderr)
            return e.returncode


def run_datagen_job_main():
    """Entry point for running datagen jobs from sbatch scripts.

    Usage:
        python -m hpc.datagen_launch_utils --mode taskgen --config /path/to/config.json
        python -m hpc.datagen_launch_utils --mode tracegen --config /path/to/config.json
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run datagen job from config JSON")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["taskgen", "tracegen"],
        help="Job mode: taskgen or tracegen",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to job config JSON file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config_data = json.loads(config_path.read_text())

    if args.mode == "taskgen":
        config = TaskgenJobConfig(**config_data)
        runner = TaskgenJobRunner(config)
    else:  # tracegen
        config = TracegenJobConfig(**config_data)
        runner = TracegenJobRunner(config)

    exit_code = runner.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    run_datagen_job_main()


__all__ = [
    # Constants
    "DATAGEN_CONFIG_DIR",
    "HARBOR_CONFIG_DIR",
    "DEFAULT_RAY_CGRAPH_TIMEOUT",
    "DEFAULT_RAY_CGRAPH_MAX_INFLIGHT",
    # Re-exports from launch_utils (for backwards compatibility)
    "derive_datagen_job_name",
    "default_vllm_endpoint_path",
    # Config utilities
    "_normalize_cli_args",
    "_prepare_datagen_configuration",
    "resolve_harbor_config_path",
    # Universal launcher
    "launch_datagen_job_v2",
    # Job runner classes for universal sbatch scripts
    "TaskgenJobConfig",
    "TaskgenJobRunner",
    "TracegenJobConfig",
    "TracegenJobRunner",
    "run_datagen_job_main",
]
