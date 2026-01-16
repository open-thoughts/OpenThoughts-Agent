"""Shared utilities for datagen-oriented HPC launches."""

from __future__ import annotations

import json
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
    hosted_vllm_alias,
    strip_hosted_vllm_alias,
    set_or_pop,
    resolve_job_and_paths,
    substitute_template,
)
from hpc.harbor_utils import (
    get_harbor_env_from_config,
    HARBOR_CONFIG_DIR,
    resolve_harbor_config_path,
)
from hpc.hf_utils import resolve_dataset_path, derive_default_hf_repo_id, sanitize_hf_repo_id

# Backward compatibility aliases
_normalize_cli_args = normalize_cli_args

DIRENV = os.path.dirname(__file__)
DATAGEN_CONFIG_DIR = os.path.join(DIRENV, "datagen_yaml")
DEFAULT_RAY_CGRAPH_TIMEOUT = os.environ.get("RAY_CGRAPH_TIMEOUT_DEFAULT", "86500")
DEFAULT_RAY_CGRAPH_MAX_INFLIGHT = os.environ.get("RAY_CGRAPH_MAX_INFLIGHT_DEFAULT", "")
HARBOR_MODEL_PLACEHOLDER = "placeholder/override-at-runtime"


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
            "conda_activate": hpc.conda_activate or "# No conda activation configured",
            "cluster_env_file": cluster_env_file,
            "config_path": str(task_config_path),
            "email_address": os.environ.get("EMAIL_ADDRESS", ""),
        }

        sbatch_text = substitute_template(template_text, substitutions)

        task_sbatch_output = exp_paths.sbatch / f"{job_name}_taskgen.sbatch"
        task_sbatch_output.write_text(sbatch_text)
        os.chmod(task_sbatch_output, 0o750)

        if exp_args.get("dry_run"):
            print(f"DRY RUN: Taskgen sbatch script written to {task_sbatch_output}")
            task_job_id = "dry_run_task_job_id"
        else:
            task_job_id = launch_sbatch(str(task_sbatch_output))
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
        from hpc.harbor_utils import collect_extra_agent_kwargs
        agent_kwargs = collect_extra_agent_kwargs(
            datagen_extras=exp_args.get("_datagen_extra_agent_kwargs"),
            cli_kwargs=exp_args.get("trace_agent_kwargs"),
        )

        # Convert vllm_cfg dataclass to dict for pass-through (if not already done)
        trace_vllm_server_config = asdict(vllm_cfg) if vllm_cfg else {}

        trace_config = TracegenJobConfig(
            job_name=f"{job_name}_traces",
            harbor_config=harbor_config_resolved,
            trace_script=trace_script or "",
            experiments_dir=experiments_subdir,
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
            agent=exp_args.get("trace_agent_name") or "",
            trace_env=exp_args.get("trace_env") or get_harbor_env_from_config(harbor_config_resolved),
            n_concurrent=int(exp_args.get("trace_n_concurrent") or 64),
            n_attempts=int(exp_args.get("trace_n_attempts") or 3),
            agent_kwargs=agent_kwargs,
            num_nodes=int(exp_args.get("num_nodes") or 1),
            gpus_per_node=gpus_per_node,
            cpus_per_node=cpus_per_node,
            vllm_server_config=trace_vllm_server_config,
            # HF upload settings (use trace_target_repo as default HF repo)
            hf_repo_id=exp_args.get("upload_hf_repo") or trace_target_repo,
            hf_private=bool(exp_args.get("upload_hf_private")),
            hf_episodes=exp_args.get("upload_hf_episodes") or "last",
        )

        # Write trace config JSON
        trace_config_path = exp_paths.configs / f"{job_name}_tracegen_config.json"
        trace_config_path.write_text(json.dumps(asdict(trace_config), indent=2))

        # Load and populate tracegen template
        template_path = Path(__file__).parent / "sbatch_data" / "universal_tracegen.sbatch"
        if not template_path.exists():
            raise FileNotFoundError(f"Universal tracegen template not found: {template_path}")

        template_text = template_path.read_text()

        # Build SBATCH directives using shared utility
        sbatch_directives = build_sbatch_directives(hpc, exp_args)

        substitutions = {
            "time_limit": exp_args.get("time_limit") or "24:00:00",
            "num_nodes": str(exp_args.get("num_nodes") or 1),
            "cpus_per_node": str(exp_args.get("cpus_per_node") or hpc.cpus_per_node),
            "experiments_dir": experiments_subdir,
            "job_name": f"{job_name}_traces",
            "sbatch_extra_directives": "\n".join(sbatch_directives),
            "module_commands": hpc.get_module_commands(),
            "conda_activate": hpc.conda_activate or "# No conda activation configured",
            "cluster_env_file": cluster_env_file,
            "config_path": str(trace_config_path),
            "email_address": os.environ.get("EMAIL_ADDRESS", ""),
        }

        sbatch_text = substitute_template(template_text, substitutions)

        trace_sbatch_output = exp_paths.sbatch / f"{job_name}_tracegen.sbatch"
        trace_sbatch_output.write_text(sbatch_text)
        os.chmod(trace_sbatch_output, 0o750)

        # Set dependency on task job if both are enabled
        dependency = f"afterok:{task_job_id}" if task_enabled and task_job_id and task_job_id != "dry_run_task_job_id" else None

        if exp_args.get("dry_run"):
            print(f"DRY RUN: Tracegen sbatch script written to {trace_sbatch_output}")
        else:
            if dependency:
                job_id = launch_sbatch(str(trace_sbatch_output), dependency=dependency)
            else:
                job_id = launch_sbatch(str(trace_sbatch_output))
            print(f"✓ Trace generation job submitted: {job_id}")


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


class TaskgenJobRunner:
    """Runs task generation jobs with optional vLLM management.

    Usage (from sbatch):
        python -m hpc.datagen_launch_utils --mode taskgen --config /path/to/config.json
    """

    def __init__(self, config: TaskgenJobConfig):
        self.config = config
        self._hpc = None

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
        return self._hpc

    def run(self) -> int:
        """Execute the task generation job.

        Returns:
            Exit code (0 for success)
        """
        print(f"=== TaskgenJobRunner: {self.config.job_name} ===")

        try:
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
            object_store_memory=DEFAULT_OBJECT_STORE_MEMORY_BYTES,
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
    n_attempts: int = 3

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
        return self._hpc

    def run(self) -> int:
        """Execute the trace generation job.

        Returns:
            Exit code (0 for success)
        """
        print(f"=== TracegenJobRunner: {self.config.job_name} ===")

        try:
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
        """Upload traces to HuggingFace after Harbor completes."""
        if not self.config.hf_repo_id:
            print("[upload] No HF repo configured; skipping upload.")
            return

        # Determine job directory
        jobs_dir = Path(self.config.experiments_dir) / "trace_jobs"
        job_dir = jobs_dir / self.config.job_name
        if not job_dir.exists():
            print(f"[upload] Job directory {job_dir} does not exist; skipping upload.")
            return

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
            object_store_memory=DEFAULT_OBJECT_STORE_MEMORY_BYTES,
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
            vllm_server = VLLMServer(
                config=vllm_cfg,
                ray_cluster=ray_cluster,
                log_path=vllm_log,
                extra_env_vars=extra_env_vars if extra_env_vars else None,
            )
            with vllm_server:
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
