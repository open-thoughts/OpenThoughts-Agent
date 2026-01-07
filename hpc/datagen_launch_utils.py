"""Shared utilities for datagen-oriented HPC launches."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

from data.generation.utils import load_datagen_config, resolve_engine_runtime
from hpc.launch_utils import (
    default_vllm_endpoint_path,
    derive_datagen_job_name,  # Re-exported for backwards compatibility
    launch_sbatch,
    cleanup_endpoint_file,
    normalize_cli_args,
    resolve_config_path,
    coerce_positive_int,
    build_sbatch_directives,
    generate_served_model_id,
    hosted_vllm_alias,
    strip_hosted_vllm_alias,
)

# Backward compatibility aliases
_normalize_cli_args = normalize_cli_args
_coerce_positive_int = coerce_positive_int

DIRENV = os.path.dirname(__file__)
DATAGEN_CONFIG_DIR = os.path.join(DIRENV, "datagen_yaml")
HARBOR_CONFIG_DIR = os.path.join(DIRENV, "harbor_yaml")
DEFAULT_RAY_CGRAPH_TIMEOUT = os.environ.get("RAY_CGRAPH_TIMEOUT_DEFAULT", "86500")
DEFAULT_RAY_CGRAPH_MAX_INFLIGHT = os.environ.get("RAY_CGRAPH_MAX_INFLIGHT_DEFAULT", "")
HARBOR_MODEL_PLACEHOLDER = "placeholder/override-at-runtime"


def _validate_sbatch_templates(hpc_obj) -> None:
    """Validate that universal sbatch templates exist.

    Since Phase 3 refactoring, we use universal templates for all clusters.
    """
    if getattr(hpc_obj, "local_mode", False):
        print(f"Local execution detected for {hpc_obj.name}; skipping sbatch template validation.")
        return

    # Validate universal templates exist
    universal_templates = [
        Path(__file__).parent / "sbatch_data" / "universal_taskgen.sbatch",
        Path(__file__).parent / "sbatch_data" / "universal_tracegen.sbatch",
        Path(__file__).parent / "sbatch_eval" / "universal_eval.sbatch",
    ]

    missing = [str(t) for t in universal_templates if not t.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing universal sbatch templates: " + ", ".join(missing)
        )


def resolve_datagen_config_path(raw_value: str) -> Path:
    """Resolve ``raw_value`` to an absolute datagen config path."""
    return resolve_config_path(raw_value, DATAGEN_CONFIG_DIR, "datagen")


def resolve_harbor_config_path(raw_value: str) -> Path:
    """Resolve ``raw_value`` to an absolute Harbor job config path."""
    return resolve_config_path(raw_value, HARBOR_CONFIG_DIR, "harbor job")


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
        engine_cfg.model = trace_model_override
        try:
            loaded.raw.engine.model = trace_model_override
        except AttributeError:
            pass

        engine_type = (engine_cfg.type or "").lower()
        if engine_type == "vllm_local" and getattr(engine_cfg, "vllm_local", None):
            engine_cfg.vllm_local.model_name = trace_model_override  # type: ignore[assignment]
            try:
                loaded.raw.engine.vllm_local.model_name = trace_model_override  # type: ignore[attr-defined]
            except AttributeError:
                pass

        vllm_cfg = loaded.config.vllm_server
        if vllm_cfg:
            vllm_cfg.model_path = trace_model_override
            try:
                loaded.raw.vllm_server.model_path = trace_model_override  # type: ignore[attr-defined]
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
    runtime_model = runtime.engine_kwargs.get("model") or runtime.engine_kwargs.get("model_name")
    if runtime_model:
        exp_args["datagen_model"] = runtime_model
    elif trace_model_override:
        exp_args["datagen_model"] = trace_model_override
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


def launch_datagen_job_v2(exp_args: dict, hpc) -> None:
    """Launch datagen job using the new universal template system.

    1. Creating TaskgenJobConfig and/or TracegenJobConfig from exp_args
    2. Writing configs to JSON
    3. Using universal_taskgen.sbatch and universal_tracegen.sbatch templates
    4. Submitting the jobs
    """
    # asdict and launch_sbatch are imported at module level

    print("\n=== DATA GENERATION MODE (Universal Launcher) ===")

    hpc_name = str(getattr(hpc, "name", "")).lower()
    if hpc_name == "nyutorch":
        raise RuntimeError("Datagen jobs are not supported on the NYU Torch cluster.")

    # Determine what to run
    task_enabled = str(exp_args.get("enable_task_gen", True)).lower() not in {"false", "0", "no", "none"}
    trace_enabled = str(exp_args.get("enable_trace_gen", False)).lower() not in {"false", "0", "no", "none"}

    if not task_enabled and not trace_enabled:
        raise ValueError("Enable at least one of task or trace generation")

    if task_enabled and not exp_args.get("datagen_script"):
        raise ValueError("--datagen-script is required for task generation")

    # Resolve paths
    experiments_subdir = exp_args.get("experiments_dir") or "experiments"
    experiments_abs = Path(experiments_subdir).expanduser().resolve()
    sbatch_dir = experiments_abs / "sbatch"
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = experiments_abs / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = experiments_abs / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    job_name = exp_args.get("job_name")
    if not job_name:
        job_name = derive_datagen_job_name(exp_args)

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

        task_config = TaskgenJobConfig(
            job_name=f"{job_name}_tasks",
            datagen_script=exp_args.get("datagen_script") or "",
            experiments_dir=experiments_subdir,
            cluster_name=hpc.name,
            output_dir=exp_args.get("datagen_output_dir"),
            input_dir=exp_args.get("datagen_input_dir"),
            target_repo=exp_args.get("datagen_target_repo"),
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
        task_config_path = configs_dir / f"{job_name}_taskgen_config.json"
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
        }

        sbatch_text = template_text
        for key, value in substitutions.items():
            sbatch_text = sbatch_text.replace("{" + key + "}", value)

        task_sbatch_output = sbatch_dir / f"{job_name}_taskgen.sbatch"
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
            raise ValueError("--trace-target-repo is required when enabling trace generation")

        harbor_config = exp_args.get("trace_harbor_config")
        if not harbor_config:
            raise ValueError("--trace-harbor-config is required for trace generation")
        harbor_config_resolved = str(resolve_harbor_config_path(harbor_config))

        tasks_input_path = exp_args.get("trace_input_path")
        if not tasks_input_path and task_enabled:
            tasks_input_path = exp_args.get("datagen_output_dir") or str(
                experiments_abs / "outputs" / "tasks"
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

        agent_kwargs = exp_args.get("_datagen_extra_agent_kwargs") or {}
        if exp_args.get("trace_agent_kwargs"):
            if isinstance(exp_args["trace_agent_kwargs"], dict):
                agent_kwargs.update(exp_args["trace_agent_kwargs"])
            else:
                try:
                    agent_kwargs.update(json.loads(str(exp_args["trace_agent_kwargs"])))
                except json.JSONDecodeError:
                    pass

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
            trace_env=exp_args.get("trace_env") or "daytona",
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
        trace_config_path = configs_dir / f"{job_name}_tracegen_config.json"
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
        }

        sbatch_text = template_text
        for key, value in substitutions.items():
            sbatch_text = sbatch_text.replace("{" + key + "}", value)

        trace_sbatch_output = sbatch_dir / f"{job_name}_tracegen.sbatch"
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
        from hpc.ray_utils import RayCluster, RayClusterConfig
        from hpc.vllm_utils import VLLMServer, VLLMConfig

        hpc = self._get_hpc()
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", self.config.num_nodes))

        # Use config values (from CLI overrides) instead of cluster defaults
        gpus_per_node = self.config.gpus_per_node or hpc.gpus_per_node
        cpus_per_node = self.config.cpus_per_node or hpc.cpus_per_node

        ray_cfg = RayClusterConfig(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            cpus_per_node=cpus_per_node,
            ray_port=self.config.ray_port,
            srun_export_env=hpc.get_srun_export_env(),
            ray_env_vars=hpc.get_ray_env_vars(),
        )

        vllm_cfg = VLLMConfig(
            model_path=self.config.vllm_model_path or "",
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
        from hpc.ray_utils import RayCluster, RayClusterConfig
        from hpc.vllm_utils import VLLMServer, VLLMConfig

        hpc = self._get_hpc()
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", self.config.num_nodes))

        # Use config values (from CLI overrides) instead of cluster defaults
        gpus_per_node = self.config.gpus_per_node or hpc.gpus_per_node
        cpus_per_node = self.config.cpus_per_node or hpc.cpus_per_node

        ray_cfg = RayClusterConfig(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            cpus_per_node=cpus_per_node,
            ray_port=self.config.ray_port,
            srun_export_env=hpc.get_srun_export_env(),
            ray_env_vars=hpc.get_ray_env_vars(),
        )

        raw_model_path = self.config.vllm_model_path or self.config.model
        model_path = strip_hosted_vllm_alias(raw_model_path) or raw_model_path

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
            )
            with vllm_server:
                return self._run_harbor(endpoint=vllm_server.endpoint)

    def _run_harbor(self, endpoint: Optional[str]) -> int:
        """Execute the Harbor CLI for trace generation."""
        cmd = [
            "harbor",
            "jobs",
            "start",
            "--config",
            self.config.harbor_config,
            "--job-name",
            self.config.job_name,
            "--env",
            self.config.trace_env,
            "--n-concurrent",
            str(self.config.n_concurrent),
            "--n-attempts",
            str(self.config.n_attempts),
        ]

        if self.config.agent:
            cmd.extend(["--agent", self.config.agent])

        if self.config.model:
            cmd.extend(["--model", self.config.model])

        if self.config.tasks_input_path:
            cmd.extend(["-p", self.config.tasks_input_path])

        # Build agent kwargs
        agent_kwargs = dict(self.config.agent_kwargs)
        if endpoint:
            agent_kwargs["api_base"] = endpoint
            metrics_endpoint = endpoint.replace("/v1", "/metrics")
            agent_kwargs["metrics_endpoint"] = metrics_endpoint

        for key, value in agent_kwargs.items():
            if isinstance(value, (dict, list)):
                cmd.extend(["--agent-kwarg", f"{key}={json.dumps(value)}"])
            else:
                cmd.extend(["--agent-kwarg", f"{key}={value}"])

        # Set jobs_dir inside experiments folder (not repo root)
        jobs_dir = str(Path(self.config.experiments_dir) / "trace_jobs")
        cmd.extend(["--jobs-dir", jobs_dir])

        # Standard export flags
        cmd.extend([
            "--export-traces",
            "--export-verifier-metadata",
            "--export-episodes", "last",
        ])

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
    "resolve_datagen_config_path",
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
