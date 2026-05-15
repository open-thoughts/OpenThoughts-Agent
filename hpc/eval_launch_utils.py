"""Utilities for launching Harbor eval jobs via the HPC launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.generation import BaseDataGenerator

from hpc.launch_utils import (
    PROJECT_ROOT,
    resolve_repo_path,
    resolve_workspace_path,
    resolve_config_path,
    default_vllm_endpoint_path,
    get_daytona_api_key_override,
    launch_sbatch,
    _parse_optional_int,
    cleanup_endpoint_file,
    validate_trace_backend,
    build_sbatch_directives,
    generate_served_model_id,
    hosted_vllm_alias,
    strip_hosted_vllm_alias,
    resolve_conda_activate,
    resolve_job_and_paths,
    substitute_template,
    derive_datagen_job_name,
    convert_parquet_to_tasks,
)
from hpc.hf_utils import resolve_hf_repo_id, resolve_dataset_path, is_raw_tasks_directory

# Import Harbor utilities from consolidated module
from hpc.harbor_utils import (
    HARBOR_CONFIG_DIR,
    resolve_harbor_config_path,
    validate_harbor_dataset_slug,
    load_harbor_config,
)

from scripts.harbor.job_config_utils import load_job_config
from hpc.cli_utils import resolve_n_concurrent


def remap_eval_cli_args(cli_args: dict) -> dict:
    """Remap CLI arguments for eval job type.

    Handles argument aliasing where eval jobs use different argument names
    than SFT jobs for the same concepts:
    - --dataset -> --harbor_dataset (for eval, --dataset is the benchmark slug)

    Args:
        cli_args: Parsed CLI arguments dict

    Returns:
        Modified cli_args dict with eval-specific remapping applied
    """
    # Allow --dataset to be used as alias for --harbor_dataset in eval jobs
    # (--dataset is also used by LlamaFactory for SFT training datasets)
    if cli_args.get("dataset") and not cli_args.get("harbor_dataset"):
        cli_args["harbor_dataset"] = cli_args["dataset"]
        # Remove from cli_args so it doesn't get passed to SFT code paths
        cli_args.pop("dataset", None)

    return cli_args



def prepare_eval_configuration(exp_args: dict) -> dict:
    """Normalize eval config inputs prior to sbatch generation."""

    if exp_args.get("_datagen_config_obj") is None:
        raise ValueError("Eval jobs reuse the datagen engine. Provide --datagen-config.")

    harbor_cfg = exp_args.get("trace_harbor_config")
    if not harbor_cfg:
        raise ValueError("Eval jobs require --trace-harbor-config pointing at an eval YAML.")
    resolved_cfg = resolve_harbor_config_path(harbor_cfg)
    if "_eval_" not in resolved_cfg.name:
        raise ValueError(
            f"Eval Harbor YAML '{resolved_cfg.name}' must include '_eval_' in the filename."
        )
    harbor_job = load_job_config(resolved_cfg)
    if not harbor_job.agents:
        raise ValueError(f"Eval Harbor YAML '{resolved_cfg.name}' must define at least one agent.")

    exp_args["_eval_harbor_config_resolved"] = str(resolved_cfg)
    exp_args["_eval_harbor_config"] = harbor_job

    dataset_path = exp_args.get("tasks_input_path")
    harbor_dataset = exp_args.get("harbor_dataset")
    if dataset_path and harbor_dataset:
        raise ValueError(
            "Eval jobs accept either --tasks-input-path or --harbor-dataset, but not both."
        )
    # Preserve original dataset path for benchmark name derivation (before HF resolution)
    original_dataset_path = dataset_path
    if dataset_path:
        # Use shared utility to handle both HF repos and local paths
        resolved_dataset = resolve_dataset_path(dataset_path, verbose=True)
        # Auto-detect parquet datasets and convert to task directories
        if not is_raw_tasks_directory(resolved_dataset):
            resolved_dataset = convert_parquet_to_tasks(resolved_dataset, dataset_path)
        exp_args["_eval_dataset_path_resolved"] = resolved_dataset
        exp_args["tasks_input_path"] = resolved_dataset
    if harbor_dataset:
        slug = harbor_dataset.strip()
        if not slug:
            raise ValueError("--harbor-dataset cannot be empty.")
        validate_harbor_dataset_slug(slug)
        exp_args["harbor_dataset"] = slug

    if not (exp_args.get("harbor_dataset") or exp_args.get("_eval_dataset_path_resolved")):
        raise ValueError(
            "Eval jobs require either --harbor-dataset or --tasks-input-path to specify tasks."
        )

    # Derive eval_benchmark_repo using shared utility
    # Use the ORIGINAL dataset path (e.g., "DCAgent2/my-dataset") not the resolved HF snapshot path
    from hpc.launch_utils import derive_benchmark_repo
    benchmark_repo = derive_benchmark_repo(
        harbor_dataset=exp_args.get("harbor_dataset"),
        dataset_path=original_dataset_path,
        explicit_repo=exp_args.get("eval_benchmark_repo"),
    )
    if benchmark_repo == "unknown-benchmark":
        raise ValueError(
            "Unable to derive eval_benchmark_repo. Provide "
            "--harbor-dataset or --tasks-input-path."
        )
    exp_args["eval_benchmark_repo"] = benchmark_repo
    print(f"[eval] Using benchmark repo: {benchmark_repo}")

    model_name = exp_args.get("trace_model")
    if not model_name:
        vllm_cfg = exp_args.get("_datagen_vllm_server_config")
        if vllm_cfg and getattr(vllm_cfg, "model_path", None):
            model_name = vllm_cfg.model_path
    if not model_name:
        model_name = exp_args.get("datagen_model")
    if not model_name and harbor_job.agents:
        model_name = harbor_job.agents[0].model_name
    if not model_name:
        raise ValueError("Eval jobs require --model (or --trace-model / --datagen-model).")
    exp_args["_eval_model_name"] = model_name
    exp_args["trace_model"] = model_name

    agent_cfg = harbor_job.agents[0]
    # Debug: print what we're getting from the Harbor config
    print(f"[prepare_eval] Harbor agent config: name={agent_cfg.name!r}, import_path={agent_cfg.import_path!r}")
    print(f"[prepare_eval] CLI trace_agent_name={exp_args.get('trace_agent_name')!r}")
    agent_name = (
        exp_args.get("trace_agent_name")
        or agent_cfg.name
        or (agent_cfg.import_path or "terminus-2")
    )
    print(f"[prepare_eval] Resolved agent_name={agent_name!r}")
    exp_args["_eval_agent_name"] = agent_name
    if "trace_agent_name" not in exp_args:
        exp_args["trace_agent_name"] = agent_name

    # Collect extra agent kwargs from datagen config and CLI
    # NOTE: Do NOT include Harbor YAML base kwargs here - merge_agent_kwargs() handles that
    from hpc.harbor_utils import collect_extra_agent_kwargs, derive_vllm_supports_tool_calling
    exp_args["_eval_agent_kwargs"] = collect_extra_agent_kwargs(
        datagen_extras=exp_args.get("_datagen_extra_agent_kwargs"),
        cli_kwargs=exp_args.get("trace_agent_kwargs"),
    )
    if agent_name == "swe-agent":
        vllm_cfg = exp_args.get("_datagen_vllm_server_config")
        supports_tool_calling = derive_vllm_supports_tool_calling(vllm_cfg)
        if supports_tool_calling is not None:
            exp_args["_eval_agent_kwargs"].setdefault(
                "supports_tool_calling", supports_tool_calling
            )

    if exp_args.get("trace_env"):
        eval_env = exp_args["trace_env"]
    else:
        env_cfg = getattr(harbor_job.environment, "type", None)
        if hasattr(env_cfg, "value"):
            eval_env = env_cfg.value
        elif env_cfg:
            eval_env = str(env_cfg)
        else:
            eval_env = "daytona"
    exp_args["_eval_env"] = str(eval_env)

    trace_backend_value = exp_args.get("trace_backend") or exp_args.get("datagen_backend")
    trace_backend = validate_trace_backend(
        trace_backend_value,
        allow_vllm=True,
        job_type="eval",
    )
    exp_args["trace_backend"] = trace_backend

    exp_args["_eval_n_concurrent"] = resolve_n_concurrent(
        exp_args.get("trace_n_concurrent"), harbor_job,
    )

    default_n_attempts = harbor_job.n_attempts or 3
    n_attempts_override = _parse_optional_int(
        exp_args.get("trace_n_attempts"),
        "--trace_n_attempts",
    )
    n_attempts_int = n_attempts_override or default_n_attempts
    exp_args["_eval_n_attempts"] = max(1, int(n_attempts_int))

    return exp_args


# =============================================================================
# EvalJobRunner - New universal job runner for Phase 2 refactoring
# =============================================================================


@dataclass
class EvalJobConfig:
    """Configuration for an eval job (serialized to JSON for sbatch)."""

    job_name: str
    harbor_config: str
    model: str
    agent: str
    served_model_id: Optional[str] = None
    dataset: Optional[str] = None
    dataset_path: Optional[str] = None
    n_concurrent: int = 64
    n_attempts: int = 3
    eval_benchmark_repo: str = ""
    eval_env: str = "daytona"
    experiments_dir: str = "experiments"
    cluster_name: str = ""

    # Resource allocation (from CLI overrides, None = use HPC cluster defaults)
    gpus_per_node: Optional[int] = None
    cpus_per_node: Optional[int] = None

    # vLLM settings (if launching inline)
    needs_vllm: bool = False
    vllm_model_path: Optional[str] = None
    model_hf_name: Optional[str] = None  # Original HF repo name (before pre-download to local path)
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

    # Agent kwargs (serialized as JSON)
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Upload settings
    upload_username: str = ""
    upload_mode: str = "skip_on_error"
    hf_repo_prefix: str = "DCAgent2"
    upload_to_database: bool = False
    hf_repo_id: Optional[str] = None
    hf_private: bool = False
    hf_episodes: str = "last"
    upload_forced_update: bool = False

    # Pinggy tunnel settings (for cloud backends that can't reach local vLLM)
    pinggy_persistent_url: Optional[str] = None
    pinggy_token: Optional[str] = None

    # Ray object store size in GB (default: 40)
    ray_object_store_gb: float = 40.0


class EvalJobRunner:
    """Runs Harbor eval jobs with optional vLLM management.

    This class encapsulates the eval job logic that was previously
    spread across 600+ lines of sbatch scripts.

    Usage (from sbatch):
        python -m hpc.eval_launch_utils --config /path/to/config.json
    """

    def __init__(self, config: EvalJobConfig):
        self.config = config
        self._hpc = None
        self._proxychains_binary = ""

    def _get_hpc(self):
        """Lazy-load HPC configuration."""
        if self._hpc is None:
            from hpc.hpc import detect_hpc, clusters
            if self.config.cluster_name:
                # Find by name
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
        """Execute the eval job.

        Returns:
            Exit code (0 for success)
        """
        print(f"=== EvalJobRunner: {self.config.job_name} ===")

        try:
            # Ensure HPC is loaded (sets _proxychains_binary for no-internet clusters)
            self._get_hpc()

            if self.config.needs_vllm:
                exit_code = self._run_with_vllm()
            else:
                exit_code = self._run_harbor(endpoint=None)

            if exit_code == 0:
                print(f"Eval job '{self.config.job_name}' completed successfully")
                # Attempt upload after successful Harbor run
                self._maybe_upload_results()
            else:
                print(f"Eval job '{self.config.job_name}' failed with code {exit_code}")

            return exit_code

        except Exception as e:
            print(f"Eval job failed with exception: {e}", file=sys.stderr)
            raise

    def _maybe_upload_results(self) -> None:
        """Upload eval results to HuggingFace and/or Supabase database after Harbor completes."""
        if not self.config.upload_to_database and not self.config.hf_repo_id:
            print("[upload] No upload configured; skipping.")
            return

        # Determine job directory
        jobs_dir = Path(self.config.experiments_dir) / "trace_jobs"
        job_dir = jobs_dir / self.config.job_name
        if not job_dir.exists():
            print(f"[upload] Job directory {job_dir} does not exist; skipping upload.")
            return

        from hpc.launch_utils import sync_eval_to_database, upload_traces_to_hf

        # Derive model name for DB registration.
        # Prefer the original HF name over local paths or hosted_vllm aliases.
        model_name = self.config.model_hf_name or self.config.model
        if model_name and model_name.startswith("hosted_vllm/"):
            model_name = self.config.vllm_model_path or model_name
        # Strip any remaining local snapshot paths back to HF name
        if model_name and "/snapshots/" in model_name:
            # /path/hub/models--org--name/snapshots/hash -> org/name
            import re
            match = re.search(r"models--(.+?)--(.+?)/snapshots/", model_name)
            if match:
                model_name = f"{match.group(1)}/{match.group(2)}"

        if self.config.upload_to_database:
            # Full database sync (includes optional HF upload)
            try:
                result = sync_eval_to_database(
                    job_dir=job_dir,
                    username=self.config.upload_username or None,
                    error_mode=self.config.upload_mode,
                    agent_name=self.config.agent,
                    model_name=model_name,
                    benchmark_name=self.config.eval_benchmark_repo or self.config.dataset,
                    register_benchmark=True,
                    hf_repo_id=self.config.hf_repo_id,
                    hf_private=self.config.hf_private,
                    hf_episodes=self.config.hf_episodes,
                    forced_update=self.config.upload_forced_update,
                )
                if result.get("success"):
                    print(f"[upload] Database sync successful: job_id={result.get('job_id')}")
                else:
                    print(f"[upload] Database sync failed: {result.get('error', 'unknown error')}")
            except Exception as e:
                print(f"[upload] Database sync error: {e}", file=sys.stderr)
        elif self.config.hf_repo_id:
            # HF upload only (no database sync)
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
        """Run eval with managed Ray cluster and vLLM server."""
        from hpc.ray_utils import (
            RayCluster,
            RayClusterConfig,
            compute_ray_memory_from_slurm,
            DEFAULT_OBJECT_STORE_MEMORY_BYTES,
        )
        from hpc.vllm_utils import VLLMServer, VLLMConfig

        hpc = self._get_hpc()
        # Stash proxychains binary for wrapping Harbor CLI (needed on no-internet clusters like JSC)
        self._proxychains_binary = getattr(hpc, "proxychains_binary", "") or ""
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

        # Use config values (from CLI overrides) instead of cluster defaults
        gpus_per_node = self.config.gpus_per_node or hpc.gpus_per_node
        cpus_per_node = self.config.cpus_per_node or hpc.cpus_per_node

        # Compute Ray memory limit from SLURM allocation (prevents OOM from over-detection)
        ray_memory = compute_ray_memory_from_slurm()
        if ray_memory:
            print(f"[EvalJobRunner] Ray memory limit: {ray_memory / (1024**3):.1f} GB", flush=True)

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
            if ray_cluster.total_nodes > 1 and self.config.eval_env in local_backends:
                os.environ["HARBOR_DISTRIBUTED_CONTAINERS"] = "1"
                print(f"[EvalJobRunner] Enabled distributed {self.config.eval_env} "
                      f"across {ray_cluster.total_nodes} nodes", flush=True)

            vllm_server = VLLMServer(
                config=vllm_cfg,
                ray_cluster=ray_cluster,
                log_path=vllm_log,
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
                needs_tunnel = needs_pinggy_tunnel(self.config.agent, self.config.eval_env)
                use_pinggy = has_url and has_token and needs_tunnel

                print(f"[EvalJobRunner] Pinggy check: url={has_url}, token={has_token}, "
                      f"needs_tunnel={needs_tunnel} (agent={self.config.agent}, env={self.config.eval_env})")
                print(f"[EvalJobRunner] use_pinggy={use_pinggy}, vllm_endpoint={vllm_server.endpoint}")

                if use_pinggy:
                    # Parse the vLLM endpoint to get the actual host:port
                    # (vLLM may bind to a specific IP, not localhost)
                    local_host, local_port = parse_endpoint_host_port(vllm_server.endpoint)
                    print(f"[EvalJobRunner] Starting Pinggy tunnel: {local_host}:{local_port} -> {self.config.pinggy_persistent_url}")
                    # On no-internet clusters (e.g., JSC Jupiter) the ssh to
                    # pro.pinggy.io:443 must go through proxychains or it fails
                    # with "Network is unreachable" and the tunnel never comes up.
                    proxychains_wrapper = None
                    pc_bin = getattr(self, "_proxychains_binary", "")
                    pc_conf = os.environ.get("PROXYCHAINS_CONF_FILE", "") if pc_bin else ""
                    if pc_bin and pc_conf:
                        proxychains_wrapper = f"{pc_bin} -f {pc_conf}"
                    elif pc_bin:
                        proxychains_wrapper = pc_bin
                    pinggy_cfg = PinggyConfig(
                        persistent_url=self.config.pinggy_persistent_url,
                        token=self.config.pinggy_token,
                        local_port=local_port,
                        local_host=local_host,
                        proxychains_wrapper=proxychains_wrapper,
                    )
                    pinggy_log = log_dir / f"{self.config.job_name}_pinggy.log"
                    pinggy_tunnel = PinggyTunnel(pinggy_cfg, log_path=pinggy_log)

                    with pinggy_tunnel:
                        # Use Pinggy's public endpoint instead of local vLLM endpoint
                        public_endpoint = pinggy_tunnel.public_endpoint
                        print(f"[EvalJobRunner] Using Pinggy endpoint for Harbor: {public_endpoint}")
                        return self._run_harbor(endpoint=public_endpoint)
                else:
                    # Use local vLLM endpoint directly
                    print(f"[EvalJobRunner] Using local vLLM endpoint for Harbor: {vllm_server.endpoint}")
                    return self._run_harbor(endpoint=vllm_server.endpoint)

    def _run_harbor(self, endpoint: Optional[str]) -> int:
        """Execute the Harbor CLI."""
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
            env_type=self.config.eval_env,
            n_concurrent=self.config.n_concurrent,
            n_attempts=self.config.n_attempts,
            endpoint_meta=endpoint_meta,
            agent_kwarg_overrides=[],  # CLI overrides already merged into config.agent_kwargs
            harbor_extra_args=[],
            dataset_slug=self.config.dataset,
            dataset_path=self.config.dataset_path,
            jobs_dir=jobs_dir,
            extra_agent_kwargs=self.config.agent_kwargs or None,
        )

        # Wrap with proxychains on no-internet clusters (e.g., JSC)
        # This routes Daytona API calls through the SSH tunnel proxy
        proxychains_binary = getattr(self, "_proxychains_binary", "")
        proxychains_conf = os.environ.get("PROXYCHAINS_CONF_FILE", "") if proxychains_binary else ""
        if proxychains_binary and proxychains_conf:
            print(f"[EvalJobRunner] Using proxychains: {proxychains_binary} -f {proxychains_conf}", flush=True)
            cmd = [proxychains_binary, "-f", proxychains_conf] + cmd
        elif proxychains_binary:
            print(f"[EvalJobRunner] Using proxychains: {proxychains_binary} (no conf file)", flush=True)
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


def launch_eval_job_v2(exp_args: dict, hpc) -> None:
    """Launch eval job using the new universal template system.

    This replaces the old launch_eval_job() function by:
    1. Creating an EvalJobConfig from exp_args
    2. Writing the config to JSON
    3. Using the universal_eval.sbatch template
    4. Submitting the job
    """
    from hpc.launch_utils import launch_sbatch

    print("\n=== EVAL MODE (Universal Launcher) ===")

    # Resolve job_name and paths (auto-derives job_name if not provided)
    job_setup = resolve_job_and_paths(
        exp_args,
        job_type_label="Eval",
        derive_job_name_fn=derive_datagen_job_name,  # Handles both datagen and eval
    )
    job_name = job_setup.job_name
    exp_paths = job_setup.paths
    experiments_subdir = str(exp_paths.root)  # String form for config dicts

    # Extract config values
    harbor_cfg = exp_args.get("_eval_harbor_config_resolved")
    dataset_path = exp_args.get("_eval_dataset_path_resolved")
    dataset_slug = exp_args.get("harbor_dataset")

    model_name = exp_args.get("_eval_model_name") or exp_args.get("trace_model") or exp_args.get("datagen_model")
    if not model_name:
        raise ValueError("Unable to determine eval model; provide --trace-model or set it in the datagen config.")

    agent_name = exp_args.get("_eval_agent_name") or exp_args.get("trace_agent_name")
    if not agent_name:
        raise ValueError("Eval jobs require an agent name (set one in the Harbor YAML or pass --trace-agent-name).")

    agent_kwargs = exp_args.get("_eval_agent_kwargs") or {}

    # vLLM settings
    vllm_cfg = exp_args.get("_datagen_vllm_server_config")
    trace_engine = str(exp_args.get("trace_engine") or exp_args.get("datagen_engine") or "").lower()
    requires_vllm = bool(vllm_cfg and trace_engine == "vllm_local")

    # Pre-download model for no-internet clusters (JSC).
    # Must happen on the login node (which has internet) before sbatch submission.
    from hpc.checkpoint_utils import pre_download_model, is_huggingface_repo
    eval_model = getattr(vllm_cfg, "model_path", None) if vllm_cfg else None
    if not eval_model:
        eval_model = model_name
    if eval_model and is_huggingface_repo(eval_model):
        print(f"Pre-downloading model: {eval_model}")
        dl_result = pre_download_model(eval_model)
        if vllm_cfg and hasattr(vllm_cfg, "model_path"):
            vllm_cfg.model_path = dl_result.local_path
        # Use local path for vLLM serving, but preserve original HF name for DB registration
        exp_args["_eval_model_hf_name"] = eval_model  # original HF name for DB
        exp_args["_eval_model_name"] = dl_result.local_path
        exp_args["trace_model"] = dl_result.local_path
        model_name = dl_result.local_path
        print(f"Model available at: {dl_result.local_path}")

    gpus_per_node = int(exp_args.get("gpus_per_node") or getattr(hpc, "gpus_per_node", 1) or 1)
    cpus_per_node = int(exp_args.get("cpus_per_node") or getattr(hpc, "cpus_per_node", 24) or 24)
    tensor_parallel_size = getattr(vllm_cfg, "tensor_parallel_size", None) or 1
    pipeline_parallel_size = getattr(vllm_cfg, "pipeline_parallel_size", None) or 1
    data_parallel_size = getattr(vllm_cfg, "data_parallel_size", None) or 1

    endpoint_json_path = None
    if requires_vllm:
        endpoint_json_path = exp_args.get("vllm_endpoint_json_path") or str(
            default_vllm_endpoint_path(experiments_subdir)
        )
        cleanup_endpoint_file(endpoint_json_path, descriptor="stale eval endpoint file")

    # Convert vllm_cfg dataclass to dict for pass-through
    vllm_server_config = asdict(vllm_cfg) if vllm_cfg else {}

    served_model_id = None
    harbor_model_name = model_name
    if requires_vllm:
        served_model_id = generate_served_model_id()
        harbor_model_name = hosted_vllm_alias(served_model_id)

    # Build the job config
    job_config = EvalJobConfig(
        job_name=job_name,
        harbor_config=harbor_cfg or "",
        model=harbor_model_name,
        served_model_id=served_model_id,
        agent=agent_name,
        dataset=dataset_slug,
        dataset_path=dataset_path,
        n_concurrent=int(exp_args.get("_eval_n_concurrent", 64)),
        n_attempts=int(exp_args.get("_eval_n_attempts", 3)),
        eval_benchmark_repo=exp_args.get("eval_benchmark_repo") or "",
        eval_env=exp_args.get("_eval_env") or "daytona",
        experiments_dir=experiments_subdir,
        cluster_name=hpc.name,
        gpus_per_node=gpus_per_node,
        cpus_per_node=cpus_per_node,
        needs_vllm=requires_vllm,
        vllm_model_path=model_name or (getattr(vllm_cfg, "model_path", None) if vllm_cfg else None),
        model_hf_name=exp_args.get("_eval_model_hf_name"),
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        endpoint_json_path=endpoint_json_path,
        ray_port=int(exp_args.get("datagen_ray_port") or 6379),
        api_port=int(exp_args.get("datagen_api_port") or 8000),
        health_max_attempts=int(exp_args.get("trace_health_max_attempts") or 120),
        health_retry_delay=int(exp_args.get("trace_health_retry_delay") or 15),
        agent_kwargs=agent_kwargs,
        upload_username=exp_args.get("job_creator") or os.environ.get("USER", "unknown"),
        vllm_server_config=vllm_server_config,
        # Upload settings
        upload_to_database=bool(exp_args.get("upload_to_database")),
        upload_mode=exp_args.get("upload_error_mode") or "skip_on_error",
        hf_repo_id=resolve_hf_repo_id(
            explicit_repo=exp_args.get("upload_hf_repo"),
            upload_to_database=bool(exp_args.get("upload_to_database")),
            job_name=job_name,
        ),
        hf_private=bool(exp_args.get("upload_hf_private")),
        hf_episodes=exp_args.get("upload_hf_episodes") or "last",
        upload_forced_update=bool(exp_args.get("upload_forced_update")),
        # Pinggy tunnel settings
        pinggy_persistent_url=exp_args.get("pinggy_persistent_url"),
        pinggy_token=exp_args.get("pinggy_token"),
        ray_object_store_gb=float(exp_args.get("ray_object_store_gb", 40.0)),
    )

    # Write config JSON
    config_path = exp_paths.configs / f"{job_name}_eval_config.json"
    config_path.write_text(json.dumps(asdict(job_config), indent=2))

    # Load and populate universal template
    template_path = Path(__file__).parent / "sbatch_eval" / "universal_eval.sbatch"
    if not template_path.exists():
        raise FileNotFoundError(f"Universal eval template not found: {template_path}")

    template_text = template_path.read_text()

    # Determine cluster env file
    cluster_env_file = hpc.dotenv_filename if hasattr(hpc, "dotenv_filename") else f"{hpc.name.lower()}.env"

    # Build SBATCH directives using shared utility
    sbatch_directives = build_sbatch_directives(hpc, exp_args)

    substitutions = {
        "time_limit": exp_args.get("time_limit") or "24:00:00",
        "num_nodes": str(exp_args.get("num_nodes") or 1),
        "cpus_per_node": str(exp_args.get("cpus_per_node") or hpc.cpus_per_node),
        "experiments_dir": experiments_subdir,
        "job_name": job_name,
        "sbatch_extra_directives": "\n".join(sbatch_directives),
        "module_commands": hpc.get_module_commands(),
        "conda_activate": resolve_conda_activate(hpc, exp_args),
        "cluster_env_file": cluster_env_file,
        "config_path": str(config_path),
        "email_address": os.environ.get("EMAIL_ADDRESS", ""),
        "harbor_env": exp_args.get("_eval_env", "daytona"),
        "env_exports": hpc.get_env_exports(),
        "ray_env_exports": hpc.get_ray_env_exports(experiments_subdir),
        "daytona_api_key_override": get_daytona_api_key_override(exp_args),
        "ssh_tunnel_setup": hpc.get_ssh_tunnel_setup(),
        "proxy_setup": hpc.get_proxy_setup(),
    }

    sbatch_text = substitute_template(template_text, substitutions)

    # Write sbatch script
    sbatch_output = exp_paths.sbatch / f"{job_name}_eval.sbatch"
    sbatch_output.write_text(sbatch_text)
    os.chmod(sbatch_output, 0o750)

    # Get dependency if specified
    dependency = exp_args.get("dependency")

    if exp_args.get("dry_run"):
        print(f"DRY RUN: Eval sbatch script written to {sbatch_output}")
        if dependency:
            print(f"  Would submit with dependency: {dependency}")
        print(f"Config JSON: {config_path}")
        print("--------")
        print(sbatch_text)
        print("--------")
        return

    job_id = launch_sbatch(str(sbatch_output), dependency=dependency)
    print(f"\nEval job submitted via {sbatch_output}")
    print(f"Config: {config_path}")
    print(f"SLURM Job ID: {job_id}")


def run_eval_job_main():
    """Entry point for running eval jobs from sbatch scripts.

    Usage:
        python -m hpc.eval_launch_utils --config /path/to/config.json
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run eval job from config JSON")
    parser.add_argument("--config", required=True, help="Path to job config JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config_data = json.loads(config_path.read_text())
    config = EvalJobConfig(**config_data)
    runner = EvalJobRunner(config)
    sys.exit(runner.run())


if __name__ == "__main__":
    run_eval_job_main()
