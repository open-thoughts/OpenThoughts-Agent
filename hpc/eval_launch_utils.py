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
    coerce_agent_kwargs,
    default_vllm_endpoint_path,
    launch_sbatch,
    _parse_optional_int,
    cleanup_endpoint_file,
    validate_trace_backend,
    build_sbatch_directives,
    generate_served_model_id,
    hosted_vllm_alias,
    strip_hosted_vllm_alias,
)

# Config directory paths (same as datagen_launch_utils)
_DIRENV = os.path.dirname(__file__)
HARBOR_CONFIG_DIR = os.path.join(_DIRENV, "harbor_yaml")

from scripts.harbor.job_config_utils import load_job_config


def resolve_harbor_config_path(raw_value: str) -> Path:
    """Resolve ``raw_value`` to an absolute Harbor job config path.

    Checks in order: raw_value as-is, then HARBOR_CONFIG_DIR fallback.
    """
    return resolve_config_path(raw_value, HARBOR_CONFIG_DIR, "harbor job")

DEFAULT_REGISTRY_HINTS = [
    Path(os.environ.get("HARBOR_REGISTRY_PATH", "")).expanduser()
    if os.environ.get("HARBOR_REGISTRY_PATH")
    else None,
    PROJECT_ROOT.parent / "harbor" / "registry.json",
]


def _load_harbor_registry() -> dict | None:
    for candidate in DEFAULT_REGISTRY_HINTS:
        if candidate and candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except Exception:
                return None
    return None


def _build_dataset_slug_set(registry: dict | None) -> set[str]:
    if not registry:
        return set()
    entries: set[str] = set()
    for item in registry:
        name = item.get("name")
        version = item.get("version")
        if not name:
            continue
        if version:
            entries.add(f"{name}@{version}")
        entries.add(name)
    return entries


def _validate_harbor_dataset_slug(slug: str) -> None:
    registry = _load_harbor_registry()
    if not registry:
        return
    valid = _build_dataset_slug_set(registry)
    if slug not in valid:
        raise ValueError(
            f"Dataset '{slug}' is not in the local Harbor registry "
            f"(known datasets: {sorted(list(valid))[:8]} ...). "
            "Specify --eval-dataset-path instead or update the registry hint."
        )


def _coerce_agent_kwargs(value: Any) -> Dict[str, Any]:
    """Wrapper for backwards compatibility - use coerce_agent_kwargs from launch_utils."""
    return coerce_agent_kwargs(value)


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

    dataset_path = exp_args.get("trace_input_path")
    harbor_dataset = exp_args.get("harbor_dataset")
    if dataset_path and harbor_dataset:
        raise ValueError(
            "Eval jobs accept either --trace-input-path or --harbor-dataset, but not both."
        )
    if dataset_path:
        resolved_dataset = resolve_repo_path(dataset_path)
        exp_args["_eval_dataset_path_resolved"] = str(resolved_dataset)
        exp_args["trace_input_path"] = str(resolved_dataset)
    if harbor_dataset:
        slug = harbor_dataset.strip()
        if not slug:
            raise ValueError("--harbor-dataset cannot be empty.")
        _validate_harbor_dataset_slug(slug)
        exp_args["harbor_dataset"] = slug

    if not (exp_args.get("harbor_dataset") or exp_args.get("_eval_dataset_path_resolved")):
        raise ValueError(
            "Eval jobs require either --harbor-dataset or --trace-input-path to specify tasks."
        )

    # Derive eval_benchmark_repo using shared utility
    from hpc.launch_utils import derive_benchmark_repo
    benchmark_repo = derive_benchmark_repo(
        harbor_dataset=exp_args.get("harbor_dataset"),
        dataset_path=exp_args.get("_eval_dataset_path_resolved"),
        explicit_repo=exp_args.get("eval_benchmark_repo"),
    )
    if benchmark_repo == "unknown-benchmark":
        raise ValueError(
            "Unable to derive eval_benchmark_repo. Provide "
            "--harbor-dataset or --trace-input-path."
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
        raise ValueError("Eval jobs require --trace-model (or --datagen-model).")
    exp_args["_eval_model_name"] = model_name
    exp_args["trace_model"] = model_name

    agent_cfg = harbor_job.agents[0]
    agent_name = (
        exp_args.get("trace_agent_name")
        or agent_cfg.name
        or (agent_cfg.import_path or "terminus-2")
    )
    exp_args["_eval_agent_name"] = agent_name
    if "trace_agent_name" not in exp_args:
        exp_args["trace_agent_name"] = agent_name

    base_agent_kwargs = dict(agent_cfg.kwargs or {})
    datagen_agent_defaults = dict(exp_args.get("_datagen_extra_agent_kwargs") or {})
    base_agent_kwargs.update(datagen_agent_defaults)
    cli_agent_kwargs = _coerce_agent_kwargs(exp_args.get("trace_agent_kwargs"))
    agent_kwargs: Dict[str, Any] = dict(base_agent_kwargs)
    agent_kwargs.update(cli_agent_kwargs)
    exp_args["_eval_agent_kwargs"] = agent_kwargs

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

    default_n_concurrent = harbor_job.orchestrator.n_concurrent_trials
    if default_n_concurrent is None or default_n_concurrent < 1:
        default_n_concurrent = 64
    n_concurrent_override = _parse_optional_int(
        exp_args.get("trace_n_concurrent"),
        "--trace_n_concurrent",
    )
    n_concurrent_int = n_concurrent_override or default_n_concurrent
    exp_args["_eval_n_concurrent"] = max(1, int(n_concurrent_int))

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
        return self._hpc

    def run(self) -> int:
        """Execute the eval job.

        Returns:
            Exit code (0 for success)
        """
        print(f"=== EvalJobRunner: {self.config.job_name} ===")

        try:
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

        # Derive model name (strip hosted_vllm prefix if present)
        model_name = self.config.model
        if model_name and model_name.startswith("hosted_vllm/"):
            # Use original model path for database records
            model_name = self.config.vllm_model_path or model_name

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
        from hpc.ray_utils import RayCluster, RayClusterConfig
        from hpc.vllm_utils import VLLMServer, VLLMConfig

        hpc = self._get_hpc()
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

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
        """Execute the Harbor CLI."""
        cmd = [
            "harbor",
            "jobs",
            "start",
            "--config",
            self.config.harbor_config,
            "--job-name",
            self.config.job_name,
            "--agent",
            self.config.agent,
            "--model",
            self.config.model,
            "--env",
            self.config.eval_env,
            "--n-concurrent",
            str(self.config.n_concurrent),
            "--n-attempts",
            str(self.config.n_attempts),
        ]

        if self.config.dataset:
            cmd.extend(["--dataset", self.config.dataset])
        elif self.config.dataset_path:
            cmd.extend(["-p", self.config.dataset_path])

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

    # Resolve paths
    experiments_subdir = exp_args.get("experiments_dir") or "experiments"
    experiments_abs = resolve_workspace_path(experiments_subdir)
    sbatch_dir = experiments_abs / "sbatch"
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = experiments_abs / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = experiments_abs / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    job_name = exp_args.get("job_name")
    if not job_name:
        raise ValueError("Eval jobs require a --job_name.")

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
        vllm_model_path=getattr(vllm_cfg, "model_path", None) if vllm_cfg else model_name,
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
        hf_repo_id=exp_args.get("upload_hf_repo") or None,
        hf_private=bool(exp_args.get("upload_hf_private")),
        hf_episodes=exp_args.get("upload_hf_episodes") or "last",
        upload_forced_update=bool(exp_args.get("upload_forced_update")),
    )

    # Write config JSON
    config_path = configs_dir / f"{job_name}_eval_config.json"
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
        "conda_activate": hpc.conda_activate or "# No conda activation configured",
        "cluster_env_file": cluster_env_file,
        "config_path": str(config_path),
    }

    sbatch_text = template_text
    for key, value in substitutions.items():
        sbatch_text = sbatch_text.replace("{" + key + "}", value)

    # Write sbatch script
    sbatch_output = sbatch_dir / f"{job_name}_eval.sbatch"
    sbatch_output.write_text(sbatch_text)
    os.chmod(sbatch_output, 0o750)

    if exp_args.get("dry_run"):
        print(f"DRY RUN: Eval sbatch script written to {sbatch_output}")
        print(f"Config JSON: {config_path}")
        print("--------")
        print(sbatch_text)
        print("--------")
        return

    job_id = launch_sbatch(str(sbatch_output))
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
