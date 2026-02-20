#!/usr/bin/env python3
"""
Launch OpenThoughts evals on a cloud VM via SkyPilot.

This wrapper mirrors the key arguments from eval/local/run_eval.py, then wraps the
whole invocation inside a SkyPilot Task so we can bring up short-lived GPU nodes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# Add repo root to sys.path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

# Handle --list-providers before importing anything heavy
if "--list-providers" in sys.argv:
    from hpc.cloud_providers import list_providers
    print(list_providers(verbose=True))
    sys.exit(0)

import argparse

from hpc.launch_utils import PROJECT_ROOT
from hpc.cloud_launch_utils import CloudLauncher, repo_relative, parse_gpu_count, infer_harbor_env_from_config
from hpc.cloud_sync_utils import sync_outputs
from hpc.arg_groups import (
    add_harbor_args,
    add_harbor_env_arg,
    add_model_compute_args,
    add_hf_upload_args,
    add_database_upload_args,
)
from hpc.harbor_utils import load_harbor_config
from hpc.datagen_config_utils import parse_datagen_config


class EvalCloudLauncher(CloudLauncher):
    """Cloud launcher for eval/local/run_eval.py."""

    task_name = "ot-eval-cloud"
    job_name_prefix = "eval"  # For auto-derived job names
    default_output_subdir = "cloud_runs"
    default_n_concurrent = 16

    def add_task_specific_args(self, parser) -> None:
        """Add eval-specific arguments using shared arg_groups."""
        # Harbor core config (--harbor_config, --agent, --job_name, --agent_kwarg, --harbor_extra_arg)
        add_harbor_args(parser, config_required=True)

        # Model and compute (--model, --n_concurrent, --n_attempts, --gpus, --dry_run)
        # model_required=False: can be inferred from --datagen_config (engine.model)
        add_model_compute_args(
            parser,
            model_required=False,  # Can be inferred from datagen_config
            default_n_concurrent=self.default_n_concurrent,  # 16 for eval
            default_n_attempts=3,  # Eval: multiple runs for standard error
            n_attempts_help="Times to run each task for standard error calculation (default: 3).",
        )

        # Harbor environment backend (unified --harbor_env, with legacy aliases)
        # Default=None to allow inference from harbor config's environment.type field
        add_harbor_env_arg(parser, default=None, legacy_names=["--eval-env", "--eval_env"])

        # Eval-specific arguments (underscore primary, kebab alias)
        parser.add_argument("--datagen_config",
                            help="Optional datagen config to seed defaults.")
        parser.add_argument("--datagen-config", dest="datagen_config", help=argparse.SUPPRESS)

        parser.add_argument("--dataset",
                            help="Harbor dataset slug (exclusive with --dataset_path).")
        parser.add_argument("--dataset_path",
                            help="Path to tasks directory (exclusive with --dataset).")
        parser.add_argument("--dataset-path", dest="dataset_path", help=argparse.SUPPRESS)

        # Ray memory configuration (for cloud VMs with limited RAM)
        parser.add_argument("--ray_object_store_gb", "--ray-object-store-gb",
                            type=float, default=None,
                            help="Ray object store (plasma) size in GB. Default 40GB may OOM on small VMs.")

        # Upload options (shared from arg_groups)
        add_hf_upload_args(parser)
        add_database_upload_args(parser)

    def get_dataset_arg_name(self) -> Optional[str]:
        """Return the dataset argument name for HF handling."""
        return "dataset_path"

    def normalize_paths(self, args) -> None:
        """Normalize repo-relative paths and infer defaults."""
        # Validate mutually exclusive dataset options
        if args.dataset and args.dataset_path:
            raise ValueError("Specify either --dataset or --dataset-path (not both).")
        if not args.dataset and not args.dataset_path:
            raise ValueError("Must provide --dataset or --dataset-path for eval workloads.")

        # Infer --gpus from --accelerator if not explicitly provided
        if args.gpus is None:
            args.gpus = parse_gpu_count(args.accelerator)

        # Normalize paths
        args.harbor_config = repo_relative(args.harbor_config, self.repo_root)
        if args.datagen_config:
            args.datagen_config = repo_relative(args.datagen_config, self.repo_root)
        if args.dataset_path and not args.dataset_path.startswith("/"):
            args.dataset_path = repo_relative(args.dataset_path, self.repo_root)

        # Infer --harbor_env from harbor config if not provided
        infer_harbor_env_from_config(args, args.harbor_config, log_prefix="[eval-cloud]")

        # Infer --agent from harbor config if not provided
        if not args.agent:
            harbor_cfg = load_harbor_config(args.harbor_config)
            agents = harbor_cfg.get("agents", [])
            if agents and isinstance(agents, list) and len(agents) > 0:
                inferred_agent = agents[0].get("name")
                if inferred_agent:
                    args.agent = inferred_agent
                    print(f"[eval-cloud] Inferred --agent={inferred_agent} from harbor config")

        # Infer --model from datagen config if not provided
        if not args.model and args.datagen_config:
            try:
                parsed = parse_datagen_config(args.datagen_config)
                if parsed.model:
                    args.model = parsed.model
                    print(f"[eval-cloud] Inferred --model={parsed.model} from datagen config")
            except Exception as e:
                print(f"[eval-cloud] Warning: Could not parse datagen config for model: {e}")

        # Validate required fields after inference
        if not args.model:
            raise ValueError(
                "Must provide --model or --datagen_config (to infer model from engine.model)"
            )
        if not args.agent:
            raise ValueError(
                "Must provide --agent or ensure harbor config has agents[0].name"
            )

    def build_task_command(self, args, remote_output_dir: str) -> List[str]:
        """Build the run_eval.py command."""
        cmd: List[str] = [
            "python", "eval/local/run_eval.py",
            "--harbor_config", args.harbor_config,
            "--model", args.model,
        ]

        if args.datagen_config:
            cmd.extend(["--datagen_config", args.datagen_config])
        if args.dataset:
            cmd.extend(["--dataset", args.dataset])
        elif args.dataset_path:
            cmd.extend(["--dataset_path", args.dataset_path])

        cmd.extend([
            "--agent", args.agent,
            "--n_concurrent", str(args.n_concurrent),
            "--n_attempts", str(args.n_attempts),
            "--gpus", str(args.gpus),
            "--experiments_dir", remote_output_dir,
        ])

        # Only pass --harbor_env if explicitly specified (otherwise infer from config)
        if args.harbor_env:
            cmd.extend(["--harbor_env", args.harbor_env])

        if args.job_name:
            cmd.extend(["--job_name", args.job_name])
        if args.dry_run:
            cmd.append("--dry_run")

        # Ray memory configuration
        if args.ray_object_store_gb is not None:
            cmd.extend(["--ray_object_store_gb", str(args.ray_object_store_gb)])

        for kwarg in args.agent_kwarg:
            cmd.extend(["--agent_kwarg", kwarg])
        for extra in args.harbor_extra_arg:
            cmd.extend(["--harbor_extra_arg", extra])

        # Upload options
        if args.upload_to_database:
            cmd.append("--upload_to_database")
        if args.upload_username:
            cmd.extend(["--upload_username", args.upload_username])
        if args.upload_error_mode:
            cmd.extend(["--upload_error_mode", args.upload_error_mode])
        if args.upload_hf_repo:
            cmd.extend(["--upload_hf_repo", args.upload_hf_repo])
        if args.upload_hf_token:
            cmd.extend(["--upload_hf_token", args.upload_hf_token])
        if args.upload_hf_private:
            cmd.append("--upload_hf_private")
        if args.upload_hf_episodes:
            cmd.extend(["--upload_hf_episodes", args.upload_hf_episodes])
        if args.upload_forced_update:
            cmd.append("--upload_forced_update")

        return cmd

    def get_periodic_sync_paths(self, args, remote_output_dir: str, remote_workdir: str) -> List[tuple]:
        """Return paths to sync periodically during job execution.

        Syncs logs and Harbor trace_jobs directory to track eval progress.
        """
        return [
            (f"{remote_output_dir}/logs", str(Path(args.local_sync_dir) / "logs")),
            (f"{remote_workdir}/trace_jobs", str(Path(args.local_sync_dir) / "trace_jobs")),
        ]

    def sync_additional_outputs(self, cluster_name: str, args, remote_workdir: str) -> None:
        """Sync Harbor trace_jobs directory (final sync after job completes)."""
        trace_jobs_remote = f"{remote_workdir}/trace_jobs"
        trace_jobs_local = str(Path(args.local_sync_dir) / "trace_jobs")
        print(f"[cloud-sync] Also syncing Harbor trace_jobs from {trace_jobs_remote}...")
        sync_outputs(
            cluster_name=cluster_name,
            remote_path=trace_jobs_remote,
            local_dir=trace_jobs_local,
        )


def main() -> None:
    launcher = EvalCloudLauncher(PROJECT_ROOT)
    parser = launcher.create_argument_parser(
        description="Launch eval/local/run_eval.py on a cloud GPU node via SkyPilot."
    )
    args = parser.parse_args()
    launcher.run(args)


if __name__ == "__main__":
    main()
