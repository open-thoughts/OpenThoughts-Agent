#!/usr/bin/env python3
"""
Launch OpenThoughts trace generation on a cloud VM via SkyPilot.

This wrapper mirrors the key arguments from data/local/run_tracegen.py, then wraps the
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
from hpc.cloud_launch_utils import CloudLauncher, repo_relative, parse_gpu_count
from hpc.cloud_sync_utils import sync_outputs
from hpc.arg_groups import (
    add_harbor_args,
    add_harbor_env_arg,
    add_model_compute_args,
    add_hf_upload_args,
    add_tasks_input_arg,
)
from hpc.cloud_launch_utils import infer_harbor_env_from_config


class TracegenCloudLauncher(CloudLauncher):
    """Cloud launcher for data/local/run_tracegen.py."""

    task_name = "ot-tracegen-cloud"
    job_name_prefix = "tracegen"  # For auto-derived job names
    default_output_subdir = "cloud_runs"
    default_n_concurrent = 64

    def add_task_specific_args(self, parser) -> None:
        """Add tracegen-specific arguments using shared arg_groups."""
        # Harbor core config (--harbor_config, --agent, --job_name, --agent_kwarg, --harbor_extra_arg)
        add_harbor_args(parser, config_required=True)

        # Model and compute (--model, --n_concurrent, --n_attempts, --gpus, --dry_run)
        add_model_compute_args(
            parser,
            model_required=False,
            default_n_concurrent=self.default_n_concurrent,  # 64 for tracegen
            default_n_attempts=1,  # Tracegen: run once per task
            n_attempts_help="Times to run each task for repeated trials (default: 1).",
        )

        # Harbor environment backend (unified --harbor_env, with legacy aliases)
        # Default=None to allow inference from harbor config's environment.type field
        add_harbor_env_arg(parser, default=None, legacy_names=["--trace-env", "--trace_env"])

        # Tracegen-specific required arguments (underscore primary, kebab alias)
        parser.add_argument("--datagen_config", required=True,
                            help="Datagen config with vLLM settings (required).")
        parser.add_argument("--datagen-config", dest="datagen_config", help=argparse.SUPPRESS)

        add_tasks_input_arg(parser, required=True)

        # HuggingFace upload options (shared from arg_groups)
        add_hf_upload_args(parser)

    def get_dataset_arg_name(self) -> Optional[str]:
        """Return the dataset argument name for HF handling."""
        return "tasks_input_path"

    def normalize_paths(self, args) -> None:
        """Normalize repo-relative paths and infer defaults."""
        # Infer --gpus from --accelerator if not explicitly provided
        if args.gpus is None:
            args.gpus = parse_gpu_count(args.accelerator)

        # Normalize paths
        args.harbor_config = repo_relative(args.harbor_config, self.repo_root)
        args.datagen_config = repo_relative(args.datagen_config, self.repo_root)
        if not args.tasks_input_path.startswith("/"):
            args.tasks_input_path = repo_relative(args.tasks_input_path, self.repo_root)

        # Infer --harbor_env from harbor config if not provided
        infer_harbor_env_from_config(args, args.harbor_config, log_prefix="[tracegen-cloud]")

    def build_task_command(self, args, remote_output_dir: str) -> List[str]:
        """Build the run_tracegen.py command."""
        cmd: List[str] = [
            "python", "data/local/run_tracegen.py",
            "--harbor_config", args.harbor_config,
            "--datagen_config", args.datagen_config,
            "--tasks_input_path", args.tasks_input_path,
        ]

        if args.model:
            cmd.extend(["--model", args.model])

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

        for kwarg in args.agent_kwarg:
            cmd.extend(["--agent_kwarg", kwarg])
        for extra in args.harbor_extra_arg:
            cmd.extend(["--harbor_extra_arg", extra])

        # Upload options
        if args.upload_hf_repo:
            cmd.extend(["--upload_hf_repo", args.upload_hf_repo])
        if args.upload_hf_token:
            cmd.extend(["--upload_hf_token", args.upload_hf_token])
        if args.upload_hf_private:
            cmd.append("--upload_hf_private")

        return cmd

    def get_periodic_sync_paths(self, args, remote_output_dir: str, remote_workdir: str) -> List[tuple]:
        """Return paths to sync periodically during job execution.

        Syncs logs and Harbor trace_jobs directory to track trace generation progress.
        """
        return [
            (f"{remote_output_dir}/logs", str(Path(args.local_sync_dir) / "logs")),
            (f"{remote_workdir}/trace_jobs", str(Path(args.local_sync_dir) / "trace_jobs")),
        ]

    def sync_additional_outputs(self, cluster_name: str, args, remote_workdir: str) -> None:
        """Sync Harbor trace_jobs directory (final sync after job completes)."""
        jobs_remote = f"{remote_workdir}/trace_jobs"
        jobs_local = str(Path(args.local_sync_dir) / "trace_jobs")
        print(f"[cloud-sync] Also syncing Harbor jobs from {jobs_remote}...")
        sync_outputs(
            cluster_name=cluster_name,
            remote_path=jobs_remote,
            local_dir=jobs_local,
        )


def main() -> None:
    launcher = TracegenCloudLauncher(PROJECT_ROOT)
    parser = launcher.create_argument_parser(
        description="Launch data/local/run_tracegen.py on a cloud GPU node via SkyPilot."
    )
    args = parser.parse_args()
    launcher.run(args)


if __name__ == "__main__":
    main()
