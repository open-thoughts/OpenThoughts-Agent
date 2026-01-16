#!/usr/bin/env python3
"""
Local trace generation runner.

Starts a single-node Ray cluster + vLLM controller and then launches a Harbor job
to generate traces from tasks. Designed for non-SLURM Linux hosts where we have
exclusive access to the box.

Usage:
    python run_tracegen.py \
        --harbor_config harbor_configs/default.yaml \
        --tasks_input_path /path/to/tasks \
        --datagen_config datagen_configs/my_config.yaml \
        --upload_hf_repo my-org/my-traces
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from hpc.launch_utils import PROJECT_ROOT
from hpc.local_runner_utils import LocalHarborRunner
from hpc.arg_groups import add_harbor_env_arg, add_hf_upload_args, add_tasks_input_arg


class TracegenRunner(LocalHarborRunner):
    """Local Harbor runner for trace generation."""

    JOB_PREFIX = "tracegen"
    DEFAULT_EXPERIMENTS_SUBDIR = "trace_runs"
    DEFAULT_N_CONCURRENT = 64
    DATAGEN_CONFIG_REQUIRED = True

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser with tracegen-specific arguments."""
        parser = argparse.ArgumentParser(
            description="Run local trace generation with Ray/vLLM server.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=__doc__,
        )

        # Add common arguments from base class
        cls.add_common_arguments(parser)

        # Tracegen-specific arguments (with underscore primary, kebab alias)
        add_tasks_input_arg(parser, required=True)

        parser.add_argument(
            "--datagen_config",
            required=True,
            help="Path to datagen YAML with vLLM settings.",
        )
        parser.add_argument("--datagen-config", dest="datagen_config", help=argparse.SUPPRESS)

        # Harbor environment backend (unified --harbor_env, with legacy aliases)
        add_harbor_env_arg(parser, default="daytona", legacy_names=["--trace-env", "--trace_env"])

        parser.add_argument(
            "--experiments_dir",
            default=str(PROJECT_ROOT / cls.DEFAULT_EXPERIMENTS_SUBDIR),
            help="Directory for logs + endpoint JSON.",
        )
        parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)

        # HuggingFace upload options (shared from arg_groups)
        add_hf_upload_args(parser)

        return parser

    def get_env_type(self) -> str:
        """Get the environment type from --harbor-env or infer from Harbor config."""
        if self.args.harbor_env:
            return self.args.harbor_env
        # Infer from harbor config if not explicitly specified
        from hpc.harbor_utils import get_harbor_env_from_config
        return get_harbor_env_from_config(self.args.harbor_config)

    def get_dataset_label(self) -> str:
        """Get the dataset label for job naming."""
        return self.args.tasks_input_path

    def get_dataset_for_harbor(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (dataset_slug, dataset_path) for harbor command."""
        return (None, self.args.tasks_input_path)

    def validate_args(self) -> None:
        """Validate tracegen-specific arguments."""
        # Resolve tasks input path
        self.args.tasks_input_path = str(Path(self.args.tasks_input_path).expanduser().resolve())

    def print_banner(self) -> None:
        """Print startup banner for tracegen."""
        args = self.args
        needs_local_vllm = getattr(args, "_needs_local_vllm", True)
        engine_type = getattr(args, "_engine_type", "vllm_local")

        print("=== Local Trace Generation ===")
        print(f"  Model: {args.model}")
        print(f"  Tasks: {args.tasks_input_path}")
        if needs_local_vllm:
            print(f"  TP/PP/DP: {args.tensor_parallel_size}/{args.pipeline_parallel_size}/{args.data_parallel_size}")
            print(f"  GPUs: {args.gpus}")
        else:
            print(f"  Engine: {engine_type} (API)")
        print("==============================")

    def post_harbor_hook(self) -> None:
        """Upload traces to HuggingFace after Harbor completes."""
        args = self.args
        hf_repo = args.upload_hf_repo
        if not hf_repo:
            print("[upload] No --upload-hf-repo specified, skipping HuggingFace upload.")
            return

        job_name = self._harbor_job_name
        jobs_dir_path = getattr(args, "_jobs_dir_path", None)
        if not job_name or jobs_dir_path is None:
            print("[upload] Unable to determine job directory; upload skipped.")
            return

        run_dir = Path(jobs_dir_path) / job_name

        # Use shared upload function from launch_utils
        from hpc.launch_utils import upload_traces_to_hf

        try:
            upload_traces_to_hf(
                job_dir=run_dir,
                hf_repo_id=hf_repo,
                hf_private=args.upload_hf_private,
                hf_token=args.upload_hf_token,
                hf_episodes=args.upload_hf_episodes,
                dry_run=args.dry_run,
            )
        except Exception as e:
            print(f"[upload] HuggingFace upload failed: {e}")


def main() -> None:
    parser = TracegenRunner.create_parser()
    args = parser.parse_args()

    runner = TracegenRunner(args, PROJECT_ROOT)
    runner.setup()
    runner.run()


if __name__ == "__main__":
    main()
