#!/usr/bin/env python3
"""
Local eval runner.

Starts a single-node Ray cluster + vLLM controller and then launches a Harbor eval
job that targets the freshly booted endpoint. Designed for non-SLURM Linux hosts
where we have exclusive access to the box.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

from hpc.local_runner_utils import LocalHarborRunner
from hpc.launch_utils import sanitize_hf_repo_id
from hpc.arg_groups import add_harbor_env_arg, add_hf_upload_args, add_database_upload_args


class EvalRunner(LocalHarborRunner):
    """Local Harbor runner for evaluation."""

    JOB_PREFIX = "eval"
    DEFAULT_EXPERIMENTS_SUBDIR = "eval_runs"
    DEFAULT_N_CONCURRENT = 16
    DATAGEN_CONFIG_REQUIRED = False

    @classmethod
    def create_parser(cls) -> argparse.ArgumentParser:
        """Create argument parser with eval-specific arguments."""
        parser = argparse.ArgumentParser(
            description="Run Harbor evals against a local Ray/vLLM server."
        )

        # Add common arguments from base class
        cls.add_common_arguments(parser)

        # Eval-specific arguments
        parser.add_argument(
            "--dataset",
            help="Harbor dataset slug (e.g., terminal-bench@2.0). Mutually exclusive with --dataset-path.",
        )
        parser.add_argument(
            "--dataset-path",
            help="Path to a Harbor task directory. Mutually exclusive with --dataset.",
        )

        # Harbor environment backend (unified --harbor-env, with --eval-env as legacy alias)
        add_harbor_env_arg(parser, default="daytona", legacy_names=["--eval-env"])

        parser.add_argument(
            "--datagen-config",
            help="Optional datagen YAML whose vLLM settings will seed defaults for this script.",
        )
        parser.add_argument(
            "--experiments-dir",
            default=str(REPO_ROOT / cls.DEFAULT_EXPERIMENTS_SUBDIR),
            help="Directory for logs + endpoint JSON.",
        )

        # Upload options (shared from arg_groups)
        add_hf_upload_args(parser)
        add_database_upload_args(parser)

        return parser

    def get_env_type(self) -> str:
        """Get the environment type from --harbor-env (or legacy --eval-env)."""
        return self.args.harbor_env

    def get_dataset_label(self) -> str:
        """Get the dataset label for job naming."""
        return self.args.dataset or self.args.dataset_path or "dataset"

    def get_dataset_for_harbor(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (dataset_slug, dataset_path) for harbor command."""
        return (self.args.dataset, self.args.dataset_path)

    def validate_args(self) -> None:
        """Validate eval-specific arguments."""
        # Ensure mutually exclusive dataset args
        if self.args.dataset and self.args.dataset_path:
            raise ValueError("Specify either --dataset or --dataset-path (not both).")
        if not self.args.dataset and not self.args.dataset_path:
            raise ValueError("Must provide --dataset or --dataset-path.")

        # Resolve dataset path if provided
        if self.args.dataset_path:
            self.args.dataset_path = str(Path(self.args.dataset_path).expanduser().resolve())

    def print_banner(self) -> None:
        """Print startup banner for eval."""
        dataset_label = self.get_dataset_label()
        print("=== Local Eval Runner ===")
        print(f"  Model: {self.args.model}")
        print(f"  Dataset: {dataset_label}")
        print(f"  TP/PP/DP: {self.args.tensor_parallel_size}/{self.args.pipeline_parallel_size}/{self.args.data_parallel_size}")
        print(f"  GPUs: {self.args.gpus}")
        print("=========================")

    def post_harbor_hook(self) -> None:
        """Upload results to Supabase/HuggingFace after Harbor completes."""
        self._maybe_upload_results()

    def _derive_default_hf_repo_id(self, job_name: str) -> str:
        """Derive default HF repo ID from benchmark repo."""
        from hpc.launch_utils import derive_benchmark_repo
        benchmark_repo = derive_benchmark_repo(
            harbor_dataset=self.args.dataset,
            dataset_path=self.args.dataset_path,
            explicit_repo=getattr(self.args, "eval_benchmark_repo", None),
        )
        if "/" in benchmark_repo:
            org = benchmark_repo.split("/", 1)[0]
        else:
            org = benchmark_repo or "openthoughts-agent"
        return f"{org}/{job_name}"

    def _maybe_upload_results(self) -> None:
        """Upload eval results using the shared upload functions from hpc.launch_utils."""
        args = self.args
        if not getattr(args, "upload_to_database", False):
            return
        if args.dry_run:
            print("[upload] Skipping Supabase upload because --dry-run was set.")
            return

        job_name = self._harbor_job_name
        jobs_dir_path = getattr(args, "_jobs_dir_path", None)
        if not job_name or jobs_dir_path is None:
            print("[upload] Unable to determine job directory; upload skipped.")
            return

        run_dir = Path(jobs_dir_path) / job_name
        if not run_dir.exists():
            print(f"[upload] Expected Harbor job directory {run_dir} does not exist; upload skipped.")
            return

        # Use shared utilities from hpc.launch_utils
        from hpc.launch_utils import sync_eval_to_database, derive_benchmark_repo

        benchmark_name = derive_benchmark_repo(
            harbor_dataset=args.dataset,
            dataset_path=args.dataset_path,
        )
        hf_repo_id = args.upload_hf_repo or self._derive_default_hf_repo_id(job_name)
        if hf_repo_id:
            hf_repo_id = sanitize_hf_repo_id(hf_repo_id)

        result = sync_eval_to_database(
            job_dir=run_dir,
            username=args.upload_username,
            error_mode=args.upload_error_mode,
            agent_name=args.agent,
            model_name=args.model,
            benchmark_name=benchmark_name,
            register_benchmark=True,
            hf_repo_id=hf_repo_id,
            hf_private=args.upload_hf_private,
            hf_token=args.upload_hf_token,
            hf_episodes=args.upload_hf_episodes,
            forced_update=args.upload_forced_update,
            dry_run=args.dry_run,
        )

        if not result.get("success"):
            print(f"[upload] Upload failed: {result.get('error', 'unknown error')}")


def main() -> None:
    parser = EvalRunner.create_parser()
    args = parser.parse_args()

    runner = EvalRunner(args, REPO_ROOT)
    runner.setup()
    runner.run()


if __name__ == "__main__":
    main()
