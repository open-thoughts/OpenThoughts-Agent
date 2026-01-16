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

from hpc.launch_utils import PROJECT_ROOT
from hpc.local_runner_utils import LocalHarborRunner
from hpc.arg_groups import add_harbor_env_arg, add_hf_upload_args, add_database_upload_args
from hpc.hf_utils import resolve_hf_repo_id


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

        # Eval-specific arguments (underscore primary, kebab alias)
        parser.add_argument(
            "--dataset",
            help="Harbor dataset slug (e.g., terminal-bench@2.0). Mutually exclusive with --dataset_path.",
        )
        parser.add_argument(
            "--dataset_path",
            help="Path to a Harbor task directory. Mutually exclusive with --dataset.",
        )
        parser.add_argument("--dataset-path", dest="dataset_path", help=argparse.SUPPRESS)

        # Harbor environment backend (unified --harbor_env, with legacy aliases)
        add_harbor_env_arg(parser, default="daytona", legacy_names=["--eval-env", "--eval_env"])

        parser.add_argument(
            "--datagen_config",
            help="Optional datagen YAML whose vLLM settings will seed defaults for this script.",
        )
        parser.add_argument("--datagen-config", dest="datagen_config", help=argparse.SUPPRESS)

        parser.add_argument(
            "--experiments_dir",
            default=str(PROJECT_ROOT / cls.DEFAULT_EXPERIMENTS_SUBDIR),
            help="Directory for logs + endpoint JSON.",
        )
        parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)

        # Upload options (shared from arg_groups)
        add_hf_upload_args(parser)
        add_database_upload_args(parser)

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

        # Resolve dataset path if provided (handles both local paths and HF repo IDs)
        if self.args.dataset_path:
            from hpc.hf_utils import resolve_dataset_path
            self.args.dataset_path = resolve_dataset_path(self.args.dataset_path, verbose=True)

    def print_banner(self) -> None:
        """Print startup banner for eval."""
        args = self.args
        needs_local_vllm = getattr(args, "_needs_local_vllm", True)
        engine_type = getattr(args, "_engine_type", "vllm_local")
        dataset_label = self.get_dataset_label()

        print("=== Local Eval Runner ===")
        print(f"  Model: {args.model}")
        print(f"  Dataset: {dataset_label}")
        if needs_local_vllm:
            print(f"  TP/PP/DP: {args.tensor_parallel_size}/{args.pipeline_parallel_size}/{args.data_parallel_size}")
            print(f"  GPUs: {args.gpus}")
        else:
            print(f"  Engine: {engine_type} (API)")
        print("=========================")

    def post_harbor_hook(self) -> None:
        """Upload results to Supabase/HuggingFace after Harbor completes."""
        self._maybe_upload_results()

    def _maybe_upload_results(self) -> None:
        """Upload eval results to HuggingFace and/or Supabase database.

        Supports three modes:
        - --upload_to_database: Full DB sync + HF upload
        - --upload_hf_repo (without --upload_to_database): HF-only upload
        - Neither: No upload
        """
        args = self.args
        upload_to_db = getattr(args, "upload_to_database", False)
        hf_repo = getattr(args, "upload_hf_repo", None)

        if not upload_to_db and not hf_repo:
            return

        if args.dry_run:
            print("[upload] Skipping upload because --dry-run was set.")
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

        from hpc.launch_utils import sync_eval_to_database, upload_traces_to_hf, derive_benchmark_repo

        if upload_to_db:
            # Full database sync (includes optional HF upload)
            benchmark_name = derive_benchmark_repo(
                harbor_dataset=args.dataset,
                dataset_path=args.dataset_path,
            )

            hf_repo_id = resolve_hf_repo_id(
                explicit_repo=hf_repo,
                upload_to_database=True,
                job_name=job_name,
            )

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
                print(f"[upload] Database sync failed: {result.get('error', 'unknown error')}")
            else:
                print(f"[upload] Database sync successful: job_id={result.get('job_id')}")

        elif hf_repo:
            # HF-only upload (no database sync)
            try:
                hf_url = upload_traces_to_hf(
                    job_dir=run_dir,
                    hf_repo_id=hf_repo,
                    hf_private=getattr(args, "upload_hf_private", False),
                    hf_episodes=getattr(args, "upload_hf_episodes", "last"),
                    hf_token=getattr(args, "upload_hf_token", None),
                )
                if hf_url:
                    print(f"[upload] HuggingFace upload successful: {hf_url}")
            except Exception as e:
                print(f"[upload] HuggingFace upload error: {e}")


def main() -> None:
    parser = EvalRunner.create_parser()
    args = parser.parse_args()

    runner = EvalRunner(args, PROJECT_ROOT)
    runner.setup()
    runner.run()


if __name__ == "__main__":
    main()
