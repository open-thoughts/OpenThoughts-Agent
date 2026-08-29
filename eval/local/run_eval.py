#!/usr/bin/env python3
"""
Local eval runner.

Starts a single-node Ray cluster + vLLM controller and then launches a Harbor eval
job that targets the freshly booted endpoint. Designed for non-SLURM Linux hosts
where we have exclusive access to the box.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

from hpc.launch_utils import PROJECT_ROOT
from hpc.local_runner_utils import LocalHarborRunner
from hpc.arg_groups import (
    add_database_upload_args,
    add_harbor_env_arg,
    add_hf_upload_args,
    add_ingress_literal_args,
)
from hpc.hf_utils import resolve_hf_repo_id


GCS_MODEL_S3_PREFIX = "s3://marin-models-"
GCS_MODEL_ENV_KEYS = {
    "AWS_ENDPOINT_URL": "https://storage.googleapis.com",
    "RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING": "False",
    "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": (
        "AWS_ENDPOINT_URL,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,"
        "RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING"
    ),
}


def _configure_gcs_model_credentials(args: argparse.Namespace) -> None:
    model_uri = getattr(args, "vllm_model_uri", None)
    if not model_uri or not model_uri.startswith(GCS_MODEL_S3_PREFIX):
        return
    access_key = os.environ.get("MARIN_HMAC_ACCESS_ID")
    secret_key = os.environ.get("MARIN_HMAC_SECRET")
    if not access_key or not secret_key:
        raise ValueError(
            "GCS model streaming requires MARIN_HMAC_ACCESS_ID and MARIN_HMAC_SECRET"
        )
    env = dict(getattr(args, "_vllm_env_vars", None) or {})
    env.update(GCS_MODEL_ENV_KEYS)
    env["AWS_ACCESS_KEY_ID"] = access_key
    env["AWS_SECRET_ACCESS_KEY"] = secret_key
    args._vllm_env_vars = env


class EvalRunner(LocalHarborRunner):
    """Local Harbor runner for evaluation."""

    JOB_PREFIX = "eval"
    DEFAULT_EXPERIMENTS_SUBDIR = "eval_runs"
    DEFAULT_N_CONCURRENT = 16
    DATAGEN_CONFIG_REQUIRED = False
    # Cluster-level default for the Iris/TPU agentic-eval serve: enable vLLM prefix
    # caching (APC) so each agentic turn reuses the KV cache for the shared, growing
    # conversation prefix instead of re-prefilling it (~30x redundant-prefill fix on
    # v6e-4). Applied to ALL models on the iris eval path, NOT per-model. Scoped to
    # eval (not the shared base) as a canary while APC support on tpu-inference
    # 0.23.0 is unconfirmed — see the CAVEAT at the call site in
    # hpc/local_runner_utils.py (validate n_cache_tokens>0 on the next eval leg).
    TPU_SERVE_DEFAULT_CLI_ARGS = ["--enable-prefix-caching"]

    def setup(self) -> None:
        super().setup()
        _configure_gcs_model_credentials(self.args)

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
        parser.add_argument(
            "--dataset-path", dest="dataset_path", help=argparse.SUPPRESS
        )
        parser.add_argument(
            "--cohort_size",
            type=int,
            help="Materialize exactly this many tasks from a parquet dataset.",
        )
        parser.add_argument(
            "--cohort-size", dest="cohort_size", type=int, help=argparse.SUPPRESS
        )

        # Harbor environment backend (unified --harbor_env, with legacy aliases)
        # Default=None to allow inference from harbor config's environment.type field
        add_harbor_env_arg(
            parser, default=None, legacy_names=["--eval-env", "--eval_env"]
        )

        parser.add_argument(
            "--datagen_config",
            help="Optional datagen YAML whose vLLM settings will seed defaults for this script.",
        )
        parser.add_argument(
            "--datagen-config", dest="datagen_config", help=argparse.SUPPRESS
        )

        parser.add_argument(
            "--vllm_model_uri",
            help="Object-store URI (s3://|gs://) the vLLM server loads weights from "
            "(via runai_streamer), while --model stays the HF id used for "
            "model_config resolution + the served-model name. Set by the iris "
            "launcher's offline pre-cache; leave unset for normal HF loads.",
        )
        parser.add_argument(
            "--vllm-model-uri", dest="vllm_model_uri", help=argparse.SUPPRESS
        )

        parser.add_argument(
            "--experiments_dir",
            default=str(PROJECT_ROOT / cls.DEFAULT_EXPERIMENTS_SUBDIR),
            help="Directory for logs + endpoint JSON.",
        )
        parser.add_argument(
            "--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS
        )

        add_ingress_literal_args(parser)

        # Re-fire errored-trial pruning. On a warm-dir re-fire (an existing run
        # dir), delete trials whose exception_info.exception_type is one of these
        # BEFORE the harbor auto-resume, so those infra-errored trials re-run
        # (the gs://-capable analog of `harbor jobs resume --filter-error-type`).
        # Repeatable; empty/unset -> no pruning (auto-resume keeps errored trials,
        # i.e. the historical no-op behavior). The Iris launcher bakes the
        # resolved non-benign infra set here; a direct run_eval invocation must
        # pass the types explicitly.
        parser.add_argument(
            "--refire_filter_error_type",
            dest="refire_filter_error_types",
            action="append",
            default=None,
            help="Exception type to delete-and-re-run on a warm-dir re-fire "
            "(repeatable). Unset -> no pruning.",
        )
        parser.add_argument(
            "--refire-filter-error-type",
            dest="refire_filter_error_types",
            action="append",
            help=argparse.SUPPRESS,
        )

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
            from hpc.hf_utils import resolve_dataset_path, is_raw_tasks_directory
            from hpc.launch_utils import convert_parquet_to_tasks

            original_identifier = self.args.dataset_path
            self.args.dataset_path = resolve_dataset_path(
                self.args.dataset_path, verbose=True
            )

            # Auto-detect parquet datasets and convert to task directories
            if not is_raw_tasks_directory(self.args.dataset_path):
                self.args.dataset_path = convert_parquet_to_tasks(
                    self.args.dataset_path,
                    original_identifier,
                    cohort_size=self.args.cohort_size,
                )

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
            print(
                f"  TP/PP/DP: {args.tensor_parallel_size}/{args.pipeline_parallel_size}/{args.data_parallel_size}"
            )
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
            print(
                f"[upload] Expected Harbor job directory {run_dir} does not exist; upload skipped."
            )
            return

        from hpc.launch_utils import (
            sync_eval_to_database,
            upload_traces_to_hf,
            derive_benchmark_repo,
        )

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
                print(
                    f"[upload] Database sync failed: {result.get('error', 'unknown error')}"
                )
            else:
                print(
                    f"[upload] Database sync successful: job_id={result.get('job_id')}"
                )

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
