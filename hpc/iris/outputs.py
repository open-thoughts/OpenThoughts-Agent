"""Output path handling for Iris launchers."""

from __future__ import annotations

import argparse


# Default GCS prefix for workload outputs. EU-region matches where most
# of our v6e-preemptible TPU slices land; us-region jobs incur small
# cross-region writes (eval outputs are ~MB-scale, so this is fine).
# Override with $OT_AGENT_GCS_OUTPUT_ROOT or the --gcs-output-dir flag.
DEFAULT_GCS_OUTPUT_ROOT = "gs://marin-eu-west4/ot-agent"
DEFAULT_LOCAL_OUTPUT_ROOT = "/tmp/ot-agent-runs"
DEFAULT_S3_OUTPUT_ROOT = None


def _join_output_path(root: str, job_name: str) -> str:
    return f"{root.rstrip('/')}/{job_name.strip('/')}"


def resolve_output_mode(
    args: argparse.Namespace,
    *,
    accelerator_kind: str,
) -> str:
    """Resolve ``--output-mode auto`` after the accelerator is known."""
    output_mode = getattr(args, "output_mode", "auto")
    if output_mode != "auto":
        return output_mode
    if accelerator_kind == "gpu":
        return "s3"
    return "gcs"


def validate_output_args(
    args: argparse.Namespace,
    output_mode: str,
    *,
    accelerator_kind: str,
) -> None:
    if accelerator_kind == "gpu" and output_mode == "gcs":
        raise SystemExit(
            "CoreWeave GPU Iris runs must not write to GCS. Use "
            "--output-mode s3 --s3-output-dir s3://... for durable output."
        )
    if accelerator_kind == "tpu" and output_mode == "s3":
        raise SystemExit(
            "TPU Iris runs use --output-mode gcs. --output-mode s3 is only "
            "supported for the CoreWeave GPU eval path."
        )
    if output_mode == "gcs" and not args.gcs_output_dir:
        raise SystemExit(
            "--gcs-output-dir is required (set OT_AGENT_GCS_OUTPUT_ROOT or pass the flag)."
        )
    if output_mode == "s3":
        if getattr(args, "resume_from", None):
            raise SystemExit("--resume-from is only supported with --output-mode gcs.")
        if not getattr(args, "s3_output_dir", None):
            raise SystemExit(
                "--s3-output-dir is required for --output-mode s3 "
                "(set OT_AGENT_S3_OUTPUT_ROOT or pass the flag)."
            )
        s3_output_root = str(args.s3_output_dir).rstrip("/")
        if not s3_output_root.startswith("s3://"):
            raise SystemExit("--s3-output-dir must start with s3://.")
        args.s3_output_dir = s3_output_root
        local_output_root = str(args.local_output_dir).rstrip("/")
        if not local_output_root.startswith("/"):
            raise SystemExit("--local-output-dir must be an absolute path inside the task container.")
        args.local_output_dir = local_output_root


def resolve_remote_output_dir(
    args: argparse.Namespace,
    *,
    job_name: str,
    output_mode: str,
    resume_target: str | None,
) -> str:
    """Return the workload output path seen by the Iris task."""
    if resume_target:
        # Resume: point at the OLD job's full GCS path so harbor finds
        # its existing config.json / trial dirs. Do NOT re-join job_name.
        return args._resume_gcs_output_dir.rstrip("/")
    if output_mode == "s3":
        return _join_output_path(args.s3_output_dir, job_name)
    return _join_output_path(args.gcs_output_dir, job_name)


def resolve_work_output_dir(
    args: argparse.Namespace,
    *,
    job_name: str,
    output_mode: str,
    remote_output_dir: str,
) -> str:
    """Return the runtime scratch path used by ``run_eval.py``."""
    if output_mode == "s3":
        return _join_output_path(args.local_output_dir, job_name)
    return remote_output_dir
