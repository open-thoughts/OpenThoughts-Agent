"""Output path handling for Iris launchers."""

from __future__ import annotations

import argparse


# Default GCS prefix for workload outputs. EU-region matches where most
# of our v6e-preemptible TPU slices land; us-region jobs incur small
# cross-region writes (eval outputs are ~MB-scale, so this is fine).
# Override with $OT_AGENT_GCS_OUTPUT_ROOT or the --gcs-output-dir flag.
DEFAULT_GCS_OUTPUT_ROOT = "gs://marin-eu-west4/ot-agent"


def validate_output_args(args: argparse.Namespace) -> None:
    if not args.gcs_output_dir:
        raise SystemExit(
            "--gcs-output-dir is required (set OT_AGENT_GCS_OUTPUT_ROOT or pass the flag)."
        )


def resolve_remote_output_dir(
    args: argparse.Namespace,
    *,
    job_name: str,
    resume_target: str | None,
) -> str:
    """Return the workload output path seen by the Iris task."""
    if resume_target:
        # Resume: point at the OLD job's full GCS path so harbor finds
        # its existing config.json / trial dirs. Do NOT re-join job_name.
        return args._resume_gcs_output_dir.rstrip("/")
    return f"{args.gcs_output_dir.rstrip('/')}/{job_name}"
