"""Reusable argument groups for HPC launchers.

This module provides functions to add common argument groups to argparse parsers,
enabling code reuse across local runners (run_tracegen.py, run_eval.py) and
cloud launchers (launch_tracegen_cloud.py, launch_eval_cloud.py).

Each function adds a cohesive group of related arguments and allows callers to
customize defaults where needed (e.g., different n_concurrent for tracegen vs eval).
"""

from __future__ import annotations

import argparse
from argparse import ArgumentParser, _ArgumentGroup
from typing import List, Optional, Union

# Type alias for parser or argument group
ArgTarget = Union[ArgumentParser, _ArgumentGroup]


def add_harbor_args(parser: ArgTarget, *, config_required: bool = True) -> None:
    """Add Harbor configuration arguments.

    Args:
        parser: ArgumentParser or argument group to add arguments to.
        config_required: Whether --harbor-config is required (default True).
    """
    parser.add_argument(
        "--harbor-config",
        required=config_required,
        help="Path to Harbor job config YAML.",
    )
    parser.add_argument(
        "--agent",
        default="terminus-2",
        help="Harbor agent name.",
    )
    parser.add_argument(
        "--job-name",
        help="Optional override for Harbor job name.",
    )
    parser.add_argument(
        "--agent-kwarg",
        action="append",
        default=[],
        help="Additional --agent-kwarg entries (key=value).",
    )
    parser.add_argument(
        "--harbor-extra-arg",
        action="append",
        default=[],
        help="Extra --harbor jobs start args.",
    )


def add_harbor_env_arg(
    parser: ArgTarget,
    *,
    default: str = "daytona",
    legacy_names: Optional[List[str]] = None,
) -> None:
    """Add Harbor environment backend argument (unified name).

    Args:
        parser: ArgumentParser or argument group to add arguments to.
        default: Default environment backend (default "daytona").
        legacy_names: Optional list of legacy argument names to support as hidden aliases.
    """
    parser.add_argument(
        "--harbor-env",
        default=default,
        choices=["daytona", "docker", "modal"],
        help="Harbor environment backend: daytona (cloud), docker (local/podman), modal.",
    )

    # Support legacy names as hidden aliases for backwards compatibility
    if legacy_names:
        for name in legacy_names:
            parser.add_argument(
                name,
                dest="harbor_env",
                help=argparse.SUPPRESS,
            )


def add_model_compute_args(
    parser: ArgTarget,
    *,
    model_required: bool = False,
    default_n_concurrent: int = 16,
    default_n_attempts: int = 1,
    n_attempts_help: str = "Times to run each task (default: 1).",
) -> None:
    """Add model and compute resource arguments.

    Callers pass their specific defaults:
    - Tracegen: default_n_concurrent=64, default_n_attempts=1
    - Eval: default_n_concurrent=16, default_n_attempts=3

    Args:
        parser: ArgumentParser or argument group to add arguments to.
        model_required: Whether --model is required (default False).
        default_n_concurrent: Default concurrent trials (default 16).
        default_n_attempts: Default attempts per task (default 1).
        n_attempts_help: Help text for --n-attempts argument.
    """
    parser.add_argument(
        "--model",
        required=model_required,
        help="Model identifier.",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=default_n_concurrent,
        help=f"Concurrent trials (default: {default_n_concurrent}).",
    )
    parser.add_argument(
        "--n-attempts",
        type=int,
        default=default_n_attempts,
        help=n_attempts_help,
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )


def add_hf_upload_args(parser: ArgTarget) -> None:
    """Add HuggingFace upload arguments (common to tracegen and eval)."""
    parser.add_argument(
        "--upload-hf-repo",
        help="HuggingFace repo for traces upload.",
    )
    parser.add_argument(
        "--upload-hf-token",
        help="HuggingFace token (defaults to $HF_TOKEN).",
    )
    parser.add_argument(
        "--upload-hf-private",
        action="store_true",
        help="Create the HuggingFace repo as private.",
    )
    parser.add_argument(
        "--upload-hf-episodes",
        choices=["last", "all"],
        default="last",
        help="Which episodes to include in traces upload.",
    )


def add_database_upload_args(parser: ArgTarget) -> None:
    """Add Supabase database upload arguments (eval only)."""
    parser.add_argument(
        "--upload-to-database",
        action="store_true",
        help="Upload result abstracts to Supabase and traces to HuggingFace.",
    )
    parser.add_argument(
        "--upload-username",
        help="Username for Supabase result attribution (defaults to $UPLOAD_USERNAME or current user).",
    )
    parser.add_argument(
        "--upload-error-mode",
        choices=["skip_on_error", "rollback_on_error"],
        default="skip_on_error",
        help="Supabase upload error handling.",
    )
    parser.add_argument(
        "--upload-forced-update",
        action="store_true",
        help="Allow overwriting existing Supabase records.",
    )


def add_ray_vllm_args(parser: ArgTarget) -> None:
    """Add Ray cluster and vLLM server arguments (local runners only)."""
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host/IP for Ray and vLLM.",
    )
    parser.add_argument(
        "--ray-port",
        type=int,
        default=6379,
        help="Ray head node port.",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="vLLM OpenAI-compatible API port.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=None,
        help="Pipeline parallel size for vLLM.",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=None,
        help="Data parallel replicas for vLLM.",
    )
    parser.add_argument(
        "--health-max-attempts",
        type=int,
        default=100,
        help="Max health check attempts for vLLM.",
    )
    parser.add_argument(
        "--health-retry-delay",
        type=int,
        default=30,
        help="Seconds between health checks.",
    )


def add_log_path_args(parser: ArgTarget) -> None:
    """Add log file path arguments (local runners only)."""
    parser.add_argument(
        "--harbor-binary",
        default="harbor",
        help="Harbor CLI executable path.",
    )
    parser.add_argument(
        "--controller-log",
        default=None,
        help="Path for vLLM controller logs.",
    )
    parser.add_argument(
        "--ray-log",
        default=None,
        help="Path for Ray logs.",
    )
    parser.add_argument(
        "--harbor-log",
        default=None,
        help="Path for Harbor CLI logs.",
    )


__all__ = [
    "add_harbor_args",
    "add_harbor_env_arg",
    "add_model_compute_args",
    "add_hf_upload_args",
    "add_database_upload_args",
    "add_ray_vllm_args",
    "add_log_path_args",
]
