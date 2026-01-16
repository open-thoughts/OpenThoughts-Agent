"""Reusable argument groups for HPC launchers.

This module provides functions to add common argument groups to argparse parsers,
enabling code reuse across local runners (run_tracegen.py, run_eval.py) and
cloud launchers (launch_tracegen_cloud.py, launch_eval_cloud.py).

Each function adds a cohesive group of related arguments and allows callers to
customize defaults where needed (e.g., different n_concurrent for tracegen vs eval).

Convention: Primary flags use underscore_case to match HPC launcher. Kebab-case
aliases are provided for backwards compatibility.
"""

from __future__ import annotations

import argparse
from argparse import ArgumentParser, _ArgumentGroup
from typing import List, Optional, Union

# Type alias for parser or argument group
ArgTarget = Union[ArgumentParser, _ArgumentGroup]


def _add_arg_with_alias(
    parser: ArgTarget,
    primary: str,
    alias: Optional[str] = None,
    **kwargs,
) -> None:
    """Add an argument with optional kebab-case alias.

    Args:
        parser: ArgumentParser or argument group.
        primary: Primary flag name (underscore_case, e.g., "--harbor_config").
        alias: Optional alias (kebab-case, e.g., "--harbor-config").
        **kwargs: Arguments passed to add_argument.
    """
    parser.add_argument(primary, **kwargs)
    if alias:
        # Add hidden alias that maps to same dest
        dest = primary.lstrip("-").replace("-", "_")
        parser.add_argument(alias, dest=dest, help=argparse.SUPPRESS)


def add_harbor_args(parser: ArgTarget, *, config_required: bool = True) -> None:
    """Add Harbor configuration arguments.

    Args:
        parser: ArgumentParser or argument group to add arguments to.
        config_required: Whether --harbor_config is required (default True).
    """
    _add_arg_with_alias(
        parser,
        "--harbor_config",
        "--harbor-config",
        required=config_required,
        help="Path to Harbor job config YAML.",
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Harbor agent name. If not specified, uses the agent from --harbor_config.",
    )
    _add_arg_with_alias(
        parser,
        "--job_name",
        "--job-name",
        help="Optional override for Harbor job name.",
    )
    _add_arg_with_alias(
        parser,
        "--agent_kwarg",
        "--agent-kwarg",
        action="append",
        default=[],
        help="Additional --agent-kwarg entries (key=value).",
    )
    _add_arg_with_alias(
        parser,
        "--harbor_extra_arg",
        "--harbor-extra-arg",
        action="append",
        default=[],
        help="Extra --harbor jobs start args.",
    )


def add_harbor_env_arg(
    parser: ArgTarget,
    *,
    default: Optional[str] = None,
    legacy_names: Optional[List[str]] = None,
) -> None:
    """Add Harbor environment backend argument (unified name).

    Args:
        parser: ArgumentParser or argument group to add arguments to.
        default: Default environment backend. If None (default), the environment
                will be inferred from the Harbor config YAML's `environment.type` field.
        legacy_names: Optional list of legacy argument names to support as hidden aliases.
    """
    _add_arg_with_alias(
        parser,
        "--harbor_env",
        "--harbor-env",
        default=default,
        choices=["daytona", "docker", "modal", None],
        help="Harbor environment backend: daytona (cloud), docker (local/podman), modal. "
             "If not specified, inferred from Harbor config YAML.",
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
        n_attempts_help: Help text for --n_attempts argument.
    """
    parser.add_argument(
        "--model",
        required=model_required,
        help="Model identifier.",
    )
    _add_arg_with_alias(
        parser,
        "--n_concurrent",
        "--n-concurrent",
        type=int,
        default=default_n_concurrent,
        help=f"Concurrent trials (default: {default_n_concurrent}).",
    )
    _add_arg_with_alias(
        parser,
        "--n_attempts",
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
    _add_arg_with_alias(
        parser,
        "--dry_run",
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )


def add_hf_upload_args(parser: ArgTarget) -> None:
    """Add HuggingFace upload arguments (common to tracegen and eval)."""
    _add_arg_with_alias(
        parser,
        "--upload_hf_repo",
        "--upload-hf-repo",
        help="HuggingFace repo for traces upload.",
    )
    _add_arg_with_alias(
        parser,
        "--upload_hf_token",
        "--upload-hf-token",
        help="HuggingFace token (defaults to $HF_TOKEN).",
    )
    _add_arg_with_alias(
        parser,
        "--upload_hf_private",
        "--upload-hf-private",
        action="store_true",
        help="Create the HuggingFace repo as private.",
    )
    _add_arg_with_alias(
        parser,
        "--upload_hf_episodes",
        "--upload-hf-episodes",
        choices=["last", "all"],
        default="last",
        help="Which episodes to include in traces upload.",
    )


def add_database_upload_args(parser: ArgTarget) -> None:
    """Add Supabase database upload arguments (eval only)."""
    _add_arg_with_alias(
        parser,
        "--upload_to_database",
        "--upload-to-database",
        action="store_true",
        help="Upload result abstracts to Supabase and traces to HuggingFace.",
    )
    _add_arg_with_alias(
        parser,
        "--upload_username",
        "--upload-username",
        help="Username for Supabase result attribution (defaults to $UPLOAD_USERNAME or current user).",
    )
    _add_arg_with_alias(
        parser,
        "--upload_error_mode",
        "--upload-error-mode",
        choices=["skip_on_error", "rollback_on_error"],
        default="skip_on_error",
        help="Supabase upload error handling.",
    )
    _add_arg_with_alias(
        parser,
        "--upload_forced_update",
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
    _add_arg_with_alias(
        parser,
        "--ray_port",
        "--ray-port",
        type=int,
        default=6379,
        help="Ray head node port.",
    )
    _add_arg_with_alias(
        parser,
        "--api_port",
        "--api-port",
        type=int,
        default=8000,
        help="vLLM OpenAI-compatible API port.",
    )
    _add_arg_with_alias(
        parser,
        "--tensor_parallel_size",
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size for vLLM.",
    )
    _add_arg_with_alias(
        parser,
        "--pipeline_parallel_size",
        "--pipeline-parallel-size",
        type=int,
        default=None,
        help="Pipeline parallel size for vLLM.",
    )
    _add_arg_with_alias(
        parser,
        "--data_parallel_size",
        "--data-parallel-size",
        type=int,
        default=None,
        help="Data parallel replicas for vLLM.",
    )
    _add_arg_with_alias(
        parser,
        "--health_max_attempts",
        "--health-max-attempts",
        type=int,
        default=100,
        help="Max health check attempts for vLLM.",
    )
    _add_arg_with_alias(
        parser,
        "--health_retry_delay",
        "--health-retry-delay",
        type=int,
        default=30,
        help="Seconds between health checks.",
    )
    # Memory configuration for Ray (prevents OOM from over-detection)
    _add_arg_with_alias(
        parser,
        "--ray_memory_gb",
        "--ray-memory-gb",
        type=float,
        default=None,
        help="Total memory (GB) for Ray. Auto-detected if not set.",
    )
    _add_arg_with_alias(
        parser,
        "--ray_object_store_gb",
        "--ray-object-store-gb",
        type=float,
        default=40.0,
        help="Ray object store (plasma) size in GB (default: 40).",
    )


def add_log_path_args(parser: ArgTarget) -> None:
    """Add log file path arguments (local runners only)."""
    _add_arg_with_alias(
        parser,
        "--harbor_binary",
        "--harbor-binary",
        default="harbor",
        help="Harbor CLI executable path.",
    )
    _add_arg_with_alias(
        parser,
        "--controller_log",
        "--controller-log",
        default=None,
        help="Path for vLLM controller logs.",
    )
    _add_arg_with_alias(
        parser,
        "--ray_log",
        "--ray-log",
        default=None,
        help="Path for Ray logs.",
    )
    _add_arg_with_alias(
        parser,
        "--harbor_log",
        "--harbor-log",
        default=None,
        help="Path for Harbor CLI logs.",
    )


def add_rl_training_args(
    parser: ArgTarget,
    *,
    default_chat_template: str = "skyrl-train/examples/terminal_bench/qwen3_thinking_acc.jinja2",
    default_eval_interval: int = 20,
) -> None:
    """Add RL (reinforcement learning) training arguments.

    These arguments are specific to SkyRL-based RL training jobs.

    Args:
        parser: ArgumentParser or argument group to add arguments to.
        default_chat_template: Default chat template path relative to SKYRL_HOME.
        default_eval_interval: Default evaluation interval in steps.
    """
    _add_arg_with_alias(
        parser,
        "--chat_template_path",
        "--chat-template-path",
        default=default_chat_template,
        help=(
            "Path to Jinja2 chat template for model inference. "
            "Relative to SKYRL_HOME or absolute path. "
            f"Default: {default_chat_template}"
        ),
    )
    _add_arg_with_alias(
        parser,
        "--eval_interval",
        "--eval-interval",
        type=int,
        default=default_eval_interval,
        help=(
            "Number of training steps between evaluations. "
            f"Set to 0 to disable periodic evaluation. Default: {default_eval_interval}"
        ),
    )
    _add_arg_with_alias(
        parser,
        "--policy_num_nodes",
        "--policy-num-nodes",
        type=int,
        default=None,
        help=(
            "Number of nodes to use for policy (actor) workers. "
            "If not set, defaults to num_nodes (symmetric setup). "
            "Use for asymmetric actor/learner configurations."
        ),
    )
    _add_arg_with_alias(
        parser,
        "--tensor_parallel_size",
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "Tensor parallel size for vLLM inference engines. "
            "Higher values needed for larger models (70B+). Default: 1"
        ),
    )
    _add_arg_with_alias(
        parser,
        "--train_batch_size",
        "--train-batch-size",
        type=int,
        default=None,
        help="Training batch size per GPU. If not set, uses SkyRL default.",
    )
    _add_arg_with_alias(
        parser,
        "--eval_batch_size",
        "--eval-batch-size",
        type=int,
        default=None,
        help="Evaluation batch size. If not set, uses SkyRL default.",
    )
    _add_arg_with_alias(
        parser,
        "--max_episodes",
        "--max-episodes",
        type=int,
        default=None,
        help=(
            "Maximum number of episodes per task during rollout. "
            "If not set, uses SkyRL default."
        ),
    )
    _add_arg_with_alias(
        parser,
        "--skyrl_export_path",
        "--skyrl-export-path",
        default=None,
        help=(
            "Path for SkyRL to export model checkpoints. "
            "If not set, derived from experiments_dir/run_name/exports."
        ),
    )
    _add_arg_with_alias(
        parser,
        "--rl_use_conda",
        "--rl-use-conda",
        action="store_true",
        default=False,
        help=(
            "Use conda environment for RL instead of venv. "
            "Useful for clusters like Perlmutter where conda is preferred."
        ),
    )
    _add_arg_with_alias(
        parser,
        "--rl_conda_env",
        "--rl-conda-env",
        default="dcagent-rl",
        help=(
            "Name of conda environment to use for RL when --rl_use_conda is set. "
            "Default: dcagent-rl"
        ),
    )


def add_tasks_input_arg(
    parser: ArgTarget,
    *,
    required: bool = True,
) -> None:
    """Add tasks input path argument with alias.

    Args:
        parser: ArgumentParser or argument group to add arguments to.
        required: Whether the argument is required (default True).
    """
    parser.add_argument(
        "--tasks_input_path",
        required=required,
        help="Path to tasks directory (input for trace generation).",
    )
    # Hyphenated alias
    parser.add_argument(
        "--tasks-input-path",
        dest="tasks_input_path",
        help=argparse.SUPPRESS,
    )


__all__ = [
    "add_harbor_args",
    "add_harbor_env_arg",
    "add_model_compute_args",
    "add_hf_upload_args",
    "add_database_upload_args",
    "add_ray_vllm_args",
    "add_log_path_args",
    "add_rl_training_args",
    "add_tasks_input_arg",
]
