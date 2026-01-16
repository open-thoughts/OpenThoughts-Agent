"""WandB utilities for HPC launchers.

This module provides utilities for:
1. WandB run initialization and metadata collection
2. Setting up WandB directories with proper permissions on HPC systems
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Optional, Tuple

import wandb


# =============================================================================
# WandB Run Initialization and Metadata
# =============================================================================


def wandb_init(kwargs: dict[str, Any]) -> None:
    """Initialize a wandb run using a normalized run name."""

    wandb_run_name = "_".join([str(value) for value in kwargs.values()])
    wandb_run_name = wandb_run_name.replace("/", "_")
    wandb_project = os.path.expandvars(os.environ.get("WANDB_PROJECT", "dcft"))
    wandb.init(project=wandb_project, name=wandb_run_name, config=kwargs)


def fetch_wandb_times(entity: str, project: str, run_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ISO timestamps for a wandb run, if accessible."""

    if not (entity and project and run_name):
        return None, None

    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        for run in runs:
            run_display = getattr(run, "display_name", None)
            run_name_attr = getattr(run, "name", None)
            if run_display == run_name or run_name_attr == run_name:
                start = getattr(run, "created_at", None)
                end = getattr(run, "finished_at", None) or getattr(run, "updated_at", None)
                start_iso = start.isoformat() if hasattr(start, "isoformat") else start
                end_iso = end.isoformat() if hasattr(end, "isoformat") else end
                return start_iso, end_iso
    except ValueError:
        return None, None
    return None, None


def collect_wandb_metadata(exp_args: dict, train_config: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return wandb link plus training start/end timestamps if wandb logging is enabled."""

    report_to = train_config.get("report_to", "")
    wandb_enabled = False
    if isinstance(report_to, str):
        wandb_enabled = report_to.lower() == "wandb"
    elif isinstance(report_to, (list, tuple, set)):
        wandb_enabled = any(str(item).lower() == "wandb" for item in report_to)

    if not wandb_enabled:
        return None, None, None

    project = os.path.expandvars(os.environ.get("WANDB_PROJECT", "dcft"))
    entity = (
        os.environ.get("WANDB_ENTITY")
        or os.environ.get("WANDB_USERNAME")
        or exp_args.get("job_creator")
    )
    run_name_value = train_config.get("run_name") or exp_args.get("job_name")
    run_name = str(run_name_value) if run_name_value else None

    wandb_link = None
    if entity and project and run_name:
        wandb_link = f"https://wandb.ai/{entity}/{project}/runs/{run_name}"

    training_start, training_end = fetch_wandb_times(entity, project, run_name)
    return wandb_link, training_start, training_end


# =============================================================================
# WandB Directory Setup (HPC-specific)
# =============================================================================


def ensure_wandb_dir(
    wandb_dir: Optional[str] = None,
    experiments_dir: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Ensure WandB directory exists and is writable.

    Creates the wandb directory if needed and fixes permissions to ensure
    WandB can write logs. Falls back to a temp directory if the primary
    location isn't writable.

    Args:
        wandb_dir: Explicit wandb directory path. If None, derives from experiments_dir.
        experiments_dir: Base experiments directory. Used if wandb_dir is None.
        verbose: Whether to print status messages.

    Returns:
        Path to the writable wandb directory.
    """
    # Determine wandb directory
    if wandb_dir:
        target_dir = Path(wandb_dir)
    elif experiments_dir:
        target_dir = Path(experiments_dir) / "wandb"
    else:
        # Fall back to environment variable or temp
        target_dir = Path(os.environ.get("WANDB_DIR", "/tmp/wandb"))

    # Create directory structure
    target_dir.mkdir(parents=True, exist_ok=True)

    # Also create the nested wandb/ subdirectory that wandb creates
    nested_dir = target_dir / "wandb"
    nested_dir.mkdir(parents=True, exist_ok=True)

    # Fix permissions
    _fix_wandb_permissions(target_dir, verbose=verbose)

    # Verify writable
    if _is_writable(target_dir):
        if verbose:
            print(f"[wandb_utils] WandB directory ready: {target_dir}")
        return str(target_dir)

    # Fall back to temp directory
    fallback_dir = Path("/tmp") / "wandb" / os.environ.get("USER", "unknown")
    fallback_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[wandb_utils] Warning: {target_dir} not writable, using fallback: {fallback_dir}")

    return str(fallback_dir)


def _fix_wandb_permissions(wandb_dir: Path, verbose: bool = True) -> None:
    """Fix permissions on WandB directory.

    Args:
        wandb_dir: Path to wandb directory.
        verbose: Whether to print status messages.
    """
    if not wandb_dir.exists():
        return

    if verbose:
        print(f"[wandb_utils] Fixing permissions on: {wandb_dir}")

    try:
        # Make directory and all contents writable by owner
        subprocess.run(
            ["chmod", "-R", "u+rwX", str(wandb_dir)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"[wandb_utils] Warning: chmod failed on {wandb_dir}: {e.stderr}")


def _is_writable(path: Path) -> bool:
    """Check if a directory is writable.

    Args:
        path: Path to check.

    Returns:
        True if directory is writable, False otherwise.
    """
    if not path.exists():
        return False

    # Try to create a test file
    test_file = path / ".wandb_write_test"
    try:
        test_file.touch()
        test_file.unlink()
        return True
    except (OSError, PermissionError):
        return False


def setup_wandb_env(
    wandb_dir: Optional[str] = None,
    experiments_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Set up WandB environment variables.

    Ensures the wandb directory is writable and sets appropriate
    environment variables for WandB logging.

    Args:
        wandb_dir: Explicit wandb directory path.
        experiments_dir: Base experiments directory.
        project_name: WandB project name.
        run_name: WandB run name.
        verbose: Whether to print status messages.

    Returns:
        Dictionary of environment variables that were set.
    """
    env_vars = {}

    # Ensure wandb directory is writable
    resolved_dir = ensure_wandb_dir(
        wandb_dir=wandb_dir,
        experiments_dir=experiments_dir,
        verbose=verbose,
    )
    os.environ["WANDB_DIR"] = resolved_dir
    env_vars["WANDB_DIR"] = resolved_dir

    # Set project and run name if provided
    if project_name:
        os.environ["WANDB_PROJECT"] = project_name
        env_vars["WANDB_PROJECT"] = project_name

    if run_name:
        os.environ["WANDB_RUN_NAME"] = run_name
        env_vars["WANDB_RUN_NAME"] = run_name

    return env_vars


__all__ = [
    # Run initialization and metadata
    "wandb_init",
    "fetch_wandb_times",
    "collect_wandb_metadata",
    # Directory setup
    "ensure_wandb_dir",
    "setup_wandb_env",
]
