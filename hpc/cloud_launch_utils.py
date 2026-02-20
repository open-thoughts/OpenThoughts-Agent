"""
Shared utilities for SkyPilot-based cloud launches.

This module provides common components for cloud launchers including:
- PeriodicRemoteSync: Background log synchronization
- Docker image utilities: Selection and normalization
- Path utilities: Repo-relative paths and workdir handling
- Resource building: SkyPilot resource configuration
- Remote command building: Setup scripts for containers
- HuggingFace dataset helpers: Pre-extract datasets for cloud sync
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union

if TYPE_CHECKING:
    import sky

# Re-export Docker image utilities from docker_runtime for backwards compatibility
from hpc.docker_runtime import (
    GHCR_IMAGE_BASE,
    DEFAULT_DOCKER_IMAGE,
    normalize_docker_image,
    select_docker_image,
    get_docker_image_for_providers,
)

# Re-export path utilities from launch_utils for backwards compatibility
from hpc.launch_utils import PROJECT_ROOT, repo_relative

# Re-export CLI utilities for backwards compatibility
from hpc.cli_utils import parse_comma_separated

# Re-export HuggingFace utilities for backwards compatibility
from hpc.hf_utils import is_hf_dataset_path

DEFAULT_LOG_SYNC_INTERVAL = 120  # 2 minutes
DEFAULT_CLOUD_OUTPUT_PREFIX = "trace_runs"  # Default output subdir prefix for cloud jobs

# Harbor git URL for cloud reinstalls (always fetch latest from branch)
HARBOR_GIT_URL = "git+https://github.com/laude-institute/harbor.git@penfever/temp-override"


# ---------------------------------------------------------------------------
# Job Name Derivation for Cloud Launchers
# ---------------------------------------------------------------------------


def derive_cloud_job_name(args: "argparse.Namespace", task_prefix: str = "cloud") -> str:
    """Construct a fallback job name for cloud launches.

    Pattern: {task_prefix}_{cloud_provider}_{model}_{dataset}
    Example: tracegen_gcp_qwen2.5-7b-instruct_swebench-verified

    Args:
        args: Parsed argparse namespace
        task_prefix: Prefix for the job name (e.g., "tracegen", "eval")

    Returns:
        Derived job name (max 63 chars for SkyPilot cluster naming)
    """
    import re

    def _sanitize_component(value: str) -> str:
        """Sanitize a string for use in job names."""
        value = value.strip().rstrip("/")
        if "/" in value:
            value = value.split("/")[-1]
        return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-_") or "task"

    parts: list[str] = [task_prefix]

    # Include cloud provider (use first if comma-separated)
    cloud_provider = getattr(args, "cloud_provider", None)
    if cloud_provider:
        first_provider = cloud_provider.split(",")[0].strip()
        parts.append(_sanitize_component(first_provider))

    # Model component
    model = getattr(args, "model", None)
    if model:
        parts.append(_sanitize_component(str(model)))

    # Dataset component (tasks_input_path, harbor_dataset, dataset, or dataset_path)
    dataset_value = (
        getattr(args, "tasks_input_path", None)
        or getattr(args, "harbor_dataset", None)
        or getattr(args, "dataset", None)
        or getattr(args, "dataset_path", None)
    )
    if dataset_value:
        parts.append(_sanitize_component(str(dataset_value)))

    job_name = "_".join(filter(None, parts))

    # SkyPilot cluster names must be valid DNS labels (<= 63 chars, lowercase)
    from hpc.launch_utils import truncate_for_cloud
    job_name = truncate_for_cloud(job_name).lower()
    return job_name


def infer_harbor_env_from_config(
    args: "argparse.Namespace",
    harbor_config_path: str,
    log_prefix: str = "[cloud]",
) -> None:
    """Infer --harbor_env from harbor config if not explicitly provided.

    Reads the harbor config YAML and extracts environment.type to set args.harbor_env.
    This allows users to omit --harbor_env when the config already specifies the backend.

    Args:
        args: Parsed argparse namespace (modified in place)
        harbor_config_path: Path to harbor config YAML file
        log_prefix: Prefix for log messages (e.g., "[eval-cloud]", "[tracegen-cloud]")
    """
    if args.harbor_env:
        return  # Already explicitly specified

    from hpc.harbor_utils import load_harbor_config

    harbor_cfg = load_harbor_config(harbor_config_path)
    env_config = harbor_cfg.get("environment", {})
    inferred_env = env_config.get("type")
    if inferred_env:
        args.harbor_env = inferred_env
        print(f"{log_prefix} Inferred --harbor_env={inferred_env} from harbor config")


# Combined cloud dependency install command.
# Use uv for better dependency resolution (pip doesn't track all installed packages).
# Installs harbor from git, pins huggingface-hub <1.0 (vLLM/transformers compat),
# and pins numpy==2.2.6 (Numba compat: "Numba needs NumPy 2.2 or less").
CLOUD_DEPS_INSTALL_CMD = (
    f'uv pip install --upgrade --force-reinstall '
    f'"harbor @ {HARBOR_GIT_URL}" '
    f'"huggingface-hub>=0.34.0,<1.0" '
    f'"numpy==2.2.6"'
)


# ---------------------------------------------------------------------------
# Periodic Remote Sync
# ---------------------------------------------------------------------------


class PeriodicRemoteSync:
    """Background thread that periodically syncs files from a remote cluster.

    Uses rsync over SSH (SkyPilot configures SSH with cluster name as host).
    """

    def __init__(
        self,
        cluster_name: str,
        remote_dir: str,
        local_dir: str,
        interval_seconds: int = DEFAULT_LOG_SYNC_INTERVAL,
    ):
        self.cluster_name = cluster_name
        self.remote_dir = remote_dir
        self.local_dir = local_dir
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sync(self) -> None:
        """Perform a single sync operation."""
        try:
            local_path = Path(self.local_dir)
            local_path.mkdir(parents=True, exist_ok=True)

            rsync_cmd = [
                "rsync", "-avz",
                f"{self.cluster_name}:{self.remote_dir}/",
                f"{self.local_dir}/",
            ]
            result = subprocess.run(
                rsync_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                synced_files = list(local_path.glob("**/*"))
                if synced_files:
                    print(f"[sync] Synced {len(synced_files)} file(s) to {self.local_dir}", flush=True)
            elif result.returncode == 23:
                # rsync code 23: partial transfer due to error (often "file/directory not found")
                stderr_lower = (result.stderr or "").lower()
                if "no such file" in stderr_lower or "does not exist" in stderr_lower or "change_dir" in stderr_lower:
                    pass  # Remote directory doesn't exist yet - expected early in job
                else:
                    print(f"[sync] Warning: rsync partial transfer (code 23): {result.stderr}", file=sys.stderr, flush=True)
            elif "No such file" in result.stderr or "does not exist" in result.stderr.lower():
                pass  # Remote directory doesn't exist yet
            else:
                print(f"[sync] Warning: rsync returned {result.returncode}", file=sys.stderr, flush=True)
        except subprocess.TimeoutExpired:
            pass  # Sync took too long, skip
        except Exception as e:
            print(f"[sync] Warning: sync failed: {e}", file=sys.stderr, flush=True)

    def _run(self) -> None:
        """Background thread loop."""
        time.sleep(5)  # Short initial delay
        while not self._stop_event.is_set():
            self._sync()
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        """Start the background sync thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[sync] Started periodic sync (every {self.interval}s) to {self.local_dir}", flush=True)

    def stop(self) -> None:
        """Stop the background sync thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._sync()  # Final sync
        print("[sync] Stopped periodic sync", flush=True)


# Backwards compatibility alias
PeriodicLogSync = PeriodicRemoteSync


# ---------------------------------------------------------------------------
# Path Utilities
# ---------------------------------------------------------------------------

def parse_gpu_count(accelerator: str) -> int:
    """Parse GPU count from accelerator spec.

    Args:
        accelerator: SkyPilot accelerator spec (e.g., "H100:2", "A100:1", "GH200:1")
                    Can also be comma-separated for fallbacks (uses first).

    Returns:
        GPU count (defaults to 1 if not parseable)
    """
    # Use first accelerator if comma-separated
    first_accel = accelerator.split(",")[0].strip()

    if ":" in first_accel:
        try:
            return int(first_accel.split(":", 1)[1])
        except ValueError:
            pass
    return 1


def get_repo_root() -> Path:
    """Get the repository root directory.

    Note: This is a backwards compatibility alias for PROJECT_ROOT.
    Prefer using PROJECT_ROOT directly.
    """
    return PROJECT_ROOT


def get_remote_workdir(no_sync: bool) -> str:
    """Get the remote working directory path.

    Args:
        no_sync: If True, use Docker's baked-in code path

    Returns:
        Remote working directory path
    """
    if no_sync:
        return "/opt/openthoughts"
    else:
        return "/sky/workdir"


# ---------------------------------------------------------------------------
# Resource Building
# ---------------------------------------------------------------------------


def build_sky_resources(
    provider_names: List[str],
    provider_configs: List,  # List[ProviderConfig]
    accelerators: List[str],
    regions: List[Optional[str]],
    zone: Optional[str],
    use_spot: bool,
    docker_image: Optional[str],
    resolve_cloud_fn,  # Callable to resolve provider name to sky.Cloud
    memory: Optional[str] = None,
    cpus: Optional[str] = None,
) -> Union["sky.Resources", Set["sky.Resources"]]:
    """Build SkyPilot resource configurations for all provider/accelerator/region combinations.

    Args:
        provider_names: List of provider names
        provider_configs: List of ProviderConfig objects (parallel to provider_names)
        accelerators: List of accelerator specs (e.g., ["H100:1", "A100:1"])
        regions: List of regions (can include None for auto)
        zone: Optional zone (applied to all)
        use_spot: Whether to use spot instances
        docker_image: Docker image (already normalized with docker: prefix)
        resolve_cloud_fn: Function to resolve provider name to sky.Cloud object
        memory: Minimum system RAM in GB (e.g., "128" or "128+")
        cpus: Minimum number of vCPUs (e.g., "32" or "32+")

    Returns:
        Single sky.Resources if only one combination, otherwise set of resources
    """
    import sky

    all_resources = []

    for pname, pconfig in zip(provider_names, provider_configs):
        for accel in accelerators:
            for region in regions:
                kwargs = {
                    "cloud": resolve_cloud_fn(pname),
                    "accelerators": accel,
                    "use_spot": use_spot if pconfig.supports_spot else False,
                }
                if region and pconfig.supports_regions:
                    kwargs["region"] = region
                if zone:
                    kwargs["zone"] = zone
                if docker_image and pconfig.supports_docker_runtime:
                    kwargs["image_id"] = docker_image
                if memory:
                    kwargs["memory"] = memory
                if cpus:
                    kwargs["cpus"] = cpus

                all_resources.append(sky.Resources(**kwargs))

    if len(all_resources) == 1:
        return all_resources[0]
    else:
        return set(all_resources)


# ---------------------------------------------------------------------------
# Remote Command Building
# ---------------------------------------------------------------------------


def build_remote_setup_script(
    workdir: str,
    main_cmd: str,
    secrets_path: Optional[str] = None,
    add_pythonpath: bool = True,
    extra_env_vars: Optional[Dict[str, str]] = None,
    pre_task_commands: Optional[List[str]] = None,
) -> str:
    """Build the remote setup script to run in the container.

    Args:
        workdir: Remote working directory
        main_cmd: Main command to execute
        secrets_path: Optional path to secrets file to source
        add_pythonpath: Whether to add workdir to PYTHONPATH
        extra_env_vars: Additional environment variables to export
        pre_task_commands: Optional list of commands to run before main_cmd
                          (e.g., package reinstalls, environment setup)

    Returns:
        Shell script as string (commands joined with &&)
    """
    cmds = [f"cd {workdir}", "set -euo pipefail"]

    if add_pythonpath:
        cmds.append(f"export PYTHONPATH={workdir}:${{PYTHONPATH:-}}")

    # Add extra environment variables
    if extra_env_vars:
        for key, value in extra_env_vars.items():
            cmds.append(f"export {key}={value}")

    if secrets_path:
        cmds.append(f"set -a && source {secrets_path} && set +a")

    # Add pre-task commands (e.g., harbor reinstall)
    if pre_task_commands:
        for pre_cmd in pre_task_commands:
            cmds.append(pre_cmd)

    cmds.append(main_cmd)

    if secrets_path:
        cmds.append(f"rm -f {secrets_path}")

    return " && ".join(cmds)


# ---------------------------------------------------------------------------
# Diagnostic Log Fetching
# ---------------------------------------------------------------------------


def fetch_diagnostic_logs(
    cluster_name: str,
    remote_logs_dir: str,
    local_logs_dir: str,
    include_ray_system: bool = True,
    verbose: bool = True,
) -> bool:
    """Fetch diagnostic logs from a failed job via rsync.

    Args:
        cluster_name: SkyPilot cluster name
        remote_logs_dir: Remote logs directory path
        local_logs_dir: Local destination for logs
        include_ray_system: Also fetch Ray system logs from /tmp/ray
        verbose: Print log contents after fetching

    Returns:
        True if any logs were fetched successfully
    """
    local_path = Path(local_logs_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    success = False

    # Sync main logs directory
    try:
        rsync_cmd = [
            "rsync", "-avz",
            f"{cluster_name}:{remote_logs_dir}/",
            f"{local_path}/",
        ]
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            success = True
            if verbose:
                # Display vLLM controller log
                vllm_log = local_path / "vllm_controller.log"
                if vllm_log.exists():
                    print(f"\n[cloud-debug] vLLM Controller Log ({vllm_log}):")
                    print(vllm_log.read_text()[-5000:])  # Last 5KB
                else:
                    print("[cloud-debug] No vLLM controller log found")

                # Display Ray log
                ray_log = local_path / "ray.log"
                if ray_log.exists():
                    lines = ray_log.read_text().splitlines()[-100:]
                    print("\n[cloud-debug] Ray Log (last 100 lines):")
                    print("\n".join(lines))
                else:
                    print("[cloud-debug] No Ray log found")
        else:
            print(f"[cloud-debug] rsync failed: {result.stderr}", file=sys.stderr)

    except subprocess.TimeoutExpired:
        print("[cloud-debug] Log retrieval timed out", file=sys.stderr)
    except Exception as e:
        print(f"[cloud-debug] Could not fetch logs: {e}", file=sys.stderr)

    # Also fetch Ray system logs
    if include_ray_system:
        try:
            ray_system_dir = local_path / "ray_system"
            ray_system_dir.mkdir(parents=True, exist_ok=True)
            ray_sys_cmd = [
                "rsync", "-avz",
                f"{cluster_name}:/tmp/ray/session_latest/logs/",
                f"{ray_system_dir}/",
            ]
            result = subprocess.run(ray_sys_cmd, capture_output=True, timeout=30)

            if result.returncode == 0 and verbose:
                raylet_log = ray_system_dir / "raylet.out"
                if raylet_log.exists():
                    lines = raylet_log.read_text().splitlines()[-50:]
                    print("\n[cloud-debug] Ray Raylet System Log (last 50 lines):")
                    print("\n".join(lines))
        except Exception:
            pass  # Non-critical

    return success


# ---------------------------------------------------------------------------
# HuggingFace Dataset Pre-Extract Helper
# ---------------------------------------------------------------------------


def prepare_hf_dataset_for_sync(
    hf_path: str,
    local_output_dir: Path,
    on_exist: str = "overwrite",
    verbose: bool = True,
) -> Path:
    """Download and prepare an HF dataset for SkyPilot file sync.

    Handles two dataset formats:
    1. Raw files (task-xxxx directories): Used directly
    2. Parquet format: Extracted to task directories first

    Args:
        hf_path: HuggingFace dataset identifier (org/repo-name)
        local_output_dir: Local directory to store the prepared dataset
        on_exist: How to handle existing directories ("skip", "overwrite", "error")
        verbose: Print progress information

    Returns:
        Path to the prepared dataset directory (ready for file_mounts)

    Raises:
        RuntimeError: If dataset download or extraction fails
    """
    local_output_dir = Path(local_output_dir)
    dataset_name = hf_path.split("/")[-1]
    output_path = local_output_dir / dataset_name

    if verbose:
        print(f"[hf-prep] Preparing HuggingFace dataset: {hf_path}")

    # Download HF dataset
    try:
        from data.commons import download_hf_dataset
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot import data.commons.download_hf_dataset. "
            f"Ensure the data module is installed: {exc}"
        ) from exc

    if verbose:
        print(f"[hf-prep] Downloading {hf_path}...")

    snapshot_dir = Path(download_hf_dataset(hf_path))

    if verbose:
        print(f"[hf-prep] Downloaded to: {snapshot_dir}")

    # Check format: raw files vs parquet
    task_dirs = list(snapshot_dir.glob("task-*"))
    parquet_files = list(snapshot_dir.rglob("*.parquet"))

    if task_dirs:
        # Raw files format - use directly
        if verbose:
            print(f"[hf-prep] Found {len(task_dirs)} task directories (raw format)")
        return snapshot_dir

    elif parquet_files:
        # Parquet format - extract first
        if verbose:
            print(f"[hf-prep] Found {len(parquet_files)} parquet file(s), extracting...")

        # Use existing extractor
        try:
            from scripts.harbor import tasks_parquet_converter as tpc
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot import parquet converter. Ensure scripts.harbor is available: {exc}"
            ) from exc

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Find the parquet file to use
        parquet_file = parquet_files[0]  # Use first if multiple

        if verbose:
            print(f"[hf-prep] Extracting {parquet_file} to {output_path}")

        # Extract tasks
        tpc.from_parquet(
            parquet_path=str(parquet_file),
            base=str(output_path),
            on_exist=on_exist,
        )

        if verbose:
            extracted_tasks = list(output_path.glob("task-*"))
            print(f"[hf-prep] Extracted {len(extracted_tasks)} task directories")

        return output_path

    else:
        raise RuntimeError(
            f"Dataset {hf_path} has no task-* directories or *.parquet files. "
            f"Contents: {list(snapshot_dir.iterdir())[:10]}"
        )


def maybe_prepare_dataset_path(
    path: str,
    local_cache_dir: Path,
    verbose: bool = True,
) -> str:
    """Check if path is an HF identifier and prepare if needed.

    Args:
        path: Path string (either local path or HF identifier)
        local_cache_dir: Directory to cache downloaded datasets
        verbose: Print progress information

    Returns:
        Local path to use (original if local, prepared path if HF)
    """
    if is_hf_dataset_path(path):
        prepared = prepare_hf_dataset_for_sync(
            hf_path=path,
            local_output_dir=local_cache_dir,
            verbose=verbose,
        )
        return str(prepared)
    else:
        return path


# ---------------------------------------------------------------------------
# Cloud Launcher Base Class
# ---------------------------------------------------------------------------


class CloudLauncher:
    """Base class for SkyPilot cloud launchers.

    Encapsulates common logic for:
    - Provider validation and configuration
    - Docker image selection
    - HuggingFace dataset preparation
    - SkyPilot task creation and launch
    - Job monitoring and log streaming
    - Output synchronization and cluster teardown

    Subclasses should implement:
    - add_task_specific_args(): Add argparse arguments specific to the task
    - build_task_command(): Build the main command to run on remote
    - get_dataset_arg_name(): Return the name of the dataset argument
    - sync_additional_outputs(): Sync any task-specific output directories
    """

    # Override these in subclasses
    task_name: str = "cloud-task"
    job_name_prefix: str = "cloud"  # Prefix for auto-derived job names
    default_output_subdir: str = "cloud_runs"
    default_n_concurrent: int = 16

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.hf_cache_dir = repo_root / ".hf_cloud_cache"

    # -------------------------------------------------------------------------
    # Argument Parsing (shared cloud args + task-specific)
    # -------------------------------------------------------------------------

    def create_argument_parser(self, description: str) -> "argparse.ArgumentParser":
        """Create argument parser with shared cloud arguments."""
        import argparse

        parser = argparse.ArgumentParser(description=description)

        # Task-specific args (subclasses implement this)
        self.add_task_specific_args(parser)

        # Cloud-specific options (shared across all launchers)
        self._add_cloud_args(parser)

        return parser

    def _add_cloud_args(self, parser: "argparse.ArgumentParser") -> None:
        """Add shared cloud arguments to parser.

        Convention: underscore_case primary flags with kebab-case aliases for compatibility.
        """
        import argparse as _argparse

        # Helper to add arg with alias
        def _add(primary, alias=None, **kw):
            parser.add_argument(primary, **kw)
            if alias:
                dest = primary.lstrip("-").replace("-", "_")
                parser.add_argument(alias, dest=dest, help=_argparse.SUPPRESS)

        _add(
            "--cloud_provider", "--cloud-provider",
            default="gcp",
            help="Cloud provider(s) to use. Comma-separated for fallbacks (e.g., 'gcp,aws,lambda'). "
            "Run with --list_providers for details.",
        )
        _add(
            "--list_providers", "--list-providers",
            action="store_true",
            help="List supported cloud providers and exit.",
        )
        parser.add_argument(
            "--region",
            help="Preferred region(s). Comma-separated for fallbacks.",
        )
        parser.add_argument("--zone", help="Preferred zone.")
        parser.add_argument(
            "--accelerator",
            default="A100:1",
            help="SkyPilot accelerator spec(s). Comma-separated for fallback options.",
        )
        parser.add_argument(
            "--memory",
            help="Minimum system RAM in GB (e.g., '128' or '128+' for at least 128GB).",
        )
        parser.add_argument(
            "--cpus",
            help="Minimum number of vCPUs (e.g., '32' or '32+' for at least 32 vCPUs).",
        )
        _add("--use_spot", "--use-spot", action="store_true", help="Use spot/preemptible instances.")
        _add(
            "--docker_image", "--docker-image",
            default=DEFAULT_DOCKER_IMAGE,
            help="Pre-built Docker image (default: auto-selects based on GPU count).",
        )
        _add("--task_name", "--task-name", help="SkyPilot task name (defaults to <job_name>).")
        _add("--cluster_name", "--cluster-name", help="SkyPilot cluster name (defaults to <job_name>).")
        _add(
            "--remote_output_subdir", "--remote-output-subdir",
            help=f"Subdirectory for outputs (defaults to '{DEFAULT_CLOUD_OUTPUT_PREFIX}/<job_name>').",
        )
        _add(
            "--local_sync_dir", "--local-sync-dir",
            help=f"Local sync directory (defaults to './{DEFAULT_CLOUD_OUTPUT_PREFIX}/<job_name>').",
        )
        _add("--secrets_env", "--secrets-env", help="Path to secrets.env to source inside the container.")
        _add(
            "--no_sync", "--no-sync",
            action="store_true",
            help="Skip syncing local codebase to VM (use code baked into Docker image).",
        )
        parser.add_argument(
            "--autostop",
            type=int,
            default=30,
            metavar="MINUTES",
            help="Auto-stop cluster after N minutes of idle time (default: 30).",
        )
        parser.add_argument(
            "--down",
            action="store_true",
            help="Tear down cluster after task completes.",
        )
        _add(
            "--log_sync_interval", "--log-sync-interval",
            type=int,
            default=DEFAULT_LOG_SYNC_INTERVAL,
            metavar="SECONDS",
            help=f"Interval for periodic log sync (default: {DEFAULT_LOG_SYNC_INTERVAL}s). Set to 0 to disable.",
        )
        _add(
            "--retry_until_up", "--retry-until-up",
            action="store_true",
            help="Keep retrying to provision until the cluster is up (useful for scarce GPU availability).",
        )

    def add_task_specific_args(self, parser: "argparse.ArgumentParser") -> None:
        """Add task-specific arguments. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement add_task_specific_args()")

    # -------------------------------------------------------------------------
    # Provider Configuration
    # -------------------------------------------------------------------------

    def validate_providers(self, args: "argparse.Namespace") -> tuple:
        """Parse and validate cloud providers, returning (names, configs)."""
        from hpc.cloud_providers import get_provider_config, check_provider_credentials

        provider_names = parse_comma_separated(args.cloud_provider)
        provider_configs = [get_provider_config(p) for p in provider_names]

        # Check credentials
        for pname, pconfig in zip(provider_names, provider_configs):
            creds_ok, creds_msg = check_provider_credentials(pname)
            if not creds_ok:
                print(f"[cloud] Warning ({pconfig.display_name}): {creds_msg}", file=sys.stderr)

        # Warn about spot limitations
        if args.use_spot and not all(pc.supports_spot for pc in provider_configs):
            no_spot = [pc.display_name for pc in provider_configs if not pc.supports_spot]
            print(
                f"[cloud] Warning: {', '.join(no_spot)} do not support spot instances.",
                file=sys.stderr,
            )

        # Warn about region limitations
        if args.region and not all(pc.supports_regions for pc in provider_configs):
            no_region = [pc.display_name for pc in provider_configs if not pc.supports_regions]
            print(
                f"[cloud] Warning: {', '.join(no_region)} do not support region selection.",
                file=sys.stderr,
            )

        # Warn about Docker backend + provider compatibility
        # Harbor's Docker backend requires host socket passthrough, which needs
        # the provider to support Docker runtime (for socket mounting)
        harbor_env = getattr(args, "harbor_env", None)
        if harbor_env == "docker":
            no_docker = [pc.display_name for pc in provider_configs if not pc.supports_docker_runtime]
            if no_docker:
                print(
                    f"[cloud] Warning: {', '.join(no_docker)} do not support Docker runtime. "
                    f"Harbor's Docker backend may not work on these providers.",
                    file=sys.stderr,
                )

        return provider_names, provider_configs

    # -------------------------------------------------------------------------
    # Docker Image Selection
    # -------------------------------------------------------------------------

    def get_docker_image(self, args: "argparse.Namespace", provider_configs: list) -> str | None:
        """Select and normalize Docker image based on args and provider support."""
        provider_docker_support = {
            pc.display_name: pc.supports_docker_runtime
            for pc in provider_configs
        }
        return get_docker_image_for_providers(
            args.docker_image,
            args.accelerator,
            provider_docker_support,
        )

    # -------------------------------------------------------------------------
    # HuggingFace Dataset Handling
    # -------------------------------------------------------------------------

    def prepare_hf_dataset_if_needed(
        self,
        dataset_path: str | None,
        file_mounts: dict,
    ) -> str | None:
        """Download HF dataset if needed and update file_mounts. Returns updated path."""
        if not dataset_path or not is_hf_dataset_path(dataset_path):
            return dataset_path

        print(f"[cloud] Detected HuggingFace dataset path: {dataset_path}")
        local_dataset_path = maybe_prepare_dataset_path(
            dataset_path,
            local_cache_dir=self.hf_cache_dir,
            verbose=True,
        )
        remote_dataset_path = f"/sky/datasets/{Path(local_dataset_path).name}"
        file_mounts[remote_dataset_path] = local_dataset_path
        print(f"[cloud] Dataset will be synced to: {remote_dataset_path}")
        return remote_dataset_path

    # -------------------------------------------------------------------------
    # File Mounts Building
    # -------------------------------------------------------------------------

    def build_file_mounts(
        self,
        args: "argparse.Namespace",
        remote_workdir: str,
        remote_secret_path: str | None,
    ) -> dict:
        """Build file mounts dictionary."""
        import os

        file_mounts = {}
        if not args.no_sync:
            file_mounts[remote_workdir] = self.repo_root.as_posix()
        if remote_secret_path and args.secrets_env:
            file_mounts[remote_secret_path] = os.path.abspath(args.secrets_env)

        # Mount Docker socket for Harbor Docker backend
        # This enables running containers via the host's Docker daemon
        harbor_env = getattr(args, "harbor_env", None)
        if harbor_env == "docker":
            file_mounts["/var/run/docker.sock"] = "/var/run/docker.sock"

        return file_mounts

    def setup_gpt_oss_if_needed(
        self,
        args: "argparse.Namespace",
        file_mounts: dict,
    ) -> Dict[str, str]:
        """Setup tiktoken encodings for GPT-OSS models if needed.

        Downloads tiktoken encoding files locally and adds file mount for
        remote access. Returns extra environment variables to set on remote.

        See: https://github.com/vllm-project/vllm/issues/22525

        Args:
            args: Parsed arguments (checks args.model and datagen_config)
            file_mounts: File mounts dict to update with tiktoken mount

        Returns:
            Dict of environment variables to export on remote (may be empty)
        """
        from hpc.model_utils import is_gpt_oss_model, setup_gpt_oss_tiktoken

        # Check model from args or datagen config
        model = getattr(args, "model", None)
        if not model:
            datagen_config = getattr(args, "datagen_config", None)
            if datagen_config:
                try:
                    import yaml
                    cfg_path = Path(datagen_config).expanduser().resolve()
                    if cfg_path.exists():
                        with cfg_path.open("r", encoding="utf-8") as f:
                            cfg = yaml.safe_load(f) or {}
                        engine_cfg = cfg.get("engine") or {}
                        model = engine_cfg.get("model")
                except Exception:
                    pass

        if not is_gpt_oss_model(model):
            return {}

        print("[cloud] Detected GPT-OSS model, setting up tiktoken encodings...")
        local_cache, env_vars = setup_gpt_oss_tiktoken()

        # Add mount for tiktoken cache
        remote_tiktoken_path = "/opt/tiktoken_encodings"
        file_mounts[remote_tiktoken_path] = str(local_cache)

        # Return env vars with remote path
        return {"TIKTOKEN_ENCODINGS_BASE": remote_tiktoken_path}

    # -------------------------------------------------------------------------
    # Resource Building
    # -------------------------------------------------------------------------

    def build_resources(
        self,
        args: "argparse.Namespace",
        provider_names: list,
        provider_configs: list,
        docker_image: str | None,
    ):
        """Build SkyPilot resources for all provider/accelerator/region combinations."""
        from hpc.cloud_providers import resolve_cloud

        accelerator_options = parse_comma_separated(args.accelerator)
        region_options = parse_comma_separated(args.region) if args.region else [None]

        return build_sky_resources(
            provider_names=provider_names,
            provider_configs=provider_configs,
            accelerators=accelerator_options,
            regions=region_options,
            zone=args.zone,
            use_spot=args.use_spot,
            docker_image=docker_image,
            resolve_cloud_fn=resolve_cloud,
            memory=getattr(args, "memory", None),
            cpus=getattr(args, "cpus", None),
        )

    # -------------------------------------------------------------------------
    # Launch Status Printing
    # -------------------------------------------------------------------------

    def print_launch_status(
        self,
        args: "argparse.Namespace",
        provider_configs: list,
        docker_image: str | None,
        remote_workdir: str,
        num_resources: int,
    ) -> None:
        """Print launch configuration summary."""
        sync_status = f"enabled -> {remote_workdir}" if not args.no_sync else "disabled (using Docker image)"
        image_status = docker_image if docker_image else "(provider default)"
        provider_status = " | ".join(pc.display_name for pc in provider_configs)
        accel_status = args.accelerator
        region_status = args.region if args.region else "auto"
        autostop_status = f"{args.autostop} min" if args.autostop > 0 else "disabled"

        print(f"[cloud] Launching SkyPilot task '{args.task_name}'")
        print(f"[cloud]   Provider(s): {provider_status}")
        print(f"[cloud]   Region(s): {region_status}")
        print(f"[cloud]   Accelerator(s): {accel_status}")
        if getattr(args, "memory", None):
            print(f"[cloud]   Memory: {args.memory} GB")
        if getattr(args, "cpus", None):
            print(f"[cloud]   CPUs: {args.cpus}")
        print(f"[cloud]   Image: {image_status}")
        print(f"[cloud]   Code sync: {sync_status}")
        print(f"[cloud]   Autostop: {autostop_status}")

        # Show Harbor environment backend if specified
        harbor_env = getattr(args, "harbor_env", None)
        if harbor_env:
            harbor_status = harbor_env
            if harbor_env == "docker":
                harbor_status += " (host socket passthrough)"
            print(f"[cloud]   Harbor backend: {harbor_status}")

        if num_resources > 1:
            print(f"[cloud]   Candidate resources: {num_resources} combinations")
        if args.down:
            print("[cloud]   Teardown: enabled (cluster will be deleted after task)")

    # -------------------------------------------------------------------------
    # Launch and Wait
    # -------------------------------------------------------------------------

    def launch_task(self, task, args: "argparse.Namespace") -> tuple:
        """Launch SkyPilot task and return (job_id, cluster_name)."""
        import sky
        from typing import Sequence

        launch_kwargs = {"cluster_name": args.cluster_name}
        if args.autostop > 0:
            launch_kwargs["idle_minutes_to_autostop"] = args.autostop
        if args.retry_until_up:
            launch_kwargs["retry_until_up"] = True

        request_id = sky.launch(task, **launch_kwargs)
        launch_result = sky.stream_and_get(request_id)

        # Parse launch result
        job_id = None
        handle = None
        if isinstance(launch_result, tuple) and len(launch_result) == 2:
            potential_job_id, potential_handle = launch_result
            if isinstance(potential_job_id, int) or potential_job_id is None:
                job_id = potential_job_id
                handle = potential_handle
            else:
                handle = launch_result[0]
        elif isinstance(launch_result, Sequence):
            handle = launch_result[0] if launch_result else None
        else:
            handle = launch_result

        cluster_name = getattr(handle, "cluster_name", None) or args.cluster_name
        return job_id, cluster_name

    def get_periodic_sync_paths(
        self,
        args: "argparse.Namespace",
        remote_output_dir: str,
        remote_workdir: str,
    ) -> List[tuple]:
        """Return list of (remote_dir, local_dir) tuples for periodic sync during job execution.

        Override in subclasses to add task-specific directories like trace_jobs.
        The base implementation syncs the logs directory.

        Args:
            args: Parsed arguments
            remote_output_dir: Remote output directory path
            remote_workdir: Remote working directory path

        Returns:
            List of (remote_path, local_path) tuples to sync periodically
        """
        return [
            (f"{remote_output_dir}/logs", str(Path(args.local_sync_dir) / "logs")),
        ]

    def wait_for_job(
        self,
        cluster_name: str,
        job_id: int | None,
        args: "argparse.Namespace",
        remote_output_dir: str,
        remote_workdir: str,
    ) -> bool:
        """Wait for job completion, streaming logs. Returns True if job failed."""
        import sky

        # Start periodic sync for all configured directories
        sync_threads: List[PeriodicRemoteSync] = []
        if cluster_name and args.log_sync_interval > 0:
            sync_paths = self.get_periodic_sync_paths(args, remote_output_dir, remote_workdir)
            for remote_dir, local_dir in sync_paths:
                sync = PeriodicRemoteSync(
                    cluster_name=cluster_name,
                    remote_dir=remote_dir,
                    local_dir=local_dir,
                    interval_seconds=args.log_sync_interval,
                )
                sync.start()
                sync_threads.append(sync)

        job_failed = False
        if cluster_name:
            print(f"[cloud] Waiting for job to complete on cluster '{cluster_name}'...")
            try:
                if job_id is not None:
                    sky.tail_logs(cluster_name, job_id=job_id, follow=True)
                else:
                    print("[cloud] Job ID unavailable; streaming latest job logs.")
                    sky.tail_logs(cluster_name, job_id=None, follow=True)
            except KeyboardInterrupt:
                print("\n[cloud] Log streaming interrupted by user (Ctrl-C).", file=sys.stderr)
                print(f"[cloud] Job may still be running. Check with: sky queue {cluster_name}", file=sys.stderr)
            except Exception as e:
                print(f"[cloud] Warning: Failed to tail logs: {e}", file=sys.stderr)
            finally:
                for sync in sync_threads:
                    sync.stop()

            # Check job status
            job_failed = self._check_job_failed(cluster_name, job_id)

            # Fetch diagnostic logs if failed
            if job_failed:
                print("[cloud] Fetching logs for diagnostics via rsync...")
                local_logs_dir = str(Path(args.local_sync_dir) / "logs")
                remote_logs_dir = f"{remote_output_dir}/logs"
                fetch_diagnostic_logs(
                    cluster_name=cluster_name,
                    remote_logs_dir=remote_logs_dir,
                    local_logs_dir=local_logs_dir,
                    include_ray_system=True,
                    verbose=True,
                )

        return job_failed

    def _check_job_failed(self, cluster_name: str, job_id: int | None) -> bool:
        """Check if the job failed."""
        import sky

        try:
            queue_result = sky.queue(cluster_name, all_users=False)
            jobs = sky.stream_and_get(queue_result) or []
            for job in jobs:
                jid = getattr(job, "job_id", None)
                status = getattr(job, "status", None)
                if job_id is not None and jid == job_id:
                    if status and "FAILED" in str(status).upper():
                        print(
                            f"[cloud] Job {job_id} FAILED. Attempting to retrieve diagnostic logs...",
                            file=sys.stderr,
                        )
                        return True
                    break
        except Exception as e:
            print(f"[cloud] Warning: Could not check job status: {e}", file=sys.stderr)

        return False

    # -------------------------------------------------------------------------
    # Output Sync and Teardown
    # -------------------------------------------------------------------------

    def sync_outputs(
        self,
        cluster_name: str,
        args: "argparse.Namespace",
        remote_output_dir: str,
        remote_workdir: str,
    ) -> None:
        """Sync outputs from remote cluster."""
        from hpc.cloud_sync_utils import sync_outputs as do_sync

        # Sync main output directory
        do_sync(
            cluster_name=cluster_name,
            remote_path=remote_output_dir,
            local_dir=args.local_sync_dir,
        )

        # Sync additional task-specific outputs
        self.sync_additional_outputs(cluster_name, args, remote_workdir)

    def sync_additional_outputs(
        self,
        cluster_name: str,
        args: "argparse.Namespace",
        remote_workdir: str,
    ) -> None:
        """Sync additional task-specific outputs. Override in subclasses."""
        pass

    def teardown_cluster(self, cluster_name: str, args: "argparse.Namespace") -> None:
        """Tear down cluster if requested."""
        import sky

        if args.down and cluster_name:
            print(f"[cloud] Tearing down cluster '{cluster_name}'...")
            try:
                down_request = sky.down(cluster_name)
                sky.stream_and_get(down_request)
                print(f"[cloud] Cluster '{cluster_name}' terminated.")
            except Exception as e:
                print(f"[cloud] Warning: Failed to tear down cluster: {e}", file=sys.stderr)
                print(f"[cloud] Run manually: sky down {cluster_name}", file=sys.stderr)

    # -------------------------------------------------------------------------
    # Job Name and Path Resolution
    # -------------------------------------------------------------------------

    def resolve_job_name_and_defaults(self, args: "argparse.Namespace") -> None:
        """Resolve job_name and set defaults for task_name, cluster_name, output dirs.

        If job_name is not provided, derives it using derive_cloud_job_name().
        Then sets defaults for task_name, cluster_name, and path-related args based on job_name.

        Args:
            args: Parsed arguments (modified in place)
        """
        # Derive job_name if not provided
        job_name = getattr(args, "job_name", None)
        if not job_name:
            job_name = derive_cloud_job_name(args, task_prefix=self.job_name_prefix)
            args.job_name = job_name
            print(f"[cloud] Auto-derived job_name: {job_name}")

        # Set task_name default to job_name
        if not getattr(args, "task_name", None):
            args.task_name = job_name

        # Set cluster_name default to job_name
        if not getattr(args, "cluster_name", None):
            args.cluster_name = job_name

        # Set remote_output_subdir default to trace_runs/<job_name>
        if not getattr(args, "remote_output_subdir", None):
            args.remote_output_subdir = f"{DEFAULT_CLOUD_OUTPUT_PREFIX}/{job_name}"

        # Set local_sync_dir default to ./trace_runs/<job_name>
        if not getattr(args, "local_sync_dir", None):
            args.local_sync_dir = (self.repo_root / DEFAULT_CLOUD_OUTPUT_PREFIX / job_name).as_posix()

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def build_task_command(self, args: "argparse.Namespace", remote_output_dir: str) -> str:
        """Build the main command to execute on remote. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement build_task_command()")

    def get_dataset_arg_name(self) -> str | None:
        """Return the name of the dataset argument, or None if N/A. Override in subclasses."""
        return None

    def normalize_paths(self, args: "argparse.Namespace") -> None:
        """Normalize repo-relative paths. Override in subclasses."""
        pass

    def get_pre_task_commands(self, args: "argparse.Namespace") -> List[str]:
        """Return commands to run before the main task on remote.

        By default, reinstalls harbor and pins dependencies to ensure
        compatibility (huggingface-hub <1.0 for vLLM, numpy <=2.2 for Numba).

        If harbor_env is "docker", also sets up DOCKER_HOST for host socket
        passthrough (enables Harbor's Docker backend in cloud environments).

        Override in subclasses to customize or disable.

        Args:
            args: Parsed arguments

        Returns:
            List of shell commands to run before the main task
        """
        commands = [
            'echo "[cloud-setup] Installing dependencies (harbor, pinned huggingface-hub, numpy)..."',
            CLOUD_DEPS_INSTALL_CMD,
        ]

        # Setup Docker host socket for Harbor Docker backend
        harbor_env = getattr(args, "harbor_env", None)
        if harbor_env == "docker":
            commands.insert(0, 'echo "[cloud-setup] Setting up Docker host socket passthrough..."')
            commands.insert(1, 'export DOCKER_HOST=unix:///var/run/docker.sock')
            commands.insert(2, 'docker info > /dev/null 2>&1 && echo "[cloud-setup] Docker daemon accessible" || echo "[cloud-setup] WARNING: Docker daemon not accessible"')

        return commands

    def run(self, args: "argparse.Namespace") -> None:
        """Main entry point to run the cloud launcher."""
        import shlex
        import sky
        from hpc.cloud_providers import list_providers

        # Handle --list-providers early
        if args.list_providers:
            print(list_providers(verbose=True))
            return

        # Resolve job_name and set defaults for cluster_name, remote_output_subdir, local_sync_dir
        self.resolve_job_name_and_defaults(args)

        # Validate providers
        provider_names, provider_configs = self.validate_providers(args)

        # Prepare HF dataset if needed
        file_mounts = {}
        dataset_arg = self.get_dataset_arg_name()
        if dataset_arg:
            dataset_path = getattr(args, dataset_arg, None)
            new_path = self.prepare_hf_dataset_if_needed(dataset_path, file_mounts)
            if new_path and new_path != dataset_path:
                setattr(args, dataset_arg, new_path)

        # Normalize paths
        self.normalize_paths(args)

        # Get Docker image
        docker_image = self.get_docker_image(args, provider_configs)

        # Set up remote paths
        remote_workdir = get_remote_workdir(args.no_sync)
        remote_output_dir = f"{remote_workdir}/{args.remote_output_subdir}"

        # Build task command
        task_cmd = self.build_task_command(args, remote_output_dir)
        task_cmd_str = " ".join(shlex.quote(part) for part in task_cmd) if isinstance(task_cmd, list) else task_cmd

        # Build remote setup script
        remote_secret_path = None
        if args.secrets_env:
            secret_src = Path(args.secrets_env).expanduser().resolve()
            if not secret_src.exists():
                raise FileNotFoundError(f"secrets env file not found: {secret_src}")
            remote_secret_path = "/tmp/openthoughts_secrets.env"

        # Build file mounts
        base_mounts = self.build_file_mounts(args, remote_workdir, remote_secret_path)
        file_mounts.update(base_mounts)

        # Setup GPT-OSS tiktoken encodings if needed (modifies file_mounts)
        extra_env_vars = self.setup_gpt_oss_if_needed(args, file_mounts)

        # Setup Docker host for Harbor Docker backend
        harbor_env = getattr(args, "harbor_env", None)
        if harbor_env == "docker":
            extra_env_vars["DOCKER_HOST"] = "unix:///var/run/docker.sock"

        # Get pre-task commands (e.g., harbor reinstall)
        pre_task_commands = self.get_pre_task_commands(args)

        final_cmd = build_remote_setup_script(
            workdir=remote_workdir,
            main_cmd=task_cmd_str,
            secrets_path=remote_secret_path,
            add_pythonpath=not args.no_sync,
            extra_env_vars=extra_env_vars,
            pre_task_commands=pre_task_commands,
        )

        # Build resources
        resources = self.build_resources(args, provider_names, provider_configs, docker_image)
        num_resources = len(resources) if isinstance(resources, set) else 1

        # Create task
        task = sky.Task(name=args.task_name, run=final_cmd)
        task.set_resources(resources)
        if file_mounts:
            task.set_file_mounts(file_mounts)

        # Print status
        self.print_launch_status(args, provider_configs, docker_image, remote_workdir, num_resources)

        # Launch and wait
        job_id, cluster_name = self.launch_task(task, args)
        self.wait_for_job(cluster_name, job_id, args, remote_output_dir, remote_workdir)

        # Sync and teardown
        self.sync_outputs(cluster_name, args, remote_output_dir, remote_workdir)
        self.teardown_cluster(cluster_name, args)


__all__ = [
    # Constants
    "GHCR_IMAGE_BASE",
    "DEFAULT_DOCKER_IMAGE",
    "DEFAULT_LOG_SYNC_INTERVAL",
    "DEFAULT_CLOUD_OUTPUT_PREFIX",
    "HARBOR_GIT_URL",
    "CLOUD_DEPS_INSTALL_CMD",
    # Job name derivation
    "derive_cloud_job_name",
    # Periodic sync
    "PeriodicRemoteSync",
    "PeriodicLogSync",
    # Docker utilities
    "normalize_docker_image",
    "parse_gpu_count",
    "select_docker_image",
    # Path utilities
    "get_repo_root",
    "repo_relative",
    "get_remote_workdir",
    # Parsing
    "parse_comma_separated",
    # Resource building
    "build_sky_resources",
    # Remote commands
    "build_remote_setup_script",
    # Diagnostics
    "fetch_diagnostic_logs",
    # HuggingFace helpers
    "is_hf_dataset_path",
    "prepare_hf_dataset_for_sync",
    "maybe_prepare_dataset_path",
    # Cloud launcher base
    "CloudLauncher",
]
