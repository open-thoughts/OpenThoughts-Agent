"""Shared utilities for local Ray/vLLM runners.

It provides managed subprocess handling for Ray clusters and vLLM servers,
datagen config parsing, Docker runtime setup, and Harbor command building.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hpc.vllm_utils import _build_vllm_cli_args, run_endpoint_health_check
from hpc.launch_utils import generate_served_model_id, hosted_vllm_alias, maybe_int, PROJECT_ROOT
from hpc.arg_groups import (
    add_harbor_args,
    add_model_compute_args,
    add_ray_vllm_args,
    add_log_path_args,
)

# Re-export model-specific utilities for backward compatibility
from hpc.model_utils import (
    is_gpt_oss_model,
    setup_gpt_oss_tiktoken,
    get_model_specific_env_vars,
    GPT_OSS_TIKTOKEN_FILES,
)

# Re-export harbor utilities for backward compatibility
from hpc.harbor_utils import (
    load_harbor_config,
    get_harbor_env_from_config,
    extract_agent_kwargs_from_config,
    apply_nested_key,
    parse_agent_kwarg_strings,
    serialize_agent_kwargs,
    default_job_name,
    build_harbor_command,
    merge_agent_kwargs,
    collect_extra_agent_kwargs,
    resolve_jobs_dir_path,
    build_endpoint_meta,
    load_endpoint_metadata,
)

# Structured harbor config parsing (same as HPC eval launcher)
from scripts.harbor.job_config_utils import load_job_config

# Re-export docker runtime utilities for backward compatibility
from hpc.docker_runtime import setup_docker_runtime_if_needed


@dataclass
class ManagedProcess:
    """A subprocess with graceful shutdown support."""

    name: str
    proc: subprocess.Popen
    _log_handle: Optional[object] = field(default=None, repr=False)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the process gracefully, falling back to kill if needed."""
        if self.proc.poll() is not None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        finally:
            if self._log_handle:
                try:
                    self._log_handle.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# File Descriptor Monitoring
# ---------------------------------------------------------------------------

DEFAULT_FD_MONITOR_INTERVAL = 120  # 2 minutes


class FileDescriptorMonitor:
    """Background thread that periodically logs file descriptor usage.

    Monitors how close the process is to hitting the NOFILE ulimit,
    which is useful for debugging "Too many open files" issues in
    high-concurrency workloads like Harbor trace generation.

    Usage:
        monitor = FileDescriptorMonitor(interval_seconds=120)
        monitor.start()
        # ... run workload ...
        monitor.stop()
    """

    def __init__(self, interval_seconds: int = DEFAULT_FD_MONITOR_INTERVAL):
        """Initialize the file descriptor monitor.

        Args:
            interval_seconds: How often to log FD usage (default: 120s)
        """
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _get_fd_usage(self) -> tuple:
        """Get current file descriptor usage.

        Returns:
            Tuple of (current_open_fds, soft_limit, hard_limit, percent_used)
        """
        try:
            # Get current open file descriptors for this process
            pid = os.getpid()
            fd_dir = Path(f"/proc/{pid}/fd")
            if fd_dir.exists():
                current_fds = len(list(fd_dir.iterdir()))
            else:
                # Fallback for non-Linux systems (macOS, etc.)
                # Count FDs by iterating through possible range
                current_fds = 0
                for fd in range(1024):
                    try:
                        os.fstat(fd)
                        current_fds += 1
                    except OSError:
                        pass

            # Get limits
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            percent_used = (current_fds / soft_limit * 100) if soft_limit > 0 else 0

            return current_fds, soft_limit, hard_limit, percent_used
        except Exception as e:
            return -1, -1, -1, 0.0

    def _log_status(self) -> None:
        """Log current file descriptor status."""
        current, soft, hard, percent = self._get_fd_usage()

        if current < 0:
            print("[fd-monitor] Unable to read file descriptor usage", flush=True)
            return

        # Determine status level
        if percent >= 90:
            level = "CRITICAL"
        elif percent >= 75:
            level = "WARNING"
        elif percent >= 50:
            level = "INFO"
        else:
            level = "OK"

        timestamp = time.strftime("%H:%M:%S")
        print(
            f"[fd-monitor] [{timestamp}] {level}: {current:,} / {soft:,} FDs open "
            f"({percent:.1f}% of soft limit, hard limit: {hard:,})",
            flush=True,
        )

        if percent >= 75:
            print(
                f"[fd-monitor] Consider reducing --n_concurrent or increasing ulimit -n",
                flush=True,
            )

    def _run(self) -> None:
        """Background thread loop."""
        # Initial status
        self._log_status()

        while not self._stop_event.is_set():
            self._stop_event.wait(self.interval)
            if not self._stop_event.is_set():
                self._log_status()

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread is not None:
            return
        if self.interval <= 0:
            print("[fd-monitor] Disabled (interval <= 0)", flush=True)
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[fd-monitor] Started monitoring (every {self.interval}s)", flush=True)

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        # Final status report
        self._log_status()
        print("[fd-monitor] Stopped", flush=True)


def _open_log_file(log_path: Optional[Path]) -> tuple:
    """Open a log file with line buffering for real-time tail access.

    Returns:
        Tuple of (stdout_dest, stderr_dest, log_file_handle)
    """
    if log_path:
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        return log_file, log_file, log_file
    return None, None, None


def start_ray(
    host: str,
    ray_port: int,
    num_gpus: int,
    num_cpus: int,
    log_path: Optional[Path] = None,
    memory: Optional[int] = None,
    object_store_memory: Optional[int] = None,
) -> ManagedProcess:
    """Start a single-node Ray cluster head.

    Args:
        host: IP address to bind to
        ray_port: Port for Ray head node
        num_gpus: Number of GPUs to expose
        num_cpus: Number of CPUs to expose
        log_path: Optional path for Ray logs (line-buffered)
        memory: Total memory Ray can use (bytes). If None, Ray auto-detects.
        object_store_memory: Ray object store (plasma) size (bytes). Default: 40GB.

    Returns:
        ManagedProcess wrapping the Ray head process
    """
    # Default object store memory to 40GB if not specified
    if object_store_memory is None:
        object_store_memory = 40 * 1024 * 1024 * 1024  # 40GB

    cmd = [
        "ray",
        "start",
        "--head",
        f"--node-ip-address={host}",
        f"--port={ray_port}",
        f"--num-gpus={num_gpus}",
        f"--num-cpus={num_cpus}",
        "--dashboard-host=0.0.0.0",
        "--block",
    ]

    # Add memory limits to prevent Ray from detecting more memory than available
    if memory is not None:
        cmd.append(f"--memory={memory}")
    if object_store_memory is not None:
        cmd.append(f"--object-store-memory={object_store_memory}")

    env = os.environ.copy()
    stdout, stderr, log_file = _open_log_file(log_path)

    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    process = ManagedProcess(name="ray", proc=popen, _log_handle=log_file)
    return process


def start_vllm_controller(
    model: str,
    host: str,
    ray_port: int,
    api_port: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    endpoint_path: Path,
    controller_script: Path,
    log_path: Optional[Path] = None,
    served_model_name: Optional[str] = None,
    extra_cli_args: Optional[List[str]] = None,
    extra_env_vars: Optional[dict] = None,
) -> ManagedProcess:
    """Start a vLLM controller process.

    Args:
        model: Model path/name for vLLM
        host: IP address to bind to
        ray_port: Ray head port to connect to
        api_port: Port for vLLM OpenAI-compatible API
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of pipeline stages
        data_parallel_size: Number of data parallel replicas
        endpoint_path: Path to write endpoint JSON
        controller_script: Path to start_vllm_ray_controller.py
        log_path: Optional path for vLLM logs (line-buffered)
        served_model_name: Optional custom model name for the API
        extra_cli_args: Additional CLI args to pass to vLLM
        extra_env_vars: Additional environment variables

    Returns:
        ManagedProcess wrapping the vLLM controller process
    """
    env = os.environ.copy()
    env["VLLM_MODEL_PATH"] = model
    env["PYTHONUNBUFFERED"] = "1"  # Ensure real-time log output

    if extra_env_vars:
        env.update(extra_env_vars)

    cmd = [
        sys.executable,
        str(controller_script),
        "--ray-address",
        f"{host}:{ray_port}",
        "--host",
        host,
        "--port",
        str(api_port),
        "--model",
        model,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--pipeline-parallel-size",
        str(pipeline_parallel_size),
        "--data-parallel-size",
        str(data_parallel_size),
        "--endpoint-json",
        str(endpoint_path),
    ]

    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])

    if extra_cli_args:
        cmd.extend(extra_cli_args)

    stdout, stderr, log_file = _open_log_file(log_path)

    popen = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)
    process = ManagedProcess(name="vllm_controller", proc=popen, _log_handle=log_file)
    return process


def wait_for_endpoint(
    endpoint_path: Path,
    controller: ManagedProcess,
    timeout: int = 300,
) -> None:
    """Wait for the vLLM endpoint JSON file to be created.

    Args:
        endpoint_path: Path to the endpoint JSON file
        controller: The vLLM controller process to monitor
        timeout: Maximum seconds to wait

    Raises:
        RuntimeError: If the controller exits before creating the endpoint
        TimeoutError: If the endpoint is not created within timeout
    """
    start = time.time()
    while time.time() - start < timeout:
        if controller.proc.poll() is not None:
            raise RuntimeError(
                "vLLM controller exited before writing the endpoint JSON. Check logs."
            )
        if endpoint_path.exists():
            return
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for endpoint JSON at {endpoint_path}")


def terminate_processes(processes: List[ManagedProcess]) -> None:
    """Terminate a list of managed processes in order."""
    for proc in processes:
        try:
            proc.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shared utilities for local runners (eval + tracegen)
# ---------------------------------------------------------------------------


def apply_datagen_defaults(args: argparse.Namespace) -> None:
    """Load datagen config and apply defaults to args.

    Uses the consolidated parse_datagen_config() to extract settings from YAML.

    Sets on args:
    - args.model (if not set)
    - args.tensor_parallel_size, pipeline_parallel_size, data_parallel_size
    - args.ray_port, api_port
    - args._vllm_cli_args, args._vllm_env_vars
    - args._engine_type (openai, anthropic, vllm_local, etc.)
    - args._needs_local_vllm (whether to start Ray/vLLM server)
    - args._extra_agent_kwargs (additional agent kwargs from datagen config)
    - args._parsed_datagen_config (the full ParsedDatagenConfig)

    Args:
        args: Parsed argparse namespace with datagen_config attribute
    """
    # Initialize defaults
    args._vllm_cli_args: List[str] = []
    args._vllm_env_vars: Dict[str, str] = {}
    args._engine_type: str = "vllm_local"
    args._needs_local_vllm: bool = True
    args._extra_agent_kwargs: Dict[str, Any] = {}
    args._parsed_datagen_config = None

    datagen_config = getattr(args, "datagen_config", None)
    if not datagen_config:
        return

    # Use consolidated parser
    from hpc.datagen_config_utils import parse_datagen_config

    parsed = parse_datagen_config(datagen_config)
    args._parsed_datagen_config = parsed
    args.datagen_config = str(parsed.config_path)

    # Apply parsed values to args
    args._engine_type = parsed.engine_type
    args._needs_local_vllm = parsed.needs_local_vllm
    args._extra_agent_kwargs = parsed.extra_agent_kwargs

    # Model path (CLI arg takes precedence)
    if getattr(args, "model", None) is None and parsed.model:
        args.model = parsed.model

    # Parallelism settings (CLI args take precedence)
    if getattr(args, "tensor_parallel_size", None) is None:
        args.tensor_parallel_size = parsed.tensor_parallel_size
    if getattr(args, "pipeline_parallel_size", None) is None:
        args.pipeline_parallel_size = parsed.pipeline_parallel_size
    if getattr(args, "data_parallel_size", None) is None:
        args.data_parallel_size = parsed.data_parallel_size

    # Port settings (CLI args take precedence)
    if getattr(args, "ray_port", None) is None:
        args.ray_port = parsed.ray_port
    if getattr(args, "api_port", None) is None:
        args.api_port = parsed.api_port

    # Build CLI args and env vars from vllm_server config
    if parsed.vllm_server_config:
        from dataclasses import asdict
        vllm_dict = asdict(parsed.vllm_server_config)
        cli_args, env_vars = _build_vllm_cli_args(vllm_dict)
        args._vllm_cli_args = cli_args
        args._vllm_env_vars = env_vars

    # Setup tiktoken encodings for GPT-OSS models
    if is_gpt_oss_model(args.model):
        _, tiktoken_env = setup_gpt_oss_tiktoken()
        args._vllm_env_vars.update(tiktoken_env)


# ---------------------------------------------------------------------------
# LocalHarborRunner base class
# ---------------------------------------------------------------------------
# Note: generate_served_model_id and hosted_vllm_alias are imported from
# hpc.launch_utils to avoid duplication.


class LocalHarborRunner:
    """Base class for local Harbor runners (tracegen, eval).

    This class encapsulates the common workflow for running Harbor jobs locally:
    1. Parse and validate arguments
    2. Set up defaults for parallelism, ports, etc.
    3. Start Ray cluster and vLLM controller
    4. Wait for endpoint to be ready
    5. Build and run Harbor command
    6. Clean up processes

    Subclasses should override:
    - JOB_PREFIX: Job name prefix (e.g., "tracegen", "eval")
    - DEFAULT_EXPERIMENTS_SUBDIR: Subdirectory for experiments (e.g., "trace_runs")
    - DEFAULT_N_CONCURRENT: Default concurrent trials
    - DATAGEN_CONFIG_REQUIRED: Whether datagen_config is required
    - get_env_type(): Return the environment type from args
    - get_dataset_label(): Return dataset label for job naming
    - get_dataset_for_harbor(): Return (dataset_slug, dataset_path) tuple
    - validate_args(): Additional argument validation
    - post_harbor_hook(): Called after Harbor completes (for uploads)
    - print_banner(): Print startup banner
    """

    # Subclass configuration - override these in subclasses
    JOB_PREFIX: str = "job"
    DEFAULT_EXPERIMENTS_SUBDIR: str = "runs"
    DEFAULT_N_CONCURRENT: int = 16
    DATAGEN_CONFIG_REQUIRED: bool = False
    DEFAULT_ENDPOINT_FILENAME: str = "vllm_endpoint.json"

    def __init__(self, args: argparse.Namespace, repo_root: Path):
        """Initialize the runner.

        Args:
            args: Parsed command-line arguments
            repo_root: Path to repository root
        """
        self.args = args
        self.repo_root = repo_root
        self.processes: List[ManagedProcess] = []
        self._endpoint_json: Optional[Path] = None
        self._endpoint_meta: Optional[Dict[str, Any]] = None
        self._harbor_job_name: Optional[str] = None
        self._fd_monitor: Optional[FileDescriptorMonitor] = None

    @classmethod
    def add_common_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add common arguments shared by all local runners.

        Uses shared argument groups from hpc.arg_groups for consistency
        with cloud launchers.

        Args:
            parser: ArgumentParser to add arguments to
        """
        # Harbor core configuration (--harbor-config, --agent, --job-name, --agent-kwarg, --harbor-extra-arg)
        add_harbor_args(parser, config_required=True)

        # Model and compute (--model, --n-concurrent, --n-attempts, --gpus, --dry-run)
        # Note: n_attempts default is 1; subclasses can override via their own defaults
        add_model_compute_args(
            parser,
            model_required=False,
            default_n_concurrent=cls.DEFAULT_N_CONCURRENT,
            default_n_attempts=1,
            n_attempts_help="Times to run each task for repeated trials (default: 1). Not retries on failure.",
        )

        # Ray/vLLM configuration (--host, --ray-port, --api-port, parallelism, health checks)
        add_ray_vllm_args(parser)

        # Log paths (--harbor-binary, --controller-log, --ray-log, --harbor-log)
        add_log_path_args(parser)

        # Local-runner-specific arguments (not in shared arg_groups)
        parser.add_argument("--cpus", type=int, help="CPUs to expose to Ray.")
        parser.add_argument(
            "--endpoint-json",
            help="Optional endpoint JSON path.",
        )

        # File descriptor monitoring
        parser.add_argument(
            "--fd_monitor_interval",
            type=int,
            default=DEFAULT_FD_MONITOR_INTERVAL,
            metavar="SECONDS",
            help=f"Interval for file descriptor monitoring (default: {DEFAULT_FD_MONITOR_INTERVAL}s). "
                 "Set to 0 to disable.",
        )
        parser.add_argument("--fd-monitor-interval", dest="fd_monitor_interval", help=argparse.SUPPRESS)

    def get_env_type(self) -> str:
        """Get the environment type from --harbor-env.

        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement get_env_type()")

    def get_dataset_label(self) -> str:
        """Get the dataset label for job naming.

        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement get_dataset_label()")

    def get_dataset_for_harbor(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (dataset_slug, dataset_path) for harbor command.

        Subclasses must override this method.

        Returns:
            Tuple of (dataset_slug, dataset_path) - one should be None
        """
        raise NotImplementedError("Subclasses must implement get_dataset_for_harbor()")

    def get_experiments_dir(self) -> Path:
        """Get the experiments directory path.

        Can be overridden by subclasses for custom logic.
        """
        if hasattr(self.args, "experiments_dir") and self.args.experiments_dir:
            return Path(self.args.experiments_dir).expanduser().resolve()
        return self.repo_root / self.DEFAULT_EXPERIMENTS_SUBDIR

    def validate_args(self) -> None:
        """Validate arguments - subclasses can override for additional checks."""
        pass

    def post_harbor_hook(self) -> None:
        """Called after Harbor completes - for uploads, etc.

        Subclasses should override this method to implement upload logic.
        """
        pass

    def print_banner(self) -> None:
        """Print startup banner - subclasses should override."""
        args = self.args
        needs_local_vllm = getattr(args, "_needs_local_vllm", True)
        engine_type = getattr(args, "_engine_type", "vllm_local")

        print(f"=== Local {self.JOB_PREFIX.title()} Runner ===")
        print(f"  Model: {args.model}")
        if needs_local_vllm:
            print(f"  TP/PP/DP: {args.tensor_parallel_size}/{args.pipeline_parallel_size}/{args.data_parallel_size}")
            print(f"  GPUs: {args.gpus}")
        else:
            print(f"  Engine: {engine_type} (API)")
        print("=" * 35)

    def setup(self) -> None:
        """Set up the runner - apply defaults, configure environment."""
        args = self.args

        # Apply datagen config defaults
        apply_datagen_defaults(args)

        # Set up Docker runtime if using docker backend
        setup_docker_runtime_if_needed(self.get_env_type())

        # Set parallelism defaults (only relevant for local vLLM)
        if args.tensor_parallel_size is None:
            args.tensor_parallel_size = 1
        if args.pipeline_parallel_size is None:
            args.pipeline_parallel_size = 1
        if args.data_parallel_size is None:
            args.data_parallel_size = 1

        # Validate model - required for local vLLM, optional for API engines
        needs_local_vllm = getattr(args, "_needs_local_vllm", True)
        if args.model is None and needs_local_vllm:
            raise ValueError("Provide --model or supply a datagen config with vllm_server.model_path.")

        # Generate served model ID (only for local vLLM)
        if needs_local_vllm:
            served_model_id = generate_served_model_id()
            args._served_model_id = served_model_id
            args._harbor_model_name = hosted_vllm_alias(served_model_id)
        else:
            # For API engines, use the model from datagen config directly
            args._served_model_id = None
            args._harbor_model_name = args.model

        # Set GPU/CPU defaults
        if args.gpus is None:
            args.gpus = max(
                1,
                args.tensor_parallel_size * args.pipeline_parallel_size * args.data_parallel_size,
            )
        if args.cpus is None:
            args.cpus = os.cpu_count() or 16

        # Set port defaults
        if args.ray_port is None:
            args.ray_port = 6379
        if args.api_port is None:
            args.api_port = 8000

        # Resolve paths
        args.harbor_config = str(Path(args.harbor_config).expanduser().resolve())

        # Load Harbor config (raw dict for backward compat)
        harbor_config_data = load_harbor_config(args.harbor_config)
        jobs_dir_value = harbor_config_data.get("jobs_dir") if isinstance(harbor_config_data, dict) else None
        args._jobs_dir_path = resolve_jobs_dir_path(jobs_dir_value, self.repo_root)
        args._harbor_config_data = harbor_config_data

        # Load structured JobConfig to extract defaults (same as HPC eval launcher)
        harbor_job = load_job_config(args.harbor_config)
        args._harbor_job_config = harbor_job

        # Apply n_concurrent from harbor config if CLI didn't override
        # (CLI default is set in add_model_compute_args, check if it's still at that default)
        config_n_concurrent = harbor_job.orchestrator.n_concurrent_trials if harbor_job.orchestrator else None
        if config_n_concurrent is not None and config_n_concurrent > 0:
            # Only override if args.n_concurrent is at the class default
            if getattr(args, "n_concurrent", None) == self.DEFAULT_N_CONCURRENT:
                args.n_concurrent = int(config_n_concurrent)

        # Apply n_attempts from harbor config if CLI didn't override
        config_n_attempts = harbor_job.n_attempts
        if config_n_attempts is not None and config_n_attempts > 0:
            # Only override if args.n_attempts is at the default of 1
            if getattr(args, "n_attempts", 1) == 1:
                args.n_attempts = int(config_n_attempts)

        # Subclass-specific validation
        self.validate_args()

    def _setup_directories(self) -> Tuple[Path, Path]:
        """Set up experiments and logs directories.

        Returns:
            Tuple of (experiments_dir, logs_dir)
        """
        experiments_dir = self.get_experiments_dir()
        experiments_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = experiments_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return experiments_dir, logs_dir

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        import signal as sig

        def _handle_signal(signum, _frame):
            print(f"\nSignal {signum} received; shutting down...", file=sys.stderr)
            self.cleanup()
            sys.exit(1)

        sig.signal(sig.SIGINT, _handle_signal)
        sig.signal(sig.SIGTERM, _handle_signal)

    def cleanup(self) -> None:
        """Clean up processes and monitoring threads."""
        # Stop file descriptor monitor
        if self._fd_monitor is not None:
            self._fd_monitor.stop()
            self._fd_monitor = None

        terminate_processes(self.processes[::-1])
        # Only stop Ray if we started it (local vLLM engines)
        needs_local_vllm = getattr(self.args, "_needs_local_vllm", True)
        if needs_local_vllm:
            subprocess.run(["ray", "stop", "--force"], check=False)

    def run(self) -> None:
        """Main entry point - start services and run Harbor."""
        args = self.args
        needs_local_vllm = getattr(args, "_needs_local_vllm", True)

        # Set up directories
        experiments_dir, logs_dir = self._setup_directories()

        # Set up endpoint JSON path (only used for local vLLM)
        self._endpoint_json = Path(args.endpoint_json or (experiments_dir / self.DEFAULT_ENDPOINT_FILENAME))
        if self._endpoint_json.exists():
            self._endpoint_json.unlink()

        # Change to repo root
        os.chdir(self.repo_root)

        # Set up log paths
        ray_log = Path(args.ray_log) if args.ray_log else logs_dir / "ray.log"
        controller_log = Path(args.controller_log) if args.controller_log else logs_dir / "vllm_controller.log"
        harbor_log = Path(args.harbor_log).expanduser().resolve() if args.harbor_log else None

        # Set up signal handlers
        self._setup_signal_handlers()

        # Print banner
        self.print_banner()

        # Start file descriptor monitor
        fd_interval = getattr(args, "fd_monitor_interval", DEFAULT_FD_MONITOR_INTERVAL)
        if fd_interval > 0:
            self._fd_monitor = FileDescriptorMonitor(interval_seconds=fd_interval)
            self._fd_monitor.start()

        # Start Ray and vLLM only if needed (local vLLM engine)
        if needs_local_vllm:
            controller_script = self.repo_root / "scripts" / "vllm" / "start_vllm_ray_controller.py"

            # Convert memory from GB to bytes if provided
            ray_memory = None
            if getattr(args, "ray_memory_gb", None) is not None:
                ray_memory = int(args.ray_memory_gb * 1024 * 1024 * 1024)
            ray_object_store = int(getattr(args, "ray_object_store_gb", 40.0) * 1024 * 1024 * 1024)

            ray_proc = start_ray(
                host=args.host,
                ray_port=args.ray_port,
                num_gpus=args.gpus,
                num_cpus=args.cpus,
                log_path=ray_log,
                memory=ray_memory,
                object_store_memory=ray_object_store,
            )
            self.processes.append(ray_proc)

            # Start vLLM controller
            vllm_proc = start_vllm_controller(
                model=args.model,
                host=args.host,
                ray_port=args.ray_port,
                api_port=args.api_port,
                tensor_parallel_size=args.tensor_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                data_parallel_size=args.data_parallel_size,
                endpoint_path=self._endpoint_json,
                controller_script=controller_script,
                log_path=controller_log,
                served_model_name=getattr(args, "_served_model_id", None),
                extra_cli_args=getattr(args, "_vllm_cli_args", []),
                extra_env_vars=getattr(args, "_vllm_env_vars", {}),
            )
            self.processes.append(vllm_proc)
        else:
            engine_type = getattr(args, "_engine_type", "unknown")
            print(f"[engine] Using {engine_type} API engine - skipping local Ray/vLLM startup")

        try:
            # Wait for endpoint and run health check (only for local vLLM)
            if needs_local_vllm:
                wait_for_endpoint(self._endpoint_json, vllm_proc)
                run_endpoint_health_check(
                    self._endpoint_json,
                    args.health_max_attempts,
                    args.health_retry_delay,
                    self.repo_root,
                )
                self._endpoint_meta = load_endpoint_metadata(self._endpoint_json)
            else:
                # For API engines, no local endpoint metadata
                self._endpoint_meta = None

            # Compute job name
            harbor_model = getattr(args, "_harbor_model_name", args.model)
            job_model_label = args.model or harbor_model or "model"
            dataset_label = self.get_dataset_label()
            job_name = args.job_name or default_job_name(self.JOB_PREFIX, dataset_label, job_model_label)
            self._harbor_job_name = job_name
            args._harbor_job_name = job_name

            # Get dataset info
            dataset_slug, dataset_path = self.get_dataset_for_harbor()

            # Build Harbor command
            harbor_cmd = build_harbor_command(
                harbor_binary=args.harbor_binary,
                harbor_config_path=args.harbor_config,
                harbor_config_data=getattr(args, "_harbor_config_data", {}),
                job_name=job_name,
                agent_name=args.agent,
                model_name=harbor_model,
                env_type=self.get_env_type(),
                n_concurrent=args.n_concurrent,
                n_attempts=args.n_attempts,
                endpoint_meta=self._endpoint_meta,
                agent_kwarg_overrides=list(args.agent_kwarg or []),
                harbor_extra_args=list(args.harbor_extra_arg or []),
                dataset_slug=dataset_slug,
                dataset_path=dataset_path,
                extra_agent_kwargs=getattr(args, "_extra_agent_kwargs", None),
            )
            print("Harbor command:", " ".join(harbor_cmd))

            if not args.dry_run:
                # Import here to avoid circular imports
                from hpc.cli_utils import run_harbor_cli
                run_harbor_cli(harbor_cmd, harbor_log)
                self.post_harbor_hook()
            else:
                print(f"[dry-run] Would run Harbor {self.JOB_PREFIX} job.")

        finally:
            self.cleanup()


__all__ = [
    # Process management
    "ManagedProcess",
    "start_ray",
    "start_vllm_controller",
    "wait_for_endpoint",
    "terminate_processes",
    # File descriptor monitoring
    "FileDescriptorMonitor",
    "DEFAULT_FD_MONITOR_INTERVAL",
    # Config and setup utilities
    "maybe_int",
    "apply_datagen_defaults",
    "setup_docker_runtime_if_needed",
    "load_harbor_config",
    "load_job_config",
    "get_harbor_env_from_config",
    "resolve_jobs_dir_path",
    # Endpoint utilities
    "run_endpoint_health_check",
    "load_endpoint_metadata",
    # Harbor command building
    "extract_agent_kwargs_from_config",
    "default_job_name",
    "apply_nested_key",
    "parse_agent_kwarg_strings",
    "serialize_agent_kwargs",
    "build_harbor_command",
    "merge_agent_kwargs",
    "collect_extra_agent_kwargs",
    # Model ID utilities (re-exported from launch_utils)
    "generate_served_model_id",
    "hosted_vllm_alias",
    # Base runner class
    "LocalHarborRunner",
    # Re-exports
    "_build_vllm_cli_args",
]
