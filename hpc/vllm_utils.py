"""vLLM server management for Ray clusters.

This module provides a context manager for managing vLLM server lifecycle
on Ray clusters, eliminating duplicated vLLM setup code across SBATCH scripts.

Usage:
    from hpc.ray_utils import RayCluster, RayClusterConfig
    from hpc.vllm_utils import VLLMServer, VLLMConfig

    ray_config = RayClusterConfig(num_nodes=4, gpus_per_node=4, cpus_per_node=48)

    with RayCluster.from_slurm(ray_config) as ray_cluster:
        vllm_config = VLLMConfig(
            model_path="meta-llama/Llama-3.1-70B-Instruct",
            tensor_parallel_size=4,
        )

        with VLLMServer(vllm_config, ray_cluster) as vllm:
            print(f"vLLM ready at {vllm.endpoint}")
            # ... run inference workloads
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hpc.ray_utils import RayCluster


# Fields from vllm_server/engine config that are for our system, not vLLM
# These are either internal fields, Harbor/Daytona-specific fields, or handled explicitly
_OUR_FIELDS = {
    # Internal/system fields
    "num_replicas",
    "time_limit",
    "endpoint_json_path",
    "model_path",
    # Harbor/Daytona engine-specific fields (not vLLM args)
    "type",                   # engine type (e.g., "vllm_local")
    "max_output_tokens",      # Harbor config, not vLLM
    "healthcheck_interval",   # Harbor config, not vLLM
    "vllm_local",            # Daytona backend config
    "model",                  # handled separately via --model
}

# Fields that map to different vLLM CLI arg names
_FIELD_RENAMES = {
    "model_path": "model",
}

# Boolean flags (passed as --flag without value when True)
_BOOLEAN_FLAGS = {
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "enable_auto_tool_choice",
    "trust_remote_code",
    "disable_log_requests",
    "enable_reasoning",
}

# Fields that are environment variables, not CLI args
_ENV_VAR_FIELDS = {
    "enable_expert_parallel": "VLLM_ENABLE_EXPERT_PARALLEL",
    "use_deep_gemm": "VLLM_USE_DEEP_GEMM",
}


def _build_vllm_cli_args(server_config: dict) -> tuple[list[str], dict[str, str]]:
    """Convert vllm_server config dict to CLI args and env vars.

    Returns:
        Tuple of (cli_args list, env_vars dict)
    """
    cli_args = []
    env_vars = {}

    for key, value in server_config.items():
        # Skip our internal fields
        if key in _OUR_FIELDS:
            continue

        # Skip None/empty values
        if value is None or value == "":
            continue

        # Handle extra_args (already a list of CLI args)
        if key == "extra_args":
            if isinstance(value, list):
                cli_args.extend(str(v) for v in value)
            continue

        # Handle env var fields
        if key in _ENV_VAR_FIELDS:
            if value:  # Only set if truthy
                env_vars[_ENV_VAR_FIELDS[key]] = "1"
            continue

        # Rename field if needed
        arg_name = _FIELD_RENAMES.get(key, key)

        # Convert underscore to dash for CLI
        arg_name = arg_name.replace("_", "-")

        # Handle boolean flags
        if key in _BOOLEAN_FLAGS:
            if value:  # Only add flag if True
                cli_args.append(f"--{arg_name}")
            continue

        # Handle regular key-value args
        if isinstance(value, bool):
            # Non-flag booleans: pass as true/false string
            cli_args.extend([f"--{arg_name}", str(value).lower()])
        else:
            cli_args.extend([f"--{arg_name}", str(value)])

    return cli_args, env_vars


@dataclass
class VLLMConfig:
    """Configuration for a vLLM server."""

    model_path: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    api_port: int = 8000
    endpoint_json_path: Optional[str] = None
    custom_model_name: Optional[str] = None

    # Health check settings
    health_max_attempts: int = 120
    health_retry_delay: int = 15
    health_path: str = "v1/models"

    # Controller script path
    controller_script: str = "scripts/vllm/start_vllm_ray_controller.py"
    wait_for_endpoint_script: str = "scripts/vllm/wait_for_endpoint.py"

    # Raw vllm_server config from YAML - passed through to vLLM
    server_config: dict = field(default_factory=dict)


@dataclass
class VLLMServer:
    """Context manager for vLLM server lifecycle.

    This class handles:
    - Starting vLLM API server on Ray cluster
    - Writing endpoint JSON for clients
    - Health checking until server is ready
    - Graceful shutdown on exit
    """

    config: VLLMConfig
    ray_cluster: "RayCluster"
    log_path: Optional[Path] = None
    extra_env_vars: Optional[Dict[str, str]] = None  # Additional env vars (e.g., tiktoken)
    _process: Optional[subprocess.Popen] = None
    _log_file: Optional[object] = None

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible API endpoint URL."""
        return f"http://{self.ray_cluster.head_ip}:{self.config.api_port}/v1"

    @property
    def base_url(self) -> str:
        """Base URL without /v1 suffix."""
        return f"http://{self.ray_cluster.head_ip}:{self.config.api_port}"

    @property
    def metrics_endpoint(self) -> str:
        """Prometheus metrics endpoint URL."""
        return f"http://{self.ray_cluster.head_ip}:{self.config.api_port}/metrics"

    def start(self) -> str:
        """Start the vLLM server.

        Returns:
            The API endpoint URL
        """
        if self._process is not None:
            print(f"vLLM server already started at {self.endpoint}")
            return self.endpoint

        print(f"=== Starting vLLM Server ===")
        print(f"  Model: {self.config.model_path}")
        print(f"  Tensor Parallel: {self.config.tensor_parallel_size}")
        print(f"  Pipeline Parallel: {self.config.pipeline_parallel_size}")
        print(f"  Data Parallel: {self.config.data_parallel_size}")
        print(f"  Host: {self.ray_cluster.head_ip}")
        print(f"  Port: {self.config.api_port}")
        print(f"  Ray Address: {self.ray_cluster.address}")
        print(f"============================")

        # Open log file if path provided (line-buffered for real-time tail access)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "w", buffering=1)  # Line buffering
            stdout_dest = self._log_file
            stderr_dest = subprocess.STDOUT
        else:
            stdout_dest = subprocess.DEVNULL
            stderr_dest = subprocess.DEVNULL

        # Build command
        cmd = [
            sys.executable,
            self.config.controller_script,
            "--ray-address",
            self.ray_cluster.address,
            "--host",
            self.ray_cluster.head_ip,
            "--port",
            str(self.config.api_port),
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.config.pipeline_parallel_size),
            "--data-parallel-size",
            str(self.config.data_parallel_size),
        ]

        if self.config.endpoint_json_path:
            cmd.extend(["--endpoint-json", self.config.endpoint_json_path])

        if self.config.custom_model_name:
            cmd.extend(["--served-model-name", self.config.custom_model_name])

        # Build CLI args and env vars from server_config (pass-through from YAML)
        if self.config.server_config:
            extra_cli_args, extra_env_vars = _build_vllm_cli_args(self.config.server_config)
            cmd.extend(extra_cli_args)
            if extra_cli_args:
                print(f"  Extra vLLM args: {' '.join(extra_cli_args[:10])}{'...' if len(extra_cli_args) > 10 else ''}")

        # Set environment
        env = os.environ.copy()
        env["VLLM_MODEL_PATH"] = self.config.model_path
        env["PYTHONUNBUFFERED"] = "1"  # Ensure real-time log output
        if self.config.server_config:
            env.update(extra_env_vars)
        # Merge extra env vars from caller (e.g., TIKTOKEN_ENCODINGS_BASE for GPT-OSS)
        if self.extra_env_vars:
            env.update(self.extra_env_vars)

        # Start the server process
        self._process = subprocess.Popen(
            cmd,
            stdout=stdout_dest,
            stderr=stderr_dest,
            env=env,
        )

        print(f"  Started vLLM controller (PID: {self._process.pid})")
        if self.log_path:
            print(f"  Log file: {self.log_path}")

        # Wait for server to be healthy
        self._wait_for_healthy()

        print(f"=== vLLM Server Ready ===")
        print(f"  Endpoint: {self.endpoint}")
        print(f"  Metrics: {self.metrics_endpoint}")
        print(f"=========================")

        return self.endpoint

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self._process is None:
            return

        print("Stopping vLLM server...")

        # Graceful termination
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
            print("  vLLM server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("  vLLM server not responding, killing...")
            self._process.kill()
            self._process.wait()

        self._process = None

        # Close log file
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _wait_for_healthy(self) -> None:
        """Wait for the vLLM server to be healthy."""
        # First wait for endpoint JSON if configured
        if self.config.endpoint_json_path:
            self._wait_for_endpoint_json()

        # Then use the wait script if available
        script_path = Path(self.config.wait_for_endpoint_script)
        if script_path.exists():
            self._wait_with_script()
        else:
            self._wait_with_http()

    def _wait_for_endpoint_json(self, timeout: int = 180) -> None:
        """Wait for endpoint JSON file to be created."""
        if not self.config.endpoint_json_path:
            return

        print(f"  Waiting for endpoint JSON: {self.config.endpoint_json_path}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process died
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM controller exited early (code {self._process.returncode}). "
                    f"Check logs at {self.log_path}"
                )

            if os.path.exists(self.config.endpoint_json_path):
                print(f"  Endpoint JSON found after {time.time() - start_time:.1f}s")
                return

            time.sleep(5)

        raise TimeoutError(
            f"Endpoint JSON not created at {self.config.endpoint_json_path} "
            f"after {timeout}s"
        )

    def _wait_with_script(self) -> None:
        """Wait using the wait_for_endpoint.py script."""
        cmd = [
            sys.executable,
            self.config.wait_for_endpoint_script,
            "--max-attempts",
            str(self.config.health_max_attempts),
            "--retry-delay",
            str(self.config.health_retry_delay),
            "--health-path",
            self.config.health_path,
        ]

        if self.config.endpoint_json_path:
            cmd.extend(["--endpoint-json", self.config.endpoint_json_path])
        else:
            # Direct URL mode
            cmd.extend(["--endpoint", self.base_url])

        print(f"  Running health check (max {self.config.health_max_attempts} attempts)...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"vLLM health check failed after {self.config.health_max_attempts} attempts"
            ) from e

    def _wait_with_http(self) -> None:
        """Fallback: wait using direct HTTP health checks."""
        import urllib.request
        import urllib.error

        health_url = f"{self.base_url}/{self.config.health_path}"
        print(f"  Waiting for health endpoint: {health_url}")

        for attempt in range(1, self.config.health_max_attempts + 1):
            # Check if process died
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM controller exited early (code {self._process.returncode})"
                )

            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        print(f"  Health check passed on attempt {attempt}")
                        return
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                pass

            if attempt < self.config.health_max_attempts:
                time.sleep(self.config.health_retry_delay)

        raise RuntimeError(
            f"vLLM health check failed after {self.config.health_max_attempts} attempts"
        )

    def get_endpoint_info(self) -> dict:
        """Get endpoint information as a dictionary.

        Useful for passing to client code or saving to a file.
        """
        return {
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            "metrics_endpoint": self.metrics_endpoint,
            "model": self.config.model_path,
            "host": self.ray_cluster.head_ip,
            "port": self.config.api_port,
        }

    def write_endpoint_json(self, path: Optional[str] = None) -> str:
        """Write endpoint information to a JSON file.

        Args:
            path: Path to write to. If None, uses config.endpoint_json_path

        Returns:
            The path that was written to
        """
        output_path = path or self.config.endpoint_json_path
        if not output_path:
            raise ValueError("No endpoint JSON path specified")

        info = self.get_endpoint_info()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(info, f, indent=2)

        return output_path

    def __enter__(self) -> VLLMServer:
        """Context manager entry - start the server."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop the server."""
        self.stop()


def create_vllm_server(
    model_path: str,
    ray_cluster: "RayCluster",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    api_port: int = 8000,
    log_dir: Optional[str] = None,
    job_name: Optional[str] = None,
) -> VLLMServer:
    """Convenience function to create a VLLMServer.

    Args:
        model_path: HuggingFace model path
        ray_cluster: RayCluster instance to run on
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of stages for pipeline parallelism
        data_parallel_size: Number of data parallel replicas
        api_port: Port for the API server
        log_dir: Directory for log files
        job_name: Job name for log file naming

    Returns:
        A configured VLLMServer instance
    """
    config = VLLMConfig(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        api_port=api_port,
    )

    log_path = None
    if log_dir and job_name:
        log_path = Path(log_dir) / f"{job_name}_vllm.log"
    elif log_dir:
        log_path = Path(log_dir) / "vllm_controller.log"

    return VLLMServer(config=config, ray_cluster=ray_cluster, log_path=log_path)


def run_endpoint_health_check(
    endpoint_json: Path,
    max_attempts: int,
    retry_delay: int,
    repo_root: Optional[Path] = None,
) -> None:
    """Run the vLLM endpoint health check script.

    This is a standalone function for running health checks outside of
    the VLLMServer context manager (e.g., for local runners that manage
    their own vLLM processes).

    Args:
        endpoint_json: Path to the endpoint JSON file
        max_attempts: Maximum number of health check attempts
        retry_delay: Delay in seconds between attempts
        repo_root: Repository root path (defaults to parent of hpc/)

    Raises:
        subprocess.CalledProcessError: If health check fails
    """
    if repo_root is None:
        # Default to repo root (parent of hpc/)
        repo_root = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "vllm" / "wait_for_endpoint.py"),
        "--endpoint-json",
        str(endpoint_json),
        "--max-attempts",
        str(max_attempts),
        "--retry-delay",
        str(retry_delay),
        "--health-path",
        "v1/models",
    ]
    subprocess.run(cmd, check=True)
