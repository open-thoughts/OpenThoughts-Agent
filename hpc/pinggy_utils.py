"""Pinggy tunnel utilities for exposing local vLLM endpoints to cloud containers.

When running evals with cloud-based sandbox environments (Daytona, Modal) that
cannot reach the HPC cluster's private network, we use Pinggy to create a
public HTTPS tunnel to the local vLLM server.

Usage:
    from hpc.pinggy_utils import PinggyTunnel, PinggyConfig, needs_pinggy_tunnel

    if needs_pinggy_tunnel(agent_name, env_type):
        config = PinggyConfig(
            persistent_url="bjfqkhfxtx.a.pinggy.link",
            ssh_command="ssh -p 443 -R0:localhost:8000 ...",
        )
        with PinggyTunnel(config) as tunnel:
            # Use tunnel.public_endpoint instead of local vLLM endpoint
            endpoint = tunnel.public_endpoint  # https://bjfqkhfxtx.a.pinggy.link
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import urllib.request
import urllib.error


@dataclass
class PinggyConfig:
    """Configuration for Pinggy tunnel.

    Only requires persistent_url and token - the SSH command is built automatically.
    """

    persistent_url: str  # e.g., "bjfqkhfxtx.a.pinggy.link"
    token: str  # Pinggy auth token (e.g., "oVxgHq855Ln")
    local_port: int = 8000  # Local vLLM port to tunnel
    local_host: str = "localhost"  # Local host to tunnel (can be IP from vLLM endpoint)
    health_check_timeout: int = 60  # Seconds to wait for tunnel to be ready
    health_check_interval: int = 2  # Seconds between health checks
    pinggy_host: str = "pro.pinggy.io"  # Pinggy server (pro.pinggy.io or free.pinggy.io)

    def get_ssh_command(self) -> str:
        """Build the SSH command for the Pinggy tunnel."""
        # Build a robust SSH command with auto-reconnect loop
        return (
            f"while true; do "
            f"ssh -p 443 "
            f"-R0:{self.local_host}:{self.local_port} "
            f"-o StrictHostKeyChecking=no "
            f"-o ServerAliveInterval=30 "
            f"-o ExitOnForwardFailure=yes "
            f"{self.token}@{self.pinggy_host}; "
            f"sleep 10; "
            f"done"
        )


@dataclass
class PinggyTunnel:
    """Manages a Pinggy tunnel process.

    The tunnel exposes a local port (typically vLLM on 8000) via Pinggy's
    public HTTPS endpoint. This allows cloud-based containers (Daytona, Modal)
    to reach the vLLM server running on an HPC compute node.
    """

    config: PinggyConfig
    _process: Optional[subprocess.Popen] = field(default=None, repr=False)
    _log_file: Optional[Any] = field(default=None, repr=False)
    log_path: Optional[Path] = None

    @property
    def public_endpoint(self) -> str:
        """Get the public HTTPS endpoint URL (OpenAI-compatible /v1 path)."""
        return f"https://{self.config.persistent_url}/v1"

    @property
    def public_base_url(self) -> str:
        """Get the public HTTPS base URL (without /v1)."""
        return f"https://{self.config.persistent_url}"

    @property
    def is_running(self) -> bool:
        """Check if the tunnel process is running."""
        return self._process is not None and self._process.poll() is None

    def start(self) -> str:
        """Start the Pinggy tunnel and return the public endpoint.

        Returns:
            The public HTTPS endpoint URL (e.g., https://xxx.a.pinggy.link/v1)

        Raises:
            RuntimeError: If tunnel fails to start or health check fails
        """
        if self._process is not None:
            print(f"Pinggy tunnel already running at {self.public_endpoint}")
            return self.public_endpoint

        print(f"=== Starting Pinggy Tunnel ===")
        print(f"  Persistent URL: {self.config.persistent_url}")
        print(f"  Local target: {self.config.local_host}:{self.config.local_port}")
        print(f"==============================")

        # Open log file if path provided
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "w", buffering=1)
            stdout_dest = self._log_file
            stderr_dest = subprocess.STDOUT
        else:
            stdout_dest = subprocess.DEVNULL
            stderr_dest = subprocess.DEVNULL

        # Parse and execute the SSH command (with host:port placeholders resolved)
        # The command is typically a shell loop, so we run it via bash
        ssh_cmd = self.config.get_ssh_command()
        cmd = ["bash", "-c", ssh_cmd]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=stdout_dest,
                stderr=stderr_dest,
                # Don't let the tunnel inherit our signal handlers
                preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start Pinggy tunnel: {e}")

        print(f"  Started Pinggy tunnel (PID: {self._process.pid})")
        if self.log_path:
            print(f"  Log file: {self.log_path}")

        # Wait for tunnel to be healthy
        self._wait_for_healthy()

        print(f"=== Pinggy Tunnel Ready ===")
        print(f"  Public endpoint: {self.public_endpoint}")
        print(f"===========================")

        return self.public_endpoint

    def stop(self) -> None:
        """Stop the Pinggy tunnel."""
        if self._process is None:
            return

        print("Stopping Pinggy tunnel...")

        # Try graceful termination first
        try:
            # Kill the process group to ensure all child processes are terminated
            if hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                self._process.terminate()

            self._process.wait(timeout=10)
            print("  Pinggy tunnel stopped gracefully")
        except subprocess.TimeoutExpired:
            print("  Pinggy tunnel not responding, killing...")
            if hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                self._process.kill()
            self._process.wait()

        self._process = None

        # Close log file
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _wait_for_healthy(self) -> None:
        """Wait for the Pinggy tunnel process to start and stabilize.

        We don't verify by hitting the external URL because HPC compute nodes
        often lack external DNS resolution. Instead, we verify the tunnel process
        is running and give it time to establish the SSH connection.
        """
        print(f"  Waiting for tunnel process to stabilize...")

        # Give the tunnel a few seconds to establish the SSH connection
        stabilize_time = 5
        for i in range(stabilize_time):
            time.sleep(1)
            # Check if process died during startup
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"Pinggy tunnel process exited unexpectedly (code {self._process.returncode}). "
                    f"Check logs at {self.log_path}"
                )

        # Final check that process is still alive
        if self._process and self._process.poll() is not None:
            raise RuntimeError(
                f"Pinggy tunnel process exited unexpectedly (code {self._process.returncode}). "
                f"Check logs at {self.log_path}"
            )

        print(f"  Tunnel process running (PID: {self._process.pid}), assuming healthy")

    def __enter__(self) -> "PinggyTunnel":
        """Context manager entry - start the tunnel."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop the tunnel."""
        self.stop()


def needs_pinggy_tunnel(agent_name: Optional[str], env_type: Optional[str]) -> bool:
    """Determine if Pinggy tunnel is needed based on agent and environment.

    Returns True if:
    - Agent is an installed agent (not terminus-2 which runs in-process)
    - Environment is a cloud backend (daytona, modal) that can't reach local network

    Args:
        agent_name: Name of the agent (e.g., "openhands", "terminus-2")
        env_type: Harbor environment type (e.g., "daytona", "docker", "apptainer")

    Returns:
        True if Pinggy tunnel is needed
    """
    # Terminus-2 runs in-process with direct LLM access, doesn't need tunnel
    if agent_name and agent_name.lower() in ("terminus-2", "terminus_2", "terminus2"):
        return False

    # Local container backends have direct network access to vLLM
    local_backends = ("docker", "podman_hpc", "apptainer", "singularity", "local")
    if env_type and env_type.lower() in local_backends:
        return False

    # Cloud backends (daytona, modal) need tunnel to reach local vLLM
    return True


def parse_endpoint_host_port(endpoint: str) -> tuple[str, int]:
    """Parse a vLLM endpoint URL and extract the host and port.

    Args:
        endpoint: vLLM endpoint URL (e.g., "http://172.24.74.235:8000/v1")

    Returns:
        Tuple of (host, port). Defaults to ("localhost", 8000) if parsing fails.

    Examples:
        >>> parse_endpoint_host_port("http://172.24.74.235:8000/v1")
        ('172.24.74.235', 8000)
        >>> parse_endpoint_host_port("http://localhost:8000/v1")
        ('localhost', 8000)
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8000
        return (host, port)
    except Exception:
        return ("localhost", 8000)


def build_pinggy_endpoint_meta(pinggy_url: str) -> Dict[str, str]:
    """Build endpoint metadata dict from a Pinggy tunnel URL.

    Args:
        pinggy_url: Pinggy public URL (e.g., "https://xxx.a.pinggy.link")

    Returns:
        Dict with 'api_base' and 'metrics_endpoint' keys
    """
    url = pinggy_url.rstrip("/")

    # Ensure we have the /v1 suffix for api_base
    if url.endswith("/v1"):
        api_base = url
        base_url = url[:-3]
    else:
        api_base = f"{url}/v1"
        base_url = url

    return {
        "api_base": api_base,
        "metrics_endpoint": f"{base_url}/metrics",
    }


def create_pinggy_config_from_args(
    persistent_url: Optional[str],
    token: Optional[str],
    local_port: int = 8000,
    local_host: str = "localhost",
) -> Optional[PinggyConfig]:
    """Create PinggyConfig from CLI arguments.

    Args:
        persistent_url: Pinggy persistent URL (e.g., "bjfqkhfxtx.a.pinggy.link")
        token: Pinggy auth token (e.g., "oVxgHq855Ln")
        local_port: Local port to tunnel (default: 8000)
        local_host: Local host to tunnel (default: "localhost")

    Returns:
        PinggyConfig if both URL and token provided, None otherwise
    """
    if not persistent_url or not token:
        return None

    return PinggyConfig(
        persistent_url=persistent_url,
        token=token,
        local_port=local_port,
        local_host=local_host,
    )


# Default Pinggy token (can be overridden via CLI or environment variable)
# This is an example token - users should use their own from https://pinggy.io
DEFAULT_PINGGY_TOKEN = "oVxgHq855Ln"


if __name__ == "__main__":
    # CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="Test Pinggy tunnel")
    parser.add_argument(
        "--persistent-url",
        default="bjfqkhfxtx.a.pinggy.link",
        help="Pinggy persistent URL",
    )
    parser.add_argument(
        "--token",
        default=DEFAULT_PINGGY_TOKEN,
        help="Pinggy auth token",
    )
    parser.add_argument(
        "--local-port",
        type=int,
        default=8000,
        help="Local port to tunnel",
    )
    parser.add_argument(
        "--local-host",
        default="localhost",
        help="Local host to tunnel",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Health check timeout in seconds",
    )

    args = parser.parse_args()

    config = PinggyConfig(
        persistent_url=args.persistent_url,
        token=args.token,
        local_port=args.local_port,
        local_host=args.local_host,
        health_check_timeout=args.timeout,
    )

    print(f"Testing Pinggy tunnel to {config.persistent_url}")
    print(f"SSH command: {config.get_ssh_command()}")
    print(f"Press Ctrl+C to stop")

    try:
        with PinggyTunnel(config) as tunnel:
            print(f"\nTunnel active at: {tunnel.public_endpoint}")
            print("Waiting... (Ctrl+C to stop)")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")


