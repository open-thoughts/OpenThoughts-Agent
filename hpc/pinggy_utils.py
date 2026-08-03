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
    # On no-internet clusters (JSC Jupiter/Jureca/Juwels, Leonardo) the compute
    # nodes cannot reach pro.pinggy.io directly, so the outbound ssh must be
    # wrapped with proxychains. Set this to e.g.
    #   "/path/to/proxychains4 -f /path/to/proxychains.conf"
    # and it will be prepended to the ssh invocation.
    proxychains_wrapper: Optional[str] = None
    # Full URL to GET to confirm the tunnel is actually forwarding (e.g.
    # "https://xxx.a.pinggy.link/v1/models"). If set, _wait_for_healthy treats
    # any HTTP response as proof the reverse tunnel is bound. Leave None on hosts
    # that can't resolve the public URL (no external DNS, or a poisoned resolver);
    # there the log-banner scan is used instead.
    health_check_url: Optional[str] = None

    def get_ssh_command(self) -> str:
        """Build the SSH command for the Pinggy tunnel."""
        ssh_prefix = f"{self.proxychains_wrapper} " if self.proxychains_wrapper else ""
        # Build a robust SSH command with auto-reconnect loop. `-n` redirects ssh's
        # stdin from /dev/null so it never reads the controlling terminal (a
        # backgrounded ssh that touches the tty gets SIGTTIN-stopped → state T →
        # silently forwards nothing). Combined with stdin=DEVNULL + setsid in
        # start(), the tunnel can't be job-control-stopped.
        return (
            f"while true; do "
            f"{ssh_prefix}ssh -p 443 -n "
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

        print("=== Starting Pinggy Tunnel ===")
        print(f"  Persistent URL: {self.config.persistent_url}")
        print(f"  Local target: {self.config.local_host}:{self.config.local_port}")
        print("==============================")

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
                # /dev/null stdin so ssh never reads the controlling tty (would
                # otherwise SIGTTIN-stop the backgrounded process → state T).
                stdin=subprocess.DEVNULL,
                stdout=stdout_dest,
                stderr=stderr_dest,
                # setsid detaches from the controlling terminal entirely, so the
                # tunnel can't be sent SIGTTIN/SIGTTOU and can't be Ctrl-Z'd; it
                # also becomes its own session/process-group leader (pgid == pid),
                # which stop()/killpg rely on.
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start Pinggy tunnel: {e}")

        print(f"  Started Pinggy tunnel (PID: {self._process.pid})")
        if self.log_path:
            print(f"  Log file: {self.log_path}")

        # Wait for tunnel to be healthy
        self._wait_for_healthy()

        print("=== Pinggy Tunnel Ready ===")
        print(f"  Public endpoint: {self.public_endpoint}")
        print("===========================")

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

    def _process_state(self) -> Optional[str]:
        """OS process-state code of the tunnel (e.g. 'S', 'R', 'T'), or None if gone."""
        if self._process is None:
            return None
        try:
            out = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(self._process.pid)],
                capture_output=True, text=True, timeout=5,
            )
        except Exception:
            return None
        return out.stdout.strip() or None

    def _resume_if_stopped(self) -> bool:
        """If the tunnel's process group is job-control-stopped (state T), SIGCONT it.

        Backstop for the SIGTTIN/SIGTTOU stop that silently kills forwarding while
        leaving the process 'alive'. start() now prevents the stop (stdin=DEVNULL +
        setsid), but resuming here costs nothing and recovers a tunnel stopped by
        some other means. Returns True if it was stopped.
        """
        state = self._process_state()
        if state and state.startswith("T"):
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGCONT)
                print("  [pinggy] tunnel was job-control-stopped (state T); sent SIGCONT to resume")
            except (ProcessLookupError, PermissionError):
                pass
            return True
        return False

    def _probe_url(self, url: str) -> bool:
        """True if the public URL returns ANY HTTP response (tunnel is forwarding).

        An HTTPError still proves the reverse tunnel is bound and reaching the
        local server; only connection-level failures (reset/timeout/DNS) mean the
        tunnel isn't forwarding.
        """
        req = urllib.request.Request(url, headers={"User-Agent": "pinggy-healthcheck"})
        try:
            with urllib.request.urlopen(req, timeout=10):
                return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            return False

    def _log_shows_bound(self) -> bool:
        """True once Pinggy has echoed the public URL to the log (reverse tunnel bound)."""
        if not self.log_path or not self.log_path.exists():
            return False
        try:
            text = self.log_path.read_text(errors="ignore")
        except Exception:
            return False
        return self.config.persistent_url in text or "pinggy.link" in text

    def _wait_for_healthy(self) -> None:
        """Wait until the tunnel is confirmed forwarding (or the process dies).

        Verification, in order of preference:
          1. ``health_check_url`` set → GET it; any HTTP response proves the tunnel
             forwards. (Don't set it on hosts that can't resolve the public URL —
             no external DNS, or a resolver that poisons ``*.a.pinggy.link``.)
          2. else, scan the log for Pinggy's URL banner → the reverse tunnel bound.
          3. else (no URL, no log: e.g. DEVNULL output on an air-gapped node) →
             fall back to a brief process-alive stabilize, as before.

        Throughout, a job-control-stopped process is auto-resumed (SIGCONT).
        """
        can_verify = bool(self.config.health_check_url) or bool(self.log_path)
        if not can_verify:
            print("  Waiting for tunnel process to stabilize (no URL/log to verify against)...")
            for _ in range(5):
                time.sleep(1)
                if self._process and self._process.poll() is not None:
                    raise RuntimeError(
                        f"Pinggy tunnel exited (code {self._process.returncode}); "
                        f"check logs at {self.log_path}"
                    )
                self._resume_if_stopped()
            print(f"  Tunnel process running (PID: {self._process.pid}), assuming healthy")
            return

        print("  Waiting for tunnel to confirm forwarding...")
        deadline = time.time() + self.config.health_check_timeout
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
                tail = ""
                if self.log_path and self.log_path.exists():
                    tail = self.log_path.read_text(errors="ignore")[-800:]
                raise RuntimeError(
                    f"Pinggy tunnel exited (code {self._process.returncode}). Log tail:\n{tail}"
                )
            self._resume_if_stopped()
            if self.config.health_check_url:
                if self._probe_url(self.config.health_check_url):
                    print(f"  [pinggy] tunnel confirmed forwarding via {self.config.health_check_url}")
                    return
            elif self._log_shows_bound():
                print(f"  [pinggy] tunnel bound (URL banner in {self.log_path})")
                return
            time.sleep(self.config.health_check_interval)

        if self.config.health_check_url:
            raise RuntimeError(
                f"Pinggy tunnel process is alive but {self.config.health_check_url} never "
                f"responded within {self.config.health_check_timeout}s — tunnel not forwarding."
            )
        print(
            f"  [pinggy] WARNING: could not confirm tunnel binding from log within "
            f"{self.config.health_check_timeout}s; process alive, proceeding."
        )

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
    print("Press Ctrl+C to stop")

    try:
        with PinggyTunnel(config) as tunnel:
            print(f"\nTunnel active at: {tunnel.public_endpoint}")
            print("Waiting... (Ctrl+C to stop)")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")


