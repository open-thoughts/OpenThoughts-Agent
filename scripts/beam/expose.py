"""Cluster exposure methods: Pinggy tunnel or GKE LoadBalancer."""

import atexit
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from scripts.beam.config import Beta9Config, LoadBalancerConfig, PinggyConfig


def load_secrets_env(secrets_path: Optional[str] = None) -> dict[str, str]:
    """Load environment variables from secrets file.

    Args:
        secrets_path: Path to secrets.env file. If None, checks:
            1. SECRETS_ENV environment variable
            2. ~/Documents/secrets.env
            3. ~/.secrets.env

    Returns:
        Dictionary of loaded environment variables.
    """
    loaded = {}

    # Determine secrets file path
    if secrets_path is None:
        secrets_path = os.environ.get("SECRETS_ENV")

    if secrets_path is None:
        # Check common locations
        candidates = [
            Path.home() / "Documents" / "secrets.env",
            Path.home() / ".secrets.env",
        ]
        for candidate in candidates:
            if candidate.exists():
                secrets_path = str(candidate)
                break

    if secrets_path is None or not Path(secrets_path).exists():
        return loaded

    logger.info(f"Loading secrets from: {secrets_path}")

    try:
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Handle export VAR=value format
                if line.startswith("export "):
                    line = line[7:]
                # Parse VAR=value
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Remove surrounding quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    os.environ[key] = value
                    loaded[key] = value
    except Exception as e:
        logger.warning(f"Failed to load secrets from {secrets_path}: {e}")

    return loaded

logger = logging.getLogger(__name__)

# Global registry for cleanup
_active_processes: list[subprocess.Popen] = []


def _cleanup_processes():
    """Cleanup handler for port-forward and tunnel processes."""
    for proc in _active_processes:
        if proc.poll() is None:  # Still running
            logger.info(f"Terminating process {proc.pid}")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


atexit.register(_cleanup_processes)


# =============================================================================
# Pinggy Tunnel Exposure
# =============================================================================


def start_port_forward(
    namespace: str,
    service: str,
    local_port: int,
    remote_port: int,
    dry_run: bool = False,
) -> Optional[subprocess.Popen]:
    """Start kubectl port-forward to the service.

    Args:
        namespace: Kubernetes namespace.
        service: Service name.
        local_port: Local port to forward to.
        remote_port: Service port to forward from.
        dry_run: If True, print command without executing.

    Returns:
        Popen process or None if dry_run.
    """
    cmd = [
        "kubectl", "port-forward",
        f"service/{service}",
        f"{local_port}:{remote_port}",
        "--namespace", namespace,
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return None

    logger.info(f"Starting port-forward: localhost:{local_port} -> {service}:{remote_port}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    _active_processes.append(proc)

    # Wait a moment for port-forward to establish
    time.sleep(3)

    if proc.poll() is not None:
        _, stderr = proc.communicate()
        logger.error(f"Port-forward failed: {stderr.decode()}")
        return None

    logger.info("Port-forward established")
    return proc


def start_pinggy_tunnel(config: PinggyConfig, dry_run: bool = False, secrets_path: Optional[str] = None) -> Optional[subprocess.Popen]:
    """Start Pinggy SSH tunnel.

    Args:
        config: Pinggy configuration.
        dry_run: If True, print command without executing.
        secrets_path: Optional path to secrets.env file to load.

    Returns:
        Popen process or None if dry_run.
    """
    # Load secrets if path provided or from default locations
    load_secrets_env(secrets_path)

    # Token identifies the persistent URL endpoint (SSH username)
    token = config.token
    if not token:
        logger.error("No Pinggy token provided (identifies which persistent URL to use)")
        return None

    logger.info(f"Using Pinggy token: {token}")

    identity_file = config.identity_file or os.environ.get("PINGGY_IDENTITY_FILE") or os.environ.get("PINGGY_SSH_KEY")
    if identity_file:
        identity_file = os.path.expanduser(identity_file)
        if not os.path.exists(identity_file):
            logger.warning(f"PINGGY identity file not found: {identity_file}")
            identity_file = None
        else:
            logger.info(f"Using Pinggy identity file: {identity_file}")

    if "SSH_AUTH_SOCK" not in os.environ and not identity_file:
        logger.warning(
            "No SSH agent or identity file configured. Pinggy requires a non-interactive SSH key; "
            "start ssh-agent and add your key (e.g., `ssh-add ~/.ssh/id_rsa`) or set PINGGY_IDENTITY_FILE."
        )

    # Match Pinggy's documented example and avoid interactive password prompts.
    ssh_cmd = [
        "ssh", "-v", "-p", "443",
        "-R", f"0:{config.local_host}:{config.local_port}",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "BatchMode=yes",
        "-o", "PasswordAuthentication=no",
        "-o", "KbdInteractiveAuthentication=no",
        "-o", "IdentitiesOnly=yes",
    ]

    if identity_file:
        ssh_cmd += ["-i", identity_file]

    ssh_cmd += [
        f"{token}@{config.pinggy_host}",
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(ssh_cmd)}")
        return None

    logger.info(f"Starting Pinggy tunnel: {config.local_host}:{config.local_port} -> {config.persistent_url}")

    # Pass through environment variables including any loaded secrets
    env = os.environ.copy()

    # Mirror Pinggy's recommended auto-reconnect loop.
    loop_cmd = (
        "while true; do "
        + " ".join(ssh_cmd)
        + " ; sleep 10; done"
    )

    proc = subprocess.Popen(
        ["bash", "-lc", loop_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    _active_processes.append(proc)

    # Wait for tunnel to establish
    time.sleep(5)

    if proc.poll() is not None:
        _, stderr = proc.communicate()
        logger.error(f"Pinggy tunnel failed: {stderr.decode()}")
        return None

    logger.info(f"Pinggy tunnel established: {config.get_public_url()}")
    return proc


def setup_pinggy_exposure(
    beta9_config: Beta9Config,
    pinggy_config: PinggyConfig,
    dry_run: bool = False,
    secrets_path: Optional[str] = None,
) -> tuple[Optional[subprocess.Popen], Optional[subprocess.Popen], str]:
    """Set up full Pinggy exposure (port-forward + tunnel).

    Beta9 SDK uses gRPC for sandbox operations, so we tunnel the gRPC port (1993),
    not the HTTP port (1994). The SDK then connects via TLS on port 443.

    Args:
        beta9_config: Beta9 configuration.
        pinggy_config: Pinggy configuration.
        dry_run: If True, print commands without executing.
        secrets_path: Optional path to secrets.env file to load.

    Returns:
        Tuple of (port_forward_proc, tunnel_proc, public_url).
    """
    # Start port-forward for gRPC port (1993)
    # The pinggy_config.local_port should be 1993 for gRPC
    pf_proc = start_port_forward(
        namespace=beta9_config.namespace,
        service=beta9_config.gateway_service_name,
        local_port=pinggy_config.local_port,
        remote_port=beta9_config.gateway_grpc_port,  # Use gRPC port, not HTTP
        dry_run=dry_run,
    )

    if not dry_run and pf_proc is None:
        return None, None, ""

    # Start Pinggy tunnel
    tunnel_proc = start_pinggy_tunnel(pinggy_config, dry_run, secrets_path)

    if not dry_run and tunnel_proc is None:
        if pf_proc:
            pf_proc.terminate()
        return None, None, ""

    return pf_proc, tunnel_proc, pinggy_config.get_public_url()


# =============================================================================
# LoadBalancer Exposure
# =============================================================================


def patch_service_to_loadbalancer(
    namespace: str,
    service: str,
    dry_run: bool = False,
) -> bool:
    """Patch a ClusterIP service to LoadBalancer type.

    Args:
        namespace: Kubernetes namespace.
        service: Service name.
        dry_run: If True, print command without executing.

    Returns:
        True if patched successfully.
    """
    patch = {"spec": {"type": "LoadBalancer"}}
    patch_json = json.dumps(patch)

    cmd = [
        "kubectl", "patch", "service", service,
        "--namespace", namespace,
        "--type", "merge",
        "--patch", patch_json,
    ]

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    logger.info(f"Patching service '{service}' to LoadBalancer type...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to patch service: {result.stderr}")
        return False

    logger.info("Service patched to LoadBalancer")
    return True


def wait_for_external_ip(
    config: LoadBalancerConfig,
    timeout: int = 300,
) -> Optional[str]:
    """Wait for LoadBalancer to get an external IP.

    Args:
        config: LoadBalancer configuration.
        timeout: Maximum seconds to wait.

    Returns:
        External IP address or None if timeout.
    """
    logger.info("Waiting for LoadBalancer external IP...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = subprocess.run(
            [
                "kubectl", "get", "service",
                config.service_name,
                "--namespace", config.namespace,
                "-o", "jsonpath={.status.loadBalancer.ingress[0].ip}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            ip = result.stdout.strip()
            logger.info(f"LoadBalancer IP assigned: {ip}")
            config.external_ip = ip
            return ip

        time.sleep(10)

    logger.error(f"LoadBalancer IP not assigned within {timeout} seconds")
    return None


def setup_loadbalancer_exposure(
    beta9_config: Beta9Config,
    lb_config: LoadBalancerConfig,
    dry_run: bool = False,
) -> tuple[str, bool]:
    """Set up LoadBalancer exposure.

    Args:
        beta9_config: Beta9 configuration.
        lb_config: LoadBalancer configuration.
        dry_run: If True, print commands without executing.

    Returns:
        Tuple of (public_url, success).
    """
    # Patch service to LoadBalancer
    if not patch_service_to_loadbalancer(
        namespace=beta9_config.namespace,
        service=beta9_config.gateway_service_name,
        dry_run=dry_run,
    ):
        return "", False

    if dry_run:
        return "http://<pending-ip>:1994", True

    # Wait for external IP
    ip = wait_for_external_ip(lb_config)
    if not ip:
        return "", False

    return lb_config.get_public_url(), True


# =============================================================================
# Common Utilities
# =============================================================================


def verify_endpoint_health(public_url: str, timeout: int = 30, grpc_port: int = 1993) -> bool:
    """Verify the public endpoint is reachable and healthy.

    Args:
        public_url: Public URL to check (e.g., https://mybeam.a.pinggy.link).
        timeout: Request timeout in seconds.

    Returns:
        True if endpoint is healthy.
    """
    from urllib.parse import urlparse

    parsed = urlparse(public_url if "://" in public_url else f"http://{public_url}")
    host = parsed.hostname or public_url

    if parsed.port == 1994:
        logger.info(f"Verifying endpoint health via gRPC: {host}:{grpc_port}")
        try:
            from scripts.beam.validate_cluster import get_or_create_token
            token = get_or_create_token(host, grpc_port)
            if token:
                logger.info("gRPC endpoint is healthy")
                return True
            logger.warning("gRPC health check failed (no token)")
            return False
        except Exception as e:
            logger.warning(f"gRPC endpoint check failed: {e}")
            return False

    health_url = f"{public_url.rstrip('/')}/api/v1/health"

    logger.info(f"Verifying endpoint health: {health_url}")

    try:
        response = requests.get(health_url, timeout=timeout)
        if response.status_code == 200:
            logger.info("Endpoint is healthy")
            return True
        else:
            logger.warning(f"Endpoint returned status {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.warning(f"Endpoint check failed: {e}")
        return False


def stop_exposure_processes():
    """Stop all active port-forward and tunnel processes."""
    for proc in _active_processes:
        if proc.poll() is None:
            logger.info(f"Stopping process {proc.pid}")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    _active_processes.clear()
    logger.info("All exposure processes stopped")
