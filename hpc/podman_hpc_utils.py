"""podman-hpc utilities for NERSC Perlmutter and similar HPC clusters.

This module provides utilities for working with podman-hpc, a wrapper around
Podman that provides HPC-specific optimizations:
- Squashed images for efficient shared storage
- Multi-threaded container execution
- Integration with SLURM and MPI

podman-hpc documentation: https://docs.nersc.gov/development/containers/podman-hpc/overview/

Usage:
    from hpc.podman_hpc_utils import (
        is_podman_hpc_available,
        setup_podman_hpc_env,
        ensure_podman_socket_running,
        verify_podman_hpc_connectivity,
    )
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Exit codes for fail-fast behavior
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_RUNTIME_NOT_FOUND = 2
EXIT_SOCKET_STARTUP_FAILED = 3
EXIT_CONNECTIVITY_FAILED = 4

# Log prefix for consistent messaging
LOG_PREFIX = "[podman-hpc]"


@dataclass
class PodmanHPCConfig:
    """Configuration for podman-hpc runtime."""

    available: bool = False
    socket_path: Optional[str] = None
    docker_host: Optional[str] = None
    storage_path: Optional[str] = None
    use_squashed_images: bool = True
    extra_env: Dict[str, str] = field(default_factory=dict)


def log_info(msg: str) -> None:
    """Log info message with prefix."""
    print(f"{LOG_PREFIX} {msg}")


def log_error(msg: str) -> None:
    """Log error message with prefix."""
    print(f"{LOG_PREFIX} ERROR: {msg}")


def log_debug(msg: str) -> None:
    """Log debug message with prefix."""
    print(f"{LOG_PREFIX} DEBUG: {msg}")


def log_hint(msg: str) -> None:
    """Log hint message with prefix."""
    print(f"{LOG_PREFIX} HINT: {msg}")


def is_podman_hpc_available() -> bool:
    """Check if podman-hpc command is available on this system.

    Returns:
        True if podman-hpc is found in PATH, False otherwise.
    """
    return shutil.which("podman-hpc") is not None


def get_podman_socket_path() -> str:
    """Get the standard podman socket path for the current user.

    Returns:
        Socket path string like '/run/user/{uid}/podman/podman.sock'
    """
    try:
        uid = os.getuid()
    except AttributeError:
        # Windows fallback (shouldn't happen on HPC)
        result = subprocess.run(["id", "-u"], capture_output=True, text=True)
        uid = result.stdout.strip()
    return f"/run/user/{uid}/podman/podman.sock"


def socket_exists(socket_path: Optional[str] = None) -> bool:
    """Check if the podman socket file exists.

    Args:
        socket_path: Path to socket, or None to use default.

    Returns:
        True if socket file exists.
    """
    path = socket_path or get_podman_socket_path()
    return os.path.exists(path)


def start_podman_socket(timeout: int = 10) -> bool:
    """Start the podman socket service using podman-hpc.

    Attempts to start the rootless podman socket that Docker-compatible
    clients can connect to via DOCKER_HOST.

    Args:
        timeout: Maximum seconds to wait for socket to appear.

    Returns:
        True if socket is available after startup, False otherwise.
    """
    socket_path = get_podman_socket_path()

    # Already running?
    if socket_exists(socket_path):
        log_info(f"Podman socket already exists at {socket_path}")
        return True

    log_info("Starting podman socket service...")

    # Try systemctl first (preferred)
    try:
        result = subprocess.run(
            ["systemctl", "--user", "start", "podman.socket"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Wait for socket to appear
            for _ in range(timeout):
                if socket_exists(socket_path):
                    log_info(f"Podman socket started via systemctl at {socket_path}")
                    return True
                time.sleep(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: start podman system service directly
    try:
        log_debug("Trying 'podman-hpc system service --time=0'...")
        # Start in background
        subprocess.Popen(
            ["podman-hpc", "system", "service", "--time=0"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for socket to appear
        for _ in range(timeout):
            if socket_exists(socket_path):
                log_info(f"Podman socket started via podman-hpc at {socket_path}")
                return True
            time.sleep(1)
    except Exception as e:
        log_debug(f"Failed to start podman-hpc service: {e}")

    log_error(f"Podman socket not found at {socket_path}")
    log_debug(f"Attempted to start via 'podman-hpc system service --time=0'")
    log_hint("Run 'systemctl --user start podman.socket' or check podman-hpc installation")
    return False


def verify_podman_hpc_connectivity(timeout: int = 10) -> bool:
    """Verify that we can connect to the podman daemon.

    Uses 'docker info' or 'podman-hpc info' to verify connectivity.

    Args:
        timeout: Maximum seconds to wait for response.

    Returns:
        True if daemon responds, False otherwise.
    """
    socket_path = get_podman_socket_path()
    docker_host = f"unix://{socket_path}"

    # Set DOCKER_HOST for the subprocess
    env = os.environ.copy()
    env["DOCKER_HOST"] = docker_host

    try:
        # Try docker info first (works with podman socket)
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode == 0:
            log_info("Docker daemon responding via podman-hpc socket")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback to podman-hpc info
    try:
        result = subprocess.run(
            ["podman-hpc", "info"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            log_info("podman-hpc daemon responding")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    log_error(f"Docker daemon not responding (timeout after {timeout}s)")
    log_debug(f"DOCKER_HOST={docker_host}")
    log_hint("Check 'podman-hpc info' manually, verify socket permissions")
    return False


def setup_podman_hpc_env() -> Dict[str, str]:
    """Set up environment variables for podman-hpc.

    Returns:
        Dict of environment variables to set for podman-hpc usage.
    """
    socket_path = get_podman_socket_path()
    return {
        "DOCKER_HOST": f"unix://{socket_path}",
        "CONTAINER_RUNTIME": "podman_hpc",
    }


def ensure_podman_hpc_ready(timeout: int = 10) -> PodmanHPCConfig:
    """Ensure podman-hpc is available and ready for use.

    This is the main entry point for setting up podman-hpc. It:
    1. Checks if podman-hpc is available
    2. Starts the socket if needed
    3. Verifies connectivity
    4. Returns configuration for use

    Args:
        timeout: Maximum seconds to wait for socket/connectivity.

    Returns:
        PodmanHPCConfig with runtime details.

    Raises:
        RuntimeError: If podman-hpc cannot be made ready (fail-fast).
    """
    if not is_podman_hpc_available():
        log_error("podman-hpc command not found")
        log_debug("Checked: which podman-hpc")
        log_hint("On Perlmutter, podman-hpc should be available. Contact support if missing.")
        raise RuntimeError(
            f"podman-hpc not available. "
            f"Checked 'which podman-hpc'. "
            f"On NERSC Perlmutter, this should be available by default."
        )

    socket_path = get_podman_socket_path()

    # Start socket if needed
    if not start_podman_socket(timeout=timeout):
        raise RuntimeError(
            f"Failed to start podman socket at {socket_path}. "
            f"Try: systemctl --user start podman.socket"
        )

    # Verify connectivity
    if not verify_podman_hpc_connectivity(timeout=timeout):
        raise RuntimeError(
            f"podman-hpc socket exists but daemon not responding. "
            f"DOCKER_HOST=unix://{socket_path}"
        )

    log_info("podman-hpc is ready")
    return PodmanHPCConfig(
        available=True,
        socket_path=socket_path,
        docker_host=f"unix://{socket_path}",
        extra_env=setup_podman_hpc_env(),
    )


# ---------------------------------------------------------------------------
# Image Management (for HPC squashed images)
# ---------------------------------------------------------------------------


def pull_image(image: str, squash: bool = True, timeout: int = 600) -> bool:
    """Pull a container image using podman-hpc.

    For HPC clusters, podman-hpc pull automatically creates squashed images
    that are optimized for shared storage access.

    Args:
        image: Image reference (e.g., "docker.io/library/ubuntu:22.04")
        squash: Whether to use podman-hpc pull (squashes by default)
        timeout: Pull timeout in seconds

    Returns:
        True if successful, False otherwise.
    """
    cmd = ["podman-hpc", "pull", image] if squash else ["podman", "pull", image]

    try:
        log_info(f"Pulling image: {image}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            log_info(f"Successfully pulled: {image}")
            return True
        else:
            log_error(f"Failed to pull image: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        log_error(f"Image pull timed out after {timeout}s")
        return False
    except Exception as e:
        log_error(f"Image pull failed: {e}")
        return False


def migrate_image(image: str, timeout: int = 600) -> bool:
    """Migrate an existing image to podman-hpc squashed format.

    This is useful for images already present in the local cache that
    need to be optimized for HPC shared storage.

    Args:
        image: Image reference to migrate
        timeout: Migration timeout in seconds

    Returns:
        True if successful, False otherwise.
    """
    try:
        log_info(f"Migrating image to squashed format: {image}")
        result = subprocess.run(
            ["podman-hpc", "migrate", image],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            log_info(f"Successfully migrated: {image}")
            return True
        else:
            log_error(f"Failed to migrate image: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        log_error(f"Image migration timed out after {timeout}s")
        return False
    except Exception as e:
        log_error(f"Image migration failed: {e}")
        return False


def get_squashed_images() -> List[str]:
    """List all images that have been migrated to squashed format.

    Returns:
        List of squashed image references.
    """
    try:
        result = subprocess.run(
            ["podman-hpc", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    except Exception:
        pass
    return []


def pre_migrate_images(images: List[str], timeout_per_image: int = 600) -> Dict[str, bool]:
    """Pre-migrate multiple images for HPC shared storage.

    This is similar to pre_download_dataset() for JSC clusters - it runs
    on the login node before job submission to ensure images are ready.

    Args:
        images: List of image references to migrate
        timeout_per_image: Timeout per image migration

    Returns:
        Dict mapping image to success/failure status.
    """
    results = {}
    for image in images:
        results[image] = migrate_image(image, timeout=timeout_per_image)
    return results
