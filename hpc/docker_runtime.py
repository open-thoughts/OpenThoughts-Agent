"""Docker runtime detection and configuration for Harbor Docker backend.

This module provides utilities for auto-detecting and configuring Docker or Podman
runtimes for use with Harbor's Docker environment backend. It supports:
- Native Docker daemon
- Podman with Docker CLI emulation
- Remote Docker via SSH tunnel (for SLURM clusters)

Usage:
    from hpc.docker_runtime import detect_docker_runtime, setup_docker_environment

    runtime = detect_docker_runtime()
    if runtime.runtime_type != DockerRuntimeType.UNAVAILABLE:
        env = setup_docker_environment(runtime)
        os.environ.update(env)
"""

import os
import subprocess
import shutil
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict


class DockerRuntimeType(Enum):
    """Detected Docker runtime type."""

    DOCKER = "docker"  # Native Docker daemon
    PODMAN = "podman"  # Podman with Docker CLI emulation
    REMOTE = "remote"  # Remote Docker via SSH tunnel or TCP
    UNAVAILABLE = "unavailable"


@dataclass
class DockerRuntimeConfig:
    """Configuration for Docker runtime."""

    runtime_type: DockerRuntimeType
    docker_host: Optional[str] = None  # DOCKER_HOST value to set
    socket_path: Optional[str] = None  # Local socket path if applicable
    requires_tunnel: bool = False  # Whether SSH tunnel is needed
    tunnel_port: Optional[int] = None  # Port for SSH tunnel if applicable
    extra_env: Dict[str, str] = field(default_factory=dict)  # Additional env vars


def get_podman_socket_path() -> Optional[str]:
    """Get Podman socket path for current user.

    Returns:
        Path to Podman socket if found, None otherwise.
    """
    try:
        user_id = subprocess.run(
            ["id", "-u"], capture_output=True, text=True, check=True
        ).stdout.strip()
        socket_path = f"/run/user/{user_id}/podman/podman.sock"
        return socket_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _is_podman_docker() -> bool:
    """Check if 'docker' command is actually podman."""
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        output = (result.stdout + result.stderr).lower()
        return "podman" in output
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _socket_exists(path: str) -> bool:
    """Check if a Unix socket exists at the given path."""
    try:
        import stat

        mode = os.stat(path).st_mode
        return stat.S_ISSOCK(mode)
    except (OSError, FileNotFoundError):
        return False


def detect_docker_runtime() -> DockerRuntimeConfig:
    """Auto-detect available Docker runtime.

    Detection order:
    1. Check if DOCKER_HOST is already set (user override / tunnel)
    2. Check for Podman with Docker emulation
    3. Check for native Docker socket
    4. Return unavailable if none found

    Returns:
        DockerRuntimeConfig with detected runtime settings.
    """
    # 1. Check for existing DOCKER_HOST (user override or pre-configured tunnel)
    existing_host = os.environ.get("DOCKER_HOST")
    if existing_host:
        if existing_host.startswith("tcp://"):
            return DockerRuntimeConfig(
                runtime_type=DockerRuntimeType.REMOTE,
                docker_host=existing_host,
                requires_tunnel=True,
            )
        elif existing_host.startswith("unix://"):
            socket_path = existing_host.replace("unix://", "")
            # Determine if it's podman or docker based on socket path
            if "podman" in socket_path:
                return DockerRuntimeConfig(
                    runtime_type=DockerRuntimeType.PODMAN,
                    docker_host=existing_host,
                    socket_path=socket_path,
                )
            else:
                return DockerRuntimeConfig(
                    runtime_type=DockerRuntimeType.DOCKER,
                    docker_host=existing_host,
                    socket_path=socket_path,
                )

    # 2. Check for Podman (either native or masquerading as docker)
    podman_socket = get_podman_socket_path()

    # Check if docker command is actually podman
    if _is_podman_docker():
        if podman_socket and _socket_exists(podman_socket):
            return DockerRuntimeConfig(
                runtime_type=DockerRuntimeType.PODMAN,
                docker_host=f"unix://{podman_socket}",
                socket_path=podman_socket,
            )
        # Podman is aliased but socket may need to be started
        if podman_socket:
            return DockerRuntimeConfig(
                runtime_type=DockerRuntimeType.PODMAN,
                docker_host=f"unix://{podman_socket}",
                socket_path=podman_socket,
                extra_env={"_PODMAN_SOCKET_NEEDS_START": "1"},
            )

    # Check for native podman command
    if shutil.which("podman") and podman_socket:
        if _socket_exists(podman_socket):
            return DockerRuntimeConfig(
                runtime_type=DockerRuntimeType.PODMAN,
                docker_host=f"unix://{podman_socket}",
                socket_path=podman_socket,
            )
        # Podman available but socket may need to be started
        return DockerRuntimeConfig(
            runtime_type=DockerRuntimeType.PODMAN,
            docker_host=f"unix://{podman_socket}",
            socket_path=podman_socket,
            extra_env={"_PODMAN_SOCKET_NEEDS_START": "1"},
        )

    # 3. Check for native Docker socket
    docker_socket = "/var/run/docker.sock"
    if _socket_exists(docker_socket):
        return DockerRuntimeConfig(
            runtime_type=DockerRuntimeType.DOCKER,
            docker_host=f"unix://{docker_socket}",
            socket_path=docker_socket,
        )

    # 4. Check Docker Desktop socket on macOS
    home = os.path.expanduser("~")
    docker_desktop_socket = f"{home}/.docker/run/docker.sock"
    if _socket_exists(docker_desktop_socket):
        return DockerRuntimeConfig(
            runtime_type=DockerRuntimeType.DOCKER,
            docker_host=f"unix://{docker_desktop_socket}",
            socket_path=docker_desktop_socket,
        )

    # 5. No runtime found
    return DockerRuntimeConfig(runtime_type=DockerRuntimeType.UNAVAILABLE)


def setup_docker_environment(
    config: DockerRuntimeConfig, env: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Configure environment variables for Docker runtime.

    Args:
        config: DockerRuntimeConfig from detect_docker_runtime()
        env: Optional existing environment dict to update

    Returns:
        Updated environment dict with DOCKER_HOST and any extra vars.
    """
    if env is None:
        env = {}

    if config.docker_host:
        env["DOCKER_HOST"] = config.docker_host

    # Add any extra environment variables
    env.update(config.extra_env)

    return env


def try_start_podman_socket(timeout: int = 3) -> bool:
    """Try to start the Podman socket service.

    Args:
        timeout: Max seconds to wait for socket activation

    Returns:
        True if socket was started or already running, False otherwise.
    """
    try:
        # Try systemctl --user first (most common on modern Linux)
        subprocess.run(
            ["systemctl", "--user", "start", "podman.socket"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        # Try direct podman system service command
        subprocess.run(
            ["podman", "system", "service", "--time=0"],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return False


def check_docker_connectivity(timeout: int = 5) -> bool:
    """Test if Docker/Podman daemon is accessible.

    Args:
        timeout: Max seconds to wait for response

    Returns:
        True if daemon is accessible, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=timeout, check=False
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_runtime_info() -> Dict[str, str]:
    """Get diagnostic info about Docker runtime.

    Returns:
        Dict with runtime diagnostic information.
    """
    runtime = detect_docker_runtime()
    info = {
        "runtime_type": runtime.runtime_type.value,
        "docker_host": runtime.docker_host or "not set",
        "socket_path": runtime.socket_path or "not applicable",
        "requires_tunnel": str(runtime.requires_tunnel),
    }

    # Add connectivity check
    info["is_connected"] = str(check_docker_connectivity())

    # Add version info if available
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            info["version"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["version"] = "not available"

    return info


def create_docker_tunnel_config(
    remote_host: str,
    remote_port: int = 2375,
    local_port: int = 23750,
    ssh_key_path: Optional[str] = None,
) -> Dict[str, str]:
    """Generate SSH tunnel configuration for remote Docker access.

    This is useful for SLURM clusters that need to access a remote Docker daemon
    via SSH tunnel.

    Args:
        remote_host: Remote host running Docker daemon
        remote_port: Port Docker is listening on (default: 2375)
        local_port: Local port to forward to (default: 23750)
        ssh_key_path: Optional path to SSH private key

    Returns:
        Dict with tunnel configuration including SSH command and env vars.
    """
    ssh_cmd_parts = ["ssh", "-N", "-f", "-o", "ExitOnForwardFailure=yes"]

    if ssh_key_path:
        ssh_cmd_parts.extend(["-i", ssh_key_path])

    ssh_cmd_parts.extend(
        ["-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}", remote_host]
    )

    return {
        "ssh_command": " ".join(ssh_cmd_parts),
        "docker_host": f"tcp://127.0.0.1:{local_port}",
        "remote_host": remote_host,
        "remote_port": str(remote_port),
        "local_port": str(local_port),
    }


def ensure_docker_runtime(
    required_type: Optional[DockerRuntimeType] = None,
) -> DockerRuntimeConfig:
    """Ensure a Docker runtime is available, attempting to start if needed.

    Args:
        required_type: Optional specific runtime type required

    Returns:
        DockerRuntimeConfig for the available runtime

    Raises:
        RuntimeError: If no suitable runtime is available
    """
    runtime = detect_docker_runtime()

    # If podman socket needs starting, try to start it
    if runtime.extra_env.get("_PODMAN_SOCKET_NEEDS_START"):
        if try_start_podman_socket():
            # Re-detect after starting
            runtime = detect_docker_runtime()

    if runtime.runtime_type == DockerRuntimeType.UNAVAILABLE:
        raise RuntimeError(
            "No Docker/Podman runtime found. Please ensure Docker or Podman is installed "
            "and running, or set DOCKER_HOST to point to a remote Docker daemon."
        )

    if required_type and runtime.runtime_type != required_type:
        raise RuntimeError(
            f"Required runtime type {required_type.value} but found {runtime.runtime_type.value}"
        )

    # Verify connectivity
    env = setup_docker_environment(runtime)
    os.environ.update(env)

    if not check_docker_connectivity():
        raise RuntimeError(
            f"Docker runtime detected ({runtime.runtime_type.value}) but daemon is not "
            f"responding. DOCKER_HOST={runtime.docker_host}"
        )

    return runtime


def setup_docker_runtime_if_needed(env_type: str) -> None:
    """Configure Docker/Podman runtime if using docker backend.

    Detects available Docker/Podman runtime, sets DOCKER_HOST environment
    variable, and verifies connectivity. This is a user-friendly wrapper
    that prints status messages and exits on failure.

    Args:
        env_type: Harbor environment type (daytona, docker, modal, apptainer)

    Raises:
        SystemExit: If docker backend requested but no runtime found
    """
    import sys

    if env_type.lower() != "docker":
        return

    print("[docker] Detecting Docker/Podman runtime...")
    runtime = detect_docker_runtime()

    if runtime.runtime_type == DockerRuntimeType.UNAVAILABLE:
        print("[docker] ERROR: Docker backend requested but no Docker/Podman runtime found.")
        print("[docker] Please ensure Docker or Podman is installed and running,")
        print("[docker] or set DOCKER_HOST to point to a remote Docker daemon.")
        sys.exit(1)

    # Set up environment variables
    env = setup_docker_environment(runtime)
    os.environ.update(env)

    print(f"[docker] Runtime type: {runtime.runtime_type.value}")
    print(f"[docker] DOCKER_HOST: {runtime.docker_host}")

    # Verify connectivity
    if not check_docker_connectivity(timeout=10):
        print("[docker] WARNING: Docker daemon not responding. Continuing anyway...")
    else:
        print("[docker] Docker daemon is accessible.")


# ---------------------------------------------------------------------------
# Docker Image Utilities (for cloud/SkyPilot launches)
# ---------------------------------------------------------------------------

# Default Docker image configuration
GHCR_IMAGE_BASE = "ghcr.io/open-thoughts/openthoughts-agent"
DEFAULT_DOCKER_IMAGE = f"{GHCR_IMAGE_BASE}:gpu-1x"


def normalize_docker_image(image: str) -> str:
    """Ensure docker image has the 'docker:' prefix required by SkyPilot.

    Args:
        image: Docker image name (with or without docker: prefix)

    Returns:
        Image name with docker: prefix
    """
    if not image.startswith("docker:"):
        return f"docker:{image}"
    return image


def select_docker_image(
    accelerator: str,
    default_image: str = DEFAULT_DOCKER_IMAGE,
    base_image: str = GHCR_IMAGE_BASE,
) -> str:
    """Select appropriate Docker image variant based on GPU count.

    If accelerator specifies count > 1, selects gpu-4x or gpu-8x variants.

    Args:
        accelerator: SkyPilot accelerator spec (e.g., "H100:2", "A100:1")
        default_image: Default image to use for single GPU
        base_image: Base image name for constructing variants

    Returns:
        Docker image name (without docker: prefix)
    """
    # Import here to avoid circular dependency
    from hpc.cloud_launch_utils import parse_gpu_count

    count = parse_gpu_count(accelerator)

    # Select appropriate variant
    if count <= 1:
        return f"{base_image}:gpu-1x"
    elif count <= 4:
        return f"{base_image}:gpu-4x"
    else:
        return f"{base_image}:gpu-8x"


def get_docker_image_for_providers(
    docker_image: str,
    accelerator: str,
    provider_docker_support: Dict[str, bool],
) -> Optional[str]:
    """Select and normalize Docker image based on provider support.

    This is the core logic for Docker image selection in cloud launches.
    It handles auto-selection based on GPU count and provider compatibility.

    Args:
        docker_image: User-specified Docker image (or DEFAULT_DOCKER_IMAGE)
        accelerator: SkyPilot accelerator spec (e.g., "H100:2")
        provider_docker_support: Dict mapping provider display names to
            whether they support Docker runtime (True/False)

    Returns:
        Normalized Docker image with 'docker:' prefix, or None if no
        providers support Docker runtime.
    """
    import sys

    # Determine base image (custom or auto-selected)
    if docker_image != DEFAULT_DOCKER_IMAGE:
        base_image = docker_image
    else:
        base_image = select_docker_image(accelerator)

    # Check provider docker support
    supports_docker = [name for name, supported in provider_docker_support.items() if supported]
    no_docker = [name for name, supported in provider_docker_support.items() if not supported]

    if not supports_docker:
        # No providers support Docker
        print(
            "[cloud] Note: Selected providers do not support Docker as runtime.",
            file=sys.stderr,
        )
        return None
    elif no_docker:
        # Some providers don't support Docker
        print(
            f"[cloud] Note: {', '.join(no_docker)} do not support Docker as runtime.",
            file=sys.stderr,
        )

    return normalize_docker_image(base_image)


if __name__ == "__main__":
    # CLI for testing runtime detection
    import json

    print("Docker Runtime Detection")
    print("=" * 40)

    info = get_runtime_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    print()
    print("Raw config:")
    print(json.dumps(info, indent=2))
