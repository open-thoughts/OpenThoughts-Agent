#!/bin/bash
# Setup Docker/Podman runtime for Harbor Docker backend on SLURM clusters.
#
# This script auto-detects and configures the Docker runtime environment.
# It supports:
# - podman-hpc (NERSC Perlmutter HPC wrapper) - checked first
# - Native Podman with Docker CLI emulation
# - Native Docker daemon
# - Pre-configured DOCKER_HOST (e.g., SSH tunnel to remote Docker)
#
# Usage:
#   source setup_docker_runtime.sh
#   # or
#   eval "$(setup_docker_runtime.sh --export)"
#   # or
#   ./setup_docker_runtime.sh --verify && echo "Ready"
#
# The script will:
# 1. Check if DOCKER_HOST is already set (preserves user configuration)
# 2. Detect podman-hpc (NERSC Perlmutter) and configure its socket
# 3. Detect Podman and configure its socket
# 4. Fall back to native Docker socket
# 5. Exit with error if no runtime found
#
# Exit codes:
# 0 - Success
# 1 - General error
# 2 - Runtime not found
# 3 - Socket startup failed
# 4 - Connectivity test failed

set -euo pipefail

# Exit codes for fail-fast behavior
EXIT_SUCCESS=0
EXIT_GENERAL_ERROR=1
EXIT_RUNTIME_NOT_FOUND=2
EXIT_SOCKET_STARTUP_FAILED=3
EXIT_CONNECTIVITY_FAILED=4

# Container runtime type (set after detection)
CONTAINER_RUNTIME=""

# Colors for output (disabled if not a terminal)
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

log_info() {
    echo -e "${GREEN}[docker-runtime]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[docker-runtime]${NC} $1"
}

log_error() {
    echo -e "${RED}[docker-runtime] ERROR:${NC} $1" >&2
}

log_debug() {
    echo -e "${YELLOW}[docker-runtime] DEBUG:${NC} $1" >&2
}

log_hint() {
    echo -e "${GREEN}[docker-runtime] HINT:${NC} $1" >&2
}

detect_podman_hpc() {
    # Check if podman-hpc is available (NERSC Perlmutter)
    if command -v podman-hpc &> /dev/null; then
        log_info "Detected podman-hpc (NERSC HPC environment)"
        CONTAINER_RUNTIME="podman_hpc"

        local user_id
        user_id=$(id -u)
        local podman_sock="/run/user/${user_id}/podman/podman.sock"

        if [[ -S "$podman_sock" ]]; then
            export DOCKER_HOST="unix://$podman_sock"
            export CONTAINER_RUNTIME="podman_hpc"
            log_info "DOCKER_HOST=$DOCKER_HOST (podman-hpc)"
            return 0
        fi

        # Try to start podman socket via podman-hpc
        log_info "Starting podman socket for podman-hpc..."

        # Try systemctl first
        if systemctl --user start podman.socket 2>/dev/null; then
            sleep 2
            if [[ -S "$podman_sock" ]]; then
                export DOCKER_HOST="unix://$podman_sock"
                export CONTAINER_RUNTIME="podman_hpc"
                log_info "Started podman-hpc socket via systemctl, DOCKER_HOST=$DOCKER_HOST"
                return 0
            fi
        fi

        # Fallback: start podman system service directly
        if podman-hpc system service --time=0 &>/dev/null & then
            sleep 2
            if [[ -S "$podman_sock" ]]; then
                export DOCKER_HOST="unix://$podman_sock"
                export CONTAINER_RUNTIME="podman_hpc"
                log_info "Started podman-hpc socket, DOCKER_HOST=$DOCKER_HOST"
                return 0
            fi
        fi

        log_error "podman-hpc found but could not start socket service"
        log_debug "Socket path: $podman_sock, exists: $(test -S "$podman_sock" && echo true || echo false)"
        log_hint "Run 'systemctl --user start podman.socket' or check podman-hpc installation"
        return $EXIT_SOCKET_STARTUP_FAILED
    fi
    return 1  # podman-hpc not available, continue to next detection
}

detect_runtime() {
    # Check if DOCKER_HOST already set (user override or pre-configured tunnel)
    if [[ -n "${DOCKER_HOST:-}" ]]; then
        log_info "Using existing DOCKER_HOST: $DOCKER_HOST"
        return 0
    fi

    # Check for podman-hpc first (NERSC Perlmutter)
    if detect_podman_hpc; then
        return 0
    fi
    # detect_podman_hpc returns 1 if not available, continue to next detection

    # Check for Podman
    if command -v podman &> /dev/null; then
        local user_id
        user_id=$(id -u)
        local podman_sock="/run/user/${user_id}/podman/podman.sock"

        if [[ -S "$podman_sock" ]]; then
            export DOCKER_HOST="unix://$podman_sock"
            log_info "Detected Podman socket, DOCKER_HOST=$DOCKER_HOST"
            return 0
        fi

        # Try to start Podman socket service
        log_info "Podman found but socket not running. Attempting to start..."
        if systemctl --user start podman.socket 2>/dev/null; then
            sleep 1
            if [[ -S "$podman_sock" ]]; then
                export DOCKER_HOST="unix://$podman_sock"
                log_info "Started Podman socket, DOCKER_HOST=$DOCKER_HOST"
                return 0
            fi
        fi

        # Try podman system service directly (for systems without systemd)
        if podman system service --time=0 &>/dev/null & then
            sleep 1
            if [[ -S "$podman_sock" ]]; then
                export DOCKER_HOST="unix://$podman_sock"
                log_info "Started Podman service, DOCKER_HOST=$DOCKER_HOST"
                return 0
            fi
        fi

        log_warn "Podman found but could not start socket service"
    fi

    # Check if 'docker' command is actually Podman (aliased)
    if command -v docker &> /dev/null; then
        local docker_version
        docker_version=$(docker --version 2>&1 || true)
        if [[ "$docker_version" == *"podman"* ]]; then
            local user_id
            user_id=$(id -u)
            local podman_sock="/run/user/${user_id}/podman/podman.sock"

            if [[ -S "$podman_sock" ]]; then
                export DOCKER_HOST="unix://$podman_sock"
                log_info "Detected Podman (via docker alias), DOCKER_HOST=$DOCKER_HOST"
                return 0
            fi
            log_warn "Docker is aliased to Podman but socket not found at $podman_sock"
        fi
    fi

    # Check for native Docker socket
    if [[ -S "/var/run/docker.sock" ]]; then
        export DOCKER_HOST="unix:///var/run/docker.sock"
        log_info "Detected Docker, DOCKER_HOST=$DOCKER_HOST"
        return 0
    fi

    # Check Docker Desktop socket on macOS
    local docker_desktop_sock="${HOME}/.docker/run/docker.sock"
    if [[ -S "$docker_desktop_sock" ]]; then
        export DOCKER_HOST="unix://$docker_desktop_sock"
        log_info "Detected Docker Desktop, DOCKER_HOST=$DOCKER_HOST"
        return 0
    fi

    log_error "No Docker/Podman runtime found"
    log_debug "Checked: podman-hpc, podman, docker commands and socket paths"
    log_hint "Install Docker/Podman, start the daemon, or set DOCKER_HOST"
    return $EXIT_RUNTIME_NOT_FOUND
}

verify_connectivity() {
    local timeout="${1:-5}"
    if timeout "$timeout" docker info &>/dev/null; then
        log_info "Docker daemon is accessible"
        return 0
    else
        log_warn "Docker daemon not responding (DOCKER_HOST=$DOCKER_HOST)"
        return 1
    fi
}

print_runtime_info() {
    echo "Docker Runtime Information"
    echo "=========================="
    echo "DOCKER_HOST: ${DOCKER_HOST:-not set}"

    if command -v docker &> /dev/null; then
        echo ""
        echo "Docker Version:"
        docker --version 2>&1 || echo "  (not available)"
        echo ""
        echo "Docker Info:"
        docker info 2>&1 | head -20 || echo "  (not available)"
    fi
}

# Handle --export flag for eval usage
if [[ "${1:-}" == "--export" ]]; then
    # Just output export statements, no logging
    if [[ -n "${DOCKER_HOST:-}" ]]; then
        echo "export DOCKER_HOST=\"$DOCKER_HOST\""
        echo "export CONTAINER_RUNTIME=\"${CONTAINER_RUNTIME:-unknown}\""
        exit 0
    fi

    user_id=$(id -u)
    podman_sock="/run/user/${user_id}/podman/podman.sock"

    # Check for podman-hpc first (NERSC Perlmutter)
    if command -v podman-hpc &> /dev/null; then
        if [[ -S "$podman_sock" ]]; then
            echo "export DOCKER_HOST=\"unix://$podman_sock\""
            echo "export CONTAINER_RUNTIME=\"podman_hpc\""
            exit 0
        fi
    fi

    # Check for standard Podman
    if [[ -S "$podman_sock" ]]; then
        echo "export DOCKER_HOST=\"unix://$podman_sock\""
        echo "export CONTAINER_RUNTIME=\"podman\""
        exit 0
    fi

    if [[ -S "/var/run/docker.sock" ]]; then
        echo "export DOCKER_HOST=\"unix:///var/run/docker.sock\""
        echo "export CONTAINER_RUNTIME=\"docker\""
        exit 0
    fi

    docker_desktop_sock="${HOME}/.docker/run/docker.sock"
    if [[ -S "$docker_desktop_sock" ]]; then
        echo "export DOCKER_HOST=\"unix://$docker_desktop_sock\""
        echo "export CONTAINER_RUNTIME=\"docker\""
        exit 0
    fi

    exit $EXIT_RUNTIME_NOT_FOUND
fi

# Handle --info flag
if [[ "${1:-}" == "--info" ]]; then
    detect_runtime || true
    print_runtime_info
    exit 0
fi

# Handle --verify flag
if [[ "${1:-}" == "--verify" ]]; then
    if ! detect_runtime; then
        log_error "No Docker/Podman runtime found"
        log_debug "Checked: podman-hpc, podman, docker, DOCKER_HOST"
        log_hint "Install Docker/Podman, start the daemon, or set DOCKER_HOST"
        exit $EXIT_RUNTIME_NOT_FOUND
    fi
    if ! verify_connectivity 10; then
        log_error "Docker daemon not responding"
        log_debug "DOCKER_HOST=${DOCKER_HOST:-not set}, CONTAINER_RUNTIME=${CONTAINER_RUNTIME:-unknown}"
        log_hint "Check daemon status: 'docker info' or 'podman-hpc info'"
        exit $EXIT_CONNECTIVITY_FAILED
    fi
    log_info "Runtime verified: ${CONTAINER_RUNTIME:-unknown}"
    exit $EXIT_SUCCESS
fi

# Default: run detection
detect_runtime
