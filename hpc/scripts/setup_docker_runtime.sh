#!/bin/bash
# Setup Docker/Podman runtime for Harbor Docker backend on SLURM clusters.
#
# This script auto-detects and configures the Docker runtime environment.
# It supports:
# - Native Docker daemon
# - Podman with Docker CLI emulation
# - Pre-configured DOCKER_HOST (e.g., SSH tunnel to remote Docker)
#
# Usage:
#   source setup_docker_runtime.sh
#   # or
#   eval "$(setup_docker_runtime.sh --export)"
#
# The script will:
# 1. Check if DOCKER_HOST is already set (preserves user configuration)
# 2. Detect Podman and configure its socket
# 3. Fall back to native Docker socket
# 4. Exit with error if no runtime found

set -euo pipefail

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
    echo -e "${RED}[docker-runtime]${NC} $1" >&2
}

detect_runtime() {
    # Check if DOCKER_HOST already set (user override or pre-configured tunnel)
    if [[ -n "${DOCKER_HOST:-}" ]]; then
        log_info "Using existing DOCKER_HOST: $DOCKER_HOST"
        return 0
    fi

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
    log_error "Please ensure Docker or Podman is installed and running,"
    log_error "or set DOCKER_HOST to point to a remote Docker daemon."
    return 1
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
        exit 0
    fi

    user_id=$(id -u)
    podman_sock="/run/user/${user_id}/podman/podman.sock"

    if [[ -S "$podman_sock" ]]; then
        echo "export DOCKER_HOST=\"unix://$podman_sock\""
        exit 0
    fi

    if [[ -S "/var/run/docker.sock" ]]; then
        echo "export DOCKER_HOST=\"unix:///var/run/docker.sock\""
        exit 0
    fi

    docker_desktop_sock="${HOME}/.docker/run/docker.sock"
    if [[ -S "$docker_desktop_sock" ]]; then
        echo "export DOCKER_HOST=\"unix://$docker_desktop_sock\""
        exit 0
    fi

    exit 1
fi

# Handle --info flag
if [[ "${1:-}" == "--info" ]]; then
    detect_runtime || true
    print_runtime_info
    exit 0
fi

# Handle --verify flag
if [[ "${1:-}" == "--verify" ]]; then
    detect_runtime || exit 1
    verify_connectivity || exit 1
    exit 0
fi

# Default: run detection
detect_runtime
