#!/bin/bash
# =============================================================================
# RL Environment Activation Script
# =============================================================================
# This script activates the RL training environment.
#
# Usage: source hpc/activate_rl_env.sh
#
# If the environment doesn't exist, run setup first:
#   ./hpc/setup_rl_env.sh
# =============================================================================

# Determine base directory
if [[ -n "${DCFT_PRIVATE:-}" ]]; then
    BASE_DIR="$DCFT_PRIVATE"
elif [[ -n "${DCFT:-}" ]]; then
    BASE_DIR="$DCFT"
else
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# RL environment location
RL_ENV_DIR="$BASE_DIR/envs/rl"

# Check if environment exists
if [[ ! -d "$RL_ENV_DIR" ]]; then
    echo "ERROR: RL environment not found at: $RL_ENV_DIR"
    echo ""
    echo "Please run the setup script first:"
    echo "  ./hpc/setup_rl_env.sh"
    echo ""
    return 1 2>/dev/null || exit 1
fi

# Activate the environment
if [[ -f "$RL_ENV_DIR/bin/activate" ]]; then
    source "$RL_ENV_DIR/bin/activate"
    echo "Activated RL environment: $RL_ENV_DIR"
else
    echo "ERROR: Cannot find activation script at: $RL_ENV_DIR/bin/activate"
    return 1 2>/dev/null || exit 1
fi
