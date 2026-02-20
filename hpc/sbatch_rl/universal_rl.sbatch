#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user={email_address}
{sbatch_extra_directives}

# ==============================================================================
# Universal RL Training SBATCH Template
# ==============================================================================
# This template handles SkyRL-based reinforcement learning training jobs.
# It sets up Ray cluster, environment variables, and runs the SkyRL entrypoint.
#
# Usage: The launcher writes a JSON config file and substitutes {config_path}
# ==============================================================================

set -euo pipefail
ulimit -c 0  # Disable core dumps to avoid filling disk space
ulimit -n 65536 2>/dev/null || true  # Raise fd limit for high-concurrency Harbor trials

# Handle bash completion scripts that use BASH_COMPLETION_DEBUG
if [ -z "${BASH_COMPLETION_DEBUG+x}" ]; then
  export BASH_COMPLETION_DEBUG=""
fi

# --- Clean up /tmp to prevent state pollution from previous jobs ---
# Some HPC systems retain /tmp contents across job allocations on the same node.
# This can cause issues with tmux sessions, container state, and other temporary files.
# Wrapped in set +u/set -u in case some clusters error out on these commands.
set +u
rm -rf /tmp/tmux-* 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
rm -rf /tmp/hf_home 2>/dev/null || true
rm -rf /tmp/containers 2>/dev/null || true
rm -rf /tmp/podman-* 2>/dev/null || true
set -u

# --- Module and Conda Setup ---
# Disable unbound variable check for module loading and conda activation.
# Module reloads can trigger conda deactivation scripts that reference unset variables.
set +u

# --- Module loading (cluster-specific, substituted by launcher) ---
{module_commands}

# --- Environment setup ---
if [ -n "${DCFT_PRIVATE:-}" ]; then
  WORKDIR="$DCFT_PRIVATE"
elif [ -n "${DCFT:-}" ]; then
  WORKDIR="$DCFT"
else
  WORKDIR="$PWD"
fi
cd "$WORKDIR"

if [ -z "${DCFT:-}" ]; then
  export DCFT="$WORKDIR"
fi

# --- Conda activation (cluster-specific, substituted by launcher) ---
{conda_activate}

# --- Source environment files (before RL env activation) ---
# Keep set +u active - dotenv files and RL env activation may trigger module reloads
# which can trigger conda deactivation scripts with unset variables
# Source dotenv and secrets first, but NOT DCFT_ACTIVATE_ENV which would clobber RL env
if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DCFT/hpc/dotenv/{cluster_env_file}"
fi
if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "$DC_AGENT_SECRET_ENV" ]; then
  set -a
  source "$DC_AGENT_SECRET_ENV"
  set +a
fi

# --- RL Environment Activation ---
# The RL environment is separate from datagen due to dependency conflicts.
# Options:
#   1. venv created by ./hpc/setup_rl_env.sh (default)
#   2. Conda environment specified by --rl_use_conda --rl_conda_env NAME
# NOTE: We intentionally skip DCFT_ACTIVATE_ENV here - it would clobber the RL environment.
# The RL environment is explicitly specified via --rl_use_conda/--rl_conda_env flags.
{rl_env_activation}

# Skip DCFT_ACTIVATE_ENV for RL jobs - we use the RL environment activation above instead
# (DCFT_ACTIVATE_ENV typically activates the datagen/SFT environment which conflicts with RL)

# --- CUDA path detection (Perlmutter and similar) ---
{cuda_setup}

# --- NCCL/Networking settings (cluster-specific) ---
{nccl_exports}

# Re-enable strict unbound variable checking after all env setup is complete
set -u

# --- Standard environment variables ---
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# --- RL-specific environment variables ---
{rl_env_exports}

# --- Ray defaults ---
{ray_env_exports}

# --- vLLM/Ray settings ---
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# --- Triton/TorchInductor cache settings (node-local to avoid shared FS issues) ---
export TRITON_CACHE_VERBOSE=1
source "$WORKDIR/hpc/shell_utils/triton_cache.sh"

# --- Hydra debugging ---
export HYDRA_FULL_ERROR=1

# --- HuggingFace/WandB paths ---
export HF_HOME="${HF_HOME:-${HF_HUB_CACHE:-/tmp/hf_home}}"
export WANDB_DIR="${DCFT_WANDB_DIR:-{experiments_dir}/wandb}"

# --- Create experiment directories ---
# Note: {experiments_dir} is an absolute path substituted by the launcher
mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"
mkdir -p "{experiments_dir}/exports"
mkdir -p "{experiments_dir}/ray_logs"

# --- Cleanup trap to preserve Ray logs ---
# Ray uses /tmp by default for logs/sockets. This trap copies them to the
# experiments directory for debugging after the job terminates.
cleanup_ray_logs() {
  echo "Preserving Ray logs to {experiments_dir}/ray_logs/"
  # Find and copy Ray session directories from /tmp on head node
  if [[ -d /tmp/ray ]]; then
    rsync -a --ignore-errors /tmp/ray/ "{experiments_dir}/ray_logs/" 2>/dev/null || true
  fi
  # Also try common Ray log locations
  for ray_dir in /tmp/ray_logs /tmp/ray_tmp; do
    if [[ -d "$ray_dir" ]]; then
      rsync -a --ignore-errors "$ray_dir/" "{experiments_dir}/ray_logs/$(basename $ray_dir)/" 2>/dev/null || true
    fi
  done
  echo "Ray log preservation complete"
}
trap cleanup_ray_logs EXIT

# --- SSH Tunneling (JSC clusters only) ---
{ssh_tunnel_setup}

# --- SOCKS5 Proxy Setup (JSC clusters - alternative to SSH tunnel) ---
{proxy_setup}

# --- Container Runtime Setup ---
source "$WORKDIR/hpc/shell_utils/container_runtime.sh"
setup_container_runtime "{harbor_env}" "$WORKDIR" || exit $?

# --- Run the RL training job via Python runner ---
# Ray cluster management is handled by hpc/ray_utils.py using proper srun calls
echo "=== Universal RL Training Runner ==="
echo "Config: {config_path}"
echo "Working directory: $WORKDIR"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# --- Proxy setup for no-internet clusters ---
# Uses LD_PRELOAD set by ssh_tunnel_setup (proxychains library approach)
# This routes external traffic (Daytona API) through proxy while internal traffic (Ray, NCCL) goes direct
# LD_PRELOAD is inherited by all child processes including Ray workers
if [ -n "${LD_PRELOAD:-}" ]; then
    echo "Proxy: ENABLED via LD_PRELOAD"
    echo "  LD_PRELOAD: $LD_PRELOAD"
    echo "  PROXYCHAINS_CONF_FILE: ${PROXYCHAINS_CONF_FILE:-not set}"
    echo "  PROXYCHAINS_SOCKS5_HOST: ${PROXYCHAINS_SOCKS5_HOST:-not set}"
    echo "  PROXYCHAINS_SOCKS5_PORT: ${PROXYCHAINS_SOCKS5_PORT:-not set}"
else
    echo "Proxy: DISABLED (direct internet or not configured)"
fi
echo "========================================"

python -m hpc.rl_launch_utils --config "{config_path}"
