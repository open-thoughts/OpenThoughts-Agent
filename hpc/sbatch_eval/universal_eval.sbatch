#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user={email_address}
{sbatch_extra_directives}

# ==============================================================================
# Universal Eval SBATCH Template
# ==============================================================================
# This template replaces the 600+ line cluster-specific scripts by delegating
# all logic to the EvalJobRunner Python class.
#
# Usage: The launcher writes a JSON config file and substitutes {config_path}
# ==============================================================================

set -euo pipefail
ulimit -c 0  # Disable core dumps to avoid filling disk space

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

# Some module command templates toggle set -u; ensure we stay in +u mode
# through conda activation and env sourcing to avoid unbound variable errors.
set +u

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

# --- Source environment files ---
# Keep set +u active - dotenv files and DCFT_ACTIVATE_ENV may trigger module reloads
# which can trigger conda deactivation scripts with unset variables
if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DCFT/hpc/dotenv/{cluster_env_file}"
fi
if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "$DC_AGENT_SECRET_ENV" ]; then
  source "$DC_AGENT_SECRET_ENV"
fi
if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  eval "$DCFT_ACTIVATE_ENV"
fi

# Re-enable strict unbound variable checking after all env setup is complete
set -u

# --- Standard environment variables ---
export HF_HOME="/tmp/hf_home"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=23
export PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# --- Ray/vLLM GPU visibility fix ---
# Prevent Ray from modifying CUDA_VISIBLE_DEVICES per-actor.
# Without these, Ray sets CVD to a single physical GPU index (e.g. "3") per actor,
# but vLLM uses Ray-reported indices as torch ordinals, causing "invalid device ordinal".
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_NOSET_CUDA_VISIBLE_DEVICES=1

# --- Ray defaults ---
{ray_env_exports}

# --- Triton/TorchInductor cache settings (node-local to avoid shared FS issues) ---
export TRITON_CACHE_VERBOSE=1
source "$WORKDIR/hpc/shell_utils/triton_cache.sh"

# --- Container Runtime Setup ---
source "$WORKDIR/hpc/shell_utils/container_runtime.sh"
setup_container_runtime "{harbor_env}" "$WORKDIR" || exit $?

# --- Run the eval job via Python runner ---
echo "=== Universal Eval Runner ==="
echo "Config: {config_path}"
echo "Working directory: $WORKDIR"
echo "============================="

python -m hpc.eval_launch_utils --config "{config_path}"
