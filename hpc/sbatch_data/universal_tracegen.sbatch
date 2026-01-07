#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
{sbatch_extra_directives}

# ==============================================================================
# Universal Trace Generation SBATCH Template
# ==============================================================================
# This template replaces the 600+ line cluster-specific scripts by delegating
# all logic to the TracegenJobRunner Python class.
#
# Usage: The launcher writes a JSON config file and substitutes {config_path}
# ==============================================================================

set -euo pipefail

# Handle bash completion scripts that use BASH_COMPLETION_DEBUG
if [ -z "${BASH_COMPLETION_DEBUG+x}" ]; then
  export BASH_COMPLETION_DEBUG=""
fi

# --- Module loading (cluster-specific, substituted by launcher) ---
# Disable unbound variable check for module loading (modules may reference unset vars)
set +u
{module_commands}
set -u

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
# Disable unbound variable check for conda (conda scripts may reference unset vars)
set +u
{conda_activate}
set -u

# --- Source environment files ---
if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DCFT/hpc/dotenv/{cluster_env_file}"
fi
if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "$DC_AGENT_SECRET_ENV" ]; then
  source "$DC_AGENT_SECRET_ENV"
fi
if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  eval "$DCFT_ACTIVATE_ENV"
fi

# --- Standard environment variables ---
export HF_HOME="/tmp/hf_home"
export PYTHONFAULTHANDLER=1
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=23
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# --- Run the trace generation job via Python runner ---
echo "=== Universal Trace Generation Runner ==="
echo "Config: {config_path}"
echo "Working directory: $WORKDIR"
echo "========================================="

python -m hpc.datagen_launch_utils --mode tracegen --config "{config_path}"
