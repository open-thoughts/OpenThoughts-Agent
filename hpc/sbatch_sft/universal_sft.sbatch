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
# Universal SFT Training SBATCH Template
# ==============================================================================
# This template replaces the cluster-specific *_train.sbatch scripts by delegating
# all logic to the SFTJobRunner Python class.
#
# Usage: The launcher writes a JSON config file and substitutes {config_path}
# ==============================================================================

set -euo pipefail
ulimit -c 0  # Disable core dumps to avoid filling disk space

# Handle bash completion scripts that use BASH_COMPLETION_DEBUG
if [ -z "${BASH_COMPLETION_DEBUG+x}" ]; then
  export BASH_COMPLETION_DEBUG=""
fi

# Guard conda deactivate scripts from set -u complaints
export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CONDA_BACKUP_CC="${CONDA_BACKUP_CC:-}"
export CONDA_BACKUP_FC="${CONDA_BACKUP_FC:-}"

# --- Module loading (cluster-specific, substituted by launcher) ---
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
set +u
{conda_activate}
set -u

# --- Source environment files ---
if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DCFT/hpc/dotenv/{cluster_env_file}"
fi
if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "$DC_AGENT_SECRET_ENV" ]; then
  set -a
  source "$DC_AGENT_SECRET_ENV"
  set +a
fi
if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
fi

# --- CUDA path detection (Perlmutter and similar) ---
{cuda_setup}

# --- NCCL/Networking settings (cluster-specific) ---
{nccl_exports}

# --- Standard environment variables ---
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1  # Skip LlamaFactory transformers version check

# --- Distributed training setup ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT={master_port}
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS_PER_NODE={gpus_per_node}
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))

# --- HuggingFace/WandB paths ---
export HF_HOME="${HF_HOME:-${HF_HUB_CACHE:-/tmp/hf_home}}"
export WANDB_DIR="${DCFT_WANDB_DIR:-$DCFT/{experiments_dir}/wandb}"

# --- Triton/TorchInductor cache settings (node-local to avoid shared FS issues) ---
export TRITON_CACHE_VERBOSE=1
source "$WORKDIR/hpc/shell_utils/triton_cache.sh"

# --- Create experiment directories ---
mkdir -p "$DCFT/{experiments_dir}"
mkdir -p "$DCFT/{experiments_dir}/logs"
mkdir -p "$DCFT/{experiments_dir}/tmp"

# --- Supabase environment variables for DB registration ---
for _supabase_var in SUPABASE_URL SUPABASE_ANON_KEY SUPABASE_SERVICE_ROLE_KEY; do
    if [[ -n "${!_supabase_var:-}" ]]; then
        export "${_supabase_var}=${!_supabase_var}"
    else
        echo "Warning: ${_supabase_var} is not set; Supabase registration may fail." >&2
    fi
done

# --- SSH Tunneling (JSC clusters only) ---
{ssh_tunnel_setup}

# --- Run the SFT job via Python runner ---
echo "=== Universal SFT Training Runner ==="
echo "Config: {config_path}"
echo "Working directory: $WORKDIR"
echo "Nodes: $NUM_NODES, GPUs/node: $NUM_GPUS_PER_NODE"
echo "======================================"

{srun_command}
