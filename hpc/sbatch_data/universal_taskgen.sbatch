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
# Universal Task Generation SBATCH Template
# ==============================================================================
# This template replaces the 400+ line cluster-specific scripts by delegating
# all logic to the TaskgenJobRunner Python class.
#
# Usage: The launcher writes a JSON config file and substitutes {config_path}
# ==============================================================================

set -eo pipefail
ulimit -c 0  # Disable core dumps to avoid filling disk space

# Handle bash completion scripts that use BASH_COMPLETION_DEBUG
if [ -z "${BASH_COMPLETION_DEBUG+x}" ]; then
  export BASH_COMPLETION_DEBUG=""
fi

# --- Module and Conda Setup ---

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

# --- WORKDIR sanity guard (HARD FAIL — do not remove) ---
# If WORKDIR does not point at the OpenThoughts-Agent repo root, every downstream
# `source "$WORKDIR/hpc/..."` and `python -m hpc....` fails cryptically and the job
# dies instantly. This happens when a job is submitted from outside the repo with
# DCFT/DCFT_PRIVATE unset, so the fallback resolves WORKDIR=$PWD (the wrong dir).
# Fail loudly and immediately instead of running from the wrong place.
if [ ! -f "$WORKDIR/hpc/shell_utils/triton_cache.sh" ]; then
  echo "FATAL: WORKDIR='$WORKDIR' is not the OpenThoughts-Agent repo root." >&2
  echo "       (marker hpc/shell_utils/triton_cache.sh not found under it)." >&2
  echo "       Submit the job FROM the repo dir with DCFT exported — on Jupiter:" >&2
  echo "         cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent && export DCFT=\$PWD" >&2
  echo "       or pass --chdir <repo> to sbatch. Do NOT submit from \$HOME or a scratch subdir." >&2
  echo "       See .claude/skills/datagen-launch for the correct launch procedure." >&2
  exit 1
fi
cd "$WORKDIR"

if [ -z "${DCFT:-}" ]; then
  export DCFT="$WORKDIR"
fi

# --- Conda activation (cluster-specific, substituted by launcher) ---
{conda_activate}

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

# --- Daytona API key override (from --daytona_api_key CLI argument) ---
DAYTONA_API_KEY_OVERRIDE="{daytona_api_key_override}"
if [ -n "$DAYTONA_API_KEY_OVERRIDE" ]; then
  export DAYTONA_API_KEY="$DAYTONA_API_KEY_OVERRIDE"
fi

# --- Standard environment variables ---
export HF_HOME="/tmp/hf_home"
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_TIMEOUT=1800
export NCCL_IB_TIMEOUT=23
export PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# --- Ray/vLLM GPU visibility ---
# Previously we set RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 and
# RAY_NOSET_CUDA_VISIBLE_DEVICES=1 to keep Ray from rewriting CUDA_VISIBLE_DEVICES
# per actor (which caused "invalid device ordinal" on an earlier vLLM build).
# That now breaks the new wheel's multi-node DP placement: with NOSET set,
# vllm/v1/engine/core.py's per-actor `get_accelerator_ids()` returns an empty
# list (Ray isn't binding accelerators to actors), and the legacy fallback's
# `get_device_indices` formula crashes with `IndexError: list index out of
# range` on cross-node DP.
# Leave NOSET unset so Ray's placement-group bundle drives per-actor GPU
# selection, matching the patched core.py + ray_executor_v2.
# If the original "invalid device ordinal" regression returns for single-node
# taskgen on the new wheel, scope NOSET back on conditionally on NUM_NODES==1.

# --- Cluster-specific environment variables ---
{env_exports}

# --- Ray defaults ---
{ray_env_exports}

# --- SSH Tunneling (JSC clusters only) ---
{ssh_tunnel_setup}

# --- SOCKS5 Proxy Setup (JSC clusters - alternative to SSH tunnel) ---
{proxy_setup}

# --- Run the task generation job via Python runner ---
echo "=== Universal Task Generation Runner ==="
echo "Config: {config_path}"
echo "Working directory: $WORKDIR"
echo "========================================"

python -m hpc.datagen_launch_utils --mode taskgen --config "{config_path}"
