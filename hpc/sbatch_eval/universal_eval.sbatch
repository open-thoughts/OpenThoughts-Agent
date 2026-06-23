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

set -eo pipefail
ulimit -c 0  # Disable core dumps to avoid filling disk space

# Handle bash completion scripts that use BASH_COMPLETION_DEBUG
if [ -z "${BASH_COMPLETION_DEBUG+x}" ]; then
  export BASH_COMPLETION_DEBUG=""
fi

# --- Clean up /tmp to prevent state pollution from previous jobs ---
# Some HPC systems retain /tmp contents across job allocations on the same node.
# This can cause issues with tmux sessions, container state, and other temporary files.
rm -rf /tmp/tmux-* 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
rm -rf /tmp/hf_home 2>/dev/null || true
rm -rf /tmp/containers 2>/dev/null || true
rm -rf /tmp/podman-* 2>/dev/null || true

# --- Module and Conda Setup ---

# --- Module loading (cluster-specific, substituted by launcher) ---
{module_commands}

# --- Conda activation (cluster-specific, substituted by launcher) ---
# Must run before WORKDIR detection so DCFT is available from the dotenv.
{conda_activate}

# --- Source environment files (sets DCFT, SCRATCH, etc.) ---
# Try common DCFT locations so the dotenv can be sourced even before DCFT is set.
for _try_dcft in "${DCFT:-}" "${DCFT_PRIVATE:-}" "$PWD"; do
  if [ -n "$_try_dcft" ] && [ -f "$_try_dcft/hpc/dotenv/{cluster_env_file}" ]; then
    source "$_try_dcft/hpc/dotenv/{cluster_env_file}"
    break
  fi
done
if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "$DC_AGENT_SECRET_ENV" ]; then
  source "$DC_AGENT_SECRET_ENV"
fi
if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  eval "$DCFT_ACTIVATE_ENV"
fi

# --- Environment setup (WORKDIR) ---
# Now DCFT should be set from the dotenv sourced above.
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
  echo "       See .claude/skills/eval-agentic-launch for the correct launch procedure." >&2
  exit 1
fi
cd "$WORKDIR"

if [ -z "${DCFT:-}" ]; then
  export DCFT="$WORKDIR"
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
# eval on the new wheel, scope NOSET back on conditionally on NUM_NODES==1.

# --- Cluster-specific environment variables ---
{env_exports}

# --- Ray defaults ---
{ray_env_exports}

# --- Triton/TorchInductor cache settings (node-local to avoid shared FS issues) ---
export TRITON_CACHE_VERBOSE=1
source "$WORKDIR/hpc/shell_utils/triton_cache.sh"

# --- SSH Tunneling (JSC clusters only) ---
{ssh_tunnel_setup}

# --- SOCKS5 Proxy Setup (JSC clusters - alternative to SSH tunnel) ---
{proxy_setup}

# --- Container Runtime Setup ---
source "$WORKDIR/hpc/shell_utils/container_runtime.sh"
setup_container_runtime "{harbor_env}" "$WORKDIR" || exit $?

# --- Run the eval job via Python runner ---
echo "=== Universal Eval Runner ==="
echo "Config: {config_path}"
echo "Working directory: $WORKDIR"

# --- Proxy setup for no-internet clusters ---
# Uses LD_PRELOAD set by ssh_tunnel_setup (proxychains library approach)
# This routes external traffic (Daytona API) through proxy while internal traffic goes direct
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
echo "============================="

python -m hpc.eval_launch_utils --config "{config_path}"
