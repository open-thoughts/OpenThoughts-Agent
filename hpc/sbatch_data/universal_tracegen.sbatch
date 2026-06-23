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
# Universal Trace Generation SBATCH Template
# ==============================================================================
# This template replaces the 600+ line cluster-specific scripts by delegating
# all logic to the TracegenJobRunner Python class.
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
# range` on cross-node DP (see Jupiter 445905/447100/447578 logs).
# Leave NOSET unset on this path so Ray's placement-group bundle drives the
# per-actor GPU selection, matching the patched core.py + ray_executor_v2.
# If the original "invalid device ordinal" regression returns for single-node
# datagen on the new wheel, scope NOSET back on conditionally on $NUM_NODES==1.

# --- Cluster-specific environment variables ---
{env_exports}

# --- Ray defaults ---
{ray_env_exports}

# --- Triton/TorchInductor cache settings (node-local to avoid shared FS issues) ---
export TRITON_CACHE_VERBOSE=1
source "$WORKDIR/hpc/shell_utils/triton_cache.sh"

# --- Create experiment directories ---
# Note: {experiments_dir} is an absolute path substituted by the launcher
mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"
mkdir -p "{experiments_dir}/ray_logs"

# --- SSH Tunneling (JSC clusters only) ---
{ssh_tunnel_setup}

# --- SOCKS5 Proxy Setup (JSC clusters - alternative to SSH tunnel) ---
{proxy_setup}

# --- Cleanup trap to preserve Ray logs ---
# Ray uses /tmp by default for logs/sockets. This trap copies them to the
# experiments directory for debugging after the job terminates.
cleanup_ray_logs() {
  echo "Preserving Ray logs to {experiments_dir}/ray_logs/"

  # --- Head node logs ---
  if [[ -d /tmp/ray ]]; then
    rsync -a --ignore-errors /tmp/ray/ "{experiments_dir}/ray_logs/" 2>/dev/null || true
  fi
  for ray_dir in /tmp/ray_logs /tmp/ray_tmp; do
    if [[ -d "$ray_dir" ]]; then
      rsync -a --ignore-errors "$ray_dir/" "{experiments_dir}/ray_logs/$(basename $ray_dir)/" 2>/dev/null || true
    fi
  done

  # --- Worker node logs (clusters with intra-node SSH, e.g. Jupiter) ---
  local WORKER_LOG_SCRIPT="$WORKDIR/scripts/ray/collect_worker_ray_logs.sh"
  if [[ -f "$WORKER_LOG_SCRIPT" ]]; then
    source "$WORKER_LOG_SCRIPT"
    collect_worker_ray_logs "{experiments_dir}/ray_logs"
  fi

  echo "Ray log preservation complete"
}
trap cleanup_ray_logs EXIT

# --- Container Runtime Setup ---
source "$WORKDIR/hpc/shell_utils/container_runtime.sh"
setup_container_runtime "{harbor_env}" "$WORKDIR" || exit $?

# --- Node health probe sidecar (every 5 min, per node) ---
# Logs /dev/shm usage + free RAM + top shm consumers to
# {experiments_dir}/logs/node_health_<NODE>.log
#
# Motivation: Jupiter docs note /dev/shm is tmpfs backed by node RAM and
# "can contribute to Out-of-Memory errors." We've observed datagen jobs hit
# shm_broadcast.py:698 watchdogs after ~55-80 min of sustained serving
# (GLM-4.7 job 493343, MiniMax job 493421). If /dev/shm or RAM is growing
# toward its ceiling over the serving window, that's a strong contributing
# signal. Flat usage rules out the tmpfs angle and points the diagnosis at
# vLLM-internal cross-node coordination.
#
# Cheap: 1 process per node, sleeping 300s between samples; logs are
# ~150 bytes/sample, ~10 KB/job at 2h × per-node. srun --overlap shares
# the allocation without contending for the main job step's tasks.
NODE_HEALTH_DIR="{experiments_dir}/logs"
mkdir -p "$NODE_HEALTH_DIR"
srun --overlap --ntasks-per-node=1 --nodes="$SLURM_JOB_NUM_NODES" \
     --kill-on-bad-exit=0 \
     bash -c '
NODE=$(hostname -s)
LOG="'"$NODE_HEALTH_DIR"'/node_health_$NODE.log"
echo "[probe] started on $NODE at $(date -Is) pid=$$" >> "$LOG"
while true; do
    TS=$(date -Is)
    # /dev/shm capacity (Linux tmpfs: used / total / use%)
    SHM=$(df -B1 /dev/shm 2>/dev/null | tail -1 \
          | awk "{printf \"used=%s total=%s pct=%s\", \$3, \$2, \$5}")
    # RAM (used / total / free / available)
    MEM=$(awk "/^Mem:/ {printf \"mem_used=%s mem_total=%s mem_free=%s mem_avail=%s\", \$3, \$2, \$4, \$7}" \
          <(free -b 2>/dev/null))
    # Top 5 /dev/shm consumers (size-sorted)
    SHM_TOP=$(du -sb /dev/shm/* 2>/dev/null | sort -rn | head -5 \
              | awk "{printf \"%s:%s \", \$2, \$1}")
    echo "$TS $SHM $MEM top_shm=[$SHM_TOP]" >> "$LOG"
    sleep 300
done
' >/dev/null 2>&1 &
NODE_HEALTH_PID=$!
echo "[node-health] sidecar started PID=$NODE_HEALTH_PID — logs at $NODE_HEALTH_DIR/node_health_<NODE>.log"

# --- Run the trace generation job via Python runner ---
echo "=== Universal Trace Generation Runner ==="
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
echo "========================================="

python -m hpc.datagen_launch_utils --mode tracegen --config "{config_path}"
