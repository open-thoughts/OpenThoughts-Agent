#!/bin/bash
#SBATCH --time={time_limit}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user={email_address}
# Send a batch-shell-only warning while the Ray job steps are still alive. The
# USR1 trap below translates this into a cooperative TERM for the RL runner.
# Five minutes covers MarinSkyRL's 120-second entrypoint shutdown deadline and
# leaves time for artifact-store and Ray-log cleanup before the hard time limit.
#SBATCH --signal=B:USR1@300
{sbatch_extra_directives}

# ==============================================================================
# Universal RL Training SBATCH Template
# ==============================================================================
# This template handles SkyRL-based reinforcement learning training jobs.
# It sets up Ray cluster, environment variables, and runs the SkyRL entrypoint.
#
# Usage: The launcher writes a JSON config file and substitutes {config_path}
# ==============================================================================

set -eo pipefail
ulimit -c 0  # Disable core dumps to avoid filling disk space
ulimit -n 65536 2>/dev/null || true  # Raise fd limit for high-concurrency Harbor trials

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
  echo "       See .agents/skills/rl-agentic-launch-jupiter for the correct launch procedure." >&2
  exit 1
fi
cd "$WORKDIR"

if [ -z "${DCFT:-}" ]; then
  export DCFT="$WORKDIR"
fi

# --- Conda activation (cluster-specific, substituted by launcher) ---
{conda_activate}

# --- Source environment files (before RL env activation) ---
# Source dotenv and secrets first, but NOT DCFT_ACTIVATE_ENV which would clobber RL env
if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DCFT/hpc/dotenv/{cluster_env_file}"
fi
if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "$DC_AGENT_SECRET_ENV" ]; then
  set -a
  source "$DC_AGENT_SECRET_ENV"
  set +a
fi

# --- Daytona API key override (from --daytona_api_key CLI argument) ---
DAYTONA_API_KEY_OVERRIDE="{daytona_api_key_override}"
if [ -n "$DAYTONA_API_KEY_OVERRIDE" ]; then
  export DAYTONA_API_KEY="$DAYTONA_API_KEY_OVERRIDE"
fi

# --- Daytona target ---
# Intentionally NOT setting DAYTONA_TARGET. Harbor's daytona.py treats unset
# DAYTONA_TARGET as "no region" — snapshot names omit the region segment and
# region_id is left None when creating snapshots. This must stay aligned with
# prebuild_daytona_snapshots() in hpc/rl_launch_utils.py, which is also region-less.

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

# --- Ensure CUDA_HOME/PATH survive conda activation ---
# Module loads set CUDA_HOME but conda activation can reset PATH.
# Re-export to ensure vLLM and DeepSpeed find the right CUDA.
if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# --- NCCL/Networking settings (cluster-specific) ---
{nccl_exports}

# --- RL container env (Apptainer SIF overlays / pydeps / extra exports) ---
# Emitted by the launcher from the RL yaml `container:` section (if present).
# Sets RL_CONTAINER_OVERLAYS / RL_CONTAINER_PYDEPS (read at runtime by
# hpc/rl_launch_utils.py build_apptainer_prefix + _build_container_pythonpath)
# plus any container.extra_env exports (e.g. SKYRL_GDN_MASK_FLA, alloc-conf).
# Empty (comment-only) when the yaml has no container section — byte-identical
# to the prior non-container path. Exported here so every Ray daemon/driver
# (apptainer --env passthrough + srun --export=ALL) inherits them.
{rl_container_env}

# --- Standard environment variables ---
# UV_USE_IO_URING=0 forces libuv (bundled in uvloop 0.22.1 as libuv 1.48.0)
# onto the plain, immediate epoll_ctl path instead of the io_uring EPOLL_CTL
# batching path. libuv 1.48.0 defers EPOLL_CTL_DEL through an io_uring SQ; under
# high socket churn (timeout-heavy RL rollouts → constant connect/teardown to
# vLLM + Daytona) a deferred op runs against a closed/recycled fd and libuv
# aborts() in uv__epoll_ctl_flush — surfacing as `uv__epoll_ctl_prep ->
# Fatal Python error: Aborted` in the skyrl_entrypoint Ray driver at ~30-70 min.
# Fixed upstream in libuv 1.49.0 ("don't delay EPOLL_CTL_DEL operations"), which
# no uvloop release ships yet. This var is exported in the top-level batch shell
# and propagates to every Ray head/worker via `srun --export=ALL` (ray_utils.py),
# so the skyrl_entrypoint Ray task inherits it. Harmless on io_uring-disabled
# nodes (they already run the plain-epoll path). See
# notes/ot-agent/agent_logs/2026-05-28_uvloop_libuv_io_uring_epoll_abort_rootcause.md
export UV_USE_IO_URING=0
export PYTHONFAULTHANDLER=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# NCCL_BLOCKING_WAIT=1 makes the NCCL collective itself error out at the
# call boundary rather than spinning while the watchdog races to abort.
# Combined with TORCH_NCCL_ASYNC_ERROR_HANDLING above, this surfaces the
# real failure point (often a peer-node HSN socket close on Perlmutter)
# instead of a downstream SIGABRT in a NCCL destructor.
export NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

# --- RL-specific environment variables ---
{rl_env_exports}

# --- RL container env (LATE override — config container.extra_env wins) ---
# Re-emit the yaml container.extra_env exports AFTER both the hardcoded
# export TORCH_NCCL_* lines above and the RL-specific env exports block,
# so a config that intentionally sets a normally-hardcoded var (e.g. #232 lowering
# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC to 600 so NCCL's flight-recorder fires BEFORE
# vLLM's 900s execute_model watchdog and actually dumps) wins by shell last-write.
# No-op for every config whose extra_env does NOT collide with a hardcoded var
# (the earlier container-env block already exported those, re-export is
# idempotent), so this is byte-identical behavior for all existing configs.
{rl_container_env_late}

# --- Ray defaults ---
{ray_env_exports}

# --- vLLM/Ray settings ---
# RL uses its own conda env pinned to an older vLLM wheel that requires the
# V1 engine. The eval/datagen path uses the newer wheel and runs V2 instead.
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

# Preserve rank-by-rank collective history on the shared experiment filesystem.
# PyTorch appends the global rank to this prefix when its NCCL watchdog fires.
source "$WORKDIR/hpc/shell_utils/nccl_flight_recorder.sh"
setup_nccl_flight_recorder "{experiments_dir}" || exit $?

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
  local WORKER_LOG_SCRIPT="$WORKDIR/hpc/ray/collect_worker_ray_logs.sh"
  if [[ -f "$WORKER_LOG_SCRIPT" ]]; then
    source "$WORKER_LOG_SCRIPT"
    collect_worker_ray_logs "{experiments_dir}/ray_logs"
  fi

  echo "Ray log preservation complete"
}

ARTIFACT_STORE_ENABLED="{artifact_store_enabled}"
ARTIFACT_STORE_IMAGE="{artifact_store_image}"
ARTIFACT_STORE_MOUNT="{artifact_store_mount}"
ARTIFACT_STORE_LEASE="${ARTIFACT_STORE_IMAGE}.mount-lease"
ARTIFACT_STORE_MOUNTED=0
ARTIFACT_STORE_LEASE_HELD=0
RL_RUNNER_PID=""

release_artifact_store_lease() {
  if [[ "$ARTIFACT_STORE_LEASE_HELD" != "1" ]]; then
    return
  fi
  rm -f "$ARTIFACT_STORE_LEASE/owner"
  rmdir "$ARTIFACT_STORE_LEASE"
  ARTIFACT_STORE_LEASE_HELD=0
}

acquire_artifact_store_lease() {
  if ! mkdir "$ARTIFACT_STORE_LEASE" 2>/dev/null; then
    local previous_job=""
    local previous_state=""
    if [[ -f "$ARTIFACT_STORE_LEASE/owner" ]]; then
      previous_job=$(sed -n 's/^job_id=//p' "$ARTIFACT_STORE_LEASE/owner" | head -1)
    fi
    if [[ -z "$previous_job" ]]; then
      echo "FATAL: artifact store has an active non-Slurm mount lease: $ARTIFACT_STORE_LEASE" >&2
      return 1
    fi
    if ! previous_state=$(squeue -h -j "$previous_job" -o '%T'); then
      echo "FATAL: could not validate artifact-store lease owner job $previous_job" >&2
      return 1
    fi
    if [[ -n "$previous_state" ]]; then
      echo "FATAL: artifact store is owned by active job $previous_job ($previous_state)" >&2
      return 1
    fi
    echo "Recovering stale artifact-store lease from completed job $previous_job"
    rm -f "$ARTIFACT_STORE_LEASE/owner"
    rmdir "$ARTIFACT_STORE_LEASE"
    mkdir "$ARTIFACT_STORE_LEASE"
  fi
  printf 'kind=writer\njob_id=%s\nnode=%s\n' "$SLURM_JOB_ID" "$(hostname)" > "$ARTIFACT_STORE_LEASE/owner"
  ARTIFACT_STORE_LEASE_HELD=1
}

cleanup_artifact_store() {
  local status=0
  if [[ "$ARTIFACT_STORE_MOUNTED" != "1" ]]; then
    release_artifact_store_lease
    return
  fi
  echo "Flushing artifact store: $ARTIFACT_STORE_IMAGE"
  if ! sync; then
    echo "FATAL: sync failed for artifact store $ARTIFACT_STORE_IMAGE" >&2
    status=1
  fi
  if ! fusermount3 -u "$ARTIFACT_STORE_MOUNT"; then
    echo "FATAL: could not unmount artifact store $ARTIFACT_STORE_MOUNT" >&2
    return 1
  fi
  ARTIFACT_STORE_MOUNTED=0
  flock -u 9 || true
  exec 9>&-
  rmdir "$ARTIFACT_STORE_MOUNT" 2>/dev/null || true
  release_artifact_store_lease
  echo "Artifact store unmounted"
  return "$status"
}

cleanup_job() {
  local status=$?
  local cleanup_status=0
  trap - EXIT TERM USR1
  set +e
  cleanup_artifact_store || cleanup_status=1
  cleanup_ray_logs || cleanup_status=1
  if [[ "$status" == "0" ]] && [[ "$cleanup_status" != "0" ]]; then
    status=$cleanup_status
  fi
  exit "$status"
}

source "$WORKDIR/hpc/shell_utils/rl_shutdown.sh"

forward_termination() {
  echo "Received Slurm shutdown signal; forwarding SIGTERM to the RL runner"
  if ! forward_termination_to_rl_runner "$RL_RUNNER_PID"; then
    exit 143
  fi
}

trap cleanup_job EXIT
trap forward_termination TERM USR1

if [[ "$ARTIFACT_STORE_ENABLED" == "1" ]]; then
  if [[ ! -f "$ARTIFACT_STORE_IMAGE" ]]; then
    echo "FATAL: artifact-store image is missing: $ARTIFACT_STORE_IMAGE" >&2
    exit 1
  fi
  mkdir -p "$ARTIFACT_STORE_MOUNT"
  acquire_artifact_store_lease
  exec 9>"${ARTIFACT_STORE_IMAGE}.lock"
  if ! flock -n 9; then
    echo "FATAL: artifact store already has an active writer: $ARTIFACT_STORE_IMAGE" >&2
    exit 1
  fi
  set +e
  e2fsck -p "$ARTIFACT_STORE_IMAGE"
  E2FSCK_STATUS=$?
  set -e
  if [[ "$E2FSCK_STATUS" -gt 1 ]]; then
    echo "FATAL: e2fsck failed for $ARTIFACT_STORE_IMAGE (status $E2FSCK_STATUS)" >&2
    exit "$E2FSCK_STATUS"
  fi
  fuse2fs -o rw,fakeroot "$ARTIFACT_STORE_IMAGE" "$ARTIFACT_STORE_MOUNT"
  if ! mountpoint -q "$ARTIFACT_STORE_MOUNT"; then
    echo "FATAL: fuse2fs returned without mounting $ARTIFACT_STORE_MOUNT" >&2
    exit 1
  fi
  ARTIFACT_STORE_MOUNTED=1
  mkdir -p "$ARTIFACT_STORE_MOUNT/trace_jobs"
  echo "Artifact store mounted at $ARTIFACT_STORE_MOUNT"
fi

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
echo "UV_USE_IO_URING: ${UV_USE_IO_URING:-<unset>}"  # libuv io_uring epoll-abort fix; must be 0

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

python -m hpc.rl_launch_utils --config "{config_path}" &
RL_RUNNER_PID=$!
set +e
while true; do
  wait "$RL_RUNNER_PID"
  RL_STATUS=$?
  if ! kill -0 "$RL_RUNNER_PID" 2>/dev/null; then
    break
  fi
  echo "RL runner is still shutting down; continuing to wait"
done
set -e
RL_RUNNER_PID=""
exit "$RL_STATUS"
