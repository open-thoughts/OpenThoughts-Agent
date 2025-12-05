#!/usr/bin/env bash
set -euo pipefail
export TERMINAL_BENCH_CACHE_DIR="/p/project/laionize/marianna/terminal_bench/terminal-bench/.cache"

UV_CACHE_DIR=/p/scratch/synthlaion/marianna/uv_cache
mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

UV_TOOL_DIR=/p/scratch/synthlaion/marianna/uv_tool
mkdir -p "$UV_TOOL_DIR"
export UV_TOOL_DIR

export TUNNEL_PORT=7001

export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export DOCKER_HOST=tcp://127.0.0.1:$TUNNEL_PORT

export T_BENCH_HOME="/p/project/laionize/marianna/terminal_bench/terminal-bench"
export IMAGE=/p/data1/mmlaion/shared/containers/docker_dind-rootless.sif
export SCRIPT=/p/project/laionize/marianna/terminal_bench/test_tbench.py
# export SCRIPT=/p/project/laionize/marianna/terminal_bench/bash-scripts/test_litellm.py
# Conda / env
export CONDA_ENV="/p/project1/ccstdl/envs/marianna/t-bench"
export MINICONDA_PATH="/p/project1/ccstdl/nezhurina1/miniconda/miniconda"
export PYTHON_EXEC="${CONDA_ENV}/bin/python"


# vllm serve
export VLLM_SERVE_SCRIPT=/p/project/laionize/marianna/terminal_bench/bash-scripts/vllm_serve.sh
export MODEL_PATH="/p/data1/mmlaion/marianna/models/Qwen/Qwen2.5-7B-Instruct"
# export MODEL_PATH="/p/data1/mmlaion/marianna/models/openai/gpt-oss-20b"
export SERVE_PORT=21000


# SSH
USER_NAME="$(whoami)"
SSH_KEY="${HOME}/.ssh/docker"   # do NOT quote ~; ${HOME} is fine


# SLURM resources
SBATCH_JOB_NAME="tbench-test"
SBATCH_ACCOUNT="cstdl"
SBATCH_PARTITION="dc-cpu"
SBATCH_NODES=1
SBATCH_NUM_GPUS=0
SBATCH_NTASKS_PER_NODE=1
SBATCH_CPUS_PER_TASK=48
SBATCH_TIME="08:00:00"
OUT_DIR="/p/project/laionize/marianna/terminal_bench/slurm-output"
mkdir -p "$OUT_DIR"

# ------------ RUNTIME STAGING ------------
ts="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_DIR}/run_${ts}"
mkdir -p "$RUN_DIR"
export RUN_DIR
export NODEFILE="${RUN_DIR}/node.txt"
export READYFILE="${RUN_DIR}/tunnel.ready"
export ALLOC_LOG="${RUN_DIR}/alloc.log"
export SRUN_LOG="${RUN_DIR}/srun.log"
export TUNNEL_LOG="${RUN_DIR}/tunnel.log"


# Pre-fetch images
export PREFETCH_SCRIPT="/p/project/laionize/marianna/terminal_bench/prefetch_docker_images.py"
export PREFETCH_TARS_SAVE_DIR="/p/scratch/laionize/marianna/terminal_bench/tars"
mkdir -p "$PREFETCH_TARS_SAVE_DIR"
export PREFETCH_TBENCH_DIR=$T_BENCH_HOME
export PREFETCH_CMD="apptainer exec \
  --bind $PREFETCH_TBENCH_DIR:$PREFETCH_TBENCH_DIR \
  --bind $PREFETCH_TARS_SAVE_DIR:$PREFETCH_TARS_SAVE_DIR $IMAGE\
  $PYTHON_EXEC $PREFETCH_SCRIPT \
    --tars_save_dir=$PREFETCH_TARS_SAVE_DIR \
    --tbench_dir=$PREFETCH_TBENCH_DIR"

echo "[info] Pre-fetch command: $PREFETCH_CMD"
echo "[info] Starting pre-fetch..."
bash -c "$PREFETCH_CMD" | tee -a "$RUN_DIR/prefetch.log"

echo "[info] Pre-fetch completed."


JOBID=""    # will fill later
TUNNEL_PID=""

cleanup() {
  set +e
  if [[ -n "${TUNNEL_PID}" ]] && ps -p "${TUNNEL_PID}" >/dev/null 2>&1; then
    echo "[cleanup] Killing tunnel pid ${TUNNEL_PID}" | tee -a "$TUNNEL_LOG"
    kill "${TUNNEL_PID}" || true
  fi
  if [[ -n "${JOBID}" ]]; then
    # If job still running, cancel it
    if squeue -j "${JOBID}" -h >/dev/null 2>&1; then
      echo "[cleanup] Cancelling allocation ${JOBID}" | tee -a "$ALLOC_LOG"
      scancel "${JOBID}" || true
    fi
  fi
}
trap cleanup EXIT

echo "[info] Logs & handshakes under: ${RUN_DIR}"
echo "[info] Will forward login:${TUNNEL_PORT} ⇄ compute:${TUNNEL_PORT}"

# ------------ START ALLOCATION (background) ------------
# We run a mini “init” on the compute node: write hostname -> wait for READYFILE -> run payload
ALLOC_CMD=$(cat <<'EOS'
  set -euo pipefail

  export T_BENCH_HOME="$T_BENCH_HOME"
  export TERMINAL_BENCH_CACHE_DIR="$TERMINAL_BENCH_CACHE_DIR"
  export UV_CACHE_DIR="$UV_CACHE_DIR"
  export SSL_CERT_FILE="$SSL_CERT_FILE"
  export DOCKER_HOST="$DOCKER_HOST"
  export IMAGE="$IMAGE"
  export SCRIPT="$SCRIPT"
  export CONDA_ENV="$CONDA_ENV"
  export MINICONDA_PATH="$MINICONDA_PATH"
  export NODEFILE="$NODEFILE"
  export READYFILE="$READYFILE"
  export SRUN_LOG="$SRUN_LOG"

  cd "$T_BENCH_HOME"

  # Discover node and signal to login
  NODE_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  NODE_HOST="${NODE_HOST}i"
  echo "$NODE_HOST" > "$NODEFILE"
  echo "[compute] Node host: $NODE_HOST"

  
  # Start vllm serve
  # bash "$VLLM_SERVE_SCRIPT" "$MODEL_PATH" "$SERVE_PORT" "$RUN_DIR"
  # echo "[compute] starting VLLM server. Model: $MODEL_PATH"


  export API_BASE=http://localhost:${SERVE_PORT}/v1

  echo "[compute] API base: $API_BASE"

  # Run the payload on the allocated node
  srun --overlap --output="$SRUN_LOG" \
    bash -lc "apptainer exec \
        --bind ${CONDA_ENV}:${CONDA_ENV} \
        --bind ${T_BENCH_HOME}:${T_BENCH_HOME} \
        ${IMAGE} ${MINICONDA_PATH}/bin/conda run -p ${CONDA_ENV} python ${SCRIPT} \
          --api_base $API_BASE --agent oracle --model_name $MODEL_PATH"
EOS
)

# Launch salloc in background; capture the JobID from its first line (parsable)
# Note: salloc holds the allocation for the duration of the inner bash -lc block
{
  # shellcheck disable=SC2086
  salloc \
    --job-name="${SBATCH_JOB_NAME}" \
    --account="${SBATCH_ACCOUNT}" \
    --partition="${SBATCH_PARTITION}" \
    --nodes="${SBATCH_NODES}" \
    --ntasks-per-node="${SBATCH_NTASKS_PER_NODE}" \
    --cpus-per-task="${SBATCH_CPUS_PER_TASK}" \
    --time="${SBATCH_TIME}" \
    --gres=gpu:"${SBATCH_NUM_GPUS}" \
    bash -lc "${ALLOC_CMD}"
} >"${ALLOC_LOG}" 2>&1 &

ALLOC_PID=$!


# Wait for compute node to write hostname
echo "[login] Waiting for compute node hostname in ${NODEFILE} ..."
for i in {1..300}; do
  if [[ -s "${NODEFILE}" ]]; then
    break
  fi
  # Check if allocation died early
  if ! ps -p "${ALLOC_PID}" >/dev/null 2>&1; then
    echo "[error] Allocation process exited early. See ${ALLOC_LOG}"
    exit 1
  fi
  sleep 1
done

NODE_HOST="$(cat "${NODEFILE}")"
if [[ -z "${NODE_HOST}" ]]; then
  echo "[error] Node hostname file empty. See ${ALLOC_LOG}"
  exit 1
fi
echo "[login] Compute node: ${NODE_HOST}"

# Open reverse SSH tunnel login -> compute
TUNNEL_CMD=(ssh -i "${SSH_KEY}"
  -fN -o ExitOnForwardFailure=yes
  -R 127.0.0.1:${TUNNEL_PORT}:127.0.0.1:${TUNNEL_PORT}
  "${USER_NAME}@${NODE_HOST}"
)

echo "[login] Creating SSH tunnel: ${TUNNEL_CMD[*]}" | tee -a "$TUNNEL_LOG"
"${TUNNEL_CMD[@]}" >>"$TUNNEL_LOG" 2>&1 || {
  echo "[error] Tunnel creation failed. See ${TUNNEL_LOG}"
  exit 1
}
# Record tunnel PID (best-effort)
TUNNEL_PID="$(pgrep -f "ssh -i ${SSH_KEY} -fN -o ExitOnForwardFailure=yes -R 127.0.0.1:${TUNNEL_PORT}:127.0.0.1:${TUNNEL_PORT} ${USER_NAME}@${NODE_HOST}" | head -n1 || true)"
echo "[login] Tunnel PID: ${TUNNEL_PID:-unknown}" | tee -a "$TUNNEL_LOG"

# Signal compute node to proceed
touch "${READYFILE}"
echo "[login] Tunnel ready signal dropped: ${READYFILE}"

echo
echo "=== Submitted ==="
echo " JobID:        ${JOBID}"
echo " Run dir:      ${RUN_DIR}"
echo " Alloc log:    ${ALLOC_LOG}"
echo " Tunnel log:   ${TUNNEL_LOG}"
echo " srun log:     ${SRUN_LOG}"
echo
echo "Use:   tail -f ${SRUN_LOG}"
echo "Or:    squeue -j ${JOBID}"
echo

# destroy the tunnel and allocation on exit
echo "[info] Press Ctrl+C to clean up and exit."
wait "${ALLOC_PID}" || true

# We’re done — leave allocation + tunnel running in background
exit 0
