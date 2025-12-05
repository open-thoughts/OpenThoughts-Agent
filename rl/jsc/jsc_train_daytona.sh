#!/usr/bin/env bash

export TERMINAL_BENCH_CACHE_DIR="/p/project/laionize/marianna/terminal_bench/terminal-bench/.cache"

UV_CACHE_DIR=/p/scratch/synthlaion/marianna/uv_cache
mkdir -p "$UV_CACHE_DIR"
export UV_CACHE_DIR

UV_TOOL_DIR=/p/scratch/synthlaion/marianna/uv_tool
mkdir -p "$UV_TOOL_DIR"
export UV_TOOL_DIR

export TUNNEL_PORT=7003  # Changed to 7003 for direct tunnel

export T_BENCH_HOME="/p/project/laionize/marianna/terminal_bench/terminal-bench"
export IMAGE=/p/data1/mmlaion/shared/containers/docker_dind-rootless.sif
export SCRIPT=/p/project/laionize/marianna/terminal_bench/test_tbench.py

# Conda / env
export TERMINAL_BENCH_CACHE_DIR="/p/project/laionize/marianna/terminal_bench/terminal-bench-main/.cache"
export CONDA_ENV="/p/project1/ccstdl/envs/marianna/py3.12"
export MINICONDA_PATH="/p/project1/ccstdl/nezhurina1/miniconda/miniconda"
source ${MINICONDA_PATH}/bin/activate ${CONDA_ENV}
export PYTHONPATH="${PYTHONPATH:-}:/p/project/laionize/marianna/terminal_bench/terminal-bench-main"
export PYTHONPATH=$PYTHONPATH:/p/project/laionize/marianna/terminal_bench/terminal-bench-main
echo "Starting HEAD at $head_node"

export PYTHON_EXEC="${CONDA_ENV}/bin/python"
export DOCKER_CONFIG="/p/project/laionize/etashg/docker_cache"

# vllm serve
export VLLM_SERVE_SCRIPT=/p/project/laionize/marianna/terminal_bench/bash-scripts/vllm_serve.sh
export MODEL_PATH="/p/data1/mmlaion/marianna/models/Qwen/Qwen2.5-0.5B-Instruct"
export SERVE_PORT=21000


# SSH
USER_NAME="guha1"
SSH_KEY="/p/project/laionize/marianna/terminal_bench/keys/docker"   # do NOT quote ~; ${HOME} is fine


# SLURM resources
SBATCH_JOB_NAME="tbench-test"
SBATCH_ACCOUNT="synthlaion"
SBATCH_PARTITION="dc-gpu-devel"
SBATCH_NODES=1
SBATCH_NUM_GPUS=4
SBATCH_NTASKS_PER_NODE=1
SBATCH_CPUS_PER_TASK=48

SBATCH_TIME="1:00:00"
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



JOBID=""    # will fill later

cleanup() {
  set +e
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

# ------------ START ALLOCATION (background) ------------
# We run a mini "init" on the compute node: write hostname -> wait for READYFILE -> run payload
ALLOC_CMD=$(cat <<'EOS'
  export T_BENCH_HOME="$T_BENCH_HOME"
  export TERMINAL_BENCH_CACHE_DIR="$TERMINAL_BENCH_CACHE_DIR"
  export UV_CACHE_DIR="$UV_CACHE_DIR"
  export DOCKER_HOST="$DOCKER_HOST"
  export IMAGE=/p/data1/mmlaion/shared/containers/docker_dind-rootless.sif
  export SCRIPT="$SCRIPT"
  export CONDA_ENV="/p/project1/ccstdl/envs/marianna/py3.12"
  export MINICONDA_PATH="/p/project1/ccstdl/nezhurina1/miniconda/miniconda"
  export NODEFILE="$NODEFILE"
  export READYFILE="$READYFILE"
  export SRUN_LOG="$SRUN_LOG"
  export SSH_KEY="$SSH_KEY"
  export USER_NAME="$USER_NAME"
  cd "$T_BENCH_HOME"
  module load CUDA/12
  module load numactl
  # Discover node and signal to login
  NODE_HOST="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
  NODE_HOST="${NODE_HOST}i"
  echo "$NODE_HOST" > "$NODEFILE"
  echo "[compute] Node host: $NODE_HOST"
  nvidia-smi
  export CONDA_ENV="/p/project1/ccstdl/envs/marianna/py3.12"
  export MINICONDA_PATH="/p/project1/ccstdl/nezhurina1/miniconda/miniconda"
  source ${MINICONDA_PATH}/bin/activate ${CONDA_ENV}
  export PYTHONPATH="${PYTHONPATH:-}:/p/project/laionize/marianna/terminal_bench/terminal-bench-main"
  export PYTHONPATH=$PYTHONPATH:/p/project/laionize/marianna/terminal_bench/terminal-bench-main

  export API_BASE=localhost:${SERVE_PORT}
  echo "[compute] API base: $API_BASE"
  export FLASHINFER_WORKSPACE_BASE=/p/project/laionize/marianna/terminal_bench/flashinfer_workspace
  export VLLM_CONFIG_ROOT=/p/project/laionize/marianna/terminal_bench/vllm_config
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
  export MASTER_PORT=12345

  # export NCCL_DEBUG=INFO
  # export NCCL_NET_GDR_LEVEL=PIX # Use GPU Direct RDMA when GPU and NIC are on the same PCI switch
  export NCCL_SOCKET_IFNAME=ib0
  export GLOO_SOCKET_IFNAME=ib0
  export NCCL_IB_TIMEOUT=120
  export NCCL_ALGO=Ring

  export NUM_GPUS_PER_NODE=4

  export nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
  export nodes_array=($nodes)
  export head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export head_node_ip=$(hostname -I | awk '{print $1}')
  export NUMACTL_BASE=$(dirname $(dirname $(which numactl)))

  # ============== WORKING TUNNEL SETUP ==============
  echo "[compute] Setting up direct SSH tunnel to login node..."
  sleep 2
  # Add keepalive options to SSH tunnel
  export DAYTONA_MAX_RETRIES=5
  export DAYTONA_RETRY_DELAY=30
  export DAYTONA_BACKOFF_FACTOR=2
  ssh -f -N -D 7003 \
      -o StrictHostKeyChecking=no \
      -o ConnectTimeout=1000 \
      -o ServerAliveInterval=10 \
      -o ServerAliveCountMax=30 \
      -o TCPKeepAlive=yes \
      -o ExitOnForwardFailure=yes \
      -o BatchMode=yes \
      -i /p/project/laionize/marianna/terminal_bench/keys/docker \
      guha1@jrlogin05i

  # Give tunnel time to establish
  sleep 5
  
  # Set timeout and SSL settings for better reliability
  export DAYTONA_TIMEOUT=1800  # 30 minutes
  export AIOHTTP_CLIENT_TIMEOUT=900  # 15 minutes
  export AIOHTTP_CONNECTOR_TIMEOUT=900
  export AIOHTTP_SOCK_CONNECT_TIMEOUT=300
  export AIOHTTP_TOTAL_TIMEOUT=1800
  export PYTHONHTTPSVERIFY=0
    
  # Clear problematic SSL certificate variables
  unset SSL_CERT_FILE
  unset CURL_CA_BUNDLE 
  unset REQUESTS_CA_BUNDLE
  unset SSL_CERT_DIR
  
  # Disable SSL verification for Python
  export PYTHONHTTPSVERIFY=0

  # Run the actual script
  echo "[compute] Running Daytona script..."

  # ============== RAY SETUP ==============
  echo "[compute] Cleaning up existing Ray processes..."
  ray stop --force || true
  rm -rf /tmp/ray/session_* || true
  sleep 5

  # Set Ray environment
  export RAY_DISABLE_IMPORT_WARNING=1
  export RAY_DEDUP_LOGS=0

  # Start Ray with explicit ports - using the INTERNAL IP
  RAY_GCS_PORT=6379
  RAY_DASHBOARD_PORT=8265
  if [[ -z "${DAYTONA_API_KEY:-}" ]]; then
    echo "[compute] ERROR: DAYTONA_API_KEY is not set. Export it in your environment or via the dotenv before running." >&2
    exit 1
  fi
  echo "[compute] Starting Ray cluster on internal IP: $head_node_ip"

  cat > /tmp/ray_env.json << EOF
{
  "env_vars": {
    "PYTHONHTTPSVERIFY": "0",
    "DAYTONA_API_KEY": "${DAYTONA_API_KEY}"
  }
}
EOF
  export RAY_RUNTIME_ENV_VARS="DAYTONA_TIMEOUT,AIOHTTP_CLIENT_TIMEOUT,AIOHTTP_CONNECTOR_TIMEOUT,AIOHTTP_SOCK_CONNECT_TIMEOUT,PYTHONHTTPSVERIFY"
  ray start --head \
    --node-ip-address=$head_node_ip \
    --port=$RAY_GCS_PORT \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --dashboard-host=0.0.0.0 \
    --num-cpus 48 \
    --num-gpus=$NUM_GPUS_PER_NODE \
    --temp-dir=/tmp/ray \
    --verbose

  # Wait for initialization
  echo "[compute] Waiting for Ray to initialize..."
  sleep 15

  # Set connection address using the internal IP
  export ip_head=$head_node_ip:$RAY_GCS_PORT
  echo "[compute] Ray Head Address: $ip_head"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  unset AIOHTTP_SOCKS_PROXY ALL_PROXY all_proxy SOCKS_PROXY socks_proxy
  
  # Run the original test_daytona.py (now works without proxy interference)
  bash /p/project/laionize/marianna/terminal_bench/ot-agent/train/jsc/helper_training.sh
EOS
)

# Launch salloc in background; capture the JobID from its first line (parsable)
# Note: salloc holds the allocation for the duration of the inner bash -lc block
{
  # shellcheck disable=SC2086
  srun --account synthlaion --partition dc-gpu-devel  --time=01:00:00 bash -lc "${ALLOC_CMD}"
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

NODE_HOST="$(cat "${NODEFILE}" 2>/dev/null || true)"
if [[ -z "${NODE_HOST}" ]]; then
  echo "[error] Node hostname file empty or missing. See ${ALLOC_LOG}"
  exit 1
fi
echo "[login] Compute node: ${NODE_HOST}"

# Signal compute node to proceed (no complex tunnel setup needed)
touch "${READYFILE}"
echo "[login] Ready signal dropped: ${READYFILE}"

echo
echo "=== Submitted ==="
echo " Run dir:      ${RUN_DIR}"
echo " Alloc log:    ${ALLOC_LOG}"
echo " srun log:     ${SRUN_LOG}"
echo
echo "Use:   tail -f ${ALLOC_LOG}"
echo

# Wait for the job to complete
echo "[info] Press Ctrl+C to clean up and exit."
wait "${ALLOC_PID}" || true

exit 0
