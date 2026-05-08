#!/bin/bash
#SBATCH -p booster
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --output=eval/logs/%x_%j.out
#SBATCH --job-name=eval_dp

# ==============================================================================
# Unified Eval Harbor DP — Jupiter Cluster (JSC GH200)
#
# Multi-node data-parallel version of unified_eval_harbor.sbatch.
# Runs N vLLM replicas (one per node/shard) with Ray, shards the dataset,
# and runs N parallel harbor jobs — one per shard.
#
# IMPORTANT: Must pass --nodes N to sbatch (or set via EVAL_NUM_SHARDS).
#
# Positional args (same as single-node):
#   $1 = MODEL (HF model name)
#   $2 = REPO_ID (HF dataset repo or local path)
#   $3 = BENCHMARK_ID (optional, DB benchmark UUID)
#   $4 = RUN_TAG_ARG (optional, override run tag)
#
# Extra env vars (DP-specific):
#   EVAL_NUM_SHARDS              (default: $SLURM_JOB_NUM_NODES)
#   EVAL_NODES_PER_SHARD         (default: 1)
#   EVAL_VLLM_PIPELINE_PARALLEL_SIZE (default: 1)
#
# All other env vars from unified_eval_harbor.sbatch are supported:
#   EVAL_N_CONCURRENT, EVAL_N_ATTEMPTS, EVAL_GPU_MEMORY_UTIL, etc.
# ==============================================================================

set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# --- Parse positional args ---
MODEL="${1:-mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1}"
REPO_ID="${2:-DCAgent/dev_set_71_tasks}"
BENCHMARK_ID="${3:-}"
RUN_TAG_ARG="${4:-}"

# --- Read env vars from listener (with defaults) ---
N_CONCURRENT="${EVAL_N_CONCURRENT:-128}"
N_ATTEMPTS="${EVAL_N_ATTEMPTS:-3}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.95}"
ERROR_THRESHOLD="${EVAL_DAYTONA_THRESHOLD:-999999}"
VLLM_MAX_RETRIES="${EVAL_VLLM_MAX_RETRIES:-20}"
AGENT_PARSER="${EVAL_AGENT_PARSER:-}"
ENABLE_THINKING="${EVAL_ENABLE_THINKING:-false}"
AGENT_NAME="${EVAL_AGENT_NAME:-terminus-2}"
EVAL_STARTS_LOG="${EVAL_STARTS_LOG:-}"
TIMEOUT_MULTIPLIER="${EVAL_TIMEOUT_MULTIPLIER:-1.0}"
CONFIG_YAML="${EVAL_CONFIG_YAML:-dcagent_eval_config.yaml}"
DB_JOB_ID="${EVAL_DB_JOB_ID:-}"
UPLOAD_USERNAME="${EVAL_UPLOAD_USERNAME:-$USER}"

# --- DP-specific env vars ---
GPUS_PER_NODE="${EVAL_GPUS_PER_NODE:-4}"  # From cluster config (Jupiter=4, MBZ=8)
TP_SIZE="${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-2}"
PP_SIZE="${EVAL_VLLM_PIPELINE_PARALLEL_SIZE:-1}"
NODES_PER_SHARD="${EVAL_NODES_PER_SHARD:-1}"
# Shards per node: how many vLLM replicas fit on one node (default: 4/TP)
SHARDS_PER_NODE="${EVAL_SHARDS_PER_NODE:-$((GPUS_PER_NODE / TP_SIZE))}"
GPUS_PER_SHARD=$((TP_SIZE * PP_SIZE))
# Total shards (default: nodes * shards_per_node)
NUM_SHARDS="${EVAL_NUM_SHARDS:-$((SLURM_JOB_NUM_NODES * SHARDS_PER_NODE))}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
if [[ "$REPO_ID" == /* ]]; then
    SAFE_REPO=$(basename "$REPO_ID")
else
    SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')
fi

# Benchmark shorthand for squeue
declare -A BENCH_SHORT=(
    ["DCAgent_dev_set_v2"]="v2"
    ["DCAgent2_swebench-verified-random-100-folders"]="swe"
    ["DCAgent2_terminal_bench_2"]="tb2"
    ["DCAgent2_aider_polyglot"]="aider"
    ["DCAgent2_bfcl-parity"]="bfcl"
    ["DCAgent_dev_set_71_tasks"]="v1"
)
BENCH_TAG="${BENCH_SHORT[$SAFE_REPO]:-${SAFE_REPO:0:12}}"
if [ "$SLURM_JOB_NAME" = "eval_dp" ]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="eval_dp_${BENCH_TAG}"
fi

echo "=============================================="
echo "Jupiter Eval Harbor DP (${NUM_SHARDS}-way)"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $REPO_ID"
echo "Benchmark ID: ${BENCHMARK_ID:-<auto>}"
echo "Nodes: $SLURM_JOB_NUM_NODES ($SHARDS_PER_NODE shards/node, $NUM_SHARDS total shards)"
echo "GPUs: TP=$TP_SIZE, PP=$PP_SIZE, $GPUS_PER_SHARD GPUs/shard"
echo "N concurrent (per shard): $N_CONCURRENT"
echo "N attempts: $N_ATTEMPTS"
echo "GPU memory util: $GPU_MEMORY_UTIL"
echo "Error threshold: $ERROR_THRESHOLD"
echo "vLLM max retries: $VLLM_MAX_RETRIES"
echo "Agent: $AGENT_NAME"
echo "Thinking: $ENABLE_THINKING"
echo "Timeout multiplier: $TIMEOUT_MULTIPLIER"
echo "Pipeline parallel: $PP_SIZE"
echo "Config YAML: $CONFIG_YAML"
echo "DB Job ID (pending): ${DB_JOB_ID:-<none>}"
echo "=============================================="

# ==============================================================================
# Cluster-Agnostic Environment Setup (same as single-node sbatch)
# ==============================================================================

DCFT="${EVAL_PROJECT_ROOT:-${DCFT:-/e/scratch/jureap59/$USER/OpenThoughts-Agent}}"
CLUSTER_NAME="${EVAL_CLUSTER_NAME:-jupiter}"
unset PYTHONPATH

# Source cluster-specific dotenv
DOTENV_FILE="$DCFT/hpc/dotenv/${CLUSTER_NAME}.env"
if [ -f "$DOTENV_FILE" ]; then
    source "$DOTENV_FILE"
    echo "Sourced dotenv: $DOTENV_FILE"
else
    echo "WARNING: dotenv not found: $DOTENV_FILE"
fi

DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$HOME/secrets.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

if [ -z "${DAYTONA_API_KEY:-}" ]; then
    echo "WARNING: DAYTONA_API_KEY not set. Check $DC_AGENT_SECRET_ENV"
fi
echo "Daytona API key: ${DAYTONA_API_KEY:0:12}..."

# vLLM / Ray / Triton env vars
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_CGRAPH_get_timeout=900
export VLLM_CONFIG_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_config_${USER}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER}}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_CACHE_DIR:-/tmp/flashinfer_cache_${USER}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache_${USER}}"
export HYDRA_FULL_ERROR=1
export HF_HUB_CACHE="${EVAL_HF_CACHE:-${HF_HUB_CACHE:-/e/data1/datasets/playground/ot/hf_hub}}"
export HF_HOME="${HF_HUB_CACHE}"
export HF_CACHE_DIR="$HF_HUB_CACHE"
export HF_XET_CACHE="${HF_XET_CACHE:-/tmp/hf_xet_cache_${USER}}"
mkdir -p "$HF_XET_CACHE"

HARBOR_SRC="${EVAL_HARBOR_SRC:-/e/scratch/jureap59/feuer1/harbor/src}"
export PYTHONPATH="${HARBOR_SRC}:${DCFT}:${PYTHONPATH:-}"

# ==============================================================================
# LD_LOADER Wrappers (needed on aarch64 Jupiter where shared conda lacks exec perms)
# ==============================================================================

OTAGENT_DIR="${OTAGENT_DIR:-/e/scratch/jureap59/feuer1/miniforge3/envs/otagent}"
export PATH="$OTAGENT_DIR/bin:$PATH"
export CONDA_PREFIX="$OTAGENT_DIR"

PYTHON_REAL="$OTAGENT_DIR/bin/python3.12"

# LD_LOADER Wrappers: required on aarch64 (Jupiter GH200) where shared conda
# lacks execute permissions.  Skipped on x86_64 where the conda env is directly usable.
if [ "$(uname -m)" = "aarch64" ] && [ -f /lib/ld-linux-aarch64.so.1 ]; then
    LD_LOADER="/lib/ld-linux-aarch64.so.1"
    # Use shared filesystem (not /tmp) so all nodes can see the wrappers
    WRAPPER_DIR="${DCFT}/eval/${CLUSTER_NAME}/.wrappers/${SLURM_JOB_ID}"
    mkdir -p "$WRAPPER_DIR"

    cat > "$WRAPPER_DIR/python3" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL "\$@"
WEOF
    cat > "$WRAPPER_DIR/python" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL "\$@"
WEOF
    cat > "$WRAPPER_DIR/ray" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL -m ray.scripts.scripts "\$@"
WEOF
    cat > "$WRAPPER_DIR/harbor" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL -c "import sys; from harbor.cli.main import app; sys.exit(app())" "\$@"
WEOF
    chmod +x "$WRAPPER_DIR/python3" "$WRAPPER_DIR/python" "$WRAPPER_DIR/ray" "$WRAPPER_DIR/harbor"
    export PATH="$WRAPPER_DIR:$PATH"
    PYTHON_BIN="$WRAPPER_DIR/python3"
    RAY_BIN="$WRAPPER_DIR/ray"
else
    PYTHON_BIN="$PYTHON_REAL"
    RAY_BIN="$(dirname "$PYTHON_REAL")/ray"
    WRAPPER_DIR=""
fi

# Set LD_LIBRARY_PATH so vLLM can find CUDA runtime libs (libcudart.so.12, etc.)
# The nvidia pip packages install libs under site-packages/nvidia/*/lib/
NVIDIA_LIB_DIR="$OTAGENT_DIR/lib/python3.12/site-packages/nvidia"
if [ -d "$NVIDIA_LIB_DIR" ]; then
    for libdir in "$NVIDIA_LIB_DIR"/*/lib; do
        [ -d "$libdir" ] && export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
    done
fi

echo "Python: $($PYTHON_BIN --version)"
harbor --help >/dev/null
export RAY_DEDUP_LOGS=0

# Set CUDA_HOME for flashinfer JIT compilation (needs nvcc)
_CUDA_HOME="${EVAL_CUDA_HOME:-/e/software/default/stages/2026/software/CUDA/13}"
if [ -n "${_CUDA_HOME}" ] && [ -d "${_CUDA_HOME}" ]; then
    export CUDA_HOME="${_CUDA_HOME}"
    export PATH="$CUDA_HOME/bin:$PATH"
fi


# Get all nodes
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
ALL_NODES_ARRAY=($ALL_NODES)
echo "All nodes: ${ALL_NODES_ARRAY[@]}"

# Env vars to export to srun workers
export SRUN_EXPORT_ENV="ALL,LD_LIBRARY_PATH=$LD_LIBRARY_PATH,PATH=$PATH,HF_TOKEN=$HF_TOKEN"
RAY_ENV_VARS="LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH HF_TOKEN=$HF_TOKEN"

# ==============================================================================
# SSH Tunnel + Proxychains Setup (from head node = first node)
# ==============================================================================

NODE_HOST=$(hostname -s)
TUNNEL_PORT=7003
LOGIN_NODE="${EVAL_LOGIN_NODE:-jpbl-s01-02}"
PROXYCHAINS_BIN="${EVAL_PROXYCHAINS_BIN:-/e/scratch/jureap59/feuer1/proxychains-ng-aarch64/bin/proxychains4}"
TUNNEL_PID=""

setup_proxy() {
    echo "[proxy] Setting up SSH tunnel to $LOGIN_NODE..."
    if [ -z "${SSH_KEY:-}" ]; then
        echo "[proxy] WARNING: SSH_KEY not set, skipping tunnel setup"
        return 1
    fi
    NODE_IP=$(nslookup "$NODE_HOST" 2>/dev/null | grep 'Address' | tail -n1 | awk '{print $2}')
    NODE_IP="${NODE_IP:-127.0.0.1}"
    ssh -g -f -N -D ${TUNNEL_PORT} \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=1000 \
        -o ServerAliveInterval=10 \
        -o ServerAliveCountMax=30 \
        -o TCPKeepAlive=yes \
        -o ExitOnForwardFailure=yes \
        -o BatchMode=yes \
        -i "${SSH_KEY}" \
        "${USER}@${LOGIN_NODE}"
    sleep 5
    if pgrep -f "ssh.*-D.*${TUNNEL_PORT}" > /dev/null; then
        echo "[proxy] SSH tunnel started successfully"
        TUNNEL_PID=$(pgrep -f "ssh.*-D.*${TUNNEL_PORT}" | head -1)
    else
        echo "[proxy] ERROR: SSH tunnel failed to start"
        return 1
    fi
    CFG_PATH="$HOME/.proxychains/proxychains_${SLURM_JOB_ID}.conf"
    mkdir -p "$HOME/.proxychains"
    cat > "$CFG_PATH" <<PCEOF
strict_chain
quiet_mode
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
localnet 169.254.0.0/255.255.0.0
[ProxyList]
socks5 ${NODE_IP} ${TUNNEL_PORT}
PCEOF
    export PROXYCHAINS_CONF_FILE="$CFG_PATH"
    echo "[proxy] Config at $CFG_PATH"

    export DAYTONA_MAX_RETRIES=5
    export DAYTONA_RETRY_DELAY=30
    export DAYTONA_BACKOFF_FACTOR=2
    export DAYTONA_TIMEOUT=1800
    export AIOHTTP_CLIENT_TIMEOUT=900
    export AIOHTTP_CONNECTOR_TIMEOUT=900
    export AIOHTTP_SOCK_CONNECT_TIMEOUT=300
    export AIOHTTP_TOTAL_TIMEOUT=1800
    export PYTHONHTTPSVERIFY=0
    unset SSL_CERT_FILE CURL_CA_BUNDLE REQUESTS_CA_BUNDLE SSL_CERT_DIR

    if [ -x "$PROXYCHAINS_BIN" ]; then
        "$PROXYCHAINS_BIN" -f "$CFG_PATH" curl -s --connect-timeout 10 https://huggingface.co -o /dev/null 2>/dev/null \
            && echo "[proxy] HuggingFace connectivity OK" \
            || echo "[proxy] WARNING: HuggingFace connectivity test failed"
    fi

    if [ -n "${DAYTONA_API_KEY:-}" ]; then
        echo "[daytona] Testing API connectivity..."
        DAYTONA_HTTP_CODE=$("$PROXYCHAINS_BIN" -f "$CFG_PATH" curl -s --connect-timeout 15 \
            -H "Authorization: Bearer $DAYTONA_API_KEY" \
            "${DAYTONA_API_URL:-https://app.daytona.io/api}/health" \
            -o /dev/null -w "%{http_code}" 2>/dev/null) || DAYTONA_HTTP_CODE="FAIL"
        echo "[daytona] Health check: $DAYTONA_HTTP_CODE"
    fi
    echo "[proxy] Setup complete"
    return 0
}

proxied() {
    if [ -x "$PROXYCHAINS_BIN" ] && [ -n "${PROXYCHAINS_CONF_FILE:-}" ]; then
        "$PROXYCHAINS_BIN" -f "$PROXYCHAINS_CONF_FILE" "$@"
    else
        "$@"
    fi
}

if [ "${EVAL_PROXY_ENABLED:-true}" = "true" ]; then
    setup_proxy || echo "[proxy] Continuing without proxy"
else
    echo "[proxy] Proxy disabled (EVAL_PROXY_ENABLED=false), assuming direct internet access"
fi

# ==============================================================================
# Cleanup trap (DP: kill all vLLM + Ray + tunnel)
# ==============================================================================
cleanup() {
    echo "Cleaning up DP resources..."
    # Kill vLLM processes
    for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
        if [ -n "${VLLM_PIDS[$shard_idx]:-}" ]; then
            kill "${VLLM_PIDS[$shard_idx]}" 2>/dev/null || true
        fi
    done
    # Stop Ray on all nodes
    for node in "${ALL_NODES_ARRAY[@]}"; do
        srun --nodes=1 --ntasks=1 --overlap -w "$node" "$RAY_BIN" stop --force 2>/dev/null &
    done
    wait
    # Kill SSH tunnel
    if [ -n "${TUNNEL_PID:-}" ]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
    fi
    pkill -f "ssh.*-D.*${TUNNEL_PORT}" 2>/dev/null || true
    # Remove wrappers + proxychains config
    [ -n "${WRAPPER_DIR:-}" ] && rm -rf "$WRAPPER_DIR" 2>/dev/null || true
    rm -f "${PROXYCHAINS_CONF_FILE:-}" 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

# ==============================================================================
# Pre-download model (compute nodes have no internet)
# ==============================================================================
EVAL_LOGS_DIR="${EVAL_LOGS_DIR:-eval/logs}"
mkdir -p "$EVAL_LOGS_DIR"

echo "Pre-downloading model: $MODEL"
proxied $PYTHON_BIN -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ.get('HF_HUB_CACHE')
path = snapshot_download('$MODEL', cache_dir=cache)
print(f'Model cached at: {path}')
"
if [ $? -ne 0 ]; then
    echo "ERROR: Model pre-download failed. Exiting."
    exit 1
fi

# ==============================================================================
# Build vLLM command args (reuse build_vllm_cmd.sh)
# ==============================================================================
source "$DCFT/eval/build_vllm_cmd.sh"
build_vllm_cmd "$PYTHON_BIN" "$MODEL" "$GPU_MEMORY_UTIL"
# VLLM_CMD includes --host 0.0.0.0 --port 8000 by default.
# For DP, we strip --port and --host from VLLM_CMD and re-add per shard.
VLLM_CMD_BASE=()
skip_next=false
for arg in "${VLLM_CMD[@]}"; do
    if $skip_next; then
        skip_next=false
        continue
    fi
    if [ "$arg" = "--port" ] || [ "$arg" = "--host" ]; then
        skip_next=true
        continue
    fi
    VLLM_CMD_BASE+=("$arg")
done

echo "vLLM DP config: TP=$TP_SIZE, PP=$PP_SIZE, shards=$NUM_SHARDS, shards_per_node=$SHARDS_PER_NODE"

# ==============================================================================
# Start Ray Clusters + vLLM (supports multiple shards per node)
#
# Topology:
#   - Each node has GPUS_PER_NODE GPUs (4 on Jupiter GH200)
#   - Each shard uses TP_SIZE GPUs (via CUDA_VISIBLE_DEVICES)
#   - SHARDS_PER_NODE = GPUS_PER_NODE / TP_SIZE (e.g. 4/2=2)
#   - Each shard gets its own Ray cluster (separate port), vLLM (separate port)
#
# Example: 1 node, TP=2, 2 shards/node
#   Shard 0: GPUs 0,1, Ray port 6379, API port 8000
#   Shard 1: GPUs 2,3, Ray port 6479, API port 8001
#
# Example: 2 nodes, TP=2, 2 shards/node = 4 total shards
#   Node 0: Shard 0 (GPUs 0,1, port 8000), Shard 1 (GPUs 2,3, port 8001)
#   Node 1: Shard 2 (GPUs 0,1, port 8000), Shard 3 (GPUs 2,3, port 8001)
# ==============================================================================
echo ""
echo "=== Starting $NUM_SHARDS shards ($SHARDS_PER_NODE per node, TP=$TP_SIZE) ==="

declare -a VLLM_PIDS
declare -a SHARD_NODES    # which node each shard runs on
declare -a SHARD_GPU_IDS  # CUDA_VISIBLE_DEVICES for each shard
declare -a API_PORTS
declare -a RAY_PORTS
declare -a HEAD_IPS

BASE_RAY_PORT=6379
BASE_API_PORT=8000

# Compute shard → node mapping and GPU assignment
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    node_idx=$((shard_idx / SHARDS_PER_NODE))
    local_shard=$((shard_idx % SHARDS_PER_NODE))
    gpu_start=$((local_shard * TP_SIZE))

    # Build CUDA_VISIBLE_DEVICES string (e.g. "0,1" or "2,3")
    gpu_ids=""
    for ((g = gpu_start; g < gpu_start + TP_SIZE; g++)); do
        [ -n "$gpu_ids" ] && gpu_ids="${gpu_ids},"
        gpu_ids="${gpu_ids}${g}"
    done

    SHARD_NODES[$shard_idx]="${ALL_NODES_ARRAY[$node_idx]}"
    SHARD_GPU_IDS[$shard_idx]="$gpu_ids"
    # Use different Ray ports per shard (even on same node)
    RAY_PORTS[$shard_idx]=$((BASE_RAY_PORT + shard_idx * 100))
    # API ports: offset by local_shard so shards on same node don't collide
    API_PORTS[$shard_idx]=$((BASE_API_PORT + local_shard))

    echo "  Shard $shard_idx: node=${SHARD_NODES[$shard_idx]}, GPUs=${gpu_ids}, Ray port=${RAY_PORTS[$shard_idx]}, API port=${API_PORTS[$shard_idx]}"
done

# --- Step 3a: Clean up existing Ray processes on all nodes ---
echo ""
echo "Cleaning up existing Ray processes..."
for node in "${ALL_NODES_ARRAY[@]}"; do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" bash -c \
        "pkill -9 -u $USER -f 'ray::' 2>/dev/null; $RAY_BIN stop --force 2>/dev/null; rm -rf /tmp/ray_${USER}_shard* 2>/dev/null" || true &
done
wait
sleep 2

# --- Step 3b: Start Ray clusters (one per shard) ---
echo ""
echo "Starting Ray clusters..."

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    node="${SHARD_NODES[$shard_idx]}"
    gpu_ids="${SHARD_GPU_IDS[$shard_idx]}"
    ray_port="${RAY_PORTS[$shard_idx]}"
    ray_temp_dir="/tmp/ray_${USER}_shard${shard_idx}"

    # Get node IP via InfiniBand
    head_ip=""
    if srun --nodes=1 --ntasks=1 --overlap -w "$node" ip -o -4 addr show ib0 >/dev/null 2>&1; then
        head_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$node" ip -o -4 addr show ib0 | awk '{print $4}' | cut -d/ -f1)
    else
        head_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$node" hostname --ip-address)
        head_ip=${head_ip%% *}
    fi
    HEAD_IPS[$shard_idx]="$head_ip"

    echo "  Shard $shard_idx: Starting Ray head on $node ($head_ip:$ray_port), GPUs=$gpu_ids"

    # Start Ray head with only this shard's GPUs visible
    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$node" bash -c \
        "env $RAY_ENV_VARS CUDA_VISIBLE_DEVICES=$gpu_ids \
         $RAY_BIN start --head --node-ip-address=${head_ip} --port=${ray_port} \
         --num-gpus=${TP_SIZE} --num-cpus=$((72 / SHARDS_PER_NODE)) \
         --temp-dir=${ray_temp_dir}" &
    sleep 3
done

echo "Waiting for Ray clusters to stabilize..."
sleep 15

# --- Step 3c: Start vLLM on each shard ---
echo ""
echo "=== Starting $NUM_SHARDS vLLM Servers ==="

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    node="${SHARD_NODES[$shard_idx]}"
    gpu_ids="${SHARD_GPU_IDS[$shard_idx]}"
    head_ip="${HEAD_IPS[$shard_idx]}"
    ray_port="${RAY_PORTS[$shard_idx]}"
    api_port="${API_PORTS[$shard_idx]}"
    ray_address="${head_ip}:${ray_port}"
    vllm_log="$EVAL_LOGS_DIR/vllm_dp_shard${shard_idx}_${SLURM_JOB_ID}.log"

    echo "  Shard $shard_idx: vLLM on $node, GPUs=$gpu_ids, port=$api_port"

    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$node" \
        env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
            PATH="$PATH" \
            PYTHONPATH="$PYTHONPATH" \
            HF_HOME="$HF_HOME" \
            HF_HUB_CACHE="$HF_HUB_CACHE" \
            HF_HUB_OFFLINE="1" \
            HF_TOKEN="$HF_TOKEN" \
            CUDA_VISIBLE_DEVICES="$gpu_ids" \
            RAY_ADDRESS="$ray_address" \
            VLLM_USE_V1="$VLLM_USE_V1" \
            VLLM_ALL2ALL_BACKEND="pplx" \
            TORCHDYNAMO_DISABLE="1" \
            TMPDIR="/tmp" \
            TRITON_CACHE_DIR="/tmp/triton_cache_${USER}" \
            TORCH_COMPILE_CACHE_DIR="/tmp/torch_cache_${USER}" \
            TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}" \
        "${VLLM_CMD_BASE[@]}" \
            --host "0.0.0.0" \
            --port "$api_port" \
            --pipeline-parallel-size "$PP_SIZE" \
        >> "$vllm_log" 2>&1 &

    VLLM_PIDS[$shard_idx]=$!
    echo "    PID: ${VLLM_PIDS[$shard_idx]}, Log: $vllm_log"
done

# --- Step 3d: Health check all shards ---
echo ""
echo "Waiting for all vLLM servers to become healthy..."

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    node="${SHARD_NODES[$shard_idx]}"
    api_port="${API_PORTS[$shard_idx]}"
    health_url="http://127.0.0.1:${api_port}/v1/models"

    echo "  Checking shard $shard_idx (port $api_port on $node)..."
    for i in $(seq 1 $VLLM_MAX_RETRIES); do
        if srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$node" \
            curl -s "$health_url" > /dev/null 2>&1; then
            echo "    Shard $shard_idx is healthy!"
            break
        fi
        if [ "$i" -eq "$VLLM_MAX_RETRIES" ]; then
            echo "ERROR: Shard $shard_idx failed health check after $VLLM_MAX_RETRIES attempts"
            tail -50 "$EVAL_LOGS_DIR/vllm_dp_shard${shard_idx}_${SLURM_JOB_ID}.log" 2>/dev/null || true
            exit 1
        fi
        sleep 15
    done
done

echo "All vLLM servers are ready!"

# ==============================================================================
# Download and shard dataset
# ==============================================================================
echo ""
echo "=== Downloading and sharding dataset ==="

# Download/locate dataset (same logic as single-node sbatch)
if [[ "$REPO_ID" == /* ]]; then
    echo "Using local dataset path: $REPO_ID"
    DATASET_PATH="$REPO_ID"
    if [ ! -d "$DATASET_PATH" ]; then
        echo "ERROR: Local dataset path does not exist: $DATASET_PATH"
        exit 1
    fi
else
    echo "Downloading/locating dataset: $REPO_ID"
    # Check dataset directories (from EVAL_DATASETS_DIRS or fallback)
    _DS_DIRS="${EVAL_DATASETS_DIRS:-/e/data1/datasets/playground/ot/datasets:/e/scratch/jureap59/${USER}/datasets}"
    DATASET_LOCAL_DIR=""
    IFS=':' read -ra _DS_DIRS_ARR <<< "$_DS_DIRS"
    for _ds_dir in "${_DS_DIRS_ARR[@]}"; do
        if [ -d "${_ds_dir}/${SAFE_REPO}" ]; then
            DATASET_LOCAL_DIR="${_ds_dir}/${SAFE_REPO}"
            break
        fi
    done
    DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-${_DS_DIRS_ARR[0]}/${SAFE_REPO}}"
    DOWNLOAD_LOG=$(mktemp /tmp/download_XXXXXX.log)
    proxied $PYTHON_BIN "$DCFT/eval/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_LOCAL_DIR" > "$DOWNLOAD_LOG" 2>&1
    DOWNLOAD_EXIT=$?
    cat "$DOWNLOAD_LOG"
    DATASET_PATH=$(grep DATASET_PATH "$DOWNLOAD_LOG" | tail -n 1 | cut -d'=' -f2)
    rm -f "$DOWNLOAD_LOG"
    if [ $DOWNLOAD_EXIT -ne 0 ] || [ -z "${DATASET_PATH:-}" ]; then
        echo "ERROR: Failed to get dataset path (exit code: $DOWNLOAD_EXIT)"
        exit 1
    fi
fi
echo "Dataset path: $DATASET_PATH"

# Shard dataset into N parts (simple round-robin symlink)
SHARD_DIR="/tmp/eval_shards_${SLURM_JOB_ID}"
mkdir -p "$SHARD_DIR"

echo "Sharding dataset into $NUM_SHARDS parts..."
$PYTHON_BIN -c "
import os, sys
from pathlib import Path

dataset_path = Path('$DATASET_PATH')
num_shards = $NUM_SHARDS
shard_dir = Path('$SHARD_DIR')

task_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
print(f'Found {len(task_dirs)} tasks to shard into {num_shards} parts')

for shard_idx in range(num_shards):
    shard_path = shard_dir / f'shard_{shard_idx}'
    shard_path.mkdir(exist_ok=True)
    shard_tasks = task_dirs[shard_idx::num_shards]
    print(f'Shard {shard_idx}: {len(shard_tasks)} tasks')
    for task_dir in shard_tasks:
        dest = shard_path / task_dir.name
        if not dest.exists():
            os.symlink(task_dir, dest)
"
if [ $? -ne 0 ]; then
    echo "ERROR: Dataset sharding failed"
    exit 1
fi

TASK_COUNT=$(ls -d "$DATASET_PATH"/*/instruction.md 2>/dev/null | wc -l)
echo "Total tasks: $TASK_COUNT, shards: $NUM_SHARDS (~$((TASK_COUNT / NUM_SHARDS)) tasks/shard)"

# ==============================================================================
# Construct run tag and directory
# ==============================================================================
if [ -n "$RUN_TAG_ARG" ]; then
    RUN_TAG="$RUN_TAG_ARG"
else
    RUN_TAG="${SAFE_REPO}_${SAFE_MODEL}"
fi
EVAL_JOBS_DIR="${EVAL_JOBS_DIR:-/e/data1/datasets/playground/mmlaion/shared/zhuang1_eval_jobs}"  # fallback; listener sets EVAL_JOBS_DIR from cluster config

echo "Run tag: $RUN_TAG"

# Benchmark name (same logic as single-node)
declare -A BENCHMARK_NAME_MAP=(
    ["DCAgent2_aider_polyglot"]="aider_polyglot"
    ["DCAgent_dev_set_v2"]="dev_set_v2"
    ["DCAgent_dev_set_71_tasks"]="dev_set_71_tasks"
    ["DCAgent2_terminal_bench_2"]="terminal_bench_2"
    ["DCAgent_swebench_verified_eval_set"]="swebench-verified-random-100-folders"
    ["DCAgent2_bfcl-parity"]="bfcl-parity"
)
BASE_NAME="${BENCHMARK_NAME_MAP[$SAFE_REPO]:-$SAFE_REPO}"
BENCHMARK_SUFFIX=""
if [ -n "${EVAL_OVERRIDE_MEMORY_MB:-}" ] && [ "$EVAL_OVERRIDE_MEMORY_MB" != "1024" ]; then
    mem_gb=$(( EVAL_OVERRIDE_MEMORY_MB / 1024 ))
    BENCHMARK_SUFFIX="${BENCHMARK_SUFFIX}_${mem_gb}gb"
fi
if [ "$TIMEOUT_MULTIPLIER" != "1" ] && [ "$TIMEOUT_MULTIPLIER" != "1.0" ]; then
    BENCHMARK_SUFFIX="${BENCHMARK_SUFFIX}_${TIMEOUT_MULTIPLIER}x"
fi
BENCHMARK_NAME="${BASE_NAME}${BENCHMARK_SUFFIX}"
export BENCHMARK_NAME MODEL REPO_ID SLURM_JOB_ID
echo "Benchmark name: $BENCHMARK_NAME"

# ==============================================================================
# Update DB: Pending → Started
# ==============================================================================
echo "Creating/updating DB job entry..."
HARBOR_VERSION=$($PYTHON_BIN -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION RUN_TAG

if [ -n "$DB_JOB_ID" ]; then
    echo "Updating Pending job $DB_JOB_ID → Started"
    proxied $PYTHON_BIN - <<'PY' || true
import os, sys
from database.unified_db.utils import update_job_status_to_started
run_tag = os.environ["RUN_TAG"]
agent_name = os.environ.get("EVAL_AGENT_NAME", "terminus-2")
timeout_multiplier = float(os.environ.get("EVAL_TIMEOUT_MULTIPLIER", "1.0"))
harbor_version = os.environ.get("HARBOR_VERSION", "unknown")
config = {"agent": agent_name, "env": "daytona", "timeout_multiplier": timeout_multiplier}
result = update_job_status_to_started(
    job_name=run_tag,
    n_trials=int(os.environ.get("EVAL_N_CONCURRENT", "128")),
    n_rep_eval=int(os.environ.get("EVAL_N_ATTEMPTS", "3")),
    config=config,
    harbor_package_version=harbor_version,
)
if not result.get("success"):
    print(f"WARNING: Pending→Started update failed: {result.get('error')}", file=sys.stderr)
else:
    print(f"DB job {run_tag} → Started")
PY
fi

# ==============================================================================
# Resolve harbor config
# ==============================================================================
if [[ "$CONFIG_YAML" != /* ]]; then
    if [ -f "$DCFT/eval/${CLUSTER_NAME}/$CONFIG_YAML" ]; then
        HARBOR_CONFIG="$DCFT/eval/${CLUSTER_NAME}/$CONFIG_YAML"
    elif [ -f "$DCFT/eval/MBZ/$CONFIG_YAML" ]; then
        HARBOR_CONFIG="$DCFT/eval/MBZ/$CONFIG_YAML"
    else
        HARBOR_CONFIG="$DCFT/eval/jupiter/$CONFIG_YAML"
    fi
else
    HARBOR_CONFIG="$CONFIG_YAML"
fi

# Build extra harbor args
EXTRA_HARBOR_ARGS=""
if [ -n "${EVAL_SNAPSHOT_NAME:-}" ]; then
    EXTRA_HARBOR_ARGS="--environment-kwarg snapshot_template_name=$EVAL_SNAPSHOT_NAME --no-force-build"
fi
if [ "$TIMEOUT_MULTIPLIER" != "1" ] && [ "$TIMEOUT_MULTIPLIER" != "1.0" ]; then
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --timeout-multiplier $TIMEOUT_MULTIPLIER"
fi
if [ -n "${EVAL_OVERRIDE_MEMORY_MB:-}" ]; then
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --override-memory-mb $EVAL_OVERRIDE_MEMORY_MB"
fi
if [ -n "${EVAL_AUTO_SNAPSHOT:-}" ]; then
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --environment-kwarg auto_snapshot=$EVAL_AUTO_SNAPSHOT"
fi

# ==============================================================================
# Run Harbor Jobs in Parallel (one per shard)
# ==============================================================================
echo ""
echo "=== Running $NUM_SHARDS Harbor Jobs in Parallel ==="
set +e

declare -a HARBOR_PIDS
declare -a HARBOR_LOGS
declare -a SHARD_RUN_DIRS

run_harbor_for_shard() {
    local shard_idx=$1
    local head_node="${SHARD_NODES[$shard_idx]}"
    local api_port=${API_PORTS[$shard_idx]}
    local shard_dataset="$SHARD_DIR/shard_${shard_idx}"
    local shard_run_tag="${RUN_TAG}_shard${shard_idx}"
    local harbor_log="$EVAL_LOGS_DIR/harbor_dp_shard${shard_idx}_${SLURM_JOB_ID}.log"
    local shard_run_dir="${EVAL_JOBS_DIR}/${shard_run_tag}"

    HARBOR_LOGS[$shard_idx]="$harbor_log"
    SHARD_RUN_DIRS[$shard_idx]="$shard_run_dir"

    echo "  Shard $shard_idx: dataset=$shard_dataset, port=$api_port, run_tag=$shard_run_tag"

    # Check for resume (existing job dir for this shard)
    local existing_job_dir="$shard_run_dir"
    local harbor_cmd

    if [ -d "$existing_job_dir" ] && [ -f "$existing_job_dir/config.json" ]; then
        echo "    Resuming existing job at $existing_job_dir"
        harbor_cmd="harbor jobs resume \
            --job-path \"$existing_job_dir\" \
            --filter-error-type EnvironmentStartTimeoutError \
            --filter-error-type DaytonaError \
            --filter-error-type DaytonaRateLimitError \
            --filter-error-type CancelledError \
            --filter-error-type AgentEnvironmentTimeoutError \
            --filter-error-type SandboxBuildFailedError"
    else
        echo "    Starting new job"
        harbor_cmd="harbor jobs start \
            -p \"$shard_dataset\" \
            --jobs-dir \"$EVAL_JOBS_DIR\" \
            --n-concurrent \"$N_CONCURRENT\" \
            --agent \"$AGENT_NAME\" \
            --model \"hosted_vllm/$MODEL\" \
            --env \"daytona\" \
            --agent-kwarg \"api_base=http://localhost:${api_port}/v1\" \
            --agent-kwarg \"key=fake_key\" \
            --agent-kwarg \"max_tokens=16384\" \
            --agent-kwarg 'model_info={\"max_output_tokens\":16384,\"max_input_tokens\":32768,\"input_cost_per_token\":0.0,\"output_cost_per_token\":0.0}' \
            --n-attempts \"$N_ATTEMPTS\" \
            --job-name \"$shard_run_tag\" \
            --export-traces \
            --config \"$HARBOR_CONFIG\" \
            --debug"

        if [ "$ENABLE_THINKING" = "true" ]; then
            harbor_cmd="$harbor_cmd --agent-kwarg \"enable_thinking=true\""
        fi
        if [ -n "$AGENT_PARSER" ]; then
            harbor_cmd="$harbor_cmd --agent-kwarg \"parser=$AGENT_PARSER\""
        fi
        if [ -n "$EXTRA_HARBOR_ARGS" ]; then
            harbor_cmd="$harbor_cmd $EXTRA_HARBOR_ARGS"
        fi
    fi

    # Run harbor on the shard's head node (via proxychains for Daytona)
    if [ -x "$PROXYCHAINS_BIN" ] && [ -n "${PROXYCHAINS_CONF_FILE:-}" ]; then
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
            "$PROXYCHAINS_BIN" -f "$PROXYCHAINS_CONF_FILE" \
            bash -c "$harbor_cmd" \
            >> "$harbor_log" 2>&1 &
    else
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
            bash -c "$harbor_cmd" \
            >> "$harbor_log" 2>&1 &
    fi

    HARBOR_PIDS[$shard_idx]=$!
    echo "    PID: ${HARBOR_PIDS[$shard_idx]}, Log: $harbor_log"
}

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    run_harbor_for_shard $shard_idx
done

echo ""
echo "All Harbor jobs started. Waiting for completion..."

FAILED=0
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    wait ${HARBOR_PIDS[$shard_idx]} || FAILED=$((FAILED + 1))
    echo "  Shard $shard_idx completed (exit: $?)"
done

set -e

# ==============================================================================
# Save meta.env (per shard)
# ==============================================================================
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_run_dir="${SHARD_RUN_DIRS[$shard_idx]}"
    mkdir -p "$shard_run_dir"
    {
        echo "MODEL=$MODEL"
        echo "REPO_ID=$REPO_ID"
        echo "TIMESTAMP=$TIMESTAMP"
        echo "SLURM_JOB_ID=$SLURM_JOB_ID"
        echo "DB_JOB_ID=${DB_JOB_ID:-}"
        echo "BENCHMARK_ID=${BENCHMARK_ID:-}"
        echo "N_CONCURRENT=$N_CONCURRENT"
        echo "N_ATTEMPTS=$N_ATTEMPTS"
        echo "AGENT_NAME=$AGENT_NAME"
        echo "ENABLE_THINKING=$ENABLE_THINKING"
        echo "TIMEOUT_MULTIPLIER=$TIMEOUT_MULTIPLIER"
        echo "BENCHMARK_NAME=$BENCHMARK_NAME"
        echo "NUM_SHARDS=$NUM_SHARDS"
        echo "SHARD_IDX=$shard_idx"
        echo "DP_MODE=true"
    } > "$shard_run_dir/meta.env"
done

# ==============================================================================
# Error check per shard + aggregate
# ==============================================================================
echo ""
echo "=== Checking errors across all shards ==="

TOTAL_INVALID=0
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_run_dir="${SHARD_RUN_DIRS[$shard_idx]}"
    result_file="$shard_run_dir/result.json"
    if [ -f "$result_file" ]; then
        SHARD_INVALID=$($PYTHON_BIN -c "
import json
try:
    with open('$result_file') as f:
        data = json.load(f)
    BENIGN = {'AgentTimeoutError','ContextLengthExceededError','SummarizationTimeout','SummarizationTimeoutError'}
    total = 0
    if 'stats' in data and 'evals' in data['stats']:
        for ek, ev in data['stats']['evals'].items():
            if 'exception_stats' in ev:
                for et, ids in ev['exception_stats'].items():
                    if et not in BENIGN and isinstance(ids, list):
                        total += len(ids)
    print(total)
except: print('0')
" 2>&1 | tail -1)
        echo "  Shard $shard_idx: $SHARD_INVALID invalid errors"
        TOTAL_INVALID=$((TOTAL_INVALID + SHARD_INVALID))
    else
        echo "  Shard $shard_idx: no result.json"
    fi
done

echo "Total invalid errors across all shards: $TOTAL_INVALID (threshold: $ERROR_THRESHOLD)"

if [ "$TOTAL_INVALID" -gt "$ERROR_THRESHOLD" ]; then
    echo "Too many invalid errors ($TOTAL_INVALID > $ERROR_THRESHOLD), skipping upload"
    exit 0
fi

# ==============================================================================
# Upload results (merge all shard dirs)
# ==============================================================================
echo ""
echo "=== Uploading results ==="

# Collect all shard job dirs
JOB_DIRS=""
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_run_dir="${SHARD_RUN_DIRS[$shard_idx]}"
    if [ -d "$shard_run_dir" ]; then
        JOB_DIRS="$JOB_DIRS $shard_run_dir"
    fi
done

echo "Uploading from shard dirs: $JOB_DIRS"

# Use upload_traces.py to merge and upload all shard results
export PYTHONPATH="${DCFT}:${DCFT}/data/sbatches/teacher_scripts:${PYTHONPATH:-}"
proxied $PYTHON_BIN -c "
import os, sys, re, hashlib

from database.unified_db.utils import upload_eval_results


def sanitize_hf_repo_id(repo_id, max_length=96):
    def collapse(s):
        prev = None
        while s != prev:
            prev = s
            s = s.replace('--', '-').replace('..', '.')
        return s
    org, name = repo_id.split('/', 1) if '/' in repo_id else (None, repo_id)
    name = re.sub(r'[^A-Za-z0-9._-]', '-', name)
    name = collapse(name).strip('-.')
    if not name: name = 'repo'
    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest))
        base = name[:keep].rstrip('-.')
        if not base: base = 'r'
        name = f'{base}{digest}'
        name = collapse(name).strip('-.')
    if name[0] in '-.': name = 'r' + name[1:]
    if name[-1] in '-.': name = name[:-1] + '0'
    return f'{org}/{name}' if org else name


# Upload each shard separately
shard_dirs = '$JOB_DIRS'.split()
run_tag = os.environ['RUN_TAG']
username = os.environ.get('UPLOAD_USERNAME', os.environ.get('USER', 'jupiter'))
hf_token = os.environ['HF_TOKEN']
benchmark_name = os.environ.get('BENCHMARK_NAME', '')

for shard_dir in shard_dirs:
    shard_name = os.path.basename(shard_dir)
    hf_repo_id = sanitize_hf_repo_id(f'DCAgent2/{shard_name}')
    print(f'[uploader] Uploading shard {shard_name} to {hf_repo_id}')

    benchmark_version_hash = hashlib.sha256(benchmark_name.encode()).hexdigest()

    try:
        upload_eval_results(
            shard_dir,
            username=username,
            error_mode='skip_on_error',
            hf_token=hf_token,
            hf_repo_id=hf_repo_id,
            register_benchmark=True,
            benchmark_name=benchmark_name,
            benchmark_version_hash=benchmark_version_hash,
        )
        print(f'[uploader] {shard_name} done.')
    except Exception as e:
        print(f'[uploader] WARNING: {shard_name} upload failed: {e}')
" 2>&1
UPLOAD_EXIT=$?

# Clean up shard directory
rm -rf "$SHARD_DIR" 2>/dev/null || true

echo "=============================================="
if [ $FAILED -eq 0 ] && [ $UPLOAD_EXIT -eq 0 ]; then
    echo "DP Eval COMPLETED ($NUM_SHARDS shards)"
else
    echo "DP Eval completed with issues (failed shards: $FAILED, upload exit: $UPLOAD_EXIT)"
fi
echo "=============================================="
