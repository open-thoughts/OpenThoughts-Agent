#!/bin/bash
# Run vLLM via Ray Serve using dp_debug.py
# Usage: source run_vllm_dp.sh
# Must be run after sourcing start_ray_cluster.sh
#
# Environment variables for configuration (set before sourcing):
#   VLLM_MODEL_PATH, VLLM_GPU_MEMORY_UTILIZATION, VLLM_MAX_NUM_SEQS,
#   VLLM_MAX_MODEL_LEN, VLLM_MAX_NUM_BATCHED_TOKENS, VLLM_TENSOR_PARALLEL_SIZE,
#   VLLM_ENABLE_EXPERT_PARALLEL, VLLM_KV_CACHE_DTYPE, VLLM_ENABLE_PREFIX_CACHING,
#   VLLM_SWAP_SPACE, VLLM_BLOCK_SIZE, VLLM_ENABLE_CHUNKED_PREFILL,
#   VLLM_ALL2ALL_BACKEND, VLLM_ENABLE_EPLB, VLLM_EPLB_NUM_REDUNDANT_EXPERTS

set -e

# Check required environment variables from start_ray_cluster.sh
if [ -z "$RAY_ADDRESS" ]; then
    echo "ERROR: RAY_ADDRESS not set. Source start_ray_cluster.sh first."
    exit 1
fi

if [ -z "$HEAD_NODE_IP" ]; then
    echo "ERROR: HEAD_NODE_IP not set. Source start_ray_cluster.sh first."
    exit 1
fi

# Paths
PYTHON_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3"
EXPERIMENTS_DIR="${DCAGENT_DIR}/data/vllm_experiments"

# API port (Ray Serve default is 8000)
API_PORT="${API_PORT:-8000}"

echo "============================================"
echo "vLLM Ray Serve (dp_debug.py)"
echo "============================================"
echo "Ray Address: $RAY_ADDRESS"
echo "Head Node IP: $HEAD_NODE_IP"
echo "API Port: $API_PORT"
echo "============================================"

# Get head node for srun
HEAD_NODE=$(echo "$RAY_NODES_ARRAY" | awk '{print $1}')

# Create logs directory
mkdir -p "${EXPERIMENTS_DIR}/logs"
VLLM_LOG="${EXPERIMENTS_DIR}/logs/vllm_serve_${SLURM_JOB_ID}.log"

# Launch vLLM Ray Serve deployment on head node using dp_debug.py
# Pass all VLLM_* environment variables to configure the deployment
echo "Launching vLLM Ray Serve deployment via dp_debug.py..."
srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" \
    env TRITON_CC="$TRITON_CC" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        PATH="$PATH" \
        LD_PRELOAD="$LD_PRELOAD" \
        HF_HOME="/tmp/hf_home" \
        HF_TOKEN="$HF_TOKEN" \
        RAY_ADDRESS="$RAY_ADDRESS" \
        VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-QuantTrio/GLM-4.6-AWQ}" \
        VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.95}" \
        VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}" \
        VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}" \
        VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}" \
        VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}" \
        VLLM_PIPELINE_PARALLEL_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-1}" \
        VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-true}" \
        VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-auto}" \
        VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}" \
        VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-4}" \
        VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-16}" \
        VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-true}" \
        VLLM_ALL2ALL_BACKEND="${VLLM_ALL2ALL_BACKEND:-pplx}" \
        VLLM_ENABLE_EPLB="${VLLM_ENABLE_EPLB:-false}" \
        VLLM_EPLB_NUM_REDUNDANT_EXPERTS="${VLLM_EPLB_NUM_REDUNDANT_EXPERTS:-32}" \
    $PYTHON_BIN ${DCAGENT_DIR}/scripts/vllm/dp_debug.py \
    >> "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "vLLM Ray Serve launcher PID: $VLLM_PID (log: $VLLM_LOG)"

# Wait for server to be ready via health check (run from head node since server binds to 127.0.0.1)
echo "Waiting for vLLM server to become healthy..."
HEALTH_URL="http://127.0.0.1:${API_PORT}/v1/models"
for i in {1..180}; do
    # Run curl from head node since server binds to localhost
    if srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" curl -s "$HEALTH_URL" > /dev/null 2>&1; then
        echo "vLLM server is healthy!"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died. Last 50 lines of log:"
        tail -50 "$VLLM_LOG"
        exit 1
    fi
    if [ "$i" -eq 180 ]; then
        echo "ERROR: vLLM server health check failed after 180 attempts (30 min)"
        tail -50 "$VLLM_LOG"
        exit 1
    fi
    echo "Waiting for health check... ($i/180)"
    sleep 10
done

# Export endpoint info for benchmark (use localhost since benchmark runs from head node)
export VLLM_ENDPOINT_URL="http://127.0.0.1:${API_PORT}"
export VLLM_PID

echo "============================================"
echo "vLLM Ray Serve is ready!"
echo "Endpoint: $VLLM_ENDPOINT_URL"
echo "PID: $VLLM_PID"
echo "============================================"
