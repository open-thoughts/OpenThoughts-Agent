#!/bin/bash
#SBATCH -p gh
#SBATCH --time=02:00:00
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --account CCR24067
#SBATCH --output=/scratch/10000/eguha3/ot-agent/data/vllm_experiments/logs/vllm_e714e6cf_%j.out
#SBATCH --job-name=vllm_ep_pplx

set -eo pipefail

EXPERIMENTS_DIR="/scratch/10000/eguha3/ot-agent/data/vllm_experiments"
CONFIG_FILE="/scratch/10000/eguha3/ot-agent/data/vllm_experiments/configs/focused/config_11_ep_pplx_4c5137a9.json"
CONFIG_HASH="e714e6cf"
CONFIG_NAME="ep_pplx"

echo "============================================"
echo "vLLM Experiment: $CONFIG_NAME ($CONFIG_HASH)"
echo "============================================"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "============================================"

# Create results directory
mkdir -p "$EXPERIMENTS_DIR/results/$CONFIG_HASH"

# Step 1: Start Ray cluster
echo ""
echo "=== Step 1: Starting Ray Cluster ==="
source "$EXPERIMENTS_DIR/start_ray_cluster.sh"

# Verify Ray cluster
if [ -z "$RAY_ADDRESS" ]; then
    echo "ERROR: Ray cluster failed to start"
    exit 1
fi

echo "Ray cluster ready: $RAY_ADDRESS"

# Step 2: Set vLLM configuration from JSON
echo ""
echo "=== Step 2: Configuring vLLM ==="

# Parse config file and set ALL environment variables for dp_debug.py
export VLLM_MODEL_PATH=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('model_path', 'zai-org/GLM-4.6'))")
export VLLM_GPU_MEMORY_UTILIZATION=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('gpu_memory_utilization', 0.95))")
export VLLM_MAX_NUM_SEQS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_num_seqs', 256))")
export VLLM_MAX_MODEL_LEN=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_model_len', 32768))")
export VLLM_MAX_NUM_BATCHED_TOKENS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_num_batched_tokens', 16384))")
export VLLM_SWAP_SPACE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('swap_space_gb', 20))")
export VLLM_BLOCK_SIZE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('block_size', 16))")
export VLLM_TENSOR_PARALLEL_SIZE=8  # Fixed at 8 for tensor parallelism across 8 GPUs

# Boolean settings
export VLLM_ENABLE_EXPERT_PARALLEL=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_expert_parallel', True) else 'false')")
export VLLM_ENABLE_CHUNKED_PREFILL=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_chunked_prefill', True) else 'false')")
export VLLM_ENABLE_PREFIX_CACHING=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_prefix_caching', False) else 'false')")
export VLLM_ENABLE_EPLB=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_eplb', False) else 'false')")

# KV cache dtype
export VLLM_KV_CACHE_DTYPE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('kv_cache_dtype', 'auto'))")

# Expert parallel settings
export VLLM_ALL2ALL_BACKEND=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('all2all_backend', 'pplx'))")
export VLLM_EPLB_NUM_REDUNDANT_EXPERTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('eplb_num_redundant_experts', 32))")

echo "Configuration:"
echo "  Model: $VLLM_MODEL_PATH"
echo "  Tensor Parallel Size: $VLLM_TENSOR_PARALLEL_SIZE"
echo "  GPU Memory Util: $VLLM_GPU_MEMORY_UTILIZATION"
echo "  Max Model Len: $VLLM_MAX_MODEL_LEN"
echo "  Max Num Seqs: $VLLM_MAX_NUM_SEQS"
echo "  Max Batched Tokens: $VLLM_MAX_NUM_BATCHED_TOKENS"
echo "  Enable Expert Parallel: $VLLM_ENABLE_EXPERT_PARALLEL"
echo "  KV Cache Dtype: $VLLM_KV_CACHE_DTYPE"
echo "  Enable Prefix Caching: $VLLM_ENABLE_PREFIX_CACHING"
echo "  Enable Chunked Prefill: $VLLM_ENABLE_CHUNKED_PREFILL"
echo "  All2All Backend: $VLLM_ALL2ALL_BACKEND"
echo "  Enable EPLB: $VLLM_ENABLE_EPLB"

# Step 3: Start vLLM via Ray Serve
echo ""
echo "=== Step 3: Starting vLLM Ray Serve ==="
source "$EXPERIMENTS_DIR/run_vllm_dp.sh"

# Verify vLLM is running
if [ -z "$VLLM_ENDPOINT_URL" ]; then
    echo "ERROR: vLLM failed to start"
    exit 1
fi

echo "vLLM ready at: $VLLM_ENDPOINT_URL"

# Step 4: Run benchmark from head node (server binds to localhost)
echo ""
echo "=== Step 4: Running Benchmark ==="

# Get head node
HEAD_NODE=$(echo "$RAY_NODES_ARRAY" | awk '{print $1}')

# Run benchmark from head node so it can access http://127.0.0.1:8000
srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" \
    env TRITON_CC="$TRITON_CC" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        PATH="$PATH" \
        LD_PRELOAD="$LD_PRELOAD" \
    /scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3 "$EXPERIMENTS_DIR/benchmark_runner.py" \
        --server-url "http://127.0.0.1:8000" \
        --dataset "$EXPERIMENTS_DIR/datasets/large_throughput.jsonl" \
        --config-hash "$CONFIG_HASH" \
        --request-rate 10.0 \
        --output-dir "$EXPERIMENTS_DIR/results/$CONFIG_HASH"

BENCHMARK_EXIT_CODE=$?

# Step 5: Cleanup
echo ""
echo "=== Step 5: Cleanup ==="

# Stop vLLM server
if [ -n "$VLLM_PID" ]; then
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
fi

# Stop Ray on all nodes
RAY_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/ray"
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null &
done
wait

echo ""
echo "============================================"
if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo "Experiment COMPLETED: $CONFIG_NAME ($CONFIG_HASH)"
else
    echo "Experiment FAILED: $CONFIG_NAME ($CONFIG_HASH)"
fi
echo "Results: $EXPERIMENTS_DIR/results/$CONFIG_HASH"
echo "============================================"

exit $BENCHMARK_EXIT_CODE
