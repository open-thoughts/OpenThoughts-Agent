#!/bin/bash
#SBATCH -p gh
#SBATCH --time=24:00:00
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --account CCR24067
#SBATCH --output=${DCAGENT_DIR}/data/vllm_experiments/logs/harbor_batch_16384_seqs_256_113k_%j.out
#SBATCH --job-name=harbor_batch_16384_seqs_256_113k

set -eo pipefail

EXPERIMENTS_DIR="${DCAGENT_DIR}/data/vllm_experiments"
CONFIG_FILE="${DCAGENT_DIR}/data/vllm_experiments/configs/harbor_113k/config_04_batch_16384_seqs_256_d3ec4d4c_113k.json"
CONFIG_NAME="batch_16384_seqs_256_113k"

echo "============================================"
echo "vLLM + Harbor Experiment: $CONFIG_NAME"
echo "============================================"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "============================================"

mkdir -p "$EXPERIMENTS_DIR/results/$CONFIG_NAME"
mkdir -p "$EXPERIMENTS_DIR/logs"

# Environment Setup
echo ""
echo "=== Step 1: Environment Setup ==="

module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
# Increase Ray channel timeout for 113K context (default 300s is too short)
export RAY_CGRAPH_get_timeout=900
export VLLM_CACHE_ROOT=/scratch/10000/eguha3/vllm_cache
export VLLM_CONFIG_ROOT=/scratch/10000/eguha3/vllm_config
export TRITON_DUMP_DIR=/scratch/10000/eguha3/triton_dump_dir
export TRITON_OVERRIDE_DIR=/scratch/10000/eguha3/triton_override_dir
export TRITON_CACHE_DIR=/scratch/10000/eguha3/triton_cache_dir
export FLASHINFER_WORKSPACE_BASE=/scratch/08002/gsmyrnis/flashinfer_cache
export UV_CACHE_DIR=/scratch/10000/eguha3/uv_cache_dir
export HYDRA_FULL_ERROR=1
export HF_CACHE_DIR=/scratch/08134/negin/dc-agent-shared/.hf_cache
export HF_HUB_CACHE=$SCRATCH/hub

source /scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/secret.env

ln -sf /home1/apps/gcc/15.1.0/lib64/libstdc++.so.6 /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/10000/eguha3/vllm_sandboxes_backup/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5

harbor --help >/dev/null

# Start Ray Cluster
echo ""
echo "=== Step 2: Starting Ray Cluster ==="
source "$EXPERIMENTS_DIR/start_ray_cluster.sh"

if [ -z "$RAY_ADDRESS" ]; then
    echo "ERROR: Ray cluster failed to start"
    exit 1
fi

echo "Ray cluster ready: $RAY_ADDRESS"

# Configure vLLM
echo ""
echo "=== Step 3: Configuring vLLM ==="

export VLLM_MODEL_PATH=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('model_path', 'QuantTrio/GLM-4.6-AWQ'))")
export VLLM_GPU_MEMORY_UTILIZATION=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('gpu_memory_utilization', 0.92))")
export VLLM_MAX_NUM_SEQS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_num_seqs', 64))")
export VLLM_MAX_MODEL_LEN=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_model_len', 113000))")
export VLLM_MAX_NUM_BATCHED_TOKENS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_num_batched_tokens', 32768))")
export VLLM_SWAP_SPACE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('swap_space_gb', 4))")
export VLLM_BLOCK_SIZE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('block_size', 16))")
export VLLM_TENSOR_PARALLEL_SIZE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('tensor_parallel_size', 4))")
export VLLM_PIPELINE_PARALLEL_SIZE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('pipeline_parallel_size', 1))")

export VLLM_ENABLE_EXPERT_PARALLEL=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_expert_parallel', True) else 'false')")
export VLLM_ENABLE_CHUNKED_PREFILL=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_chunked_prefill', True) else 'false')")
export VLLM_ENABLE_PREFIX_CACHING=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_prefix_caching', True) else 'false')")
export VLLM_ENABLE_EPLB=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_eplb', False) else 'false')")
export VLLM_KV_CACHE_DTYPE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('kv_cache_dtype', 'auto'))")
export VLLM_ALL2ALL_BACKEND=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('all2all_backend', 'pplx'))")
export VLLM_EPLB_NUM_REDUNDANT_EXPERTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('eplb_num_redundant_experts', 32))")

echo "Configuration:"
echo "  Model: $VLLM_MODEL_PATH"
echo "  Tensor Parallel Size: $VLLM_TENSOR_PARALLEL_SIZE"
echo "  Pipeline Parallel Size: $VLLM_PIPELINE_PARALLEL_SIZE"
echo "  Max Model Len: $VLLM_MAX_MODEL_LEN"
echo "  Max Num Seqs: $VLLM_MAX_NUM_SEQS"
echo "  KV Cache Dtype: $VLLM_KV_CACHE_DTYPE"

# Start vLLM
echo ""
echo "=== Step 4: Starting vLLM Ray Serve ==="
source "$EXPERIMENTS_DIR/run_vllm_dp.sh"

if [ -z "$VLLM_ENDPOINT_URL" ]; then
    echo "ERROR: vLLM failed to start"
    exit 1
fi

echo "vLLM ready at: $VLLM_ENDPOINT_URL"

# Run Harbor
echo ""
echo "=== Step 5: Running Harbor Evaluation ==="

# Direct dataset and config paths
DATASET_PATH="/scratch/10000/eguha3/hf_cache/datasets--DCAgent--dev_set_71_tasks/snapshots/da9bcc03f95a264602be912dcdc6db9371a3e4e3"
HARBOR_CONFIG="${DCAGENT_DIR}/eval/tacc/dcagent_eval_config.yaml"

echo "Using dataset path: $DATASET_PATH"
echo "Using harbor config: $HARBOR_CONFIG"

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
RUN_TAG="${CONFIG_NAME}_${TIMESTAMP}"

echo "Run tag: $RUN_TAG"

set +e
harbor jobs start \
  -p "$DATASET_PATH" \
  --n-concurrent 64 \
  --agent terminus-2 \
  --model "hosted_vllm/glm" \
  --env "daytona" \
  --agent-kwarg "api_base=http://localhost:8000/v1" \
  --agent-kwarg "key=fake_key" \
  --n-attempts 1 \
  --max-retries 0 \
  --disable-verification \
  --job-name "$RUN_TAG" \
  --config "$HARBOR_CONFIG"
HARBOR_EXIT=$?
set -e

# Cleanup
echo ""
echo "=== Step 6: Cleanup ==="

if [ -n "$VLLM_PID" ]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
fi

RAY_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/ray"
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null &
done
wait

conda deactivate || true

echo ""
echo "============================================"
if [ $HARBOR_EXIT -eq 0 ]; then
    echo "Experiment COMPLETED: $CONFIG_NAME"
else
    echo "Experiment FAILED: $CONFIG_NAME (exit code: $HARBOR_EXIT)"
fi
echo "============================================"

exit $HARBOR_EXIT
