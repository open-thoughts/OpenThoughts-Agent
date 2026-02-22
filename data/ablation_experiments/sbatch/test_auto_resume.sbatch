#!/bin/bash
#SBATCH -p gh-dev
#SBATCH --time=01:00:00
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --account CCR24067
#SBATCH --output=/scratch/10000/eguha3/dc-agent/data/ablation_experiments/logs/test_auto_resume_%j.out
#SBATCH --job-name=test_auto_resume

set -eo pipefail

EXPERIMENT_NAME="test_auto_resume"
INPUT_REPO_ID="DCAgent/harbor-devel-sandboxes"

EXPERIMENTS_DIR="/scratch/10000/eguha3/dc-agent/data/ablation_experiments"
NUM_SHARDS=1
NODES_PER_SHARD=4

echo "============================================"
echo "Test Auto-Resume Experiment: $EXPERIMENT_NAME"
echo "============================================"
echo "Total Nodes: $SLURM_JOB_NUM_NODES"
echo "Shards: $NUM_SHARDS"
echo "Job ID: $SLURM_JOB_ID"
echo "Input Dataset: $INPUT_REPO_ID"
echo "============================================"

mkdir -p "$EXPERIMENTS_DIR/logs"
mkdir -p "$EXPERIMENTS_DIR/results/$EXPERIMENT_NAME"

echo ""
echo "=== Step 1: Environment Setup ==="

module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
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

source /scratch/10000/eguha3/old-dc-agent/secret.env

ln -sf /home1/apps/gcc/15.1.0/lib64/libstdc++.so.6 /scratch/10000/eguha3/vllm_sandboxes_backup/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/10000/eguha3/vllm_sandboxes_backup/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/vllm_sandboxes_backup

harbor --help >/dev/null

RAY_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/ray"
PYTHON_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3"
export RAY_DEDUP_LOGS=0
export TRITON_CC=$(which gcc)

ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
ALL_NODES_ARRAY=($ALL_NODES)

echo "All nodes: ${ALL_NODES_ARRAY[@]}"

export SRUN_EXPORT_ENV="ALL,TRITON_CC=$TRITON_CC,LD_LIBRARY_PATH=$LD_LIBRARY_PATH,PATH=$PATH,HF_TOKEN=$HF_TOKEN"
RAY_ENV_VARS="TRITON_CC=$TRITON_CC LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH HF_TOKEN=$HF_TOKEN"

VLLM_MODEL_PATH="QuantTrio/GLM-4.7-AWQ"
VLLM_GPU_MEMORY_UTILIZATION=0.92
VLLM_MAX_NUM_SEQS=64
VLLM_MAX_MODEL_LEN=113000
VLLM_MAX_NUM_BATCHED_TOKENS=32768
VLLM_TENSOR_PARALLEL_SIZE=4
VLLM_PIPELINE_PARALLEL_SIZE=1

echo ""
echo "=== Step 2: Downloading and Sharding Dataset ==="

export PYTHONPATH="/scratch/10000/eguha3/dc-agent:/scratch/10000/eguha3/dc-agent/scripts/harbor:$PYTHONPATH"
TEACHER_SCRIPTS_DIR="/scratch/10000/eguha3/dc-agent/data/ablation_experiments/sbatch/teacher_scripts"

# Download and shard dataset (handles caching internally)
SHARD_OUTPUT_FILE=$(mktemp)
LD_LIBRARY_PATH="" /scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3 "$TEACHER_SCRIPTS_DIR/shard_dataset.py" "$INPUT_REPO_ID" "$NUM_SHARDS" "$SHARD_OUTPUT_FILE"
source "$SHARD_OUTPUT_FILE"
rm "$SHARD_OUTPUT_FILE"

if [ -z "$SHARD_DIR" ] || [ ! -d "$SHARD_DIR" ]; then
    echo "ERROR: Failed to create shards or path does not exist: $SHARD_DIR"
    exit 1
fi

echo "Shard directory: $SHARD_DIR"
echo "Dataset fingerprint: $DATASET_FINGERPRINT"

echo ""
echo "=== Step 3: Starting Ray Cluster ==="

HEAD_NODE=${ALL_NODES_ARRAY[0]}
RAY_PORT=6379
API_PORT=8000
RAY_TEMP_DIR="/tmp/ray_${USER}_test"

echo "Head node: $HEAD_NODE"

# Stop any existing Ray processes
for node in "${ALL_NODES_ARRAY[@]}"; do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null || true
    srun --nodes=1 --ntasks=1 --overlap -w "$node" rm -rf "$RAY_TEMP_DIR" 2>/dev/null || true
done
sleep 2

# Get head node IP
HEAD_IFACE="ib0"
if srun --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" ip -o -4 addr show "$HEAD_IFACE" >/dev/null 2>&1; then
    HEAD_IP=$(srun --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" ip -o -4 addr show "$HEAD_IFACE" | awk '{print $4}' | cut -d/ -f1)
else
    HEAD_IP=$(srun --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" hostname --ip-address)
    HEAD_IP=${HEAD_IP%% *}
fi

echo "Head IP: $HEAD_IP"

# Start Ray head
srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" bash -c \
    "env $RAY_ENV_VARS $RAY_BIN start --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} --num-gpus=1 --num-cpus=64 --temp-dir=${RAY_TEMP_DIR}" &
sleep 5

# Start Ray workers
for ((i = 1; i < ${#ALL_NODES_ARRAY[@]}; i++)); do
    WORKER_NODE=${ALL_NODES_ARRAY[$i]}
    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$WORKER_NODE" bash -c \
        "env $RAY_ENV_VARS $RAY_BIN start --address ${HEAD_IP}:${RAY_PORT} --num-gpus=1 --num-cpus=64 --temp-dir=${RAY_TEMP_DIR}" &
    sleep 2
done

echo "Waiting for Ray cluster to stabilize..."
sleep 30

echo ""
echo "=== Step 4: Starting vLLM Server ==="

VLLM_LOG="${EXPERIMENTS_DIR}/logs/vllm_${EXPERIMENT_NAME}_${SLURM_JOB_ID}.log"
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

echo "Starting vLLM on $HEAD_NODE (port $API_PORT)..."

srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" \
    env TRITON_CC="$TRITON_CC" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        PATH="$PATH" \
        HF_HOME="/tmp/hf_home" \
        HF_TOKEN="$HF_TOKEN" \
        RAY_ADDRESS="$RAY_ADDRESS" \
        VLLM_MODEL_PATH="$VLLM_MODEL_PATH" \
        VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
        VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
        VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
        VLLM_MAX_NUM_BATCHED_TOKENS="$VLLM_MAX_NUM_BATCHED_TOKENS" \
        VLLM_TENSOR_PARALLEL_SIZE="$VLLM_TENSOR_PARALLEL_SIZE" \
        VLLM_PIPELINE_PARALLEL_SIZE="$VLLM_PIPELINE_PARALLEL_SIZE" \
        VLLM_ENABLE_EXPERT_PARALLEL="true" \
        VLLM_KV_CACHE_DTYPE="auto" \
        VLLM_ENABLE_PREFIX_CACHING="true" \
        VLLM_SWAP_SPACE="4" \
        VLLM_BLOCK_SIZE="16" \
        VLLM_ENABLE_CHUNKED_PREFILL="true" \
        VLLM_ALL2ALL_BACKEND="pplx" \
        VLLM_ENABLE_EPLB="false" \
        SERVE_HTTP_PORT="$API_PORT" \
    $PYTHON_BIN /scratch/10000/eguha3/dc-agent/scripts/vllm/dp_debug.py \
    >> "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID, Log: $VLLM_LOG"

echo ""
echo "Waiting for vLLM server to become healthy..."

HEALTH_URL="http://127.0.0.1:${API_PORT}/v1/models"
for i in {1..180}; do
    if srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" curl -s "$HEALTH_URL" > /dev/null 2>&1; then
        echo "vLLM server is healthy!"
        break
    fi
    if [ "$i" -eq 180 ]; then
        echo "ERROR: vLLM failed health check"
        exit 1
    fi
    sleep 10
done

echo ""
echo "=== Step 5: Running Harbor Job ==="

HARBOR_CONFIG="/scratch/10000/eguha3/dc-agent/eval/tacc/dcagent_eval_config.yaml"
# Use deterministic job name based on experiment name and dataset fingerprint (not timestamp)
CONFIG_NAME="${EXPERIMENT_NAME}_${DATASET_FINGERPRINT}"
RUN_TAG="${CONFIG_NAME}"
HARBOR_LOG="${EXPERIMENTS_DIR}/logs/harbor_${EXPERIMENT_NAME}_${SLURM_JOB_ID}.log"

echo "Starting Harbor job..."
echo "  Dataset: $SHARD_DIR/shard_0"
echo "  API port: $API_PORT"
echo "  Run tag: $RUN_TAG"

srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" \
    harbor jobs start \
        -p "$SHARD_DIR/shard_0" \
        --n-concurrent 4 \
        --agent terminus-2 \
        --model "hosted_vllm/glm" \
        --env "daytona" \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$RUN_TAG" \
        --auto-resume \
        --timeout-multiplier 2.0 \
        --config "$HARBOR_CONFIG" \
        --disable-verification \
        --agent-kwarg "api_base=http://localhost:${API_PORT}/v1" \
        --agent-kwarg "key=fake_key" \
        --agent-kwarg "temperature=1.0" \
        --agent-kwarg "max_episodes=128" \
        --agent-kwarg "enable_summarize=true" \
        --agent-kwarg "proactive_summarization_threshold=8192" \
        --agent-kwarg "interleaved_thinking=false" \
        --agent-kwarg "parser_name=json" \
        --agent-kwarg "tmux_pane_width=160" \
        --agent-kwarg "tmux_pane_height=40" \
        --agent-kwarg 'trajectory_config={"raw_content":true,"linear_history":true}' \
        --agent-kwarg 'extra_body={"chat_template_kwargs":{"enable_thinking":true},"min_p":0.01}' \
    >> "$HARBOR_LOG" 2>&1

HARBOR_EXIT=$?
echo "Harbor job completed with exit code: $HARBOR_EXIT"

echo ""
echo "=== Step 6: Cleanup ==="

kill "$VLLM_PID" 2>/dev/null || true

for node in "${ALL_NODES_ARRAY[@]}"; do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null &
done
wait

conda deactivate || true

echo ""
echo "============================================"
if [ $HARBOR_EXIT -eq 0 ]; then
    echo "Test COMPLETED: $EXPERIMENT_NAME"
else
    echo "Test FAILED: $EXPERIMENT_NAME (exit code: $HARBOR_EXIT)"
fi
echo "============================================"

exit $HARBOR_EXIT
