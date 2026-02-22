#!/bin/bash
#SBATCH -p gh-dev
#SBATCH --time=02:00:00
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --account CCR24067
#SBATCH --output=/scratch/10000/eguha3/dc-agent/data/sbatches/logs/test_harbor_ghdev_%j.out
#SBATCH --job-name=test_harbor

set -eo pipefail

if [ -z "$1" ]; then
    echo "Usage: sbatch test_harbor_ghdev.sbatch <hf_dataset_repo_id>"
    exit 1
fi

source /scratch/10000/eguha3/old-dc-agent/secret.env

INPUT_REPO_ID="$1"
DATASET_NAME="${INPUT_REPO_ID##*/}"
OUTPUT_REPO_ID="DCAgent/${DATASET_NAME}_glm_4.7_traces_test"

SBATCHES_DIR="${DCAGENT_DIR}/data/sbatches"
NUM_SHARDS=1
NODES_PER_SHARD=4

echo "============================================"
echo "TEST: vLLM + Harbor (gh-dev)"
echo "============================================"
echo "Total Nodes: $SLURM_JOB_NUM_NODES"
echo "Shards: $NUM_SHARDS"
echo "Job ID: $SLURM_JOB_ID"
echo "Input Dataset: $INPUT_REPO_ID"
echo "Output Traces: $OUTPUT_REPO_ID"
echo "DCAGENT_DIR: $DCAGENT_DIR"
echo "============================================"

mkdir -p "$SBATCHES_DIR/logs"

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

export PYTHONPATH="${DCAGENT_DIR}:${DCAGENT_DIR}/scripts/harbor:$PYTHONPATH"
TEACHER_SCRIPTS_DIR="${DCAGENT_DIR}/data/sbatches/teacher_scripts"

echo "TEACHER_SCRIPTS_DIR: $TEACHER_SCRIPTS_DIR"
echo "Checking if shard_dataset.py exists..."
ls -la "$TEACHER_SCRIPTS_DIR/shard_dataset.py"

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

shard_idx=0
ray_port=6379
api_port=8000

head_node=${ALL_NODES_ARRAY[0]}
ray_temp_dir="/tmp/ray_${USER}_shard${shard_idx}"

echo "Head node: $head_node"

for node in "${ALL_NODES_ARRAY[@]}"; do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null || true
    srun --nodes=1 --ntasks=1 --overlap -w "$node" rm -rf "/tmp/ray_${USER}_shard${shard_idx}" 2>/dev/null || true
done
sleep 2

head_iface="ib0"
if srun --nodes=1 --ntasks=1 --overlap -w "$head_node" ip -o -4 addr show "$head_iface" >/dev/null 2>&1; then
    head_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" ip -o -4 addr show "$head_iface" | awk '{print $4}' | cut -d/ -f1)
else
    head_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" hostname --ip-address)
    head_ip=${head_ip%% *}
fi

echo "Head IP: $head_ip"

srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" bash -c \
    "env $RAY_ENV_VARS $RAY_BIN start --head --node-ip-address=${head_ip} --port=${ray_port} --num-gpus=1 --num-cpus=64 --temp-dir=${ray_temp_dir}" &
sleep 5

for ((i = 1; i < NODES_PER_SHARD; i++)); do
    worker_node=${ALL_NODES_ARRAY[$i]}
    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$worker_node" bash -c \
        "env $RAY_ENV_VARS $RAY_BIN start --address ${head_ip}:${ray_port} --num-gpus=1 --num-cpus=64 --temp-dir=${ray_temp_dir}" &
    sleep 2
done

echo "Waiting for Ray cluster to stabilize..."
sleep 30

echo ""
echo "=== Step 4: Starting vLLM Server ==="

ray_address="${head_ip}:${ray_port}"
vllm_log="${SBATCHES_DIR}/logs/vllm_test_${DATASET_NAME}_${SLURM_JOB_ID}.log"

echo "Starting vLLM on $head_node (port $api_port)..."

srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
    env TRITON_CC="$TRITON_CC" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        PATH="$PATH" \
        HF_HOME="/tmp/hf_home" \
        HF_TOKEN="$HF_TOKEN" \
        RAY_ADDRESS="$ray_address" \
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
        SERVE_HTTP_PORT="$api_port" \
    $PYTHON_BIN ${DCAGENT_DIR}/scripts/vllm/dp_debug.py \
    >> "$vllm_log" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID, Log: $vllm_log"

echo ""
echo "Waiting for vLLM server to become healthy..."

health_url="http://127.0.0.1:${api_port}/v1/models"
for i in {1..180}; do
    if srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" curl -s "$health_url" > /dev/null 2>&1; then
        echo "vLLM is healthy!"
        break
    fi
    if [ "$i" -eq 180 ]; then
        echo "ERROR: vLLM failed health check"
        cat "$vllm_log"
        exit 1
    fi
    sleep 10
done

echo ""
echo "=== Step 5: Running Harbor Job (limited to 5 tasks) ==="

HARBOR_CONFIG="${DCAGENT_DIR}/eval/tacc/dcagent_eval_config.yaml"
CONFIG_NAME="${DATASET_NAME}_test_${SLURM_JOB_ID}"
shard_dataset="$SHARD_DIR/shard_0"
harbor_log="${SBATCHES_DIR}/logs/harbor_test_${DATASET_NAME}_${SLURM_JOB_ID}.log"

echo "Dataset: $shard_dataset"
echo "Config: $HARBOR_CONFIG"

srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
    harbor jobs start \
        -p "$shard_dataset" \
        --n-concurrent 5 \
        --agent terminus-2 \
        --model "hosted_vllm/glm" \
        --env "daytona" \
        --agent-kwarg "api_base=http://localhost:${api_port}/v1" \
        --agent-kwarg "key=fake_key" \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$CONFIG_NAME" \
        --timeout-multiplier 2.0 \
        --config "$HARBOR_CONFIG" \
        --disable-verification \
        --agent-kwarg 'trajectory_config={"raw_content":true,"linear_history":true}' \
        --agent-kwarg 'extra_body={"chat_template_kwargs":{"enable_thinking":true}}' \
        --limit 5 \
    >> "$harbor_log" 2>&1

HARBOR_EXIT=$?
echo "Harbor exit code: $HARBOR_EXIT"

echo ""
echo "=== Harbor Log (last 50 lines) ==="
tail -50 "$harbor_log"

echo ""
echo "=== Cleanup ==="

kill "$VLLM_PID" 2>/dev/null || true

for node in "${ALL_NODES_ARRAY[@]}"; do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null &
done
wait

conda deactivate || true

echo ""
echo "============================================"
if [ $HARBOR_EXIT -eq 0 ]; then
    echo "TEST PASSED"
else
    echo "TEST FAILED (exit code: $HARBOR_EXIT)"
fi
echo "============================================"

exit $HARBOR_EXIT
