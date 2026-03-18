#!/bin/bash
#SBATCH -p dc-gpu-devel
#SBATCH --time=1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --account=synthlaion
#SBATCH --output=/p/scratch/synthlaion/dc-agent-shared/logs/shard_test_%j.out
#SBATCH --job-name=shard_test_jsc

set -eo pipefail

echo "============================================"
echo "Sharding Test Job on JSC"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Testing: 2 shards -> run sequentially -> merge"
echo "============================================"

# === 1. Source existing JSC environment ===
source /p/project1/laionize/marianna/dc_agent/dc-agent/train/hpc/dotenv/jsc.env
source $SECRET_ENV_PATH

export PATH=$HOME/.local/bin:$PATH

# === 2. Activate conda ===
echo "Activating conda environment..."
source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate /p/scratch/laionize/raj3/vllm_sandboxes

harbor --help >/dev/null || { echo "ERROR: harbor not found"; exit 1; }

# Set cache directories early so Ray workers inherit them
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${USER}
export TRITON_DUMP_DIR=/tmp/triton_dump_${USER}
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TRITON_DUMP_DIR

# === 3. Setup proxy ===
echo "Setting up proxy..."
PROXY_HOST="10.14.0.53"
PROXY_PORT="1080"

if nc -zv -w 5 ${PROXY_HOST} ${PROXY_PORT} 2>&1 | grep -q Connected; then
    echo "Proxy reachable at ${PROXY_HOST}:${PROXY_PORT}"
else
    echo "ERROR: Cannot reach proxy at ${PROXY_HOST}:${PROXY_PORT}"
    exit 1
fi

export PROXYCHAINS_CONF_FILE="/tmp/proxychains_${SLURM_JOB_ID}.conf"
cat > "$PROXYCHAINS_CONF_FILE" << EOF
strict_chain
tcp_read_time_out 15000
tcp_connect_time_out 8000
localnet 10.0.0.0/255.0.0.0
localnet 127.0.0.0/255.0.0.0
localnet 192.168.0.0/255.255.0.0
[ProxyList]
socks5 ${PROXY_HOST} ${PROXY_PORT}
EOF

PROXYCHAINS_PRELOAD="/p/scratch/laionize/raj3/vllm_sandboxes/lib/libproxychains.so.4"

export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# === 4. Download and shard dataset into 2 shards ===
echo "Downloading and sharding dataset into 2 parts..."
INPUT_REPO_ID="DCAgent/harbor-devel-sandboxes"
NUM_SHARDS=2  # Test with 2 shards

export PYTHONPATH="/p/project1/laionize/eguha/OpenThoughts-Agent:/p/project1/laionize/eguha/OpenThoughts-Agent/scripts/harbor:$PYTHONPATH"
SHARD_OUTPUT_FILE=$(mktemp)
LD_PRELOAD="$PROXYCHAINS_PRELOAD" python /p/project1/laionize/eguha/OpenThoughts-Agent/data/sbatches/teacher_scripts/shard_dataset.py "$INPUT_REPO_ID" "$NUM_SHARDS" "$SHARD_OUTPUT_FILE"
source "$SHARD_OUTPUT_FILE"
rm "$SHARD_OUTPUT_FILE"

if [ -z "$SHARD_DIR" ] || [ ! -d "$SHARD_DIR" ]; then
    echo "ERROR: Failed to create shards"
    exit 1
fi

echo "Shard directory: $SHARD_DIR"
echo "Shards created:"
ls -la "$SHARD_DIR"

# === 5. Start Ray cluster ===
echo "Starting Ray cluster..."
bash $DC_AGENT/dc-agent/eval/JSC/start_ray_server.sh ""
sleep 30

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$(nslookup "${head_node}i" | grep 'Address' | tail -n1 | awk '{print $2}')
echo "Head node: $head_node ($head_node_ip)"

# === 6. Start vLLM server ===
echo "Starting vLLM server with GLM-4.7 (8-way tensor parallelism)..."
VLLM_LOG=/p/scratch/synthlaion/dc-agent-shared/logs/vllm_${SLURM_JOB_ID}.log

export VLLM_USE_V1=1
export RAY_ADDRESS=$head_node_ip:6379
export HF_HUB_CACHE=/p/scratch/synthlaion/dc-agent-shared/hf_hub
export HF_HOME=/p/scratch/synthlaion/dc-agent-shared/hf_hub

python -m vllm.entrypoints.openai.api_server \
    --model QuantTrio/GLM-4.7-AWQ \
    --served-model-name glm \
    --tensor-parallel-size 8 \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.92 \
    --max-model-len 113000 \
    --max-num-seqs 64 \
    --trust-remote-code \
    --port 8000 \
    >> $VLLM_LOG 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID, Log: $VLLM_LOG"

echo "Waiting for vLLM server to be healthy..."
for i in {1..240}; do
    if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo "vLLM server is healthy!"
        break
    fi
    if [ "$i" -eq 240 ]; then
        echo "ERROR: vLLM server failed to start"
        cat $VLLM_LOG
        exit 1
    fi
    sleep 10
done

# === 7. Run Harbor on each shard sequentially ===
HARBOR_CONFIG=$DC_AGENT/dc-agent/eval/tacc/dcagent_eval_config.yaml
JOB_DIRS=""
ALL_SUCCESS=true

for SHARD_IDX in 0 1; do
    DATASET_PATH="$SHARD_DIR/shard_${SHARD_IDX}"
    JOB_NAME="shard_test_${SLURM_JOB_ID}_shard${SHARD_IDX}"

    echo ""
    echo "============================================"
    echo "Running Harbor on SHARD ${SHARD_IDX}"
    echo "Dataset: $DATASET_PATH"
    echo "Job name: $JOB_NAME"
    echo "============================================"

    # Count tasks in this shard
    TASK_COUNT=$(ls -d "$DATASET_PATH"/*/ 2>/dev/null | wc -l)
    echo "Tasks in shard ${SHARD_IDX}: $TASK_COUNT"

    LD_PRELOAD="$PROXYCHAINS_PRELOAD" harbor jobs start \
        -p "$DATASET_PATH" \
        --n-concurrent 64 \
        --agent terminus-2 \
        --model "hosted_vllm/glm" \
        --env "daytona" \
        --agent-kwarg "api_base=http://127.0.0.1:8000/v1" \
        --agent-kwarg "key=fake_key" \
        --agent-kwarg 'model_info={"max_input_tokens":100000,"max_output_tokens":8000,"input_cost_per_token":0,"output_cost_per_token":0}' \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$JOB_NAME" \
        --timeout-multiplier 2.0 \
        --config "$HARBOR_CONFIG" \
        --disable-verification \
        --agent-kwarg 'trajectory_config={"raw_content":true,"linear_history":true}' \
        --agent-kwarg 'extra_body={"chat_template_kwargs":{"enable_thinking":true}}'

    SHARD_EXIT=$?

    if [ $SHARD_EXIT -eq 0 ]; then
        echo "Shard ${SHARD_IDX} completed successfully!"
        JOB_DIRS="$JOB_DIRS jobs/$JOB_NAME"
    else
        echo "ERROR: Shard ${SHARD_IDX} failed with exit code: $SHARD_EXIT"
        ALL_SUCCESS=false
    fi
done

# === 8. Merge and upload traces from all shards ===
echo ""
echo "============================================"
echo "Merging traces from all shards"
echo "============================================"
echo "Job directories: $JOB_DIRS"

if [ -n "$JOB_DIRS" ]; then
    OUTPUT_REPO_ID="DCAgent/harbor-devel-sandboxes_glm47_sharding_test"
    TEACHER_SCRIPTS_DIR="/p/project1/laionize/eguha/OpenThoughts-Agent/data/sbatches/teacher_scripts"

    UPLOAD_HF_CACHE="$HOME/.cache/huggingface"
    mkdir -p "$UPLOAD_HF_CACHE"

    echo "Uploading merged traces to: $OUTPUT_REPO_ID"
    HF_HOME="$UPLOAD_HF_CACHE" HF_HUB_CACHE="$UPLOAD_HF_CACHE" \
        LD_PRELOAD="$PROXYCHAINS_PRELOAD" LD_LIBRARY_PATH="" TRANSFORMERS_NO_TORCH=1 \
        python "$TEACHER_SCRIPTS_DIR/upload_traces.py" "$OUTPUT_REPO_ID" $JOB_DIRS

    UPLOAD_EXIT=$?
    if [ $UPLOAD_EXIT -eq 0 ]; then
        echo "Traces merged and uploaded to: $OUTPUT_REPO_ID"
    else
        echo "WARNING: Failed to upload merged traces (exit code: $UPLOAD_EXIT)"
    fi
else
    echo "ERROR: No job directories to merge"
fi

# === 9. Cleanup ===
echo "Cleaning up..."
kill $VLLM_PID 2>/dev/null || true
ray stop --force 2>/dev/null || true

echo "============================================"
if [ "$ALL_SUCCESS" = true ]; then
    echo "Sharding test COMPLETED successfully"
    echo "- 2 shards processed sequentially"
    echo "- Traces merged and uploaded"
else
    echo "Sharding test FAILED - some shards had errors"
fi
echo "============================================"

if [ "$ALL_SUCCESS" = true ]; then
    exit 0
else
    exit 1
fi
