#!/bin/bash
#SBATCH -p dc-gpu
#SBATCH --time=6:00:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --account=synthlaion
#SBATCH --output=/p/scratch/synthlaion/dc-agent-shared/logs/harbor_16node_%j.out
#SBATCH --job-name=harbor_16node

# ============================================
# Harbor 16-node job: 8 shards × 2 nodes each
# Single sbatch, parallel execution using srun
# ============================================

set -eo pipefail

echo "============================================"
echo "Harbor 16-node 8-shard Job"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "============================================"

# === 1. Source environment ===
source /p/project1/laionize/marianna/dc_agent/dc-agent/train/hpc/dotenv/jsc.env
source $SECRET_ENV_PATH
export PATH=$HOME/.local/bin:$PATH

# === 2. Get all nodes and pair them ===
ALL_NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NUM_NODES=${#ALL_NODES[@]}

if [ "$NUM_NODES" -ne 16 ]; then
    echo "ERROR: Expected 16 nodes, got $NUM_NODES"
    exit 1
fi

echo "All nodes: ${ALL_NODES[*]}"

# === 3. Setup proxy ===
PROXY_HOST="10.14.0.53"
PROXY_PORT="1080"

if ! nc -zv -w 5 ${PROXY_HOST} ${PROXY_PORT} 2>&1 | grep -q Connected; then
    echo "ERROR: Cannot reach proxy"
    exit 1
fi

PROXYCHAINS_PRELOAD="/p/scratch/laionize/raj3/vllm_sandboxes/lib/libproxychains.so.4"
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

export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# === 4. Activate conda and download dataset ===
echo "Activating conda..."
source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate /p/scratch/laionize/raj3/vllm_sandboxes

INPUT_REPO_ID="${INPUT_REPO_ID:-DCAgent/harbor-devel-sandboxes}"
NUM_SHARDS=8

export PYTHONPATH="/p/project1/laionize/eguha/OpenThoughts-Agent:/p/project1/laionize/eguha/OpenThoughts-Agent/scripts/harbor:$PYTHONPATH"

echo "Downloading and sharding dataset..."
SHARD_OUTPUT_FILE=$(mktemp)
LD_PRELOAD="$PROXYCHAINS_PRELOAD" python /p/project1/laionize/eguha/OpenThoughts-Agent/data/sbatches/teacher_scripts/shard_dataset.py "$INPUT_REPO_ID" "$NUM_SHARDS" "$SHARD_OUTPUT_FILE"
source "$SHARD_OUTPUT_FILE"
rm "$SHARD_OUTPUT_FILE"

echo "Shard directory: $SHARD_DIR"

# Build list of non-empty shards
declare -a ACTIVE_SHARDS
for i in {0..7}; do
    TASK_COUNT=$(ls -d "$SHARD_DIR/shard_$i"/*/ 2>/dev/null | wc -l)
    echo "  Shard $i: $TASK_COUNT tasks"
    if [ "$TASK_COUNT" -gt 0 ]; then
        ACTIVE_SHARDS+=($i)
    fi
done

NUM_ACTIVE=${#ACTIVE_SHARDS[@]}
echo ""
echo "Active shards: ${ACTIVE_SHARDS[*]} ($NUM_ACTIVE total)"

if [ "$NUM_ACTIVE" -eq 0 ]; then
    echo "ERROR: No tasks in any shard!"
    exit 1
fi

# === 5. Create per-shard worker script ===
WORKER_SCRIPT="/tmp/shard_worker_${SLURM_JOB_ID}.sh"
cat > "$WORKER_SCRIPT" << 'WORKER_EOF'
#!/bin/bash
# Per-shard worker script - runs on 2-node allocation
SHARD_IDX=$1
SHARD_DIR=$2
MAIN_JOB_ID=$3
PROXYCHAINS_PRELOAD=$4
PROXYCHAINS_CONF=$5

set -eo pipefail

# Get assigned nodes for this shard (use override if set)
NODELIST="${SLURM_JOB_NODELIST_OVERRIDE:-$SLURM_JOB_NODELIST}"
SHARD_NODES=($(scontrol show hostnames "$NODELIST"))
HEAD_NODE=${SHARD_NODES[0]}
WORKER_NODE=${SHARD_NODES[1]}
export SLURM_JOB_NODELIST="$NODELIST"  # Override for start_ray_server.sh

echo "[Shard $SHARD_IDX] Nodes: $HEAD_NODE, $WORKER_NODE"

# Setup
source /p/project1/laionize/marianna/dc_agent/dc-agent/train/hpc/dotenv/jsc.env
source $SECRET_ENV_PATH
source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate /p/scratch/laionize/raj3/vllm_sandboxes

export PYTHONPATH="/p/project1/laionize/eguha/OpenThoughts-Agent:/p/project1/laionize/eguha/OpenThoughts-Agent/scripts/harbor:$PYTHONPATH"
export PROXYCHAINS_CONF_FILE="$PROXYCHAINS_CONF"

# Unique cache dirs
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}_shard${SHARD_IDX}
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${USER}_shard${SHARD_IDX}
export TRITON_DUMP_DIR=/tmp/triton_dump_${USER}_shard${SHARD_IDX}
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TRITON_DUMP_DIR

# Start Ray cluster manually (can't use srun inside parallel workers)
echo "[Shard $SHARD_IDX] Starting Ray cluster manually..."
HEAD_NODE_IP=$(nslookup "${HEAD_NODE}i" | grep 'Address' | tail -n1 | awk '{print $2}')
RAY_PORT=6379

# Start Ray head on HEAD_NODE
ssh -o StrictHostKeyChecking=no $HEAD_NODE "
    source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate /p/scratch/laionize/raj3/vllm_sandboxes
    export TRITON_CACHE_DIR=$TRITON_CACHE_DIR
    export TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR
    ray start --head --port=$RAY_PORT --num-gpus=4 --block
" &
HEAD_RAY_PID=$!
sleep 15

# Start Ray worker on WORKER_NODE
ssh -o StrictHostKeyChecking=no $WORKER_NODE "
    source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate /p/scratch/laionize/raj3/vllm_sandboxes
    export TRITON_CACHE_DIR=$TRITON_CACHE_DIR
    export TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR
    ray start --address=$HEAD_NODE_IP:$RAY_PORT --num-gpus=4 --block
" &
WORKER_RAY_PID=$!
sleep 15

echo "[Shard $SHARD_IDX] Ray cluster started on $HEAD_NODE_IP:$RAY_PORT"
echo "[Shard $SHARD_IDX] Head IP: $HEAD_NODE_IP"

# Start vLLM
VLLM_LOG=/p/scratch/synthlaion/dc-agent-shared/logs/vllm_${MAIN_JOB_ID}_shard${SHARD_IDX}.log
echo "[Shard $SHARD_IDX] Starting vLLM server..."

export VLLM_USE_V1=1
export RAY_ADDRESS=$HEAD_NODE_IP:6379
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

# Wait for vLLM
echo "[Shard $SHARD_IDX] Waiting for vLLM..."
for i in {1..240}; do
    if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo "[Shard $SHARD_IDX] vLLM ready!"
        break
    fi
    if [ "$i" -eq 240 ]; then
        echo "[Shard $SHARD_IDX] ERROR: vLLM timeout"
        kill $VLLM_PID 2>/dev/null || true
        ray stop --force 2>/dev/null || true
        exit 1
    fi
    sleep 10
done

# Run Harbor
DATASET_PATH="$SHARD_DIR/shard_${SHARD_IDX}"
JOB_NAME="harbor_16node_${MAIN_JOB_ID}_shard${SHARD_IDX}"
HARBOR_CONFIG=$DC_AGENT/dc-agent/eval/tacc/dcagent_eval_config.yaml

echo "[Shard $SHARD_IDX] Running Harbor on $DATASET_PATH..."
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
    --agent-file-retention trajectories_only \
    --agent-kwarg 'trajectory_config={"raw_content":true,"linear_history":true}' \
    --agent-kwarg 'extra_body={"chat_template_kwargs":{"enable_thinking":true}}'

HARBOR_EXIT=$?

# Cleanup
kill $VLLM_PID 2>/dev/null || true
kill $HEAD_RAY_PID $WORKER_RAY_PID 2>/dev/null || true
ssh -o StrictHostKeyChecking=no $HEAD_NODE "ray stop --force" 2>/dev/null || true
ssh -o StrictHostKeyChecking=no $WORKER_NODE "ray stop --force" 2>/dev/null || true

if [ $HARBOR_EXIT -eq 0 ]; then
    echo "[Shard $SHARD_IDX] COMPLETED"
    echo "jobs/$JOB_NAME" > /tmp/shard_${MAIN_JOB_ID}_${SHARD_IDX}_done
else
    echo "[Shard $SHARD_IDX] FAILED: $HARBOR_EXIT"
fi

exit $HARBOR_EXIT
WORKER_EOF
chmod +x "$WORKER_SCRIPT"

# === 6. Launch only active shards using srun with exclusive node allocation ===
echo ""
echo "============================================"
echo "Launching $NUM_ACTIVE active shards in parallel..."
echo "============================================"

declare -a SHARD_PIDS
declare -a LAUNCHED_SHARDS
NODE_PAIR_IDX=0

for SHARD_IDX in "${ACTIVE_SHARDS[@]}"; do
    # Calculate which 2 nodes this shard gets
    NODE1_IDX=$((NODE_PAIR_IDX * 2))
    NODE2_IDX=$((NODE_PAIR_IDX * 2 + 1))
    SHARD_NODELIST="${ALL_NODES[$NODE1_IDX]},${ALL_NODES[$NODE2_IDX]}"

    echo "Launching shard $SHARD_IDX on nodes: $SHARD_NODELIST"

    # Run worker script on first node of pair, pass both nodes via env var
    SLURM_JOB_NODELIST_OVERRIDE="$SHARD_NODELIST" \
        bash "$WORKER_SCRIPT" "$SHARD_IDX" "$SHARD_DIR" "$SLURM_JOB_ID" "$PROXYCHAINS_PRELOAD" "$PROXYCHAINS_CONF_FILE" \
        > /p/scratch/synthlaion/dc-agent-shared/logs/harbor_16node_${SLURM_JOB_ID}_shard${SHARD_IDX}.log 2>&1 &
    SHARD_PIDS+=($!)
    LAUNCHED_SHARDS+=($SHARD_IDX)

    ((NODE_PAIR_IDX++))
    sleep 5  # Stagger launches
done

echo "Launched ${#SHARD_PIDS[@]} shards. PIDs: ${SHARD_PIDS[*]}"

# === 7. Wait for all launched shards ===
echo ""
echo "Waiting for all shards to complete..."
FAILED_SHARDS=0

for idx in "${!SHARD_PIDS[@]}"; do
    SHARD_IDX=${LAUNCHED_SHARDS[$idx]}
    wait ${SHARD_PIDS[$idx]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Shard $SHARD_IDX: COMPLETED"
    else
        echo "Shard $SHARD_IDX: FAILED (exit code $EXIT_CODE)"
        ((FAILED_SHARDS++)) || true
    fi
done

# === 8. Merge traces ===
echo ""
echo "============================================"
echo "Merging traces from $NUM_ACTIVE shards..."
echo "============================================"

JOB_DIRS=""
for SHARD_IDX in "${ACTIVE_SHARDS[@]}"; do
    MARKER="/tmp/shard_${SLURM_JOB_ID}_${SHARD_IDX}_done"
    if [ -f "$MARKER" ]; then
        JOB_DIR=$(cat "$MARKER")
        if [ -d "$JOB_DIR" ]; then
            JOB_DIRS="$JOB_DIRS $JOB_DIR"
            echo "Found: $JOB_DIR"
        fi
    fi
done

if [ -n "$JOB_DIRS" ]; then
    OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-DCAgent/harbor-glm47-16node-traces}"
    TEACHER_SCRIPTS_DIR="/p/project1/laionize/eguha/OpenThoughts-Agent/data/sbatches/teacher_scripts"

    UPLOAD_HF_CACHE="$HOME/.cache/huggingface"
    mkdir -p "$UPLOAD_HF_CACHE"

    echo "Uploading to: $OUTPUT_REPO_ID"
    HF_HOME="$UPLOAD_HF_CACHE" HF_HUB_CACHE="$UPLOAD_HF_CACHE" \
        LD_PRELOAD="$PROXYCHAINS_PRELOAD" LD_LIBRARY_PATH="" TRANSFORMERS_NO_TORCH=1 \
        python "$TEACHER_SCRIPTS_DIR/upload_traces.py" "$OUTPUT_REPO_ID" $JOB_DIRS

    if [ $? -eq 0 ]; then
        echo "Traces uploaded to $OUTPUT_REPO_ID"
    else
        echo "WARNING: Upload failed"
    fi
else
    echo "ERROR: No job directories found"
fi

# === 9. Summary ===
echo ""
echo "============================================"
if [ $FAILED_SHARDS -eq 0 ]; then
    echo "All 8 shards COMPLETED successfully!"
else
    echo "$FAILED_SHARDS/8 shards FAILED"
fi
echo "============================================"

exit $FAILED_SHARDS
