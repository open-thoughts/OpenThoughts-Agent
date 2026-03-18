#!/bin/bash
#SBATCH -p dc-gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --account=synthlaion
#SBATCH --output=/p/scratch/synthlaion/dc-agent-shared/logs/harbor_8shard_%A_%a.out
#SBATCH --job-name=harbor_8shard
#SBATCH --array=0-7

# ============================================
# Harbor 8-shard parallel job (16 nodes total)
# Each array task: 2 nodes, 8-way TP, 1 shard
#
# Usage: sbatch harbor_16node_8shard.sbatch <dataset>
# Example: sbatch harbor_16node_8shard.sbatch DCAgent/exp_llmve_llm-verifier-dcagent-dev-set
# ============================================

set -eo pipefail

# Accept dataset as argument (passed via sbatch)
if [ -n "$1" ]; then
    INPUT_REPO_ID="$1"
elif [ -z "$INPUT_REPO_ID" ]; then
    echo "ERROR: No dataset specified"
    echo "Usage: sbatch harbor_16node_8shard.sbatch <dataset>"
    echo "Example: sbatch harbor_16node_8shard.sbatch DCAgent/exp_llmve_llm-verifier-dcagent-dev-set"
    exit 1
fi

SHARD_IDX=$SLURM_ARRAY_TASK_ID
ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID

echo "============================================"
echo "Harbor Shard ${SHARD_IDX}/8"
echo "============================================"
echo "Array Job ID: $ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Input Dataset: $INPUT_REPO_ID"
echo "============================================"

# === 1. Source environment ===
source /p/project1/laionize/marianna/dc_agent/dc-agent/train/hpc/dotenv/jsc.env
source /p/project1/laionize/eguha/OpenThoughts-Agent/data/sbatches/secret.env

export PATH=$HOME/.local/bin:$PATH

# === 2. Activate conda ===
echo "Activating conda environment..."
source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate /p/scratch/laionize/raj3/vllm_sandboxes

harbor --help >/dev/null || { echo "ERROR: harbor not found"; exit 1; }

# Set cache directories
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}_${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${USER}_${SLURM_JOB_ID}
export TRITON_DUMP_DIR=/tmp/triton_dump_${USER}_${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $TRITON_DUMP_DIR

# === 3. Setup proxy (use existing SOCKS5 proxy on login node) ===
# NOTE: Start proxy on login node FIRST with:
#   ssh -D 10.14.0.46:1080 -f -N jrlogin05i
PROXY_HOST="10.14.0.46"
PROXY_PORT="1080"

echo "Checking proxy at ${PROXY_HOST}:${PROXY_PORT}..."
if nc -zv -w 5 ${PROXY_HOST} ${PROXY_PORT} 2>&1 | grep -q Connected; then
    echo "✓ Proxy reachable at ${PROXY_HOST}:${PROXY_PORT}"
else
    echo "ERROR: Cannot reach proxy at ${PROXY_HOST}:${PROXY_PORT}"
    echo "Start proxy on login node first: ssh -D 10.14.0.46:1080 -f -N jrlogin05i"
    exit 1
fi

# Create proxychains config
export PROXYCHAINS_CONF_FILE="/tmp/proxychains_${SLURM_JOB_ID}.conf"
cat > "$PROXYCHAINS_CONF_FILE" << EOF
strict_chain
proxy_dns
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

# === 4. Download dataset and get shard (only shard 0 downloads, others wait) ===
INPUT_REPO_ID="${INPUT_REPO_ID:-DCAgent/harbor-devel-sandboxes}"
DATASET_NAME="${INPUT_REPO_ID##*/}"
OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-DCAgent/${DATASET_NAME}_glm_4.7_traces}"
NUM_SHARDS=8

echo "Input: $INPUT_REPO_ID"
echo "Output: $OUTPUT_REPO_ID"

export PYTHONPATH="/p/project1/laionize/eguha/OpenThoughts-Agent:/p/project1/laionize/eguha/OpenThoughts-Agent/scripts/harbor:$PYTHONPATH"

# Use SHARED scratch storage for coordination (not /tmp which is node-local)
SHARED_SCRATCH="/p/data1/mmlaion/eguha"
MARKER_DIR="${SHARED_SCRATCH}/harbor_sharding_${ARRAY_JOB_ID}"
SHARED_TMP="${SHARED_SCRATCH}/hf_datasets_cache"
mkdir -p "$MARKER_DIR" "$SHARED_TMP"
SHARD_INFO_FILE="$MARKER_DIR/shard_info.env"

if [ "$SHARD_IDX" -eq 0 ]; then
    echo "Shard 0: Downloading and sharding dataset..."
    SHARD_OUTPUT_FILE=$(mktemp)
    LD_PRELOAD="$PROXYCHAINS_PRELOAD" python /p/project1/laionize/eguha/OpenThoughts-Agent/data/sbatches/teacher_scripts/shard_dataset.py "$INPUT_REPO_ID" "$NUM_SHARDS" "$SHARD_OUTPUT_FILE" "$SHARED_TMP"
    source "$SHARD_OUTPUT_FILE"

    # Write shard info for other tasks
    echo "SHARD_DIR=$SHARD_DIR" > "$SHARD_INFO_FILE"
    echo "DATASET_FINGERPRINT=$DATASET_FINGERPRINT" >> "$SHARD_INFO_FILE"
    rm "$SHARD_OUTPUT_FILE"
    echo "Dataset sharded. Info written to $SHARD_INFO_FILE"
else
    echo "Shard $SHARD_IDX: Waiting for shard 0 to prepare dataset..."
    for i in {1..120}; do
        if [ -f "$SHARD_INFO_FILE" ]; then
            source "$SHARD_INFO_FILE"
            echo "Found shard info, continuing..."
            break
        fi
        if [ "$i" -eq 120 ]; then
            echo "ERROR: Timeout waiting for shard info"
            exit 1
        fi
        sleep 5
    done
fi

DATASET_PATH="$SHARD_DIR/shard_${SHARD_IDX}"
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Shard directory not found: $DATASET_PATH"
    exit 1
fi

TASK_COUNT=$(ls -d "$DATASET_PATH"/*/ 2>/dev/null | wc -l)
echo "Shard ${SHARD_IDX} dataset: $DATASET_PATH ($TASK_COUNT tasks)"

# === 5. Start Ray cluster for this shard ===
echo "Starting Ray cluster..."
RAY_BIN="/p/scratch/laionize/raj3/vllm_sandboxes/bin/ray"

# Get nodes
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(nslookup "${head_node}i" | grep 'Address' | tail -n1 | awk '{print $2}')
port=6379

echo "Starting HEAD at $head_node ($head_node_ip)"
srun --nodes=1 --ntasks=1 --gres=gpu:4 --overlap -w "$head_node" \
    $RAY_BIN start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 48 --num-gpus 4 --block &
sleep 15

# Start worker
if [ ${#nodes_array[@]} -gt 1 ]; then
    worker_node=${nodes_array[1]}
    worker_ip=$(nslookup "${worker_node}i" | grep 'Address' | tail -n1 | awk '{print $2}')
    echo "Starting WORKER at $worker_node ($worker_ip)"
    srun --nodes=1 --ntasks=1 --gres=gpu:4 --overlap -w "$worker_node" \
        $RAY_BIN start --address "$head_node_ip:$port" --node-ip-address="$worker_ip" \
        --num-cpus 48 --num-gpus 4 --block &
    sleep 15
fi

export RAY_ADDRESS=$head_node_ip:$port
echo "RAY ADDRESS: $RAY_ADDRESS"
echo "Head node: $head_node ($head_node_ip)"

# === 6. Start vLLM server ===
echo "Starting vLLM server with GLM-4.7 (8-way TP)..."
VLLM_LOG=/p/scratch/synthlaion/dc-agent-shared/logs/vllm_${SLURM_JOB_ID}.log

export VLLM_USE_V1=1
export RAY_ADDRESS=$head_node_ip:6379
export HF_HUB_CACHE=/p/scratch/synthlaion/dc-agent-shared/hf_hub
export HF_HOME=/p/scratch/synthlaion/dc-agent-shared/hf_hub

# Use offline mode - model must be pre-cached in HF_HUB_CACHE
# Pre-download with: huggingface-cli download QuantTrio/GLM-4.7-AWQ
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "HF_HUB_OFFLINE=1 - using pre-cached model from $HF_HUB_CACHE"

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
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM server..."
for i in {1..360}; do
    if curl -s http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
        echo "vLLM server is healthy!"
        break
    fi
    if [ "$i" -eq 360 ]; then
        echo "ERROR: vLLM server failed to start"
        tail -100 $VLLM_LOG
        exit 1
    fi
    sleep 10
done

# === 7. Run Harbor on this shard ===
JOB_NAME="${DATASET_NAME}_8shard_${DATASET_FINGERPRINT}_shard${SHARD_IDX}"
HARBOR_CONFIG=$DC_AGENT/dc-agent/eval/tacc/dcagent_eval_config.yaml
JOBS_DIR="/p/data1/mmlaion/eguha/harbor_jobs"
mkdir -p "$JOBS_DIR"

echo "Starting Harbor on shard ${SHARD_IDX}..."
LD_PRELOAD="$PROXYCHAINS_PRELOAD" harbor jobs start \
    -p "$DATASET_PATH" \
    --jobs-dir "$JOBS_DIR" \
    --n-concurrent 64 \
    --agent terminus-2 \
    --model "hosted_vllm/glm" \
    --env "daytona" \
    --agent-kwarg "api_base=http://127.0.0.1:8000/v1" \
    --agent-kwarg "key=fake_key" \
    --n-attempts 1 \
    --max-retries 0 \
    --job-name "$JOB_NAME" \
    --auto-resume \
    --auto-resume-filter-error-type DaytonaRateLimitError \
    --auto-resume-filter-error-type EnvironmentStartTimeoutError \
    --auto-resume-filter-error-type DaytonaError \
    --timeout-multiplier 2.0 \
    --config "$HARBOR_CONFIG" \
    --disable-verification \
    --agent-file-retention trajectories_only \
    --agent-kwarg 'trajectory_config={"raw_content":true,"linear_history":true}' \
    --agent-kwarg 'extra_body={"chat_template_kwargs":{"enable_thinking":true}}'

HARBOR_EXIT=$?

# === 8. Write completion marker ===
if [ $HARBOR_EXIT -eq 0 ]; then
    echo "$JOBS_DIR/$JOB_NAME" > "$MARKER_DIR/shard_${SHARD_IDX}_done"
    echo "Shard ${SHARD_IDX} COMPLETED successfully"
else
    echo "FAILED" > "$MARKER_DIR/shard_${SHARD_IDX}_done"
    echo "Shard ${SHARD_IDX} FAILED with exit code: $HARBOR_EXIT"
fi

# === 8.5. Clean up unnecessary files (keep only trajectory*.json) ===
echo "Cleaning up unnecessary files in $JOBS_DIR/$JOB_NAME..."
JOB_DIR="$JOBS_DIR/$JOB_NAME"
if [ -d "$JOB_DIR" ]; then
    # Delete episode directories (contain debug.json, prompt.txt, response.txt)
    find "$JOB_DIR" -type d -name "episode-*" -exec rm -rf {} + 2>/dev/null || true
    # Delete recording and pane files
    find "$JOB_DIR" -path "*/agent/*" -type f \( -name "recording.cast" -o -name "terminus_2.pane" \) -delete 2>/dev/null || true
    echo "Cleanup completed for shard ${SHARD_IDX}"
fi

# === 9. Cleanup ===
kill $VLLM_PID 2>/dev/null || true
ray stop --force 2>/dev/null || true

# === 10. Upload traces (shard 0 waits for all shards and uploads) ===
if [ "$SHARD_IDX" -eq 0 ]; then
    echo "============================================"
    echo "Shard 0: Waiting for all shards to complete before uploading..."
    echo "============================================"

    # Wait for all shards to finish (up to 5 hours)
    for attempt in {1..600}; do
        completed=0
        for s in $(seq 0 $((NUM_SHARDS - 1))); do
            if [ -f "$MARKER_DIR/shard_${s}_done" ]; then
                completed=$((completed + 1))
            fi
        done

        echo "$(date): $completed / $NUM_SHARDS shards completed"

        if [ "$completed" -eq "$NUM_SHARDS" ]; then
            echo "All shards completed!"
            break
        fi

        if [ "$attempt" -eq 600 ]; then
            echo "WARNING: Timeout waiting for all shards. Uploading available traces..."
            break
        fi

        sleep 30
    done

    # Collect job directories from completion markers
    JOB_DIRS=""
    for s in $(seq 0 $((NUM_SHARDS - 1))); do
        marker="$MARKER_DIR/shard_${s}_done"
        if [ -f "$marker" ]; then
            job_dir=$(cat "$marker")
            if [ "$job_dir" != "FAILED" ] && [ -d "$job_dir" ]; then
                JOB_DIRS="$JOB_DIRS $job_dir"
                echo "Found job dir: $job_dir"
            fi
        fi
    done

    if [ -n "$JOB_DIRS" ]; then
        # Upload to HuggingFace
        echo "============================================"
        echo "Uploading traces to: $OUTPUT_REPO_ID"
        echo "============================================"

        # Unset offline mode for upload (needs HuggingFace access)
        unset HF_HUB_OFFLINE
        unset TRANSFORMERS_OFFLINE

        # Use temp HF cache to avoid XET permission errors on shared storage
        export HF_HOME=/tmp/hf_upload_cache_${SLURM_JOB_ID}
        export HF_HUB_CACHE=/tmp/hf_upload_cache_${SLURM_JOB_ID}
        mkdir -p $HF_HOME

        LD_PRELOAD="$PROXYCHAINS_PRELOAD" python /p/project1/laionize/eguha/OpenThoughts-Agent/data/sbatches/teacher_scripts/upload_traces.py \
            "$OUTPUT_REPO_ID" $JOB_DIRS

        UPLOAD_EXIT=$?
        if [ $UPLOAD_EXIT -eq 0 ]; then
            echo "Upload COMPLETED successfully to $OUTPUT_REPO_ID"
        else
            echo "Upload FAILED with exit code: $UPLOAD_EXIT"
        fi
    else
        echo "ERROR: No valid job directories found for upload"
    fi
fi

exit $HARBOR_EXIT
