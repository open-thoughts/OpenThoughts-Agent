#!/bin/bash
#SBATCH -p gh
#SBATCH --time=24:00:00
#SBATCH --nodes 32
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --account CCR24067
#SBATCH --output=${DCAGENT_DIR}/data/ablation_experiments/logs/top_k_64_%j.out
#SBATCH --job-name=abl_top_k_64

set -eo pipefail

EXPERIMENT_NAME="top_k_64"
INPUT_REPO_ID="mlfoundations-dev/stackexchange-tezos-sandboxes"
OUTPUT_REPO_ID="DCAgent/exp_tas_top_k_64_traces"

EXPERIMENTS_DIR="${DCAGENT_DIR}/data/ablation_experiments"
NUM_SHARDS=8
NODES_PER_SHARD=4

echo "============================================"
echo "Ablation Experiment: $EXPERIMENT_NAME"
echo "============================================"
echo "Total Nodes: $SLURM_JOB_NUM_NODES"
echo "Shards: $NUM_SHARDS"
echo "Job ID: $SLURM_JOB_ID"
echo "Input Dataset: $INPUT_REPO_ID"
echo "Output Traces: $OUTPUT_REPO_ID"
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

export PYTHONPATH="${DCAGENT_DIR}:${DCAGENT_DIR}/scripts/harbor:$PYTHONPATH"
TEACHER_SCRIPTS_DIR="${DCAGENT_DIR}/data/ablation_experiments/sbatch/teacher_scripts"

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
echo "=== Step 3: Starting $NUM_SHARDS Ray Clusters ==="

declare -a VLLM_PIDS
declare -a RAY_PORTS
declare -a API_PORTS
declare -a HEAD_IPS

BASE_RAY_PORT=6379
BASE_API_PORT=8000

start_ray_cluster_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local ray_port=$((BASE_RAY_PORT + shard_idx * 100))
    local api_port=$((BASE_API_PORT + shard_idx))

    local shard_nodes=()
    for ((i = 0; i < NODES_PER_SHARD; i++)); do
        shard_nodes+=(${ALL_NODES_ARRAY[$((start_node_idx + i))]})
    done

    local head_node=${shard_nodes[0]}
    local ray_temp_dir="/tmp/ray_${USER}_shard${shard_idx}"

    echo "  Shard $shard_idx: Nodes ${shard_nodes[@]}"
    echo "    Head: $head_node, Ray port: $ray_port, API port: $api_port"

    for node in "${shard_nodes[@]}"; do
        srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null || true
        srun --nodes=1 --ntasks=1 --overlap -w "$node" rm -rf "/tmp/ray_${USER}_shard${shard_idx}" 2>/dev/null || true
    done
    sleep 2

    local head_iface="ib0"
    local head_ip
    if srun --nodes=1 --ntasks=1 --overlap -w "$head_node" ip -o -4 addr show "$head_iface" >/dev/null 2>&1; then
        head_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" ip -o -4 addr show "$head_iface" | awk '{print $4}' | cut -d/ -f1)
    else
        head_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" hostname --ip-address)
        head_ip=${head_ip%% *}
    fi

    HEAD_IPS[$shard_idx]=$head_ip
    RAY_PORTS[$shard_idx]=$ray_port
    API_PORTS[$shard_idx]=$api_port

    echo "    Head IP: $head_ip"

    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" bash -c \
        "env $RAY_ENV_VARS $RAY_BIN start --head --node-ip-address=${head_ip} --port=${ray_port} --num-gpus=1 --num-cpus=64 --temp-dir=${ray_temp_dir}" &
    sleep 5

    for ((i = 1; i < NODES_PER_SHARD; i++)); do
        local worker_node=${shard_nodes[$i]}
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$worker_node" bash -c \
            "env $RAY_ENV_VARS $RAY_BIN start --address ${head_ip}:${ray_port} --num-gpus=1 --num-cpus=64 --temp-dir=${ray_temp_dir}" &
        sleep 2
    done
}

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    start_ray_cluster_for_shard $shard_idx
done

echo "Waiting for Ray clusters to stabilize..."
sleep 30

echo ""
echo "=== Step 4: Starting $NUM_SHARDS vLLM Servers ==="

start_vllm_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    local head_ip=${HEAD_IPS[$shard_idx]}
    local ray_port=${RAY_PORTS[$shard_idx]}
    local api_port=${API_PORTS[$shard_idx]}
    local ray_address="${head_ip}:${ray_port}"

    local vllm_log="${EXPERIMENTS_DIR}/logs/vllm_${EXPERIMENT_NAME}_shard${shard_idx}_${SLURM_JOB_ID}.log"

    echo "  Starting vLLM for shard $shard_idx on $head_node (port $api_port)..."

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

    VLLM_PIDS[$shard_idx]=$!
    echo "    PID: ${VLLM_PIDS[$shard_idx]}, Log: $vllm_log"
}

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    start_vllm_for_shard $shard_idx
done

echo ""
echo "Waiting for all vLLM servers to become healthy..."

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    start_node_idx=$((shard_idx * NODES_PER_SHARD))
    head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    api_port=${API_PORTS[$shard_idx]}
    health_url="http://127.0.0.1:${api_port}/v1/models"

    echo "  Checking shard $shard_idx (port $api_port)..."
    for i in {1..180}; do
        if srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" curl -s "$health_url" > /dev/null 2>&1; then
            echo "    Shard $shard_idx is healthy!"
            break
        fi
        if [ "$i" -eq 180 ]; then
            echo "ERROR: Shard $shard_idx failed health check"
            exit 1
        fi
        sleep 10
    done
done

echo "All vLLM servers are ready!"

echo ""
echo "=== Step 5: Running $NUM_SHARDS Harbor Jobs in Parallel ==="

HARBOR_CONFIG="${DCAGENT_DIR}/eval/tacc/dcagent_eval_config.yaml"
# Use deterministic job name based on experiment name and dataset fingerprint (not timestamp)
CONFIG_NAME="${EXPERIMENT_NAME}_${DATASET_FINGERPRINT}"

declare -a HARBOR_PIDS
declare -a HARBOR_LOGS

run_harbor_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    local api_port=${API_PORTS[$shard_idx]}
    local shard_dataset="$SHARD_DIR/shard_${shard_idx}"
    local run_tag="${CONFIG_NAME}_shard${shard_idx}"
    local harbor_log="${EXPERIMENTS_DIR}/logs/harbor_${EXPERIMENT_NAME}_shard${shard_idx}_${SLURM_JOB_ID}.log"

    HARBOR_LOGS[$shard_idx]="$harbor_log"

    echo "  Starting Harbor for shard $shard_idx..."
    echo "    Dataset: $shard_dataset"
    echo "    API port: $api_port"
    echo "    Run tag: $run_tag"

    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
        harbor jobs start \
            -p "$shard_dataset" \
            --n-concurrent 64 \
            --agent terminus-2 \
            --model "hosted_vllm/glm" \
            --env "daytona" \
            --n-attempts 1 \
            --max-retries 0 \
            --job-name "$run_tag" \
            --auto-resume \
            --timeout-multiplier 2.0 \
            --config "$HARBOR_CONFIG" \
            --disable-verification \
            --agent-kwarg "api_base=http://localhost:${api_port}/v1" \
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
            --agent-kwarg 'extra_body={"chat_template_kwargs":{"enable_thinking":true},"top_k":64}'         >> "$harbor_log" 2>&1 &

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

echo ""
echo "=== Step 6: Exporting and Uploading Expert Traces ==="

JOB_DIRS=""
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    harbor_log="${HARBOR_LOGS[$shard_idx]}"
    job_dir=$(grep "Results written to" "$harbor_log" 2>/dev/null | sed 's/.*Results written to \(jobs\/[^/]*\).*/\1/' | head -1)
    if [ -n "$job_dir" ]; then
        JOB_DIRS="$JOB_DIRS $job_dir"
    fi
done

echo "Job directories: $JOB_DIRS"
echo "Uploading traces to: $OUTPUT_REPO_ID"

# Upload traces using external script
LD_LIBRARY_PATH="" TRANSFORMERS_NO_TORCH=1 /scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3 "$TEACHER_SCRIPTS_DIR/upload_traces.py" "$OUTPUT_REPO_ID" $JOB_DIRS

EXPORT_EXIT=$?
if [ $EXPORT_EXIT -eq 0 ]; then
    echo "Expert traces uploaded successfully to: $OUTPUT_REPO_ID"
else
    echo "WARNING: Failed to upload expert traces (exit code: $EXPORT_EXIT)"
fi

echo ""
echo "=== Step 7: Cleanup ==="

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    if [ -n "${VLLM_PIDS[$shard_idx]}" ]; then
        kill "${VLLM_PIDS[$shard_idx]}" 2>/dev/null || true
    fi
done

for node in "${ALL_NODES_ARRAY[@]}"; do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null &
done
wait

conda deactivate || true

echo ""
echo "============================================"
if [ $FAILED -eq 0 ]; then
    echo "Experiment COMPLETED: $EXPERIMENT_NAME"
    echo "  Input: $INPUT_REPO_ID"
    echo "  Output: $OUTPUT_REPO_ID"
    echo "  All $NUM_SHARDS shards completed successfully!"
else
    echo "Experiment PARTIALLY FAILED: $EXPERIMENT_NAME"
    echo "  $FAILED out of $NUM_SHARDS shards failed"
fi
echo "============================================"

exit $FAILED
