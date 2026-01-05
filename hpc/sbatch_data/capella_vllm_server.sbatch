#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --exclusive
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -euo pipefail

if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/zih_capella.env" ]; then
    # shellcheck disable=SC1090
    source "$DCFT/hpc/dotenv/zih_capella.env"
elif [ -n "${DC_AGENT:-}" ] && [ -f "$DC_AGENT/hpc/dotenv/zih_capella.env" ]; then
    # shellcheck disable=SC1090
    source "$DC_AGENT/hpc/dotenv/zih_capella.env"
fi

module load CUDA/12.8.0

if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
    set +u
    eval "$DCFT_ACTIVATE_ENV"
    set -u
fi

SECRET_FILE="${DC_AGENT_SECRET_ENV:-${KEYS:-}}"
if [[ -n "${SECRET_FILE}" && -f "${SECRET_FILE}" ]]; then
    echo "Sourcing secrets from ${SECRET_FILE}"
    set -a
    # shellcheck disable=SC1090
    source "${SECRET_FILE}"
    set +a
fi

export PYTHONFAULTHANDLER=1
export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/tmp/.outlines}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/triton_cache}"

mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"

cd "${DCFT_PRIVATE:-$DCFT}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BACKEND="${VLLM_BACKEND:-vllm}"

MODEL_PATH="${VLLM_MODEL_PATH:?VLLM_MODEL_PATH is required}"
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
PP_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-1}"
DP_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
CUSTOM_MODEL_NAME="${VLLM_CUSTOM_MODEL_NAME:-}"
ENDPOINT_JSON="${VLLM_ENDPOINT_JSON_PATH:-$PWD/vllm_endpoint.json}"
RAY_PORT="${VLLM_RAY_PORT:-6379}"
API_PORT="${VLLM_API_PORT:-8000}"
GPUS_PER_NODE="${VLLM_GPUS_PER_NODE:-{gpus_per_node}}"
export VLLM_GPUS_PER_NODE="$GPUS_PER_NODE"

# Ray GPU visibility:
# vLLM's Ray executor expects CUDA ordinals to match Ray's GPU ids. On some
# clusters, Ray's default CUDA_VISIBLE_DEVICES rewriting can remap ordinals and
# trigger "CUDA error: invalid device ordinal" inside vLLM workers.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_NOSET_CUDA_VISIBLE_DEVICES="${RAY_NOSET_CUDA_VISIBLE_DEVICES:-1}"

SRUN_EXPORT_ENV="ALL,PATH=$PATH,LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-},PYTHONPATH=$PYTHONPATH,HF_HOME=$HF_HOME,RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=$RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES,RAY_NOSET_CUDA_VISIBLE_DEVICES=$RAY_NOSET_CUDA_VISIBLE_DEVICES"
RAY_ENV_VARS="PATH=$PATH LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} PYTHONPATH=$PYTHONPATH HF_HOME=$HF_HOME RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=$RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES RAY_NOSET_CUDA_VISIBLE_DEVICES=$RAY_NOSET_CUDA_VISIBLE_DEVICES"

if [ "$BACKEND" = "ray" ]; then

echo "=== Ray-backed vLLM Server Configuration (Capella) ==="
echo "Model: $MODEL_PATH"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Pipeline Parallel Size: $PP_SIZE"
echo "Data Parallel Size: $DP_SIZE"
echo "Custom Model Name: ${CUSTOM_MODEL_NAME:-<none>}"
echo "Endpoint JSON: $ENDPOINT_JSON"
echo "Ray Port: $RAY_PORT"
echo "API Port: $API_PORT"
echo "GPUs per node: $GPUS_PER_NODE"
echo "==========================================="

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}

head_node_ip=$(srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" hostname --ip-address)
head_node_ip=${head_node_ip%% *}

ip_head="${head_node_ip}:${RAY_PORT}"
export RAY_ADDRESS="$ip_head"

ray_pids=()

start_ray_node() {
    local node_name=$1
    local start_cmd=$2
    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$node_name" bash -c "env $RAY_ENV_VARS $start_cmd" &
    ray_pids+=($!)
}

head_cmd="ray start --head --node-ip-address=${head_node_ip} --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --num-cpus={cpus_per_node} --block"
echo "Starting Ray head on ${head_node} (${head_node_ip})"
start_ray_node "$head_node" "$head_cmd"

for ((i = 1; i < SLURM_JOB_NUM_NODES; i++)); do
    node_i=${nodes_array[$i]}
    worker_cmd="ray start --address ${ip_head} --num-gpus=${GPUS_PER_NODE} --num-cpus={cpus_per_node} --block"
    echo "Starting Ray worker on ${node_i}"
    start_ray_node "$node_i" "$worker_cmd"
    sleep 3
done

cleanup() {
    echo "Stopping vLLM server and Ray cluster..."
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" >/dev/null 2>&1 || true
        wait "$VLLM_PID" >/dev/null 2>&1 || true
    fi
    for node in "${nodes_array[@]}"; do
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$node" ray stop --force >/dev/null 2>&1 || true
    done
    for pid in "${ray_pids[@]}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT

TOTAL_GPUS=$(python3 - <<'PY'
import os
print(int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) * int(os.environ.get("VLLM_GPUS_PER_NODE", "0")))
PY
)

python3 scripts/ray/wait_for_cluster.py \
    --address "$ip_head" \
    --expected-gpus "$TOTAL_GPUS" \
    --expected-nodes "$SLURM_JOB_NUM_NODES" \
    --timeout 600 \
    --poll-interval 10

controller_log="{experiments_dir}/logs/{job_name}_controller.log"
echo "Launching vLLM API server on ${head_node_ip}:${API_PORT}"

CONTROLLER_ENV_CMD="env PATH=$PATH LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} PYTHONPATH=$PYTHONPATH HF_HOME=$HF_HOME"
srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
    $CONTROLLER_ENV_CMD \
        python3 scripts/vllm/start_vllm_ray_controller.py \
            --ray-address "$ip_head" \
            --host "$head_node_ip" \
            --port "$API_PORT" \
            --endpoint-json "$ENDPOINT_JSON" \
            --tensor-parallel-size "$TP_SIZE" \
            --pipeline-parallel-size "$PP_SIZE" \
            --data-parallel-size "$DP_SIZE" \
        >> "$controller_log" 2>&1 &
VLLM_PID=$!

python3 scripts/vllm/wait_for_endpoint.py \
    --endpoint-json "$ENDPOINT_JSON" \
    --max-attempts "${VLLM_HEALTH_MAX_ATTEMPTS:-120}" \
    --retry-delay "${VLLM_HEALTH_RETRY_DELAY:-15}" \
    --health-path "v1/models"

echo "=== vLLM Ray server is ready ==="
echo "Endpoint JSON: $ENDPOINT_JSON"
echo "Ray address: $RAY_ADDRESS"
echo "Controller log: $controller_log"

wait "$VLLM_PID"
exit $?
fi

echo "=== Local vLLM Server Configuration (Capella) ==="
echo "Model: $MODEL_PATH"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Pipeline Parallel Size: $PP_SIZE"
echo "Custom Model Name: ${CUSTOM_MODEL_NAME:-<none>}"
echo "Endpoint JSON: $ENDPOINT_JSON"
HF_OVERRIDES="${VLLM_HF_OVERRIDES:-}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-}"
SWAP_SPACE="${VLLM_SWAP_SPACE:-}"
MAX_SEQ_LEN_TO_CAPTURE="${VLLM_MAX_SEQ_LEN_TO_CAPTURE:-}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-}"
DISABLE_LOG_REQUESTS="${VLLM_DISABLE_LOG_REQUESTS:-}"
CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}"
KV_OFFLOADING_SIZE="${VLLM_KV_OFFLOADING_SIZE:-}"
KV_OFFLOADING_BACKEND="${VLLM_KV_OFFLOADING_BACKEND:-}"

CMD=(python3 scripts/vllm/start_vllm_cluster.py
    --model "$MODEL_PATH"
    --num-replicas "${VLLM_NUM_REPLICAS:-1}"
    --tensor-parallel-size "$TP_SIZE"
    --pipeline-parallel-size "$PP_SIZE"
    --enable-pinggy
    --pinggy-output-path "$ENDPOINT_JSON"
    --verbose
)

if [ -n "$CUSTOM_MODEL_NAME" ]; then
    CMD+=(--custom-model-name "$CUSTOM_MODEL_NAME")
fi
if [ -n "$HF_OVERRIDES" ]; then
    CMD+=(--hf-overrides "$HF_OVERRIDES")
fi
if [ -n "$MAX_NUM_SEQS" ]; then
    CMD+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
    CMD+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
if [ "$ENABLE_EXPERT_PARALLEL" = "1" ]; then
    CMD+=(--enable-expert-parallel)
fi
if [ -n "$SWAP_SPACE" ]; then
    CMD+=(--swap-space "$SWAP_SPACE")
fi
if [ -n "$CPU_OFFLOAD_GB" ]; then
    CMD+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi
if [ -n "$KV_OFFLOADING_SIZE" ]; then
    CMD+=(--kv-offloading-size "$KV_OFFLOADING_SIZE")
fi
if [ -n "$KV_OFFLOADING_BACKEND" ]; then
    CMD+=(--kv-offloading-backend "$KV_OFFLOADING_BACKEND")
fi
if [ -n "$MAX_SEQ_LEN_TO_CAPTURE" ]; then
    CMD+=(--max-seq-len-to-capture "$MAX_SEQ_LEN_TO_CAPTURE")
fi
if [ -n "$MAX_MODEL_LEN" ]; then
    CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [ "$TRUST_REMOTE_CODE" = "1" ]; then
    CMD+=(--trust-remote-code)
fi
if [ "$DISABLE_LOG_REQUESTS" = "1" ]; then
    CMD+=(--disable-log-requests)
fi

echo "Launching standalone vLLM server:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}" 2>&1 | tee {experiments_dir}/logs/{job_name}_standalone_%j.log
