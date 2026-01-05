#!/bin/bash
#SBATCH -A {account}
#SBATCH -p batch
#SBATCH -q {qos}
#SBATCH --exclusive
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -euo pipefail

frontier_env_loaded=0
if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/frontier.env" ]; then
    # shellcheck disable=SC1090
    source "$DCFT/hpc/dotenv/frontier.env"
    frontier_env_loaded=1
elif [ -n "${DC_AGENT:-}" ] && [ -f "$DC_AGENT/hpc/dotenv/frontier.env" ]; then
    # shellcheck disable=SC1090
    source "$DC_AGENT/hpc/dotenv/frontier.env"
    frontier_env_loaded=1
fi

# Ensure DCFT points to a valid checkout even if SCRATCH is unavailable.
DEFAULT_OT_ROOT="${OPEN_THOUGHTS_AGENT_ROOT:-$HOME/OpenThoughts-Agent}"
if [ -z "${DCFT:-}" ] && [ -d "$DEFAULT_OT_ROOT" ]; then
    export DCFT="$DEFAULT_OT_ROOT"
elif [ -n "${DCFT:-}" ] && [ ! -d "$DCFT" ] && [ -d "$DEFAULT_OT_ROOT" ]; then
    echo "DCFT path $DCFT not found; falling back to $DEFAULT_OT_ROOT"
    export DCFT="$DEFAULT_OT_ROOT"
fi

if [ "$frontier_env_loaded" -eq 0 ] && [ -f "$DCFT/hpc/dotenv/frontier.env" ]; then
    # shellcheck disable=SC1090
    source "$DCFT/hpc/dotenv/frontier.env"
fi

# Frontier jobs should always use the NVMe-backed cache paths.
if [ -n "${FRONTIER_NVME_HF_HOME:-}" ]; then
    export HF_HOME="$FRONTIER_NVME_HF_HOME"
fi
if [ -n "${FRONTIER_NVME_HF_HUB_CACHE:-}" ]; then
    export HF_HUB_CACHE="$FRONTIER_NVME_HF_HUB_CACHE"
fi

module load PrgEnv-amd
module load amd/6.4.0
module load rocm/6.4.0
module load craype-accel-amd-gfx90a
module load cray-mpich

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

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [ -n "${LIBRARY_PATH:-}" ]; then
    export LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LIBRARY_PATH"
else
    export LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64"
fi

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64"
fi

export PATH="$ROCM_PATH/bin:$PATH"
export MPICH_GPU_SUPPORT_ENABLED=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-90a}

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

echo "=== Ray-backed vLLM Server Configuration (Frontier) ==="
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

echo "=== Local vLLM Server Configuration (Frontier) ==="
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
NUM_SCHEDULER_STEPS="${VLLM_NUM_SCHEDULER_STEPS:-}"

CMD=(python3 scripts/vllm/start_vllm_local_controller.py
     --model "$MODEL_PATH"
     --tensor-parallel-size "$TP_SIZE"
     --pipeline-parallel-size "$PP_SIZE"
     --endpoint-json "$ENDPOINT_JSON")

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
if [ -n "$ENABLE_EXPERT_PARALLEL" ]; then
    CMD+=(--enable-expert-parallel "$ENABLE_EXPERT_PARALLEL")
fi
if [ -n "$SWAP_SPACE" ]; then
    CMD+=(--swap-space "$SWAP_SPACE")
fi
if [ -n "$MAX_SEQ_LEN_TO_CAPTURE" ]; then
    CMD+=(--max-seq-len-to-capture "$MAX_SEQ_LEN_TO_CAPTURE")
fi
if [ -n "$MAX_MODEL_LEN" ]; then
    CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [ -n "$TRUST_REMOTE_CODE" ]; then
    CMD+=(--trust-remote-code "$TRUST_REMOTE_CODE")
fi
if [ -n "$DISABLE_LOG_REQUESTS" ]; then
    CMD+=(--disable-log-requests "$DISABLE_LOG_REQUESTS")
fi
if [ -n "$CPU_OFFLOAD_GB" ]; then
    CMD+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
fi
if [ -n "$NUM_SCHEDULER_STEPS" ]; then
    CMD+=(--num-scheduler-steps "$NUM_SCHEDULER_STEPS")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"
