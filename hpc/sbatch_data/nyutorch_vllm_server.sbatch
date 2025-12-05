#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --constraint="h200"
#SBATCH --gpus-per-node={num_gpus}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -euo pipefail

export DCFT=/scratch/bf996/dcagent_scratch

# Explicit dotenv path to avoid brittle expansions
export DCFT_DOTENV=${DCFT_DOTENV:-/scratch/bf996/dc-agent/hpc/dotenv/nyutorch.env}
source "$DCFT_DOTENV"
if [ -f "$DCFT_PRIVATE/database/access.env" ]; then
    source "$DCFT_PRIVATE/database/access.env"
else
    echo "[nyutorch_vllm_server] database/access.env not found; continuing without DB creds" >&2
fi

export PYTHONFAULTHANDLER=1
export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/tmp/.outlines}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
export RUN_SINGULARITY=/scratch/bf996/dc-agent/hpc/scripts/run-singularity-torch.bash

if [ ! -x "$RUN_SINGULARITY" ]; then
    echo "Missing singularity launcher at $RUN_SINGULARITY" >&2
    exit 1
fi

mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"

MODEL_PATH="${VLLM_MODEL_PATH}"
NUM_REPLICAS="${VLLM_NUM_REPLICAS:-1}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-1}"
CUSTOM_MODEL_NAME="${VLLM_CUSTOM_MODEL_NAME:-}"
ENDPOINT_JSON_PATH="${VLLM_ENDPOINT_JSON_PATH:-$DCFT_PRIVATE/vllm_endpoint.json}"
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

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: VLLM_MODEL_PATH is required" >&2
    exit 1
fi

cat <<EOC
=== VLLM Server Configuration ===
Model: $MODEL_PATH
Replicas: $NUM_REPLICAS
Tensor Parallel Size: $TENSOR_PARALLEL_SIZE
Pipeline Parallel Size: $PIPELINE_PARALLEL_SIZE
Custom Model Name: ${CUSTOM_MODEL_NAME:-<none>}
Endpoint JSON: $ENDPOINT_JSON_PATH
HF Overrides: ${HF_OVERRIDES:-<none>}
Max num seqs: ${MAX_NUM_SEQS:-<default>}
GPU memory utilization: ${GPU_MEMORY_UTILIZATION:-<default>}
Expert parallel: $([ "$ENABLE_EXPERT_PARALLEL" = "1" ] && echo enabled || echo disabled)
Swap space: ${SWAP_SPACE:-<default>}
Max seq len to capture: ${MAX_SEQ_LEN_TO_CAPTURE:-<default>}
Max model len: ${MAX_MODEL_LEN:-<default>}
CPU offload (GiB): ${CPU_OFFLOAD_GB:-<none>}
KV offloading size (GiB): ${KV_OFFLOADING_SIZE:-<none>}
KV offloading backend: ${KV_OFFLOADING_BACKEND:-<default>}
Trust remote code: $([ "$TRUST_REMOTE_CODE" = "1" ] && echo enabled || echo disabled)
Request logging: $([ "$DISABLE_LOG_REQUESTS" = "1" ] && echo disabled || echo enabled)
==================================
EOC

CMD=(python3 scripts/vllm/start_vllm_cluster.py --model "$MODEL_PATH" --num-replicas "$NUM_REPLICAS" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE" --enable-pinggy --pinggy-output-path "$ENDPOINT_JSON_PATH" --verbose)
if [ -n "$CUSTOM_MODEL_NAME" ]; then
    CMD+=(--custom-model-name "$CUSTOM_MODEL_NAME")
fi
if [ -n "$HF_OVERRIDES" ]; then
    CMD+=(--hf-overrides "$HF_OVERRIDES")
fi
if [ -n "${VLLM_PINGGY_HEALTH_INTERVAL:-}" ]; then
    CMD+=(--pinggy-health-interval "$VLLM_PINGGY_HEALTH_INTERVAL")
fi
if [ -n "${VLLM_PINGGY_HEALTH_TIMEOUT:-}" ]; then
    CMD+=(--pinggy-health-timeout "$VLLM_PINGGY_HEALTH_TIMEOUT")
fi
if [ -n "${VLLM_PINGGY_MAX_RESTARTS:-}" ]; then
    CMD+=(--pinggy-max-restarts "$VLLM_PINGGY_MAX_RESTARTS")
fi
if [ -n "${VLLM_PINGGY_HEALTH_URL:-}" ]; then
    CMD+=(--pinggy-health-url "$VLLM_PINGGY_HEALTH_URL")
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

CMD_STR=$(printf '%q ' "${CMD[@]}")
CONTAINER_CMD="cd \"/scratch/bf996/dc-agent\" && export PYTHONPATH=\"/scratch/bf996/dc-agent:\\$PYTHONPATH\" && $CMD_STR"

srun --cpu_bind=v --accel-bind=v /bin/bash "$RUN_SINGULARITY" /bin/bash -c "$CONTAINER_CMD" 2>&1 | tee "{experiments_dir}/logs/vllm_server_${SLURM_JOB_ID}.log"

EXIT_CODE=${PIPESTATUS[0]}
if [ $EXIT_CODE -ne 0 ]; then
    echo "VLLM server exited with status $EXIT_CODE" >&2
fi
exit $EXIT_CODE
