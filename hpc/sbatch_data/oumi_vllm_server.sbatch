#!/bin/bash
# OUMI: VLLM server with optional Pinggy tunnel
#SBATCH --time={time_limit}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gpus-per-node={num_gpus}
#SBATCH --mem=512GB
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}

set -euo pipefail

# Activate environment (configured in dotenv/oumi.env)
activated_env=0
if [ -n "${VLLM_ACTIVATE_ENV:-}" ]; then
  set +u
  if eval "$VLLM_ACTIVATE_ENV"; then
    activated_env=1
  else
    echo "Warning: VLLM_ACTIVATE_ENV failed; attempting DCFT_ACTIVATE_ENV." >&2
  fi
  set -u
fi

if [ "${activated_env}" -ne 1 ] && [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  if eval "$DCFT_ACTIVATE_ENV"; then
    activated_env=1
  else
    echo "Warning: DCFT_ACTIVATE_ENV failed; continuing without environment activation." >&2
  fi
  set -u
fi

if [ "${activated_env}" -ne 1 ]; then
  echo "Warning: No environment activation command succeeded." >&2
fi

# Light NCCL/network defaults
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"

# Project directory
cd "$DCFT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# VLLM parameters from environment (set by launcher)
MODEL_PATH="${VLLM_MODEL_PATH}"
NUM_REPLICAS="${VLLM_NUM_REPLICAS:-1}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
PIPELINE_PARALLEL_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-1}"
CUSTOM_MODEL_NAME="${VLLM_CUSTOM_MODEL_NAME:-}"
ENDPOINT_JSON_PATH="${VLLM_ENDPOINT_JSON_PATH:-$DCFT/vllm_endpoint.json}"
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

echo "=== VLLM Server (OUMI) ==="
echo "Model: $MODEL_PATH"
echo "Replicas: $NUM_REPLICAS"
echo "TP Size: $TENSOR_PARALLEL_SIZE"
echo "PP Size: $PIPELINE_PARALLEL_SIZE"
echo "Custom Name: $CUSTOM_MODEL_NAME"
echo "Endpoint JSON: $ENDPOINT_JSON_PATH"
if [ -n "$HF_OVERRIDES" ]; then
  echo "HF Overrides: $HF_OVERRIDES"
fi
if [ "${VLLM_USE_DEEP_GEMM:-0}" = "1" ]; then
  export VLLM_USE_DEEP_GEMM=1
  echo "DeepGEMM: enabled"
else
  unset VLLM_USE_DEEP_GEMM
  echo "DeepGEMM: disabled"
fi
if [ -n "$MAX_NUM_SEQS" ]; then
  echo "Max num seqs: $MAX_NUM_SEQS"
fi
if [ -n "$GPU_MEMORY_UTILIZATION" ]; then
  echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
fi
if [ "$ENABLE_EXPERT_PARALLEL" = "1" ]; then
  echo "Expert parallel: enabled"
else
  echo "Expert parallel: disabled"
fi
if [ -n "$SWAP_SPACE" ]; then
  echo "Swap space: $SWAP_SPACE GB"
fi
if [ -n "$MAX_SEQ_LEN_TO_CAPTURE" ]; then
  echo "Max seq len to capture: $MAX_SEQ_LEN_TO_CAPTURE"
fi
if [ -n "$MAX_MODEL_LEN" ]; then
  echo "Max model len: $MAX_MODEL_LEN"
fi
if [ -n "$CPU_OFFLOAD_GB" ]; then
  echo "CPU offload: ${CPU_OFFLOAD_GB} GiB"
fi
if [ -n "$KV_OFFLOADING_SIZE" ]; then
  echo "KV offloading size: ${KV_OFFLOADING_SIZE} GiB (backend: ${KV_OFFLOADING_BACKEND:-<default>})"
fi
if [ "$TRUST_REMOTE_CODE" = "1" ]; then
  echo "Trust remote code: enabled"
else
  echo "Trust remote code: disabled"
fi
if [ "$DISABLE_LOG_REQUESTS" = "1" ]; then
  echo "Request logging: disabled"
else
  echo "Request logging: enabled"
fi

CMD=(python3 scripts/vllm/start_vllm_cluster.py
  --model "$MODEL_PATH"
  --num-replicas "$NUM_REPLICAS"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
  --enable-pinggy
  --pinggy-output-path "$ENDPOINT_JSON_PATH"
  --verbose
)

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

"${CMD[@]}" 2>&1 | tee {experiments_dir}/logs/vllm_server_%j.log
