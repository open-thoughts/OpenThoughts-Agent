#!/bin/bash
#SBATCH --account={account}
#SBATCH --qos=premium
#SBATCH --constraint="gpu&hbm80g"
#SBATCH --exclusive
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -euo pipefail

JOB_SUBMIT_ORDER=${JOB_SUBMIT_ORDER:-}
if [ -n "$JOB_SUBMIT_ORDER" ]; then
  if [[ "$JOB_SUBMIT_ORDER" =~ ^[0-9]+$ ]]; then
    SLEEP_DURATION=$((30 * JOB_SUBMIT_ORDER))
    echo "Submit order delay: sleeping ${SLEEP_DURATION}s (JOB_SUBMIT_ORDER=${JOB_SUBMIT_ORDER})"
    sleep "$SLEEP_DURATION"
  else
    echo "Warning: JOB_SUBMIT_ORDER='${JOB_SUBMIT_ORDER}' is not an integer; skipping launch delay"
  fi
fi

if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/perlmutter.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/perlmutter.env"
elif [ -n "${DC_AGENT:-}" ] && [ -f "$DC_AGENT/hpc/dotenv/perlmutter.env" ]; then
  # shellcheck disable=SC1090
  source "$DC_AGENT/hpc/dotenv/perlmutter.env"
fi

module load cudatoolkit/12.9

if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
fi

SECRET_FILE="${DC_AGENT_SECRET_ENV:-${KEYS:-}}"
if [[ -n "${SECRET_FILE}" ]]; then
    if [[ -f "${SECRET_FILE}" ]]; then
        echo "Sourcing secrets from ${SECRET_FILE}"
        set -a
        # shellcheck disable=SC1090
        source "${SECRET_FILE}"
        set +a
    else
        echo "Warning: secrets file ${SECRET_FILE} not found; trace uploads may fail." >&2
    fi
else
    echo "Warning: DC_AGENT_SECRET_ENV is not set; trace uploads may fail." >&2
fi

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/tmp/.outlines}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/triton_cache}"

if [ -z "${DCFT:-}" ]; then
  if [ -n "${DC_AGENT:-}" ]; then
    export DCFT="$DC_AGENT"
  else
    export DCFT="$PWD"
  fi
fi

mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"

cd "$DCFT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Ray GPU visibility:
# vLLM's Ray executor expects CUDA ordinals to match Ray's GPU ids. On some
# clusters, Ray's default CUDA_VISIBLE_DEVICES rewriting can remap ordinals and
# trigger "CUDA error: invalid device ordinal" inside vLLM workers.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_NOSET_CUDA_VISIBLE_DEVICES="${RAY_NOSET_CUDA_VISIBLE_DEVICES:-1}"

SRUN_EXPORT_ENV="ALL,PATH=$PATH,LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-},PYTHONPATH=$PYTHONPATH,HF_HOME=$HF_HOME,RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=$RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES,RAY_NOSET_CUDA_VISIBLE_DEVICES=$RAY_NOSET_CUDA_VISIBLE_DEVICES"
RAY_ENV_VARS="PATH=$PATH LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} PYTHONPATH=$PYTHONPATH HF_HOME=$HF_HOME RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=$RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES RAY_NOSET_CUDA_VISIBLE_DEVICES=$RAY_NOSET_CUDA_VISIBLE_DEVICES"

TRACE_BACKEND="${TRACE_BACKEND:-vllm}"

# Trace generation parameters
TRACE_SCRIPT="${TRACE_SCRIPT}"
TRACE_STAGE="${TRACE_STAGE:-traces}"
TRACE_TASKS_PATH="${TRACE_TASKS_PATH}"
TRACE_TARGET_REPO="${TRACE_TARGET_REPO}"
TRACE_MODEL="${TRACE_MODEL:-}"
TRACE_ENGINE="${TRACE_ENGINE:-}"
TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-}"
TRACE_USE_GPU="${TRACE_USE_GPU:-0}"
TRACE_EPISODES="${TRACE_EPISODES:-}"
TRACE_EXPORT_FILTER="${TRACE_EXPORT_FILTER:-}"
TRACE_DATASET_TYPE="${TRACE_DATASET_TYPE:-}"
TRACE_JOBS_DIR="${TRACE_JOBS_DIR:-}"
TRACE_ENDPOINT_JSON="${TRACE_ENDPOINT_JSON:-$DCFT/vllm_endpoint.json}"
TRACE_REQUIRE_ENDPOINT="${TRACE_REQUIRE_ENDPOINT:-0}"
TRACE_WAIT_FOR_ENDPOINT="${TRACE_WAIT_FOR_ENDPOINT:-0}"
TRACE_HARBOR_CONFIG="${TRACE_HARBOR_CONFIG:-}"
TRACE_DISABLE_VERIFICATION="${TRACE_DISABLE_VERIFICATION:-0}"
TRACE_EVAL_ONLY="${TRACE_EVAL_ONLY:-0}"
TRACE_AGENT_TIMEOUT_SEC="${TRACE_AGENT_TIMEOUT_SEC:-}"
TRACE_VERIFIER_TIMEOUT_SEC="${TRACE_VERIFIER_TIMEOUT_SEC:-}"
TRACE_ENGINE_CONFIG_PATH="${TRACE_ENGINE_CONFIG_PATH:-}"
VLLM_JOB_ID="${VLLM_JOB_ID:-}"
TRACE_TASK_TYPE="${TRACE_TASK_TYPE:-}"
TRACE_EXPORT_SUBAGENTS="${TRACE_EXPORT_SUBAGENTS:-1}"

source "$DCFT/hpc/sbatch_data/trace_helpers.sh"
load_trace_chunk_env

cleanup_vllm() {
  if [ -n "$VLLM_JOB_ID" ]; then
    echo "Tearing down VLLM job $VLLM_JOB_ID"
    scancel "$VLLM_JOB_ID" >/dev/null 2>&1 || echo "Warning: failed to cancel VLLM job $VLLM_JOB_ID"
  fi
}

trap cleanup_vllm EXIT

echo "=== Trace Generation Configuration ==="
echo "Backend: $TRACE_BACKEND"
echo "Trace Script: $TRACE_SCRIPT"
echo "Stage: $TRACE_STAGE"
echo "Tasks Input: $TRACE_TASKS_PATH"
echo "Target Repo: $TRACE_TARGET_REPO"
echo "Model: ${TRACE_MODEL:-<none>}"
echo "Output Dir: ${TRACE_OUTPUT_DIR:-<none>}"
echo "Engine: ${TRACE_ENGINE:-<none>}"
echo "Use GPU: $TRACE_USE_GPU"
echo "Task Type: ${TRACE_TASK_TYPE:-<none>}"
echo "Episodes: ${TRACE_EPISODES:-<none>}"
echo "Export Filter: ${TRACE_EXPORT_FILTER:-<none>}"
echo "Dataset Type: ${TRACE_DATASET_TYPE:-<none>}"
echo "Jobs Dir: ${TRACE_JOBS_DIR:-<none>}"
echo "Harbor Config: ${TRACE_HARBOR_CONFIG:-<none>}"
echo "Disable Verification: $TRACE_DISABLE_VERIFICATION"
echo "Eval Only: $TRACE_EVAL_ONLY"
if [ "${TRACE_CHUNK_MODE:-0}" = "1" ]; then
  echo "Chunk Index: ${TRACE_JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-?}} / ${TRACE_CHUNK_COUNT:-?}"
fi
echo "======================================"

if [ -f "$TRACE_ENDPOINT_JSON" ]; then
    echo "Removing stale endpoint JSON: $TRACE_ENDPOINT_JSON"
    rm -f "$TRACE_ENDPOINT_JSON"
fi

ensure_tasks_ready_or_exit "$TRACE_TASKS_PATH" "${TRACE_TASKS_READY_TIMEOUT:-1800}"

if [ -n "$TRACE_OUTPUT_DIR" ]; then
    mkdir -p "$TRACE_OUTPUT_DIR"
fi

CMD=(python3 "$TRACE_SCRIPT" --stage "$TRACE_STAGE" --target-repo "$TRACE_TARGET_REPO" --tasks-input "$TRACE_TASKS_PATH")

if [ -n "$TRACE_OUTPUT_DIR" ]; then
    CMD+=(--output-dir "$TRACE_OUTPUT_DIR")
fi
if [ -n "$TRACE_HARBOR_CONFIG" ]; then
    CMD+=(--trace-harbor-config "$TRACE_HARBOR_CONFIG")
fi
if [ -n "$TRACE_JOBS_DIR" ]; then
    CMD+=(--trace-jobs-dir "$TRACE_JOBS_DIR")
fi
if [ -n "$TRACE_MODEL" ]; then
    CMD+=(--trace-model "$TRACE_MODEL")
fi
if [ -n "$TRACE_EPISODES" ]; then
    CMD+=(--trace-episodes "$TRACE_EPISODES")
fi
if [ -n "$TRACE_EXPORT_FILTER" ]; then
    CMD+=(--trace-export-filter "$TRACE_EXPORT_FILTER")
fi
if [ -n "$TRACE_DATASET_TYPE" ]; then
    CMD+=(--trace-dataset-type "$TRACE_DATASET_TYPE")
fi
if [ "$TRACE_EXPORT_SUBAGENTS" = "1" ]; then
    CMD+=(--trace-export-subagents)
else
    CMD+=(--trace-skip-subagents)
fi
if [ -n "$TRACE_ENGINE" ]; then
    CMD+=(--engine "$TRACE_ENGINE")
fi
if [ -n "$TRACE_AGENT_TIMEOUT_SEC" ]; then
  CMD+=(--trace-agent-timeout-sec "$TRACE_AGENT_TIMEOUT_SEC")
fi
if [ -n "$TRACE_VERIFIER_TIMEOUT_SEC" ]; then
  CMD+=(--trace-verifier-timeout-sec "$TRACE_VERIFIER_TIMEOUT_SEC")
fi
if [ "$TRACE_DISABLE_VERIFICATION" = "1" ]; then
    CMD+=(--disable-verification)
fi
if [ "$TRACE_EVAL_ONLY" = "1" ]; then
    CMD+=(--trace-eval-only)
fi
if [ -n "$TRACE_ENGINE_CONFIG_PATH" ]; then
    CMD+=(--engine-config "$TRACE_ENGINE_CONFIG_PATH")
fi
if [ -n "$TRACE_AGENT_NAME" ]; then
    CMD+=(--trace-agent-name "$TRACE_AGENT_NAME")
fi
if [ -n "$TRACE_AGENT_KWARGS" ]; then
    CMD+=(--trace-agent-kwargs "$TRACE_AGENT_KWARGS")
fi
if [ -n "$TRACE_ENV" ]; then
    CMD+=(--trace-env "$TRACE_ENV")
fi
if [ -n "$TRACE_N_CONCURRENT" ]; then
    CMD+=(--trace-n-concurrent "$TRACE_N_CONCURRENT")
fi
if [ -n "$TRACE_CHUNK_SIZE" ]; then
    CMD+=(--chunk_size "$TRACE_CHUNK_SIZE")
fi
if [ -n "$TRACE_TASK_TYPE" ]; then
    CMD+=(--trace-task-type "$TRACE_TASK_TYPE")
fi
if [ "$TRACE_USE_GPU" = "1" ]; then
    CMD+=(--trace-use-gpu)
fi
if [ "$TRACE_BACKEND" = "vllm" ]; then
    CMD+=(--trace-backend vllm)
fi
if [ -n "$TRACE_ENDPOINT_JSON" ]; then
    CMD+=(--endpoint-json "$TRACE_ENDPOINT_JSON")
fi
if [ "$TRACE_REQUIRE_ENDPOINT" = "1" ]; then
    CMD+=(--trace-require-endpoint)
fi
if [ "$TRACE_WAIT_FOR_ENDPOINT" = "1" ]; then
    CMD+=(--trace-wait-for-endpoint)
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"
