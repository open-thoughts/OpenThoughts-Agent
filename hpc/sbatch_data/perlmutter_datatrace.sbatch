#!/bin/bash
# Perlmutter trace generation template.
#SBATCH --account={account}
#SBATCH --qos=regular
#SBATCH --constraint="gpu&hbm80g"
#SBATCH --exclusive
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}

set -euo pipefail

# Guard conda deactivate scripts from set -u complaints.
export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CONDA_BACKUP_CC="${CONDA_BACKUP_CC:-}"
export CONDA_BACKUP_FC="${CONDA_BACKUP_FC:-}"

if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/perlmutter.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/perlmutter.env"
fi

if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  eval "$DCFT_ACTIVATE_ENV"
fi

if [ -f "$HOME/secrets.env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/secrets.env"
fi

module load cuda/12.9
module load cudatoolkit/12.9

if [ -n "${LIBRARY_PATH:-}" ]; then
  export LIBRARY_PATH="$CUDA_HOME/lib64:$LIBRARY_PATH"
else
  export LIBRARY_PATH="$CUDA_HOME/lib64"
fi

if [ -n "${CONDA_PREFIX:-}" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export FC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gfortran"
  export CUDAHOSTCXX="$CXX"
  export PATH="$CONDA_PREFIX/bin:$PATH"
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
  else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
  fi
else
  echo "Warning: CONDA_PREFIX is not set; compiler environment variables were not configured." >&2
fi

cd "$DCFT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

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
TRACE_ENDPOINT_JSON="${TRACE_ENDPOINT_JSON:-}"
TRACE_REQUIRE_ENDPOINT="${TRACE_REQUIRE_ENDPOINT:-0}"
TRACE_WAIT_FOR_ENDPOINT="${TRACE_WAIT_FOR_ENDPOINT:-0}"
TRACE_HARBOR_CONFIG="${TRACE_HARBOR_CONFIG:-}"
TRACE_DISABLE_VERIFICATION="${TRACE_DISABLE_VERIFICATION:-0}"
TRACE_EVAL_ONLY="${TRACE_EVAL_ONLY:-0}"
TRACE_AGENT_TIMEOUT_SEC="${TRACE_AGENT_TIMEOUT_SEC:-}"
TRACE_VERIFIER_TIMEOUT_SEC="${TRACE_VERIFIER_TIMEOUT_SEC:-}"
TRACE_ENGINE_CONFIG_PATH="${TRACE_ENGINE_CONFIG_PATH:-}"
TRACE_HEALTH_MAX_ATTEMPTS="${TRACE_HEALTH_MAX_ATTEMPTS:-20}"
TRACE_HEALTH_RETRY_DELAY="${TRACE_HEALTH_RETRY_DELAY:-60}"
TRACE_TASKS_READY_TIMEOUT="${TRACE_TASKS_READY_TIMEOUT:-1800}"
VLLM_JOB_ID="${VLLM_JOB_ID:-}"
TRACE_TASK_TYPE="${TRACE_TASK_TYPE:-}"
TRACE_INCLUDE_REASONING="${TRACE_INCLUDE_REASONING:-1}"

source "$DCFT/hpc/sbatch_data/trace_helpers.sh"
load_trace_chunk_env

trap 'if [ -n "$VLLM_JOB_ID" ]; then
  echo "Tearing down VLLM job $VLLM_JOB_ID"
  scancel "$VLLM_JOB_ID" >/dev/null 2>&1 || echo "Warning: failed to cancel VLLM job $VLLM_JOB_ID"
fi' EXIT

echo "=== Trace Generation (Perlmutter) ==="
echo "Trace Script: $TRACE_SCRIPT"
echo "Stage: $TRACE_STAGE"
echo "Tasks Input: $TRACE_TASKS_PATH"
echo "Target Repo: $TRACE_TARGET_REPO"
echo "Model: ${TRACE_MODEL:-<none>}"
echo "Output Dir: ${TRACE_OUTPUT_DIR:-<none>}"
echo "Engine: ${TRACE_ENGINE:-<none>}"
echo "Episodes: ${TRACE_EPISODES:-<none>}"
echo "Export Filter: ${TRACE_EXPORT_FILTER:-<none>}"
echo "Dataset Type: ${TRACE_DATASET_TYPE:-<none>}"
echo "Jobs Dir: ${TRACE_JOBS_DIR:-<none>}"
echo "Harbor Config: ${TRACE_HARBOR_CONFIG:-<none>}"
echo "Disable Verification: $TRACE_DISABLE_VERIFICATION"
echo "Eval Only: $TRACE_EVAL_ONLY"
echo "Engine Config: ${TRACE_ENGINE_CONFIG_PATH:-<none>}"
echo "Use GPU: $TRACE_USE_GPU"
echo "Task Type: ${TRACE_TASK_TYPE:-<none>}"
if [ "${TRACE_CHUNK_MODE:-0}" = "1" ]; then
echo "Chunk Index: ${TRACE_JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-?}} / ${TRACE_CHUNK_COUNT:-?}"
fi

ensure_tasks_ready_or_exit "$TRACE_TASKS_PATH" "$TRACE_TASKS_READY_TIMEOUT"

if [ "$TRACE_REQUIRE_ENDPOINT" = "1" ] && [ "$TRACE_WAIT_FOR_ENDPOINT" = "1" ]; then
  if [ -z "$TRACE_ENDPOINT_JSON" ]; then
    echo "ERROR: Trace endpoint required but path is empty." >&2
    exit 1
  fi

  echo "Waiting for trace endpoint JSON..."
  TRACE_MAX_WAIT=600
  TRACE_WAIT=0
  while [ ! -f "$TRACE_ENDPOINT_JSON" ] && [ $TRACE_WAIT -lt $TRACE_MAX_WAIT ]; do
    sleep 5
    TRACE_WAIT=$((TRACE_WAIT + 5))
    if [ $((TRACE_WAIT % 30)) -eq 0 ]; then
      echo "Still waiting for $TRACE_ENDPOINT_JSON... (${TRACE_WAIT}s elapsed)"
    fi
  done

  if [ ! -f "$TRACE_ENDPOINT_JSON" ]; then
    echo "ERROR: Trace endpoint JSON not found after ${TRACE_MAX_WAIT}s." >&2
    exit 1
  fi

  echo "✓ Trace endpoint JSON present"
  cat "$TRACE_ENDPOINT_JSON"

  python3 scripts/vllm/wait_for_endpoint.py \
    --endpoint-json "$TRACE_ENDPOINT_JSON" \
    --max-attempts "$TRACE_HEALTH_MAX_ATTEMPTS" \
    --retry-delay "$TRACE_HEALTH_RETRY_DELAY" \
    --health-path "v1/models"
  if [ $? -ne 0 ]; then
    echo "ERROR: Trace endpoint healthcheck failed"
    exit 1
  fi
else
  echo "Trace endpoint wait skipped (engine=$TRACE_ENGINE, wait_flag=$TRACE_WAIT_FOR_ENDPOINT)"
fi

CMD=(python3 "$TRACE_SCRIPT" --stage "$TRACE_STAGE" --target-repo "$TRACE_TARGET_REPO" --tasks-input "$TRACE_TASKS_PATH")

if [ -n "$TRACE_OUTPUT_DIR" ]; then
  mkdir -p "$TRACE_OUTPUT_DIR"
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
if [ "$TRACE_INCLUDE_REASONING" = "1" ]; then
  CMD+=(--trace-include-reasoning)
else
  CMD+=(--trace-skip-reasoning)
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

if [ -n "$TRACE_ENGINE_CONFIG_PATH" ]; then
  CMD+=(--engine-config "$TRACE_ENGINE_CONFIG_PATH")
fi

if [ "$TRACE_EVAL_ONLY" = "1" ]; then
  CMD+=(--trace-eval-only)
fi
if [ -n "$TRACE_TASK_TYPE" ]; then
  CMD+=(--task-type "$TRACE_TASK_TYPE")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Trace generation completed"
else
  echo "✗ Trace generation failed with exit code: $EXIT_CODE"
fi
exit $EXIT_CODE
