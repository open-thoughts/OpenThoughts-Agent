#!/bin/bash
#SBATCH -A {account}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --constraint="h100"
#SBATCH --gres=gpu:{num_gpus}
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

export DCFT=/scratch/bf996/dcagent_scratch

DCFT_PRIVATE_CANDIDATE="${DCFT_PRIVATE:-}"
if [ -z "$DCFT_PRIVATE_CANDIDATE" ]; then
    if [ -d "$PWD/hpc/dotenv" ]; then
        DCFT_PRIVATE_CANDIDATE="$PWD"
    else
        DCFT_PRIVATE_CANDIDATE="/scratch/bf996/dc-agent"
    fi
fi
export DCFT_PRIVATE="$DCFT_PRIVATE_CANDIDATE"

source "$DCFT_PRIVATE/hpc/dotenv/nyugreene.env"
if [ -f "$DCFT_PRIVATE/database/access.env" ]; then
    source "$DCFT_PRIVATE/database/access.env"
else
    echo "[nyugreene_datatrace] database/access.env not found; continuing without DB creds" >&2
fi

export PYTHONFAULTHANDLER=1
export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
export RUN_SINGULARITY=/scratch/bf996/dc-agent/hpc/scripts/run-singularity.bash

if [ ! -f "$RUN_SINGULARITY" ]; then
    echo "Missing singularity launcher at $RUN_SINGULARITY" >&2
    exit 1
fi

mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"

TRACE_SCRIPT_PATH="${TRACE_SCRIPT}"
TRACE_STAGE="${TRACE_STAGE:-traces}"
TRACE_TASKS_PATH="${TRACE_TASKS_PATH}"
TRACE_TARGET_REPO="${TRACE_TARGET_REPO}"
TRACE_MODEL="${TRACE_MODEL:-}"
TRACE_ENGINE="${TRACE_ENGINE:-}"
TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-}"
TRACE_USE_GPU="${TRACE_USE_GPU:-0}"
TRACE_NUM_GPUS="${TRACE_NUM_GPUS:-0}"
TRACE_EPISODES="${TRACE_EPISODES:-}"
TRACE_EXPORT_FILTER="${TRACE_EXPORT_FILTER:-}"
TRACE_DATASET_TYPE="${TRACE_DATASET_TYPE:-}"
TRACE_JOBS_DIR="${TRACE_JOBS_DIR:-}"
TRACE_ENDPOINT_JSON="${TRACE_ENDPOINT_JSON:-}"
TRACE_REQUIRE_ENDPOINT="${TRACE_REQUIRE_ENDPOINT:-0}"
TRACE_WAIT_FOR_ENDPOINT="${TRACE_WAIT_FOR_ENDPOINT:-0}"
TRACE_HEALTH_MAX_ATTEMPTS="${TRACE_HEALTH_MAX_ATTEMPTS:-20}"
TRACE_HEALTH_RETRY_DELAY="${TRACE_HEALTH_RETRY_DELAY:-60}"
TRACE_HARBOR_CONFIG="${TRACE_HARBOR_CONFIG:-}"
TRACE_DISABLE_VERIFICATION="${TRACE_DISABLE_VERIFICATION:-0}"
TRACE_EVAL_ONLY="${TRACE_EVAL_ONLY:-0}"
TRACE_AGENT_TIMEOUT_SEC="${TRACE_AGENT_TIMEOUT_SEC:-}"
TRACE_VERIFIER_TIMEOUT_SEC="${TRACE_VERIFIER_TIMEOUT_SEC:-}"
TRACE_ENGINE_CONFIG_PATH="${TRACE_ENGINE_CONFIG_PATH:-}"
VLLM_JOB_ID="${VLLM_JOB_ID:-}"
TRACE_TASK_TYPE="${TRACE_TASK_TYPE:-}"
TRACE_INCLUDE_REASONING="${TRACE_INCLUDE_REASONING:-1}"

source "$DCFT/hpc/sbatch_data/trace_helpers.sh"
load_trace_chunk_env

cleanup_vllm() {{
  if [ -n "$VLLM_JOB_ID" ]; then
    echo "Tearing down VLLM job $VLLM_JOB_ID"
    scancel "$VLLM_JOB_ID" >/dev/null 2>&1 || echo "Warning: failed to cancel VLLM job $VLLM_JOB_ID"
  fi
}}
trap cleanup_vllm EXIT

if [ "${TRACE_CHUNK_MODE:-0}" = "1" ]; then
    echo "[nyugreene_datatrace] Chunk index ${TRACE_JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-?}} / ${TRACE_CHUNK_COUNT:-?}"
fi

ensure_tasks_ready_or_exit "$TRACE_TASKS_PATH" "${TRACE_TASKS_READY_TIMEOUT:-1800}"

if [ -n "$TRACE_OUTPUT_DIR" ]; then
    mkdir -p "$TRACE_OUTPUT_DIR"
fi

if [ "$TRACE_REQUIRE_ENDPOINT" = "1" ] && [ "$TRACE_WAIT_FOR_ENDPOINT" = "1" ]; then
    if [ -z "$TRACE_ENDPOINT_JSON" ]; then
        echo "ERROR: Trace endpoint required but path is empty" >&2
        exit 1
    fi

    echo "Waiting for trace endpoint JSON at $TRACE_ENDPOINT_JSON"
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
        echo "ERROR: Trace endpoint JSON not found after ${TRACE_MAX_WAIT}s" >&2
        exit 1
    fi

    echo "✓ Trace endpoint JSON present"

    /bin/bash "$RUN_SINGULARITY" /bin/bash -c "cd \"$DCFT_PRIVATE\" && python3 scripts/vllm/wait_for_endpoint.py --endpoint-json \"$TRACE_ENDPOINT_JSON\" --max-attempts \"$TRACE_HEALTH_MAX_ATTEMPTS\" --retry-delay \"$TRACE_HEALTH_RETRY_DELAY\" --health-path \"v1/models\""
fi

CMD=(python3 "$TRACE_SCRIPT_PATH" --stage "$TRACE_STAGE" --target-repo "$TRACE_TARGET_REPO" --tasks-input "$TRACE_TASKS_PATH")

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

CMD_STR=$(printf '%q ' "${CMD[@]}")
CONTAINER_CMD="cd \"/scratch/bf996/dc-agent\" && export PYTHONPATH=\"/scratch/bf996/dc-agent:\\$PYTHONPATH\" && $CMD_STR"

if [ "$TRACE_USE_GPU" = "1" ]; then
    if [ "$TRACE_NUM_GPUS" -le 0 ]; then
        echo "ERROR: GPU usage requested but TRACE_NUM_GPUS is 0" >&2
        exit 1
    fi
    echo "Running trace job with GPU access ($TRACE_NUM_GPUS GPUs)"
    srun --gpus-per-node="$TRACE_NUM_GPUS" --cpu_bind=v --accel-bind=v /bin/bash "$RUN_SINGULARITY" /bin/bash -c "$CONTAINER_CMD"
else
    echo "Running trace job without GPU access"
    srun /bin/bash "$RUN_SINGULARITY" /bin/bash -c "$CONTAINER_CMD"
fi

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================"
    echo "✓ Trace generation completed successfully"
    echo "======================================"
else
    echo "======================================"
    echo "✗ Trace generation failed with exit code: $EXIT_CODE"
    echo "======================================"
fi
exit $EXIT_CODE
