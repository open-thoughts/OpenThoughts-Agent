#!/bin/bash
set -euo pipefail

echo "=== Local Trace Generation (dip) ==="

WORKDIR="${DCFT:-${PROJECT_ROOT:-$PWD}}"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

SCRIPT="${TRACE_SCRIPT:?TRACE_SCRIPT must be set}"
TASKS_PATH="${TRACE_TASKS_PATH:?TRACE_TASKS_PATH must be set}"
TARGET_REPO="${TRACE_TARGET_REPO:?TRACE_TARGET_REPO must be set}"
OUTPUT_DIR="${TRACE_OUTPUT_DIR:-}"
HARDCFG="${TRACE_HARBOR_CONFIG:-}"

source "$WORKDIR/hpc/sbatch_data/trace_helpers.sh"
load_trace_chunk_env
ensure_tasks_ready_or_exit "$TASKS_PATH" "${TRACE_TASKS_READY_TIMEOUT:-1800}"

if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

if [ "${TRACE_REQUIRE_ENDPOINT:-0}" = "1" ]; then
    echo "ERROR: TRACE_REQUIRE_ENDPOINT=1 is not supported for local execution." >&2
    exit 1
fi

CMD=("${PYTHON_EXECUTABLE:-python}" "$SCRIPT" --stage "${TRACE_STAGE:-traces}" --target-repo "$TARGET_REPO" --tasks-input "$TASKS_PATH")

if [ -n "$OUTPUT_DIR" ]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi
if [ -n "$HARDCFG" ]; then
    CMD+=(--trace-harbor-config "$HARDCFG")
fi
if [ -n "${TRACE_JOBS_DIR:-}" ]; then
    CMD+=(--trace-jobs-dir "$TRACE_JOBS_DIR")
fi
if [ -n "${TRACE_MODEL:-}" ]; then
    CMD+=(--trace-model "$TRACE_MODEL")
fi
if [ -n "${TRACE_EPISODES:-}" ]; then
    CMD+=(--trace-episodes "$TRACE_EPISODES")
fi
if [ -n "${TRACE_EXPORT_FILTER:-}" ]; then
    CMD+=(--trace-export-filter "$TRACE_EXPORT_FILTER")
fi
if [ -n "${TRACE_DATASET_TYPE:-}" ]; then
    CMD+=(--trace-dataset-type "$TRACE_DATASET_TYPE")
fi
if [ "${TRACE_EXPORT_SUBAGENTS:-1}" = "1" ]; then
    CMD+=(--trace-export-subagents)
else
    CMD+=(--trace-skip-subagents)
fi
if [ -n "${TRACE_ENGINE:-}" ]; then
    CMD+=(--engine "$TRACE_ENGINE")
fi
if [ -n "${TRACE_AGENT_TIMEOUT_SEC:-}" ]; then
    CMD+=(--trace-agent-timeout-sec "$TRACE_AGENT_TIMEOUT_SEC")
fi
if [ -n "${TRACE_VERIFIER_TIMEOUT_SEC:-}" ]; then
    CMD+=(--trace-verifier-timeout-sec "$TRACE_VERIFIER_TIMEOUT_SEC")
fi
if [ "$TRACE_DISABLE_VERIFICATION" = "1" ]; then
    CMD+=(--disable-verification)
fi
if [ -n "${TRACE_ENGINE_CONFIG_PATH:-}" ]; then
    CMD+=(--engine-config "$TRACE_ENGINE_CONFIG_PATH")
fi
if [ "$TRACE_EVAL_ONLY" = "1" ]; then
    CMD+=(--trace-eval-only)
fi
if [ -n "${TRACE_TASK_TYPE:-}" ]; then
    CMD+=(--task-type "$TRACE_TASK_TYPE")
fi
# Local execution cannot support SLURM chunk arrays.
if [ "${TRACE_CHUNK_MODE:-0}" = "1" ]; then
    echo "ERROR: TRACE chunking (--chunk_size) is not supported for local execution." >&2
    exit 1
fi

echo "Running trace locally: ${CMD[*]}"
"${CMD[@]}"
