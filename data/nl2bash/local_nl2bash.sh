#!/bin/bash
set -euo pipefail

CONDA_ENV="llama-factory"
SECRETS_PATH="/Users/benjaminfeuer/Documents/secrets.env"

if [[ -f "$SECRETS_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$SECRETS_PATH"
else
  echo "Warning: secrets file not found at $SECRETS_PATH" >&2
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Error: conda command not found. Please ensure Conda is installed." >&2
  exit 1
fi

PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV")

NUM_TASKS=${1:-100}
TASK_MAX_CONCURRENT=${2:-1}
TEACHER_N_CONCURRENT=${3:-1}

if ! [[ "$NUM_TASKS" =~ ^[0-9]+$ ]] || (( NUM_TASKS <= 0 )); then
  echo "Error: number of tasks must be a positive integer" >&2
  exit 1
fi

if ! [[ "$TASK_MAX_CONCURRENT" =~ ^[0-9]+$ ]] || (( TASK_MAX_CONCURRENT <= 0 )); then
  echo "Error: task concurrency must be a positive integer" >&2
  exit 1
fi

if ! [[ "$TEACHER_N_CONCURRENT" =~ ^[0-9]+$ ]] || (( TEACHER_N_CONCURRENT <= 0 )); then
  echo "Error: teacher concurrency must be a positive integer" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ROOT="$PROJECT_ROOT/data/nl2bash/local_runs"
OUTPUT_ROOT="$RUN_ROOT/tasks"
JOBS_DIR="$RUN_ROOT/jobs"

mkdir -p "$JOBS_DIR" "$OUTPUT_ROOT"

"${PYTHON_CMD[@]}" python -m data.nl2bash.run_batch \
  --num-tasks "$NUM_TASKS" \
  --first-seed 0 \
  --dataset-prefix-base "nl2bash" \
  --split train \
  --output-root "$OUTPUT_ROOT" \
  --jobs-dir "$JOBS_DIR" \
  --task-max-concurrent "$TASK_MAX_CONCURRENT" \
  --teacher-n-concurrent "$TEACHER_N_CONCURRENT" \
  --teacher-agent terminus-2 \
  --teacher-agent-model gpt-5-mini \
  --teacher-agent-kwargs '{"max_episodes": 8}' \
  --teacher-timeout-multiplier 0.2
