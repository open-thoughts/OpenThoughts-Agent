#!/bin/bash
set -euo pipefail

CONDA_ENV="ot-agent"
SECRETS_PATH="/Users/benjaminfeuer/Documents/secrets.env"

if [[ -f "$SECRETS_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$SECRETS_PATH"
else
  echo "Warning: secrets file not found at $SECRETS_PATH" >&2
fi

PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV")

NUM_TASKS=${1:-100}
TASK_MAX_CONCURRENT=${2:-1}
TEACHER_N_CONCURRENT=${3:-1}
SAMPLING_CONFIG=${4:-""}

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

# Auto-generate run name: YYMMDD_HHMM_{num_samples}
RUN_NAME="$(date +%y%m%d_%H%M)_${NUM_TASKS}"
RUN_ROOT="$PROJECT_ROOT/data/nl2bash_sampled_verified/runs/$RUN_NAME"

OUTPUT_ROOT="$RUN_ROOT/tasks"
JOBS_DIR="$RUN_ROOT/jobs"

# Clean up old tasks and validation results for this run
echo "Cleaning up old tasks in $RUN_ROOT..."
rm -rf "$OUTPUT_ROOT" "$RUN_ROOT/tasks_flat" "$RUN_ROOT/validation_jobs"

mkdir -p "$JOBS_DIR" "$OUTPUT_ROOT"

PYTHON_ARGS=(
  --num-tasks "$NUM_TASKS"
  --first-seed 0
  --dataset-prefix-base "nl2bash"
  --split train
  --output-root "$OUTPUT_ROOT"
  --jobs-dir "$JOBS_DIR"
  --task-max-concurrent "$TASK_MAX_CONCURRENT"
  --teacher-n-concurrent "$TEACHER_N_CONCURRENT"
  --teacher-agent terminus-2
  --teacher-agent-model gpt-5-mini
  --teacher-agent-kwargs '{"max_episodes": 8}'
  --teacher-timeout-multiplier 2.0
)

if [[ -n "$SAMPLING_CONFIG" && -f "$SAMPLING_CONFIG" ]]; then
  PYTHON_ARGS+=(--sampling-config "$SAMPLING_CONFIG")
fi

"${PYTHON_CMD[@]}" python -m data.nl2bash_sampled_verified.run_batch "${PYTHON_ARGS[@]}"

# Run validation if tasks were generated
if [ -d "$OUTPUT_ROOT" ] && [ "$(ls -A "$OUTPUT_ROOT")" ]; then
  echo ""
  echo "Running validation..."
  "$SCRIPT_DIR/validation/validate.sh" "$RUN_ROOT"

  # Analyze validation results
  echo ""
  echo "Analyzing validation results..."
  "${PYTHON_CMD[@]}" python "$SCRIPT_DIR/validation/analyze_results.py" "$RUN_ROOT/validation_jobs"

  # Extract verified tasks
  echo ""
  echo "Extracting verified tasks..."
  "${PYTHON_CMD[@]}" python "$SCRIPT_DIR/validation/extract_verified.py" "$RUN_ROOT"
fi
