#!/bin/bash
# OUMI: Data generation job that consumes VLLM endpoint JSON
#SBATCH --time={time_limit}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --mem=64GB
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}

set -euo pipefail

# Activate environment (configure in dotenv/oumi.env)
if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  eval "$DCFT_ACTIVATE_ENV"
fi

cd "$DCFT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Parameters provided via environment by launcher
GENERATE_SCRIPT="${DATAGEN_SCRIPT}"
ENDPOINT_JSON="${DATAGEN_ENDPOINT_JSON:-}"
TARGET_REPO="${DATAGEN_TARGET_REPO:-}"
INPUT_DIR="${DATAGEN_INPUT_DIR:-}"
EXTRA_ARGS="${DATAGEN_EXTRA_ARGS:-}"
ENGINE="${DATAGEN_ENGINE:-vllm_local}"
REQUIRE_ENDPOINT="${DATAGEN_REQUIRE_ENDPOINT:-0}"
OUTPUT_DIR="${DATAGEN_OUTPUT_DIR:-}"
STAGE="${DATAGEN_STAGE:-tasks}"
TASK_TYPE="${DATAGEN_TASK_TYPE:-}"
CONFIG_PATH="${DATAGEN_CONFIG_PATH:?DATAGEN_CONFIG_PATH is required}"

echo "=== DataGen (OUMI) ==="
echo "Script: $GENERATE_SCRIPT"
echo "Endpoint: ${ENDPOINT_JSON:-<none>}"
echo "Target Repo: $TARGET_REPO"
echo "Input Dir: $INPUT_DIR"
echo "Output Dir: ${OUTPUT_DIR:-<none>}"
echo "Engine: $ENGINE"
echo "Config Path: $CONFIG_PATH"
echo "Stage: $STAGE"
echo "Task Type: ${TASK_TYPE:-<none>}"

WAIT_FOR_ENDPOINT="${DATAGEN_WAIT_FOR_ENDPOINT:-0}"
if [ "$REQUIRE_ENDPOINT" = "1" ] && [ "$WAIT_FOR_ENDPOINT" = "1" ]; then
  if [ -z "$ENDPOINT_JSON" ]; then
    echo "ERROR: Endpoint required but path is empty"
    exit 1
  fi

  MAX_WAIT=3600
  WAITED=0
  while [ ! -f "$ENDPOINT_JSON" ] && [ $WAITED -lt $MAX_WAIT ]; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
      echo "Waiting for $ENDPOINT_JSON... (${WAITED}s)"
    fi
  done

  if [ ! -f "$ENDPOINT_JSON" ]; then
    echo "ERROR: Endpoint JSON not found after ${MAX_WAIT}s: $ENDPOINT_JSON"
    exit 1
  fi

  echo "✓ Endpoint JSON present"
  cat "$ENDPOINT_JSON"
else
  echo "Skipping endpoint wait (engine=$ENGINE, wait_flag=$WAIT_FOR_ENDPOINT)"
fi

CMD=(python3 "$GENERATE_SCRIPT" --stage "$STAGE" --engine-config "$CONFIG_PATH")

if [ -n "$TARGET_REPO" ]; then
  CMD+=(--target-repo "$TARGET_REPO")
fi

if [ -n "$INPUT_DIR" ]; then
  CMD+=(--input-dir "$INPUT_DIR")
fi

if [ -n "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
  CMD+=(--output-dir "$OUTPUT_DIR")
fi

if [ -n "$TASK_TYPE" ]; then
  CMD+=(--task-type "$TASK_TYPE")
fi

if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "✓ Data generation completed. Target repo: $TARGET_REPO"
else
  echo "✗ Data generation failed with code $EXIT_CODE"
fi
exit $EXIT_CODE
