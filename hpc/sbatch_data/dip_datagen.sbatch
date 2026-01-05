#!/bin/bash
set -euo pipefail

echo "=== Local Datagen (dip) ==="

WORKDIR="${DCFT:-${PROJECT_ROOT:-$PWD}}"
cd "$WORKDIR"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"

SCRIPT="${DATAGEN_SCRIPT:?DATAGEN_SCRIPT must be set}"
STAGE="${DATAGEN_STAGE:-tasks}"
CONFIG_PATH="${DATAGEN_CONFIG_PATH:-}"
TARGET_REPO="${DATAGEN_TARGET_REPO:-}"
INPUT_DIR="${DATAGEN_INPUT_DIR:-}"
OUTPUT_DIR="${DATAGEN_OUTPUT_DIR:-}"
EXTRA_ARGS="${DATAGEN_EXTRA_ARGS:-}"
TASK_TYPE="${DATAGEN_TASK_TYPE:-}"

if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

if [ "${DATAGEN_REQUIRE_ENDPOINT:-0}" = "1" ]; then
    echo "ERROR: DATAGEN_REQUIRE_ENDPOINT=1 is not supported for local execution." >&2
    exit 1
fi

CMD=(python "$SCRIPT" --stage "$STAGE")
if [ -n "$CONFIG_PATH" ]; then
    CMD+=(--engine-config "$CONFIG_PATH")
fi
if [ -n "$TARGET_REPO" ]; then
    CMD+=(--target-repo "$TARGET_REPO")
fi
if [ -n "$INPUT_DIR" ]; then
    CMD+=(--input-dir "$INPUT_DIR")
fi
if [ -n "$OUTPUT_DIR" ]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi
if [ -n "$TASK_TYPE" ]; then
    CMD+=(--task-type "$TASK_TYPE")
fi

if [ -n "$EXTRA_ARGS" ]; then
    # shellcheck disable=SC2086
    CMD+=($EXTRA_ARGS)
fi

PY_BIN="${PYTHON_EXECUTABLE:-python3}"
CMD[0]="$PY_BIN"

echo "Running datagen locally: ${CMD[*]}"
"${CMD[@]}"
