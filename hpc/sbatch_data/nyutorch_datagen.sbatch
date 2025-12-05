#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --constraint="h200"
#SBATCH --gres=gpu:{num_gpus}
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
    echo "[nyutorch_datagen] database/access.env not found; continuing without DB creds" >&2
fi

export PYTHONFAULTHANDLER=1
export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
export RUN_SINGULARITY=/scratch/bf996/dc-agent/hpc/scripts/run-singularity-torch.bash

if [ ! -f "$RUN_SINGULARITY" ]; then
    echo "Missing singularity launcher at $RUN_SINGULARITY" >&2
    exit 1
fi

mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"

ENGINE=${DATAGEN_ENGINE:-}
GPU_REQUIRED=${DATAGEN_GPU_REQUIRED:-0}
REQUESTED_GPUS=${DATAGEN_GPUS_PER_NODE:-0}
ENDPOINT_PATH=${DATAGEN_ENDPOINT_JSON:-}
REQUIRE_ENDPOINT=${DATAGEN_REQUIRE_ENDPOINT:-0}
WAIT_FOR_ENDPOINT=${DATAGEN_WAIT_FOR_ENDPOINT:-0}
OUTPUT_DIR=${DATAGEN_OUTPUT_DIR:-}
CONFIG_PATH=${DATAGEN_CONFIG_PATH:?DATAGEN_CONFIG_PATH is required}
DATAGEN_TARGET_REPO=${DATAGEN_TARGET_REPO:-}
TASK_TYPE=${DATAGEN_TASK_TYPE:-}

if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "Task Type: ${TASK_TYPE:-<none>}"

if [ "$REQUIRE_ENDPOINT" = "1" ] && [ "$WAIT_FOR_ENDPOINT" = "1" ]; then
    if [ -z "$ENDPOINT_PATH" ]; then
        echo "ERROR: Endpoint required but DATAGEN_ENDPOINT_JSON is empty" >&2
        exit 1
    fi

    echo "Waiting for endpoint JSON at $ENDPOINT_PATH"
    MAX_WAIT=600
    WAIT_COUNT=0
    while [ ! -f "$ENDPOINT_PATH" ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
        if [ $((WAIT_COUNT % 30)) -eq 0 ]; then
            echo "Still waiting for $ENDPOINT_PATH... (${WAIT_COUNT}s elapsed)"
        fi
    done

    if [ ! -f "$ENDPOINT_PATH" ]; then
        echo "ERROR: Endpoint JSON not found after ${MAX_WAIT}s" >&2
        exit 1
    fi

    echo "✓ Endpoint JSON found"
fi

CMD=(python3 "$DATAGEN_SCRIPT" --stage "${DATAGEN_STAGE:-tasks}" --engine-config "$CONFIG_PATH")

if [ -n "$DATAGEN_TARGET_REPO" ]; then
    CMD+=(--target-repo "$DATAGEN_TARGET_REPO")
fi

if [ -n "${DATAGEN_INPUT_DIR:-}" ]; then
    CMD+=(--input-dir "${DATAGEN_INPUT_DIR}")
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD+=(--output-dir "$OUTPUT_DIR")
fi

if [ -n "$TASK_TYPE" ]; then
    CMD+=(--task-type "$TASK_TYPE")
fi

if [ -n "${DATAGEN_EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=(${DATAGEN_EXTRA_ARGS})
    CMD+=("${EXTRA_ARGS_ARRAY[@]}")
fi

CMD_STR=$(printf '%q ' "${CMD[@]}")
CONTAINER_CMD="cd \"/scratch/bf996/dc-agent\" && export PYTHONPATH=\"/scratch/bf996/dc-agent:\\$PYTHONPATH\" && $CMD_STR"

if [ "$GPU_REQUIRED" = "1" ]; then
    if [ "$REQUESTED_GPUS" -le 0 ]; then
        echo "ERROR: GPU usage requested but DATAGEN_GPUS_PER_NODE is 0" >&2
        exit 1
    fi
    echo "Running datagen job with GPU access ($REQUESTED_GPUS GPUs)"
    srun --gpus-per-node="$REQUESTED_GPUS" --cpu_bind=v --accel-bind=v /bin/bash "$RUN_SINGULARITY" /bin/bash -c "$CONTAINER_CMD"
else
    echo "Running datagen job without GPU access"
    srun /bin/bash "$RUN_SINGULARITY" /bin/bash -c "$CONTAINER_CMD"
fi

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================"
    echo "✓ Data generation completed successfully"
    echo "======================================"
else
    echo "======================================"
    echo "✗ Data generation failed with exit code: $EXIT_CODE"
    echo "======================================"
fi
exit $EXIT_CODE
