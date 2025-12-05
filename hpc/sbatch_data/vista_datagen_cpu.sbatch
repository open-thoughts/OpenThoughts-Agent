#!/bin/bash
# Lightweight Vista datagen launcher (no Ray/vLLM cluster)
#SBATCH -p {partition}
#SBATCH --time={time_limit}
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082
#SBATCH --account {account}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -eo pipefail

module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

export TRITON_CC=$(which gcc)
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/10635/penfever/miniconda3/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/miniconda3/envs/vllm_sandboxes

# Conda activation can reset LD_LIBRARY_PATH; restore GCC/Triton visibility
export TRITON_CC=$(which gcc)
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/10635/penfever/miniconda3/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Source environment variables
echo $DCFT
source "$DCFT/hpc/dotenv/tacc.env"
source "$SCRATCH/keys.env"

# Set HF cache
export HF_HOME="/tmp/hf_home"
export PYTHONFAULTHANDLER=1
export NCCL_TIMEOUT=1800  # 30 minutes
export NCCL_IB_TIMEOUT=23  # InfiniBand timeout

# Change to project directory
cd "$DCFT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BACKEND="${DATAGEN_BACKEND:-vllm}"

# Data generation parameters (set via environment)
GENERATE_SCRIPT="${DATAGEN_SCRIPT}"
ENDPOINT_JSON="${DATAGEN_ENDPOINT_JSON:-$DCFT/vllm_endpoint.json}"
TARGET_REPO="${DATAGEN_TARGET_REPO:-}"
INPUT_DIR="${DATAGEN_INPUT_DIR:-}"
EXTRA_ARGS="${DATAGEN_EXTRA_ARGS:-}"
ENGINE="${DATAGEN_ENGINE:-}"
REQUIRE_ENDPOINT="${DATAGEN_REQUIRE_ENDPOINT:-0}"
OUTPUT_DIR="${DATAGEN_OUTPUT_DIR:-}"
STAGE="${DATAGEN_STAGE:-tasks}"
WAIT_FOR_ENDPOINT="${DATAGEN_WAIT_FOR_ENDPOINT:-0}"
CONFIG_PATH="${DATAGEN_CONFIG_PATH:?DATAGEN_CONFIG_PATH is required}"
TASK_TYPE="${DATAGEN_TASK_TYPE:-}"

echo "=== Data Generation Configuration (Vista CPU) ==="
echo "Backend: $BACKEND"
echo "Generate Script: $GENERATE_SCRIPT"
echo "Endpoint JSON: ${ENDPOINT_JSON:-<none>}"
echo "Target Repo: $TARGET_REPO"
echo "Input Dir: $INPUT_DIR"
echo "Extra Args: $EXTRA_ARGS"
echo "Engine Hint: ${ENGINE:-<from config>}"
echo "Config Path: $CONFIG_PATH"
echo "Output Dir: ${OUTPUT_DIR:-<none>}"
echo "Task Type: ${TASK_TYPE:-<none>}"
echo "Stage: $STAGE"
echo "Wait for endpoint: $WAIT_FOR_ENDPOINT"
echo "======================================"

if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Build command
CMD=(python3 "$GENERATE_SCRIPT" --stage "$STAGE" --engine-config "$CONFIG_PATH")

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
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_ARGS)
    CMD+=("${EXTRA_ARR[@]}")
fi

run_datagen_stream() {
    echo "Running: ${CMD[*]}"
    echo "======================================"
    "${CMD[@]}" 2>&1 | while IFS= read -r line; do
        echo "$line"
        if echo "$line" | grep -q "Healthcheck passed"; then
            echo "[$(date)] ✓ Endpoint healthy"
        elif echo "$line" | grep -q "Healthcheck FAILED"; then
            echo "[$(date)] ✗ Endpoint health check FAILED"
            echo "ERROR: Healthcheck failed - terminating job"
            exit 1
        fi
    done
    return "${PIPESTATUS[0]}"
}

if [ "$REQUIRE_ENDPOINT" = "1" ] && [ "$WAIT_FOR_ENDPOINT" = "1" ]; then
    if [ -z "$ENDPOINT_JSON" ]; then
        echo "ERROR: Endpoint required but path is empty"
        exit 1
    fi

    echo "Waiting for endpoint JSON file..."
    MAX_WAIT=600
    WAIT_COUNT=0
    while [ ! -f "$ENDPOINT_JSON" ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
        if [ $((WAIT_COUNT % 30)) -eq 0 ]; then
            echo "Still waiting for $ENDPOINT_JSON... (${WAIT_COUNT}s elapsed)"
        fi
    done

    if [ ! -f "$ENDPOINT_JSON" ]; then
        echo "ERROR: Endpoint JSON not found after ${MAX_WAIT}s: $ENDPOINT_JSON"
        exit 1
    fi

    echo "✓ Endpoint JSON found: $ENDPOINT_JSON"
    cat "$ENDPOINT_JSON"

    python3 scripts/vllm/wait_for_endpoint.py \
        --endpoint-json "$ENDPOINT_JSON" \
        --max-attempts "${DATAGEN_HEALTHCHECK_MAX_ATTEMPTS:-20}" \
        --retry-delay "${DATAGEN_HEALTHCHECK_RETRY_DELAY:-30}" \
        --health-path "v1/models"
else
    echo "Skipping endpoint wait (engine=$ENGINE, wait_flag=$WAIT_FOR_ENDPOINT, backend=$BACKEND)"
fi

run_datagen_stream
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================"
    echo "✓ Data generation completed successfully"
    echo "Target repo: $TARGET_REPO"
    echo "======================================"
else
    echo "======================================"
    echo "✗ Data generation failed with exit code: $EXIT_CODE"
    echo "======================================"
    exit $EXIT_CODE
fi
