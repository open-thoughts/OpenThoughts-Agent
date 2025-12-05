#!/bin/bash
# Perlmutter datagen job; assumes optional vLLM endpoint JSON handoff.
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

if [ -n "${DATAGEN_ACTIVATE_ENV:-}" ]; then
  eval "$DATAGEN_ACTIVATE_ENV"
elif [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  eval "$DCFT_ACTIVATE_ENV"
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

GENERATE_SCRIPT="${DATAGEN_SCRIPT}"
ENDPOINT_JSON="${DATAGEN_ENDPOINT_JSON:-}"
TARGET_REPO="${DATAGEN_TARGET_REPO:-}"
INPUT_DIR="${DATAGEN_INPUT_DIR:-}"
EXTRA_ARGS="${DATAGEN_EXTRA_ARGS:-}"
ENGINE="${DATAGEN_ENGINE:-}"
REQUIRE_ENDPOINT="${DATAGEN_REQUIRE_ENDPOINT:-0}"
OUTPUT_DIR="${DATAGEN_OUTPUT_DIR:-}"
STAGE="${DATAGEN_STAGE:-tasks}"
WAIT_FOR_ENDPOINT="${DATAGEN_WAIT_FOR_ENDPOINT:-0}"
CONFIG_PATH="${DATAGEN_CONFIG_PATH:?DATAGEN_CONFIG_PATH is required}"
GPU_REQUIRED="${DATAGEN_GPU_REQUIRED:-0}"
GPUS_PER_NODE="${DATAGEN_GPUS_PER_NODE:-0}"
TASK_TYPE="${DATAGEN_TASK_TYPE:-}"

echo "=== Datagen (Perlmutter) ==="
echo "Script: $GENERATE_SCRIPT"
echo "Endpoint: ${ENDPOINT_JSON:-<none>}"
echo "Target Repo: $TARGET_REPO"
echo "Input Dir: $INPUT_DIR"
echo "Output Dir: ${OUTPUT_DIR:-<none>}"
echo "Engine: ${ENGINE:-<unset>}"
echo "Config Path: $CONFIG_PATH"
echo "Stage: $STAGE"
echo "GPUs required: $GPU_REQUIRED (per node: $GPUS_PER_NODE)"
echo "Task Type: ${TASK_TYPE:-<none>}"

if [ "$REQUIRE_ENDPOINT" = "1" ] && [ "$WAIT_FOR_ENDPOINT" = "1" ]; then
  if [ -z "$ENDPOINT_JSON" ]; then
    echo "ERROR: Endpoint required but DATAGEN_ENDPOINT_JSON is empty." >&2
    exit 1
  fi

  echo "Waiting for endpoint JSON at $ENDPOINT_JSON"
  MAX_WAIT=600
  WAITED=0
  while [ ! -f "$ENDPOINT_JSON" ] && [ $WAITED -lt $MAX_WAIT ]; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
      echo "Still waiting for $ENDPOINT_JSON... (${WAITED}s elapsed)"
    fi
  done

  if [ ! -f "$ENDPOINT_JSON" ]; then
    echo "ERROR: Endpoint JSON not found after ${MAX_WAIT}s." >&2
    exit 1
  fi

  echo "✓ Endpoint JSON discovered"
fi

if [ -n "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

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

if [ "$GPU_REQUIRED" = "1" ]; then
  if [ "$GPUS_PER_NODE" -le 0 ]; then
    echo "ERROR: GPU access requested but DATAGEN_GPUS_PER_NODE is 0." >&2
    exit 1
  fi
  echo "Running datagen with GPU access ($GPUS_PER_NODE per node)"
  srun --gpus-per-node="$GPUS_PER_NODE" --cpu_bind=v --accel-bind=v "${CMD[@]}"
else
  echo "Running datagen without GPU access"
  srun "${CMD[@]}"
fi
