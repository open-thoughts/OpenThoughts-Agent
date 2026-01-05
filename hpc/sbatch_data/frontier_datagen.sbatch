#!/bin/bash
#SBATCH -A {account}
#SBATCH -p batch
#SBATCH -q {qos}
#SBATCH --exclusive
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -euo pipefail

frontier_env_loaded=0
if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/frontier.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/frontier.env"
  frontier_env_loaded=1
elif [ -n "${DC_AGENT:-}" ] && [ -f "$DC_AGENT/hpc/dotenv/frontier.env" ]; then
  # shellcheck disable=SC1090
  source "$DC_AGENT/hpc/dotenv/frontier.env"
  frontier_env_loaded=1
fi

# Ensure DCFT points to a valid checkout even if SCRATCH is unavailable.
DEFAULT_OT_ROOT="${OPEN_THOUGHTS_AGENT_ROOT:-$HOME/OpenThoughts-Agent}"
if [ -z "${DCFT:-}" ] && [ -d "$DEFAULT_OT_ROOT" ]; then
  export DCFT="$DEFAULT_OT_ROOT"
elif [ -n "${DCFT:-}" ] && [ ! -d "$DCFT" ] && [ -d "$DEFAULT_OT_ROOT" ]; then
  echo "DCFT path $DCFT not found; falling back to $DEFAULT_OT_ROOT"
  export DCFT="$DEFAULT_OT_ROOT"
fi

if [ "$frontier_env_loaded" -eq 0 ] && [ -f "$DCFT/hpc/dotenv/frontier.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/frontier.env"
fi

# Frontier jobs should always use the NVMe-backed cache paths.
if [ -n "${FRONTIER_NVME_HF_HOME:-}" ]; then
  export HF_HOME="$FRONTIER_NVME_HF_HOME"
fi
if [ -n "${FRONTIER_NVME_HF_HUB_CACHE:-}" ]; then
  export HF_HUB_CACHE="$FRONTIER_NVME_HF_HUB_CACHE"
fi

module load PrgEnv-amd
module load amd/6.4.0
module load rocm/6.4.0
module load craype-accel-amd-gfx90a
module load cray-mpich

if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
fi

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [ -n "${LIBRARY_PATH:-}" ]; then
  export LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LIBRARY_PATH"
else
  export LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64"
fi

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64"
fi

export PATH="$ROCM_PATH/bin:$PATH"
export MPICH_GPU_SUPPORT_ENABLED=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-90a}

if [ -n "${CONDA_PREFIX:-}" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export FC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gfortran"
  export PATH="$CONDA_PREFIX/bin:$PATH"
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
  else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
  fi
else
  echo "Warning: CONDA_PREFIX is not set; compiler environment variables were not configured." >&2
fi

SECRET_FILE="${DC_AGENT_SECRET_ENV:-${KEYS:-}}"
if [[ -n "${SECRET_FILE}" ]]; then
    if [[ -f "${SECRET_FILE}" ]]; then
        echo "Sourcing secrets from ${SECRET_FILE}"
        set -a
        # shellcheck disable=SC1090
        source "${SECRET_FILE}"
        set +a
    else
        echo "Warning: secrets file ${SECRET_FILE} not found; uploads may fail." >&2
    fi
else
    echo "Warning: DC_AGENT_SECRET_ENV is not set; uploads may fail." >&2
fi

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=23
export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-/tmp/.outlines}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/triton_cache}"

if [ -z "${DCFT:-}" ]; then
  if [ -n "${DC_AGENT:-}" ]; then
    export DCFT="$DC_AGENT"
  else
    export DCFT="$PWD"
  fi
fi

mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"

cd "$DCFT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

BACKEND="${DATAGEN_BACKEND:-vllm}"
GENERATE_SCRIPT="${DATAGEN_SCRIPT:?DATAGEN_SCRIPT must be set}"
STAGE="${DATAGEN_STAGE:-tasks}"
CONFIG_PATH="${DATAGEN_CONFIG_PATH:?DATAGEN_CONFIG_PATH is required}"
TARGET_REPO="${DATAGEN_TARGET_REPO:-}"
INPUT_DIR="${DATAGEN_INPUT_DIR:-}"
OUTPUT_DIR="${DATAGEN_OUTPUT_DIR:-}"
EXTRA_ARGS="${DATAGEN_EXTRA_ARGS:-}"
TASK_TYPE="${DATAGEN_TASK_TYPE:-}"
ENDPOINT_JSON="${DATAGEN_ENDPOINT_JSON:-$DCFT/vllm_endpoint.json}"
REQUIRE_ENDPOINT="${DATAGEN_REQUIRE_ENDPOINT:-0}"
WAIT_FOR_ENDPOINT="${DATAGEN_WAIT_FOR_ENDPOINT:-0}"
ENGINE="${DATAGEN_ENGINE:-}"

echo "=== Data Generation Configuration (Frontier) ==="
echo "Backend: $BACKEND"
echo "Generate Script: $GENERATE_SCRIPT"
echo "Stage: $STAGE"
echo "Engine config: $CONFIG_PATH"
echo "Target repo: ${TARGET_REPO:-<none>}"
echo "Input dir: ${INPUT_DIR:-<none>}"
echo "Output dir: ${OUTPUT_DIR:-<none>}"
echo "Task type: ${TASK_TYPE:-<none>}"
echo "Endpoint JSON: ${ENDPOINT_JSON:-<none>}"
echo "Require endpoint: $REQUIRE_ENDPOINT (wait=$WAIT_FOR_ENDPOINT)"
echo "Extra args: ${EXTRA_ARGS:-<none>}"
echo "======================================"

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

wait_for_endpoint() {
    local endpoint_json="$1"
    local max_wait="${DATAGEN_ENDPOINT_WAIT_TIMEOUT:-600}"
    local elapsed=0
    echo "Waiting for endpoint JSON at $endpoint_json ..."
    while [ ! -f "$endpoint_json" ] && [ $elapsed -lt $max_wait ]; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  ... still waiting (${elapsed}s elapsed)"
        fi
    done
    if [ ! -f "$endpoint_json" ]; then
        echo "ERROR: Endpoint JSON not found after ${max_wait}s: $endpoint_json"
        return 1
    fi
    echo "✓ Endpoint JSON found. Running health check..."
    python3 scripts/vllm/wait_for_endpoint.py \
        --endpoint-json "$endpoint_json" \
        --max-attempts "${DATAGEN_HEALTHCHECK_MAX_ATTEMPTS:-60}" \
        --retry-delay "${DATAGEN_HEALTHCHECK_RETRY_DELAY:-15}" \
        --health-path "v1/models"
}

if [ "$REQUIRE_ENDPOINT" = "1" ] && [ "$WAIT_FOR_ENDPOINT" = "1" ]; then
    wait_for_endpoint "$ENDPOINT_JSON"
else
    echo "Skipping endpoint wait (require=$REQUIRE_ENDPOINT wait=$WAIT_FOR_ENDPOINT engine=${ENGINE:-auto})"
fi

echo "Running datagen: ${CMD[*]}"
"${CMD[@]}"
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Data generation completed successfully ==="
else
    echo "=== Data generation failed with exit code $EXIT_CODE ==="
fi

exit $EXIT_CODE
