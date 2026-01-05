#!/bin/bash

# =============================================================================
# kimi-k2-tracegen-run-v2.sh
#
# Fixed version that uses srun for Ray/vLLM process management like vista_datagen
# =============================================================================

# Source configs first (before set -u which can break bashrc)
set +u
source ~/.bashrc 2>/dev/null || true
source ~/secrets.env 2>/dev/null || true
set -u

# Now enable strict mode
set -eo pipefail

# Ensure SCRATCH is set (common on HPC systems)
if [[ -z "${SCRATCH:-}" ]]; then
    echo "WARNING: SCRATCH not set, using HOME" >&2
    export SCRATCH="$HOME"
fi

cd "$SCRATCH/OpenThoughts-Agent"

# Source environment - disable strict mode temporarily
set +u
source hpc/dotenv/nyutorch.env 2>/dev/null || {
    echo "WARNING: Could not source hpc/dotenv/nyutorch.env" >&2
}
set -u

# Ensure scripts module is importable
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

echo "=== Environment Setup ==="
echo "SCRATCH: $SCRATCH"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<not set>}"
echo "SLURM_JOB_NUM_NODES: ${SLURM_JOB_NUM_NODES:-1}"

# Activate conda
CONDA_ENV_NAME="dcagent312"
CONDA_ENV_PATH="$SCRATCH/miniconda3/envs/$CONDA_ENV_NAME"

if [[ -d "$CONDA_ENV_PATH" ]]; then
    set +u
    source "$SCRATCH/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    set -u
    echo "Activated conda env: $CONDA_ENV_NAME"
else
    echo "ERROR: Conda env not found at $CONDA_ENV_PATH" >&2
    exit 1
fi

# Detect libcuda for Triton
for candidate in \
    "/usr/lib/x86_64-linux-gnu/libcuda.so.1" \
    "/usr/lib64/libcuda.so.1" \
    "/usr/local/cuda/lib64/stubs/libcuda.so" \
    "/usr/local/cuda/compat/libcuda.so.1"; do
    if [[ -f "$candidate" ]]; then
        export TRITON_LIBCUDA_PATH="$candidate"
        break
    fi
done

export TRITON_CC=$(command -v gcc 2>/dev/null || echo "")
echo "TRITON_CC: ${TRITON_CC:-<not set>}"
echo "TRITON_LIBCUDA_PATH: ${TRITON_LIBCUDA_PATH:-<not set>}"

# GPU check
echo "=== GPU Check ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
nvidia-smi -L 2>/dev/null || echo "nvidia-smi not available"

# Parameters from environment with defaults
OVERWRITE_TASKS="${OVERWRITE_TASKS:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TRACE_EPISODES="${TRACE_EPISODES:-last}"
TRACE_EXPORT_FILTER="${TRACE_EXPORT_FILTER:-none}"
TRACE_DATASET_TYPE="${TRACE_DATASET_TYPE:-SFT}"
TIME_LIMIT="${TIME_LIMIT:-47:59:00}"

# Validate required environment variables
required_vars=(
    "EXPERIMENTS_DIR"
    "DATAGEN_CONFIG"
    "TRACE_SCRIPT"
    "TRACE_TARGET_REPO"
    "TRACE_HARBOR_CONFIG"
    "TRACE_MODEL"
    "TRACE_ENGINE"
    "TRACE_BACKEND"
    "TASKS_REPO"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "ERROR: Missing required environment variables:" >&2
    for var in "${missing_vars[@]}"; do
        echo "  - $var" >&2
    done
    echo "These should be exported by the sbatch script." >&2
    exit 1
fi

echo "=== Required Variables ==="
echo "EXPERIMENTS_DIR: $EXPERIMENTS_DIR"
echo "DATAGEN_CONFIG: $DATAGEN_CONFIG"
echo "TRACE_SCRIPT: $TRACE_SCRIPT"
echo "TRACE_TARGET_REPO: $TRACE_TARGET_REPO"
echo "TASKS_REPO: $TASKS_REPO"

mkdir -p "$EXPERIMENTS_DIR" "$EXPERIMENTS_DIR/logs" "$EXPERIMENTS_DIR/outputs/traces" "$EXPERIMENTS_DIR/trace_jobs"

# Build srun export string
SRUN_EXPORT_ENV="ALL"
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,TRITON_CC=${TRITON_CC:-}"
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-}"
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,HF_HOME=${HF_HOME:-}"
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,PYTHONPATH=${PYTHONPATH:-}"
# Ray control-plane stability knobs (prevents backlog storms / exporter hangs)
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,RAY_report_worker_backlog_to_raylet=0"
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,RAY_metrics_export_port=8090"
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,RAY_gcs_server_request_timeout_seconds=300"
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,RAY_raylet_heartbeat_timeout_milliseconds=600000"
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,RAY_worker_heartbeat_timeout_milliseconds=600000"
export SRUN_EXPORT_ENV

# =============================================================================
# Extract tasks
# =============================================================================
echo ">>> Extracting tasks via extract_tasks_from_parquet.py"

TASKS_INPUT="$EXPERIMENTS_DIR/tasks_extracted"
EXTRACT_CMD=(
  python3
  scripts/datagen/extract_tasks_from_parquet.py
  --parquet "$TASKS_REPO"
  --output_dir "$TASKS_INPUT"
)

ON_EXIST_MODE="skip"
if [[ "$OVERWRITE_TASKS" == "1" ]]; then
  ON_EXIST_MODE="overwrite"
fi
EXTRACT_CMD+=(--on_exist "$ON_EXIST_MODE")

[[ -n "${TASKS_REVISION:-}" ]] && EXTRACT_CMD+=(--tasks_revision "$TASKS_REVISION")
[[ -n "${PARQUET_NAME:-}" ]] && EXTRACT_CMD+=(--parquet_name "$PARQUET_NAME")

echo "Command: ${EXTRACT_CMD[*]}"
"${EXTRACT_CMD[@]}"

TRACE_OUTPUT_DIR="$EXPERIMENTS_DIR/outputs/traces"
TRACE_JOBS_DIR="$EXPERIMENTS_DIR/trace_jobs"

# =============================================================================
# Prepare vLLM configuration
# =============================================================================
echo ">>> Preparing datagen configuration"
mapfile -t CONFIG_EXPORTS < <(python3 - <<'PY'
import os
import shlex
from hpc.datagen_launch_utils import (
    _prepare_datagen_configuration,
    _snapshot_datagen_config,
    _build_vllm_env_vars,
    resolve_datagen_config_path,
)

datagen_config = resolve_datagen_config_path(os.environ["DATAGEN_CONFIG"])
exp_args = {
    "datagen_config": str(datagen_config),
    "experiments_dir": os.environ["EXPERIMENTS_DIR"],
}
_prepare_datagen_configuration(exp_args)
snapshot_path = _snapshot_datagen_config(exp_args)
env_vars, exp_args = _build_vllm_env_vars(exp_args)

print(f'export DATAGEN_CONFIG_RESOLVED={shlex.quote(snapshot_path)}')
for key, value in env_vars.items():
    print(f'export {key}={shlex.quote(str(value))}')
ray_port = exp_args.get("datagen_ray_port", 6379)
api_port = exp_args.get("datagen_api_port", 8000)
print(f'export DATAGEN_RAY_PORT={shlex.quote(str(ray_port))}')
print(f'export DATAGEN_API_PORT={shlex.quote(str(api_port))}')
PY
)

for line in "${CONFIG_EXPORTS[@]}"; do
  eval "$line"
done

VLLM_ENDPOINT_JSON_PATH="${VLLM_ENDPOINT_JSON_PATH:-$EXPERIMENTS_DIR/vllm_endpoint.json}"
RAY_PORT="${DATAGEN_RAY_PORT:-6379}"
API_PORT="${DATAGEN_API_PORT:-8000}"
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-8}"
PP_SIZE="${VLLM_PIPELINE_PARALLEL_SIZE:-1}"
DP_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"

# =============================================================================
# Ray cluster setup - using srun like vista_datagen
# =============================================================================
echo ">>> Setting up Ray cluster with srun"
set -x

# Get node info
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
CPUS_PER_NODE=${SLURM_CPUS_PER_TASK:-96}
HEADROOM_MB=8192
SRUN_MEM_PER_STEP=$((1572864 - HEADROOM_MB))
echo "Using SRUN_MEM_PER_STEP=${SRUN_MEM_PER_STEP} MB"

# Get head node IP
head_ip_output=$(srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --mem="$SRUN_MEM_PER_STEP" --overlap -w "$head_node" hostname --ip-address 2>&1)
if [[ $? -ne 0 ]]; then
    echo "$head_ip_output"
    echo "ERROR: Failed to resolve head node IP via srun"
    exit 1
fi
head_node_ip=$(echo "$head_ip_output" | head -n1)
head_node_ip=${head_node_ip%% *}

if [[ -z "$head_node_ip" || "$head_node_ip" == "127.0.0.1" ]]; then
    echo "WARNING: Could not get non-loopback IP, falling back to hostname -I"
    head_node_ip=$(hostname -I | awk '{print $1}')
fi

ip_head="${head_node_ip}:${RAY_PORT}"
export RAY_ADDRESS="$ip_head"

echo "Head node: $head_node ($head_node_ip)"
echo "RAY_ADDRESS: $RAY_ADDRESS"
echo "Nodes: $NUM_NODES, GPUs/node: $GPUS_PER_NODE, CPUs/node: $CPUS_PER_NODE"

# Track Ray PIDs
ray_pids=()

# Cleanup function
cleanup() {
    echo ">>> Cleanup: stopping vLLM and Ray..."
    set +e
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    for node in "${nodes_array[@]}"; do
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --mem="$SRUN_MEM_PER_STEP" --overlap -w "$node" \
            ray stop --force 2>/dev/null || true
    done
    for pid in "${ray_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo ">>> Cleanup complete"
}
trap cleanup EXIT

# Use SCRATCH-backed path for Ray temp/logs so we can inspect sessions post-mortem
RAY_TMPDIR_BASE="${SCRATCH}/ray_sessions"
RAY_TMPDIR="${RAY_TMPDIR_BASE}/ray_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
# Make sure all subsequent srun invocations inherit the tmpdir override
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,RAY_TMPDIR=$RAY_TMPDIR"
export SRUN_EXPORT_ENV
echo ">>> Ray session directory: $RAY_TMPDIR"

# Calculate object store memory (conservative for large models)
OBJECT_STORE_BYTES=$((40 * 1024 * 1024 * 1024))  # 40GB

# Start Ray head via srun
echo ">>> Starting Ray head on $head_node"
srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --mem="$SRUN_MEM_PER_STEP" --overlap -w "$head_node" \
    ray start --head \
        --node-ip-address="$head_node_ip" \
        --port="$RAY_PORT" \
        --num-gpus="$GPUS_PER_NODE" \
        --num-cpus="$CPUS_PER_NODE" \
        --temp-dir="$RAY_TMPDIR" \
        --object-store-memory="$OBJECT_STORE_BYTES" \
        --block &
ray_pids+=($!)

# Start workers on other nodes
for ((i = 1; i < NUM_NODES; i++)); do
    node_i=${nodes_array[$i]}
    echo ">>> Starting Ray worker on $node_i"
    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --mem="$SRUN_MEM_PER_STEP" --overlap -w "$node_i" \
        ray start \
            --address="$ip_head" \
            --num-gpus="$GPUS_PER_NODE" \
            --num-cpus="$CPUS_PER_NODE" \
            --temp-dir="$RAY_TMPDIR" \
            --object-store-memory="$OBJECT_STORE_BYTES" \
            --block &
    ray_pids+=($!)
    sleep 3
done

# Wait for cluster
echo ">>> Waiting for Ray cluster..."
TOTAL_GPUS=$((GPUS_PER_NODE * NUM_NODES))
sleep 15  # Give Ray time to initialize

python3 scripts/ray/wait_for_cluster.py \
    --address "$ip_head" \
    --expected-gpus "$TOTAL_GPUS" \
    --expected-nodes "$NUM_NODES" \
    --timeout 600 \
    --poll-interval 10

echo ">>> Ray cluster ready!"
set +x

# =============================================================================
# Start vLLM controller via srun
# =============================================================================
echo ">>> Launching vLLM controller"
CONTROLLER_LOG="$EXPERIMENTS_DIR/logs/vllm_controller.log"
rm -f "$VLLM_ENDPOINT_JSON_PATH"

echo "  Ray address: $ip_head"
echo "  API port: $API_PORT"
echo "  TP size: $TP_SIZE"
echo "  Log: $CONTROLLER_LOG"

srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --mem="$SRUN_MEM_PER_STEP" --overlap -w "$head_node" \
    python3 scripts/vllm/start_vllm_ray_controller.py \
        --ray-address "$ip_head" \
        --host "$head_node_ip" \
        --port "$API_PORT" \
        --endpoint-json "$VLLM_ENDPOINT_JSON_PATH" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$PP_SIZE" \
        --data-parallel-size "$DP_SIZE" \
    >> "$CONTROLLER_LOG" 2>&1 &
VLLM_PID=$!

echo ">>> vLLM controller PID: $VLLM_PID"

# Wait for endpoint JSON
echo ">>> Waiting for vLLM endpoint..."
ENDPOINT_WAIT=0
while [[ ! -f "$VLLM_ENDPOINT_JSON_PATH" && $ENDPOINT_WAIT -lt 900 ]]; do
    sleep 10
    ENDPOINT_WAIT=$((ENDPOINT_WAIT + 10))
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM controller died" >&2
        echo ">>> Controller log:" >&2
        tail -n 100 "$CONTROLLER_LOG" 2>/dev/null || true
        exit 1
    fi
    echo "  Waiting... ${ENDPOINT_WAIT}s"
done

if [[ ! -f "$VLLM_ENDPOINT_JSON_PATH" ]]; then
    echo "ERROR: vLLM endpoint not created within 900s" >&2
    tail -n 100 "$CONTROLLER_LOG" 2>/dev/null || true
    exit 1
fi

echo ">>> Endpoint JSON created, waiting for health check..."
python3 scripts/vllm/wait_for_endpoint.py \
    --endpoint-json "$VLLM_ENDPOINT_JSON_PATH" \
    --max-attempts 60 \
    --retry-delay 20 \
    --health-path "v1/models"

# =============================================================================
# Run trace generation
# =============================================================================
echo ">>> Running trace generator"

TRACE_CMD=(
    python3 "$TRACE_SCRIPT"
    --stage traces
    --engine-config "$DATAGEN_CONFIG_RESOLVED"
    --target-repo "$TRACE_TARGET_REPO"
    --tasks-input "$TASKS_INPUT"
    --output-dir "$TRACE_OUTPUT_DIR"
    --trace-harbor-config "$TRACE_HARBOR_CONFIG"
    --trace-jobs-dir "$TRACE_JOBS_DIR"
    --trace-model "$TRACE_MODEL"
    --endpoint-json "$VLLM_ENDPOINT_JSON_PATH"
    --trace-episodes "$TRACE_EPISODES"
    --trace-export-filter "$TRACE_EXPORT_FILTER"
    --trace-dataset-type "$TRACE_DATASET_TYPE"
)

[[ -n "${TRACE_AGENT_NAME:-}" ]] && TRACE_CMD+=(--trace-agent-name "$TRACE_AGENT_NAME")
[[ -n "${TRACE_AGENT_KWARGS:-}" ]] && TRACE_CMD+=(--trace-agent-kwargs "$TRACE_AGENT_KWARGS")
[[ -n "${TRACE_ENV:-}" ]] && TRACE_CMD+=(--trace-env "$TRACE_ENV")
[[ -n "${TRACE_AGENT_TIMEOUT_SEC:-}" ]] && TRACE_CMD+=(--trace-agent-timeout-sec "$TRACE_AGENT_TIMEOUT_SEC")
[[ -n "${TRACE_N_CONCURRENT:-}" ]] && TRACE_CMD+=(--trace-n-concurrent "$TRACE_N_CONCURRENT")
[[ -n "${TRACE_CHUNK_SIZE:-}" ]] && TRACE_CMD+=(--chunk_size "$TRACE_CHUNK_SIZE")

echo "Command: ${TRACE_CMD[*]}"
"${TRACE_CMD[@]}"

echo ">>> Trace generation complete!"
