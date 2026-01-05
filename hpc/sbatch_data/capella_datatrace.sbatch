#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --exclusive
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -euo pipefail

JOB_SUBMIT_ORDER=${JOB_SUBMIT_ORDER:-}
if [ -n "$JOB_SUBMIT_ORDER" ]; then
  if [[ "$JOB_SUBMIT_ORDER" =~ ^[0-9]+$ ]]; then
    SLEEP_DURATION=$((30 * JOB_SUBMIT_ORDER))
    echo "Submit order delay: sleeping ${SLEEP_DURATION}s (JOB_SUBMIT_ORDER=${JOB_SUBMIT_ORDER})"
    sleep "$SLEEP_DURATION"
  else
    echo "Warning: JOB_SUBMIT_ORDER='${JOB_SUBMIT_ORDER}' is not an integer; skipping launch delay"
  fi
fi

if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/zih_capella.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/zih_capella.env"
elif [ -n "${DC_AGENT:-}" ] && [ -f "$DC_AGENT/hpc/dotenv/zih_capella.env" ]; then
  # shellcheck disable=SC1090
  source "$DC_AGENT/hpc/dotenv/zih_capella.env"
fi

module load CUDA/12.8.0

if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
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
        echo "Warning: secrets file ${SECRET_FILE} not found; trace uploads may fail." >&2
    fi
else
    echo "Warning: DC_AGENT_SECRET_ENV is not set; trace uploads may fail." >&2
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

# Ray GPU visibility:
# vLLM's Ray executor expects CUDA ordinals to match Ray's GPU ids. On some
# clusters, Ray's default CUDA_VISIBLE_DEVICES rewriting can remap ordinals and
# trigger "CUDA error: invalid device ordinal" inside vLLM workers.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}"
export RAY_NOSET_CUDA_VISIBLE_DEVICES="${RAY_NOSET_CUDA_VISIBLE_DEVICES:-1}"

SRUN_EXPORT_ENV="ALL,PATH=$PATH,LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-},PYTHONPATH=$PYTHONPATH,HF_HOME=$HF_HOME,RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=$RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES,RAY_NOSET_CUDA_VISIBLE_DEVICES=$RAY_NOSET_CUDA_VISIBLE_DEVICES"
RAY_ENV_VARS="PATH=$PATH LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} PYTHONPATH=$PYTHONPATH HF_HOME=$HF_HOME RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=$RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES RAY_NOSET_CUDA_VISIBLE_DEVICES=$RAY_NOSET_CUDA_VISIBLE_DEVICES"

# Persist Ray sessions/logs on scratch for easier debugging
RAY_TMPDIR_BASE="${SCRATCH:-/tmp}/ray_sessions"
RAY_TMPDIR="${RAY_TMPDIR_BASE}/ray_${SLURM_JOB_ID:-$$}"
mkdir -p "$RAY_TMPDIR"
SRUN_EXPORT_ENV="$SRUN_EXPORT_ENV,RAY_TMPDIR=$RAY_TMPDIR"
RAY_ENV_VARS="$RAY_ENV_VARS RAY_TMPDIR=$RAY_TMPDIR"
echo "Ray session directory: $RAY_TMPDIR"

TRACE_BACKEND="${TRACE_BACKEND:-vllm}"

# Trace generation parameters
TRACE_SCRIPT="${TRACE_SCRIPT}"
TRACE_STAGE="${TRACE_STAGE:-traces}"
TRACE_TASKS_PATH="${TRACE_TASKS_PATH}"
TRACE_TARGET_REPO="${TRACE_TARGET_REPO}"
TRACE_MODEL="${TRACE_MODEL:-}"
TRACE_ENGINE="${TRACE_ENGINE:-}"
TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-}"
TRACE_USE_GPU="${TRACE_USE_GPU:-0}"
TRACE_EPISODES="${TRACE_EPISODES:-}"
TRACE_EXPORT_FILTER="${TRACE_EXPORT_FILTER:-}"
TRACE_DATASET_TYPE="${TRACE_DATASET_TYPE:-}"
TRACE_JOBS_DIR="${TRACE_JOBS_DIR:-}"
TRACE_ENDPOINT_JSON="${TRACE_ENDPOINT_JSON:-$DCFT/vllm_endpoint.json}"
TRACE_REQUIRE_ENDPOINT="${TRACE_REQUIRE_ENDPOINT:-0}"
TRACE_WAIT_FOR_ENDPOINT="${TRACE_WAIT_FOR_ENDPOINT:-0}"
TRACE_HARBOR_CONFIG="${TRACE_HARBOR_CONFIG:-}"
TRACE_DISABLE_VERIFICATION="${TRACE_DISABLE_VERIFICATION:-0}"
TRACE_EVAL_ONLY="${TRACE_EVAL_ONLY:-0}"
TRACE_AGENT_TIMEOUT_SEC="${TRACE_AGENT_TIMEOUT_SEC:-}"
TRACE_VERIFIER_TIMEOUT_SEC="${TRACE_VERIFIER_TIMEOUT_SEC:-}"
TRACE_ENGINE_CONFIG_PATH="${TRACE_ENGINE_CONFIG_PATH:-}"
VLLM_JOB_ID="${VLLM_JOB_ID:-}"
TRACE_TASK_TYPE="${TRACE_TASK_TYPE:-}"
TRACE_EXPORT_SUBAGENTS="${TRACE_EXPORT_SUBAGENTS:-1}"

source "$DCFT/hpc/sbatch_data/trace_helpers.sh"
load_trace_chunk_env

cleanup_vllm() {
  if [ -n "$VLLM_JOB_ID" ]; then
    echo "Tearing down VLLM job $VLLM_JOB_ID"
    scancel "$VLLM_JOB_ID" >/dev/null 2>&1 || echo "Warning: failed to cancel VLLM job $VLLM_JOB_ID"
  fi
}

trap cleanup_vllm EXIT

echo "=== Trace Generation Configuration ==="
echo "Backend: $TRACE_BACKEND"
echo "Trace Script: $TRACE_SCRIPT"
echo "Stage: $TRACE_STAGE"
echo "Tasks Input: $TRACE_TASKS_PATH"
echo "Target Repo: $TRACE_TARGET_REPO"
echo "Model: ${TRACE_MODEL:-<none>}"
echo "Output Dir: ${TRACE_OUTPUT_DIR:-<none>}"
echo "Engine: ${TRACE_ENGINE:-<none>}"
echo "Use GPU: $TRACE_USE_GPU"
echo "Task Type: ${TRACE_TASK_TYPE:-<none>}"
echo "Episodes: ${TRACE_EPISODES:-<none>}"
echo "Export Filter: ${TRACE_EXPORT_FILTER:-<none>}"
echo "Dataset Type: ${TRACE_DATASET_TYPE:-<none>}"
echo "Jobs Dir: ${TRACE_JOBS_DIR:-<none>}"
echo "Harbor Config: ${TRACE_HARBOR_CONFIG:-<none>}"
echo "Disable Verification: $TRACE_DISABLE_VERIFICATION"
echo "Eval Only: $TRACE_EVAL_ONLY"
if [ "${TRACE_CHUNK_MODE:-0}" = "1" ]; then
  echo "Chunk Index: ${TRACE_JOB_INDEX:-${SLURM_ARRAY_TASK_ID:-?}} / ${TRACE_CHUNK_COUNT:-?}"
fi
echo "======================================"

if [ -f "$TRACE_ENDPOINT_JSON" ]; then
    echo "Removing stale endpoint JSON: $TRACE_ENDPOINT_JSON"
    rm -f "$TRACE_ENDPOINT_JSON"
fi

ensure_tasks_ready_or_exit "$TRACE_TASKS_PATH" "${TRACE_TASKS_READY_TIMEOUT:-1800}"

if [ -n "$TRACE_OUTPUT_DIR" ]; then
    mkdir -p "$TRACE_OUTPUT_DIR"
fi

CMD=(python3 "$TRACE_SCRIPT" --stage "$TRACE_STAGE" --target-repo "$TRACE_TARGET_REPO" --tasks-input "$TRACE_TASKS_PATH")

if [ -n "$TRACE_OUTPUT_DIR" ]; then
    CMD+=(--output-dir "$TRACE_OUTPUT_DIR")
fi
if [ -n "$TRACE_HARBOR_CONFIG" ]; then
    CMD+=(--trace-harbor-config "$TRACE_HARBOR_CONFIG")
fi
if [ -n "$TRACE_JOBS_DIR" ]; then
    CMD+=(--trace-jobs-dir "$TRACE_JOBS_DIR")
fi
if [ -n "$TRACE_MODEL" ]; then
    CMD+=(--trace-model "$TRACE_MODEL")
fi
if [ -n "$TRACE_EPISODES" ]; then
    CMD+=(--trace-episodes "$TRACE_EPISODES")
fi
if [ -n "$TRACE_EXPORT_FILTER" ]; then
    CMD+=(--trace-export-filter "$TRACE_EXPORT_FILTER")
fi
if [ -n "$TRACE_DATASET_TYPE" ]; then
    CMD+=(--trace-dataset-type "$TRACE_DATASET_TYPE")
fi
if [ "$TRACE_EXPORT_SUBAGENTS" = "1" ]; then
    CMD+=(--trace-export-subagents)
else
    CMD+=(--trace-skip-subagents)
fi
if [ -n "$TRACE_ENGINE" ]; then
    CMD+=(--engine "$TRACE_ENGINE")
fi
if [ -n "$TRACE_AGENT_TIMEOUT_SEC" ]; then
  CMD+=(--trace-agent-timeout-sec "$TRACE_AGENT_TIMEOUT_SEC")
fi
if [ -n "$TRACE_VERIFIER_TIMEOUT_SEC" ]; then
  CMD+=(--trace-verifier-timeout-sec "$TRACE_VERIFIER_TIMEOUT_SEC")
fi
if [ "$TRACE_DISABLE_VERIFICATION" = "1" ]; then
    CMD+=(--disable-verification)
fi
if [ -n "$TRACE_ENGINE_CONFIG_PATH" ]; then
    CMD+=(--engine-config "$TRACE_ENGINE_CONFIG_PATH")
fi
if [ "$TRACE_EVAL_ONLY" = "1" ]; then
    CMD+=(--trace-eval-only)
fi
if [ -n "$TRACE_TASK_TYPE" ]; then
    CMD+=(--task-type "$TRACE_TASK_TYPE")
fi

run_trace() {
    echo "Running: ${CMD[*]}"
    echo "======================================"
    "${CMD[@]}"
    return $?
}

if [ "$TRACE_BACKEND" != "ray" ]; then
    if [ "$TRACE_REQUIRE_ENDPOINT" = "1" ] && [ "$TRACE_WAIT_FOR_ENDPOINT" = "1" ]; then
        if [ -z "$TRACE_ENDPOINT_JSON" ]; then
            echo "ERROR: Trace endpoint required but path is empty"
            exit 1
        fi

        echo "Waiting for trace endpoint JSON..."
        TRACE_MAX_WAIT=600
        TRACE_WAIT=0
        while [ ! -f "$TRACE_ENDPOINT_JSON" ] && [ $TRACE_WAIT -lt $TRACE_MAX_WAIT ]; do
            sleep 5
            TRACE_WAIT=$((TRACE_WAIT + 5))
            if [ $((TRACE_WAIT % 30)) -eq 0 ]; then
                echo "Still waiting for $TRACE_ENDPOINT_JSON... (${TRACE_WAIT}s elapsed)"
            fi
        done

        if [ ! -f "$TRACE_ENDPOINT_JSON" ]; then
            echo "ERROR: Trace endpoint JSON not found after ${TRACE_MAX_WAIT}s: $TRACE_ENDPOINT_JSON"
            exit 1
        fi

        echo "✓ Trace endpoint JSON present"
        cat "$TRACE_ENDPOINT_JSON"

        python3 scripts/vllm/wait_for_endpoint.py \
            --endpoint-json "$TRACE_ENDPOINT_JSON" \
            --max-attempts "${TRACE_HEALTH_MAX_ATTEMPTS:-20}" \
            --retry-delay "${TRACE_HEALTH_RETRY_DELAY:-60}" \
            --health-path "v1/models"
    else
        echo "Trace endpoint wait skipped (engine=$TRACE_ENGINE, wait_flag=$TRACE_WAIT_FOR_ENDPOINT)"
    fi

    run_trace
    EXIT_CODE=$?
else
    echo "=== Starting Ray-backed vLLM cluster for trace generation ==="
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)
    head_node=${nodes_array[0]}

    head_node_ip=$(srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" hostname --ip-address)
    head_node_ip=${head_node_ip%% *}

    RAY_PORT="${TRACE_RAY_PORT:-6379}"
    API_PORT="${TRACE_API_PORT:-8000}"
    TP_SIZE="${TRACE_TENSOR_PARALLEL_SIZE:-1}"
    PP_SIZE="${TRACE_PIPELINE_PARALLEL_SIZE:-${VLLM_PIPELINE_PARALLEL_SIZE:-1}}"
    DP_SIZE="${TRACE_DATA_PARALLEL_SIZE:-1}"
    GPUS_PER_NODE="${TRACE_GPUS_PER_NODE:-0}"
    export TRACE_GPUS_PER_NODE="$GPUS_PER_NODE"
    NUM_NODES=$SLURM_JOB_NUM_NODES
    TOTAL_GPUS=$(python3 - <<'PY'
import os
print(int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) * int(os.environ.get("TRACE_GPUS_PER_NODE", "0")))
PY
)

    ip_head="${head_node_ip}:${RAY_PORT}"
    export RAY_ADDRESS="$ip_head"

    ray_pids=()

    start_ray_node() {
        local node_name=$1
        local start_cmd=$2
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$node_name" bash -c "env $RAY_ENV_VARS $start_cmd" &
        ray_pids+=($!)
    }

    head_cmd="ray start --head --node-ip-address=${head_node_ip} --port=${RAY_PORT} --num-gpus=${GPUS_PER_NODE} --num-cpus={cpus_per_node} --temp-dir=${RAY_TMPDIR} --block"
    echo "Starting Ray head on ${head_node} (${head_node_ip})"
    start_ray_node "$head_node" "$head_cmd"

    for ((i = 1; i < NUM_NODES; i++)); do
        node_i=${nodes_array[$i]}
        worker_cmd="ray start --address ${ip_head} --num-gpus=${GPUS_PER_NODE} --num-cpus={cpus_per_node} --temp-dir=${RAY_TMPDIR} --block"
        echo "Starting Ray worker on ${node_i}"
        start_ray_node "$node_i" "$worker_cmd"
        sleep 3
    done

    cleanup_ray() {
        echo "Cleaning up Ray cluster and vLLM controller..."
        if [ -n "${VLLM_PID:-}" ]; then
            kill "$VLLM_PID" >/dev/null 2>&1 || true
            wait "$VLLM_PID" >/dev/null 2>&1 || true
        fi
        for node in "${nodes_array[@]}"; do
            srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$node" ray stop --force >/dev/null 2>&1 || true
        done
        for pid in "${ray_pids[@]}"; do
            wait "$pid" >/dev/null 2>&1 || true
        done
    }

    trap 'cleanup_ray; cleanup_vllm' EXIT

    echo "Waiting for Ray cluster resources..."
    python3 scripts/ray/wait_for_cluster.py \
        --address "$ip_head" \
        --expected-gpus "$TOTAL_GPUS" \
        --expected-nodes "$NUM_NODES" \
        --timeout 600 \
        --poll-interval 10

    echo "Launching vLLM Ray controller on ${head_node_ip}:${API_PORT}"
    controller_log="{experiments_dir}/logs/{job_name}_vllm.log"
    CONTROLLER_ENV_CMD="env PATH=$PATH LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} HF_HOME=$HF_HOME PYTHONPATH=$PYTHONPATH"

    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
        $CONTROLLER_ENV_CMD \
        python3 scripts/vllm/start_vllm_ray_controller.py \
            --ray-address "$ip_head" \
            --host "$head_node_ip" \
            --port "$API_PORT" \
            --endpoint-json "$TRACE_ENDPOINT_JSON" \
            --tensor-parallel-size "$TP_SIZE" \
            --pipeline-parallel-size "$PP_SIZE" \
            --data-parallel-size "$DP_SIZE" \
        >> "$controller_log" 2>&1 &
    VLLM_PID=$!

    echo "vLLM controller PID: $VLLM_PID (log: $controller_log)"

    if [ "$TRACE_REQUIRE_ENDPOINT" = "1" ] && [ "$TRACE_WAIT_FOR_ENDPOINT" = "1" ]; then
        ENDPOINT_WAIT_TIMEOUT="${TRACE_ENDPOINT_WAIT_TIMEOUT:-180}"
        ENDPOINT_WAIT_INTERVAL="${TRACE_ENDPOINT_WAIT_INTERVAL:-5}"
        wait_elapsed=0

        while true; do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                wait "$VLLM_PID" || true
                echo "ERROR: vLLM controller exited before writing endpoint JSON; see $controller_log"
                exit 1
            fi

            if [ -f "$TRACE_ENDPOINT_JSON" ]; then
                echo "✓ Trace endpoint JSON detected after ${wait_elapsed}s: $TRACE_ENDPOINT_JSON"
                break
            fi

            if [ "$wait_elapsed" -ge "$ENDPOINT_WAIT_TIMEOUT" ]; then
                echo "ERROR: Trace endpoint JSON not found within ${ENDPOINT_WAIT_TIMEOUT}s (controller PID $VLLM_PID still running)"
                exit 1
            fi

            sleep "$ENDPOINT_WAIT_INTERVAL"
            wait_elapsed=$((wait_elapsed + ENDPOINT_WAIT_INTERVAL))
            if [ $((wait_elapsed % 30)) -eq 0 ]; then
                echo "Waiting for trace endpoint JSON... ${wait_elapsed}s elapsed"
            fi
        done

        echo "Waiting for vLLM endpoint to become healthy..."
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
            python3 scripts/vllm/wait_for_endpoint.py \
                --endpoint-json "$TRACE_ENDPOINT_JSON" \
                --max-attempts "${TRACE_HEALTH_MAX_ATTEMPTS:-20}" \
                --retry-delay "${TRACE_HEALTH_RETRY_DELAY:-60}" \
                --health-path "v1/models"
    else
        echo "Skipping endpoint wait (backend=$TRACE_BACKEND, wait_flag=$TRACE_WAIT_FOR_ENDPOINT)"
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            wait "$VLLM_PID" || true
            echo "ERROR: vLLM controller exited immediately; see $controller_log"
            exit 1
        fi
    fi

    run_trace
    EXIT_CODE=$?
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================"
    echo "✓ Trace generation completed successfully"
    echo "Target repo: $TRACE_TARGET_REPO"
    echo "======================================"
else
    echo "======================================"
    echo "✗ Trace generation failed with exit code: $EXIT_CODE"
    echo "======================================"
    exit $EXIT_CODE
fi
