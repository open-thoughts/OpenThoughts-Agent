#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=production
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/macs/eval/logs/%x_%j.out

# ==============================================================================
# Unified Eval Harbor — Hazy (8x H100 80GB, single node)
#
# Everything is YAML-driven. This script only:
#   1. Starts vLLM (reading model + params from EVAL_VLLM_CONFIG)
#   2. Ensures dataset exists (downloading if needed)
#   3. Calls `harbor jobs start --config <yaml>` — nothing else on CLI
#
# Env vars:
#   EVAL_HARBOR_CONFIG   Harbor YAML (required) — has model, agent, dataset, everything
#   EVAL_VLLM_CONFIG     vLLM serving YAML (required) — has TP, context len, parser
#   DAYTONA_API_KEY      Required for Daytona sandbox backend
# ==============================================================================

set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# ==============================================================================
# Paths
# ==============================================================================

OT_AGENT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
REPO_ROOT="$(cd "$OT_AGENT_DIR/../.." && pwd)"
SCRIPT_DIR="$OT_AGENT_DIR/eval/hazy"
# Eval artifacts root. Default: <repo>/eval (per setup_env.sh). Override with
# HARBOR_EVAL_ROOT to put venv/jobs/logs/cache outside the repo on a different
# cluster.
EVAL_ROOT="${HARBOR_EVAL_ROOT:-$REPO_ROOT/eval}"
VENV_DIR="$EVAL_ROOT/venv"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "ERROR: Eval venv not found at $VENV_DIR. Run eval/hazy/setup_env.sh"
    exit 1
fi

PYTHON_BIN="$VENV_DIR/bin/python"
HARBOR_CONFIG="${EVAL_HARBOR_CONFIG:?ERROR: EVAL_HARBOR_CONFIG not set}"
VLLM_CONFIG="${EVAL_VLLM_CONFIG:?ERROR: EVAL_VLLM_CONFIG not set}"

# ==============================================================================
# Environment
# ==============================================================================

unset PYTHONPATH
export HF_HOME="${HARBOR_HF_HOME:-/data/home/macs/.cache/huggingface}"
export HF_HUB_CACHE="${HARBOR_HF_HUB_CACHE:-/data/home/macs/.cache/huggingface/hub}"
export VLLM_CONFIG_ROOT="$EVAL_ROOT/cache/vllm"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_cache_${USER}"
export TORCH_DISABLE_ADDR2LINE=1
export HYDRA_FULL_ERROR=1

# Source secrets
DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$HOME/.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

# ==============================================================================
# Read everything from YAMLs
# ==============================================================================

# vllm_serve.yaml → EVAL_VLLM_* env vars (model, TP, context len, parser, etc.)
eval "$($PYTHON_BIN "$REPO_ROOT/scripts/oa/eval/parse_vllm_config.py" "$VLLM_CONFIG")"
MODEL="${EVAL_VLLM_MODEL:?ERROR: model not set in $VLLM_CONFIG}"

# harbor YAML → dataset path + repo_id
read -r DATASET_PATH REPO_ID <<< "$($PYTHON_BIN -c "
import yaml
with open('$HARBOR_CONFIG') as f:
    cfg = yaml.safe_load(f)
ds = cfg.get('datasets', [{}])[0]
print(ds.get('path', ''), ds.get('repo_id', ''))
")"

echo "=============================================="
echo "Hazy Eval Harbor"
echo "=============================================="
echo "Model: $MODEL"
echo "Harbor config: $HARBOR_CONFIG"
echo "vLLM config: $VLLM_CONFIG"
echo "Dataset path: $DATASET_PATH"
echo "Dataset repo: $REPO_ID"
echo "Python: $($PYTHON_BIN --version)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "=============================================="

# ==============================================================================
# Cleanup trap
# ==============================================================================
cleanup() {
    echo "Cleaning up..."
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    echo "Cleanup done."
}
trap cleanup EXIT

# ==============================================================================
# Start vLLM (args from EVAL_VLLM_* env vars set by parse_vllm_config.py)
# ==============================================================================
mkdir -p "$EVAL_ROOT/logs"

VLLM_TP="${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_MAX_LEN="${EVAL_VLLM_MAX_MODEL_LEN:-}"
VLLM_TCP="${EVAL_VLLM_TOOL_CALL_PARSER:-}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.95}"
VLLM_PORT="${EVAL_VLLM_PORT:-8000}"

VLLM_ARGS="--model $MODEL --host 0.0.0.0 --port $VLLM_PORT"
VLLM_ARGS="$VLLM_ARGS --served-model-name $MODEL"
VLLM_ARGS="$VLLM_ARGS --tensor-parallel-size $VLLM_TP"
VLLM_ARGS="$VLLM_ARGS --gpu-memory-utilization $GPU_MEMORY_UTIL"
VLLM_ARGS="$VLLM_ARGS --trust-remote-code --enforce-eager"
[ -n "$VLLM_MAX_LEN" ] && VLLM_ARGS="$VLLM_ARGS --max-model-len $VLLM_MAX_LEN"
[ -n "$VLLM_TCP" ] && VLLM_ARGS="$VLLM_ARGS --enable-auto-tool-choice --tool-call-parser $VLLM_TCP"
VLLM_RP="${EVAL_VLLM_REASONING_PARSER:-}"
[ -n "$VLLM_RP" ] && VLLM_ARGS="$VLLM_ARGS --reasoning-parser $VLLM_RP"
VLLM_EXTRA="${EVAL_VLLM_EXTRA_ARGS:-}"
[ -n "$VLLM_EXTRA" ] && VLLM_ARGS="$VLLM_ARGS $VLLM_EXTRA"

echo "Starting vLLM: $VLLM_ARGS"
VLLM_LOG="$EVAL_ROOT/logs/vllm_${SLURM_JOB_ID}.log"

# Kill any stale process on the chosen vLLM port (e.g. leftover vLLM from a prior job)
if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null 2>&1; then
    echo "WARNING: port $VLLM_PORT already in use — killing stale process"
    fuser -k $VLLM_PORT/tcp 2>/dev/null || true
    sleep 2
fi

env TORCHDYNAMO_DISABLE=1 \
    TMPDIR="/tmp" \
    TRITON_CACHE_DIR="/tmp/triton_cache_${USER}" \
    TORCH_COMPILE_CACHE_DIR="/tmp/torch_cache_${USER}" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
$PYTHON_BIN -m vllm.entrypoints.openai.api_server $VLLM_ARGS \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Health check: wait up to 30 minutes
VLLM_TIMEOUT=1800
VLLM_START=$SECONDS
while ! curl -s http://localhost:$VLLM_PORT/v1/models 2>/dev/null | grep -q "$MODEL"; do
    ELAPSED=$(( SECONDS - VLLM_START ))
    if [ $ELAPSED -ge $VLLM_TIMEOUT ]; then
        echo "ERROR: vLLM server failed to start after ${VLLM_TIMEOUT}s"
        tail -50 "$VLLM_LOG" 2>/dev/null || true
        exit 1
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died"
        tail -50 "$VLLM_LOG" 2>/dev/null || true
        exit 1
    fi
    echo "Waiting for vLLM server (${ELAPSED}s/${VLLM_TIMEOUT}s)..."
    sleep 30
done
echo "vLLM server is ready (took $(( SECONDS - VLLM_START ))s)"

# ==============================================================================
# Ensure dataset exists (download to the path in the YAML if needed)
# ==============================================================================
if [ ! -d "$DATASET_PATH" ]; then
    if [ -z "$REPO_ID" ]; then
        echo "ERROR: Dataset not found at $DATASET_PATH and no repo_id in YAML"
        exit 1
    fi
    echo "Downloading dataset: $REPO_ID → $DATASET_PATH"
    $PYTHON_BIN "$SCRIPT_DIR/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_PATH"
fi
echo "Dataset: $DATASET_PATH ($(ls -d "$DATASET_PATH"/*/instruction.md 2>/dev/null | wc -l) tasks)"

# ==============================================================================
# Run Harbor — start or resume
# ==============================================================================
set +e
if [ -n "${EVAL_RESUME_JOB_PATH:-}" ]; then
    RESUME_FILTERS=""
    for etype in ${EVAL_RESUME_FILTER_ERRORS:-}; do
        RESUME_FILTERS="$RESUME_FILTERS -f $etype"
    done
    echo "Resuming Harbor job: $EVAL_RESUME_JOB_PATH"
    echo "Filtering error types: ${EVAL_RESUME_FILTER_ERRORS:-none}"
    $VENV_DIR/bin/harbor jobs resume -p "$EVAL_RESUME_JOB_PATH" $RESUME_FILTERS
else
    echo "Starting Harbor eval..."
    $VENV_DIR/bin/harbor jobs start --config "$HARBOR_CONFIG"
fi
SB_EXIT=$?
set -e

if [ ${SB_EXIT:-0} -ne 0 ]; then
    echo "Harbor eval exited with status: ${SB_EXIT}."
    exit ${SB_EXIT}
fi

echo "=============================================="
echo "Eval finished."
echo "=============================================="
