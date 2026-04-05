#!/bin/bash
#SBATCH -p main
#SBATCH --no-requeue
#SBATCH --time=24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=terminal

# ==============================================================================
# Unified Eval Harbor Script v4 — MBZ H200 Cluster
# ==============================================================================
# This script replaces all individual sbatch scripts with one parameterized version.
# Parameters are passed via environment variables (set by unified_eval_listener_v4.py).
#
# v3 Enhancements:
#   - Enhancement 1: Unified invalid error threshold (counts all errors except AgentTimeoutError and ContextLengthExceededError)
#   - Enhancement 4: Model retry tracking (appends to eval starts log)
#   - Enhancement 5: Timeout multiplier support
#
# Positional Args:
#   $1 = MODEL (HuggingFace model name, e.g., "org/model-name")
#   $2 = REPO_ID (HuggingFace dataset repo, e.g., "DCAgent/dev_set_v2")
#   $3 = BENCHMARK_ID (optional, DB benchmark UUID)
#   $4 = RUN_TAG (optional, job name - if provided, updates Pending job to Started)
#
# Environment Variables (with defaults):
#   EVAL_N_CONCURRENT       Harbor --n-concurrent (default: 64)
#   EVAL_N_ATTEMPTS         Harbor --n-attempts (default: 3)
#   EVAL_GPU_MEMORY_UTIL    VLLM --gpu-memory-utilization (default: 0.9)
#   EVAL_DAYTONA_THRESHOLD  Max invalid errors before abort (default: 3)
#   EVAL_VLLM_MAX_RETRIES   VLLM startup retries (default: 5)
#   EVAL_AGENT_PARSER       Agent parser type (default: "", use "xml" for swebench)
#   EVAL_SLURM_TIME         SLURM time limit (read at submit time)
#   EVAL_ENABLE_THINKING    Enable thinking blocks (default: false)
#   EVAL_AGENT_NAME         Agent name for harbor and DB entries (default: "terminus-2")
#   EVAL_STARTS_LOG         Path to shared eval starts log file (default: "")
#   EVAL_TIMEOUT_MULTIPLIER Harbor timeout multiplier (default: 1.0)
#   EVAL_VLLM_TENSOR_PARALLEL_SIZE  vLLM --tensor-parallel-size (default: 1)
#   EVAL_SLURM_GPUS         Number of GPUs to request (default: 1)
#   EVAL_INVALID_ERRORS_LOG Path to invalid errors log (default: jobs/invalid_errors.log)
# ==============================================================================

# === CONFIGURABLE PARAMETERS (with defaults) ===
N_CONCURRENT="${EVAL_N_CONCURRENT:-64}"
N_ATTEMPTS="${EVAL_N_ATTEMPTS:-3}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.9}"
ERROR_THRESHOLD="${EVAL_DAYTONA_THRESHOLD:-3}"
VLLM_MAX_RETRIES="${EVAL_VLLM_MAX_RETRIES:-5}"
AGENT_PARSER="${EVAL_AGENT_PARSER:-}"
ENABLE_THINKING="${EVAL_ENABLE_THINKING:-false}"
AGENT_NAME="${EVAL_AGENT_NAME:-terminus-2}"
EVAL_STARTS_LOG="${EVAL_STARTS_LOG:-}"
TIMEOUT_MULTIPLIER="${EVAL_TIMEOUT_MULTIPLIER:-1.0}"
TENSOR_PARALLEL_SIZE="${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}"

# === POSITIONAL ARGS ===
MODEL="${1:-mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1}"
REPO_ID="${2:-DCAgent/dev_set_v2}"
BENCHMARK_ID="${3:-}"
RUN_TAG_ARG="${4:-}"  # Optional: job name from listener (if provided, Pending entry exists)

# Create timestamp and safe names
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')

echo "=============================================="
echo "Unified Eval Harbor v4 - Starting job (MBZ H200)"
echo "=============================================="
echo "Model: $MODEL"
echo "Repository: $REPO_ID"
echo "Benchmark ID: ${BENCHMARK_ID:-<auto>}"
echo ""
echo "Parameters:"
echo "  N_CONCURRENT: $N_CONCURRENT"
echo "  N_ATTEMPTS: $N_ATTEMPTS"
echo "  GPU_MEMORY_UTIL: $GPU_MEMORY_UTIL"
echo "  ERROR_THRESHOLD: $ERROR_THRESHOLD"
echo "  VLLM_MAX_RETRIES: $VLLM_MAX_RETRIES"
echo "  AGENT_PARSER: ${AGENT_PARSER:-<default>}"
echo "  ENABLE_THINKING: $ENABLE_THINKING"
echo "  AGENT_NAME: $AGENT_NAME"
echo "  TIMEOUT_MULTIPLIER: $TIMEOUT_MULTIPLIER"
echo "  TENSOR_PARALLEL_SIZE: $TENSOR_PARALLEL_SIZE"
echo "  EVAL_STARTS_LOG: ${EVAL_STARTS_LOG:-<disabled>}"
echo "=============================================="

# === MBZ ENVIRONMENT SETUP ===
SCRATCH="/mnt/weka/home/richard.zhuang"
DCFT="$SCRATCH/OpenThoughts-Agent"

# Set up environment variables
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_ADDRESS=${RAY_ADDRESS:-}
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm"
export VLLM_CONFIG_ROOT="$SCRATCH/.cache/vllm_config"
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton"
export FLASHINFER_WORKSPACE_BASE="$SCRATCH/.cache/flashinfer"
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export HYDRA_FULL_ERROR=1
export HF_CACHE_DIR="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
# XET staging cache — must be user-writable (shared HF_HOME/xet/ has permission issues)
export HF_XET_CACHE="$SCRATCH/.cache/hf_xet_cache"
mkdir -p "$HF_XET_CACHE"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONPATH="${DCFT}${PYTHONPATH:+:$PYTHONPATH}"

# DB/API secrets
SECRETS_FILE="${EVAL_SECRETS_FILE:-$HOME/secrets.env}"
echo "Sourcing secrets from: $SECRETS_FILE"
source "$SECRETS_FILE"

# Conda env (configurable via EVAL_CONDA_ENV, default: otagent)
CONDA_ENV="${EVAL_CONDA_ENV:-otagent}"
eval "$(conda shell.bash hook 2>/dev/null)" || source "$SCRATCH/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "Conda env: $CONDA_ENV"

# Create cache dirs
mkdir -p "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT" "$TRITON_CACHE_DIR" \
         "$FLASHINFER_WORKSPACE_BASE" "$UV_CACHE_DIR" "$HF_CACHE_DIR" \
         experiments/logs

# === PRE-FLIGHT: Validate model architecture before starting vLLM ===
echo "Validating model architecture..."
python - "$MODEL" <<'PREFLIGHT'
import json, sys, os

model = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else ""
if not model:
    print("WARNING: No MODEL set, skipping preflight check")
    sys.exit(0)

# Blocklist of architecture prefixes that vLLM cannot serve
UNSUPPORTED_PREFIXES = ["FSDP"]

try:
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(model, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    archs = cfg.get("architectures", [])
    print(f"Model architectures: {archs}")
    for arch in archs:
        for prefix in UNSUPPORTED_PREFIXES:
            if arch.startswith(prefix):
                print(f"FATAL: Architecture '{arch}' is not supported by vLLM.", file=sys.stderr)
                print(f"This model appears to be a raw training checkpoint (e.g. FSDP).", file=sys.stderr)
                print(f"Convert it to standard HuggingFace format before evaluating.", file=sys.stderr)
                sys.exit(1)
except Exception as e:
    # If we can't download config.json, vLLM will also fail — let it fail there
    # with its own error message rather than blocking on network issues here
    print(f"WARNING: Preflight check could not validate model: {e}")
    sys.exit(0)
PREFLIGHT

if [ $? -ne 0 ]; then
    echo "FATAL: Model architecture not supported. Aborting job."
    exit 1
fi

# ==============================================================================
# Fix: Qwen3.5 VL fine-tunes missing preprocessor_config.json
# ==============================================================================
echo "Checking for Qwen3.5 preprocessor fix..."
MODEL_SNAP_DIR=$(python3 -c "
import sys
from huggingface_hub import snapshot_download
path = snapshot_download('$MODEL')
sys.stdout.write(path)
" 2>/dev/null) || true

if [ -n "$MODEL_SNAP_DIR" ] && [ -f "$MODEL_SNAP_DIR/config.json" ]; then
    IS_QWEN35_VL=$(python3 -c "
import json
with open('$MODEL_SNAP_DIR/config.json') as f:
    cfg = json.load(f)
archs = cfg.get('architectures', [])
if any('Qwen3_5ForConditional' in a for a in archs):
    print('yes')
" 2>/dev/null)

    if [ "$IS_QWEN35_VL" = "yes" ]; then
        if [ ! -f "$MODEL_SNAP_DIR/preprocessor_config.json" ]; then
            echo "Qwen3.5 VL fine-tune missing preprocessor_config.json — copying from base model"
            BASE_PREP=$(python3 -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('Qwen/Qwen3.5-9B', 'preprocessor_config.json'))
" 2>/dev/null)
            if [ -n "$BASE_PREP" ] && [ -f "$BASE_PREP" ]; then
                cp "$BASE_PREP" "$MODEL_SNAP_DIR/preprocessor_config.json"
                echo "Copied preprocessor_config.json to $MODEL_SNAP_DIR"
            fi
        else
            echo "preprocessor_config.json already present in snapshot"
        fi
        export HF_HUB_OFFLINE=1
        echo "Set HF_HUB_OFFLINE=1 for Qwen3.5 VL model loading"
    fi
fi

# Start VLLM server
VLLM_ARGS="--host 0.0.0.0 --port 8000"
VLLM_ARGS="$VLLM_ARGS --served-model-name $MODEL"
VLLM_ARGS="$VLLM_ARGS --enforce-eager"
VLLM_ARGS="$VLLM_ARGS --gpu-memory-utilization $GPU_MEMORY_UTIL"
VLLM_ARGS="$VLLM_ARGS --tensor-parallel-size $TENSOR_PARALLEL_SIZE"

# Baseline model config overrides (set by listener from baseline_model_configs.yaml)
if [ -n "${EVAL_VLLM_MAX_MODEL_LEN:-}" ]; then
    VLLM_ARGS="$VLLM_ARGS --max-model-len $EVAL_VLLM_MAX_MODEL_LEN"
    echo "  Override: --max-model-len $EVAL_VLLM_MAX_MODEL_LEN"
fi
if [ -n "${EVAL_VLLM_SWAP_SPACE:-}" ]; then
    VLLM_ARGS="$VLLM_ARGS --swap-space $EVAL_VLLM_SWAP_SPACE"
    echo "  Override: --swap-space $EVAL_VLLM_SWAP_SPACE"
fi
if [ -n "${EVAL_VLLM_TRUST_REMOTE_CODE:-}" ]; then
    VLLM_ARGS="$VLLM_ARGS --trust-remote-code"
    echo "  Override: --trust-remote-code"
fi
if [ -n "${EVAL_VLLM_TOOL_CALL_PARSER:-}" ]; then
    VLLM_ARGS="$VLLM_ARGS --tool-call-parser $EVAL_VLLM_TOOL_CALL_PARSER"
    echo "  Override: --tool-call-parser $EVAL_VLLM_TOOL_CALL_PARSER"
fi
if [ -n "${EVAL_VLLM_REASONING_PARSER:-}" ]; then
    VLLM_ARGS="$VLLM_ARGS --reasoning-parser $EVAL_VLLM_REASONING_PARSER"
    echo "  Override: --reasoning-parser $EVAL_VLLM_REASONING_PARSER"
fi
if [ -n "${EVAL_VLLM_EXTRA_ARGS:-}" ]; then
    VLLM_ARGS="$VLLM_ARGS $EVAL_VLLM_EXTRA_ARGS"
    echo "  Override: extra_args=$EVAL_VLLM_EXTRA_ARGS"
fi

echo "Starting vLLM: vllm serve $MODEL $VLLM_ARGS"
vllm serve "$MODEL" \
  $VLLM_ARGS \
  > "experiments/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Cleaning up..."
    kill $VLLM_PID 2>/dev/null || true
    conda deactivate || true
}
trap cleanup EXIT

# Wait for VLLM server to start with healthcheck
RETRY_INTERVAL=100
for i in $(seq 1 $VLLM_MAX_RETRIES); do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "VLLM server is ready"
        break
    fi
    echo "Waiting for VLLM server to start (attempt $i/$VLLM_MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
    if [ $i -eq $VLLM_MAX_RETRIES ]; then
        echo "VLLM server failed to start"
        exit 1
    fi
done

# Restore online mode for dataset download and uploads
unset HF_HUB_OFFLINE

# Get the dataset path using the specified repo_id
echo "Downloading/locating dataset: $REPO_ID"
DATASET_PATH=$(python "$DCFT/eval/MBZ/snapshot_download.py" "$REPO_ID" | grep DATASET_PATH | tail -n 1 | cut -d'=' -f2)
if [ -z "${DATASET_PATH:-}" ]; then
    echo "Failed to get dataset path"
    exit 1
fi
echo "Using dataset path: $DATASET_PATH"

# Construct run dir (harbor honors --job-name in jobs/<job-name>)
# Use provided RUN_TAG if available (from listener), otherwise generate one
if [ -n "$RUN_TAG_ARG" ]; then
    RUN_TAG="$RUN_TAG_ARG"
    echo "Using RUN_TAG from listener: $RUN_TAG"
else
    RUN_TAG="${SAFE_REPO}_${SAFE_MODEL}_${TIMESTAMP}"
    echo "Generated new RUN_TAG: $RUN_TAG"
fi
RUN_DIR="jobs/${RUN_TAG}"
echo "Job dir: ${RUN_DIR}"

# Update or create DB row with status='Started' BEFORE running eval
echo "Updating/Creating DB job entry..."
export MODEL="$MODEL"
export REPO_ID="$REPO_ID"
export RUN_TAG="$RUN_TAG"
export RUN_TAG_ARG="$RUN_TAG_ARG"
export SLURM_JOB_ID="$SLURM_JOB_ID"
export N_CONCURRENT="$N_CONCURRENT"
export N_ATTEMPTS="$N_ATTEMPTS"
export AGENT_NAME="$AGENT_NAME"
export TIMEOUT_MULTIPLIER="$TIMEOUT_MULTIPLIER"

# Get harbor package version
HARBOR_VERSION=$(python -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION

python - <<'PY'
import os, sys, json
from database.unified_db.utils import create_job_entry_started, update_job_status_to_started

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]
run_tag = os.environ["RUN_TAG"]
run_tag_arg = os.environ.get("RUN_TAG_ARG", "")
slurm_job_id = os.environ["SLURM_JOB_ID"]
harbor_version = os.environ.get("HARBOR_VERSION", "unknown")
n_concurrent = int(os.environ.get("N_CONCURRENT", "64"))
n_attempts = int(os.environ.get("N_ATTEMPTS", "3"))
agent_name = os.environ.get("AGENT_NAME", "terminus-2")
timeout_multiplier = float(os.environ.get("TIMEOUT_MULTIPLIER", "1.0"))

# Build config dict with timeout_multiplier (Enhancement 5)
config = {"agent": agent_name, "env": "daytona", "timeout_multiplier": timeout_multiplier}

# If RUN_TAG was provided from listener, a Pending entry should exist - update it
# Otherwise, create a new Started entry (backward compatibility)
# Fallback: if Pending entry not found (e.g. stale entry reused by listener),
# create a fresh Started entry instead of failing.
result = None
if run_tag_arg:
    print(f"Updating Pending job to Started: {run_tag}")
    result = update_job_status_to_started(
        job_name=run_tag,
        n_trials=n_concurrent,
        n_rep_eval=n_attempts,
        config=config,
        harbor_package_version=harbor_version
    )
    if not result.get("success"):
        print(f"WARNING: Pending entry not found ({result.get('error')}), creating Started entry directly")
        result = None  # fall through to create_job_entry_started

if result is None:
    print(f"Creating new Started job entry: {run_tag}")
    result = create_job_entry_started(
        model_hf_name=model_hf,
        benchmark_hf_name=dataset_hf,
        job_name=run_tag,
        username=os.environ.get("UPLOAD_USERNAME", os.environ.get("USER", "unknown")),
        slurm_job_id=slurm_job_id,
        harbor_package_version=harbor_version,
        agent_name=agent_name,
        config=config,
        n_trials=n_concurrent,
        n_rep_eval=n_attempts
    )

if not result.get("success"):
    print(f"ERROR: {result.get('error')}", file=sys.stderr)
    sys.exit(1)

# IMPORTANT: Save the DB job_id for later update
db_job_id = result["job"]["id"]
print(f"DB job ready with ID: {db_job_id}")
PY

if [ $? -ne 0 ]; then
    echo "Failed to create/update DB job entry"
    exit 1
fi

# Enhancement 4: Append to eval starts log (model retry tracking)
if [ -n "$EVAL_STARTS_LOG" ]; then
    mkdir -p "$(dirname "$EVAL_STARTS_LOG")"
    echo "$(date -Iseconds) $MODEL $REPO_ID $RUN_TAG" >> "$EVAL_STARTS_LOG"
    echo "Logged eval start to: $EVAL_STARTS_LOG"
fi

# Extract the job_id from Python output and save it
DB_JOB_ID=$(python - <<'PY'
import os, sys
from database.unified_db.utils import get_latest_job_for_model_benchmark

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]

result = get_latest_job_for_model_benchmark(model_hf, dataset_hf)
if result and result.get("id"):
    print(result["id"])
else:
    sys.exit(1)
PY
)

if [ -z "$DB_JOB_ID" ]; then
    echo "Failed to get DB job ID"
    exit 1
fi

echo "DB job entry ready: $DB_JOB_ID"

export HARBOR_JOBS_DIR="/tmp/harbor_jobs"
mkdir -p "$HARBOR_JOBS_DIR"

# Let Terminus2 auto-resolve model_info for hosted_vllm (includes cost fields)
export TERMINUS_MODEL_MAX_TOKENS=32768
export TERMINUS_MODEL_MAX_OUTPUT_TOKENS=16384

# Build harbor command with configurable params
HARBOR_CMD="harbor jobs start"
HARBOR_CMD="$HARBOR_CMD -p \"$DATASET_PATH\""
HARBOR_CMD="$HARBOR_CMD --n-concurrent $N_CONCURRENT"
HARBOR_CMD="$HARBOR_CMD --agent \"$AGENT_NAME\""
HARBOR_CMD="$HARBOR_CMD --model \"hosted_vllm/$MODEL\""
HARBOR_CMD="$HARBOR_CMD --env \"daytona\""
AUTO_SNAPSHOT="${EVAL_AUTO_SNAPSHOT:-false}"
HARBOR_CMD="$HARBOR_CMD --environment-kwarg auto_snapshot=$AUTO_SNAPSHOT"
HARBOR_CMD="$HARBOR_CMD --agent-kwarg \"api_base=http://localhost:8000/v1\""
HARBOR_CMD="$HARBOR_CMD --agent-kwarg \"key=fake_key\""
HARBOR_CMD="$HARBOR_CMD --agent-kwarg \"max_tokens=16384\""
HARBOR_CMD="$HARBOR_CMD --agent-kwarg 'model_info={\"max_output_tokens\":16384,\"max_input_tokens\":32768,\"input_cost_per_token\":0.0,\"output_cost_per_token\":0.0}'"
HARBOR_CMD="$HARBOR_CMD --n-attempts $N_ATTEMPTS"
HARBOR_CMD="$HARBOR_CMD --export-traces"
HARBOR_CMD="$HARBOR_CMD --job-name \"$RUN_TAG\""
CONFIG_YAML="${EVAL_CONFIG_YAML:-dcagent_eval_config.yaml}"
HARBOR_CMD="$HARBOR_CMD --config \"$DCFT/eval/MBZ/$CONFIG_YAML\""

# Add parser arg if specified (e.g., for swebench)
if [ -n "$AGENT_PARSER" ]; then
    HARBOR_CMD="$HARBOR_CMD --agent-kwarg \"parser=$AGENT_PARSER\""
fi

# Add extra_body for thinking control (only if enable_thinking is true)
if [ "$ENABLE_THINKING" = "true" ]; then
    HARBOR_CMD="$HARBOR_CMD --agent-kwarg 'extra_body={\"chat_template_kwargs\":{\"enable_thinking\":true}}'"
    echo "Thinking blocks enabled via extra_body"
fi

# Enhancement 5: Add timeout multiplier if != 1.0
if [ "$TIMEOUT_MULTIPLIER" != "1.0" ]; then
    HARBOR_CMD="$HARBOR_CMD --timeout-multiplier $TIMEOUT_MULTIPLIER"
    echo "Using timeout multiplier: $TIMEOUT_MULTIPLIER"
fi

echo "Running: $HARBOR_CMD"

# Run sandbox evaluation
set +e
eval $HARBOR_CMD
SB_EXIT=$?
set -e

# Save originals for exact round-trip later
mkdir -p "$RUN_DIR"
{
  echo "MODEL=$MODEL"
  echo "REPO_ID=$REPO_ID"
  echo "TIMESTAMP=$TIMESTAMP"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "DB_JOB_ID=$DB_JOB_ID"
  echo "RUN_TAG=$RUN_TAG"
  echo "RUN_TAG_ARG=$RUN_TAG_ARG"
  echo "N_CONCURRENT=$N_CONCURRENT"
  echo "N_ATTEMPTS=$N_ATTEMPTS"
  echo "GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
  echo "ERROR_THRESHOLD=$ERROR_THRESHOLD"
  echo "AGENT_PARSER=$AGENT_PARSER"
  echo "ENABLE_THINKING=$ENABLE_THINKING"
  echo "TIMEOUT_MULTIPLIER=$TIMEOUT_MULTIPLIER"
} > "$RUN_DIR/meta.env"

# If eval failed, don't attempt upload
if [ ${SB_EXIT:-0} -ne 0 ]; then
  echo "harbor run exited with non-zero status: ${SB_EXIT}. Skipping upload."
  exit ${SB_EXIT}
fi

# Ensure run dir exists; no fallback
if [ ! -d "$RUN_DIR" ]; then
  echo "Expected run directory not found: $RUN_DIR"
  exit 2
fi


# ---- Enhancement 1: Check for invalid errors before upload ----
# Count ALL error types except AgentTimeoutError and ContextLengthExceededError and SummarizationTimeout and SummarizationTimeoutError (normal timeout) toward the threshold.
RESULT_FILE="$RUN_DIR/result.json"
ERROR_LOG="${EVAL_INVALID_ERRORS_LOG:-jobs/invalid_errors.log}"

if [ -f "$RESULT_FILE" ]; then
  echo "Checking for invalid errors in $RESULT_FILE..."

  # Count invalid errors using Python (all errors except AgentTimeoutError and ContextLengthExceededError and SummarizationTimeout and SummarizationTimeoutError)
  INVALID_ERROR_COUNT=$(python3 -c "
import json
import sys

VALID_ERROR_TYPES = {'AgentTimeoutError', 'ContextLengthExceededError', 'SummarizationTimeout', 'SummarizationTimeoutError'}

try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)

    invalid_trials = set()
    error_counts = {}

    if 'stats' in data and 'evals' in data['stats']:
        for eval_key, eval_data in data['stats']['evals'].items():
            if 'exception_stats' in eval_data:
                for error_type, trial_names in eval_data['exception_stats'].items():
                    if error_type not in VALID_ERROR_TYPES and isinstance(trial_names, list):
                        invalid_trials.update(trial_names)
                        error_counts[error_type] = error_counts.get(error_type, 0) + len(trial_names)

    print(len(invalid_trials))

    # Log breakdown to stderr
    if invalid_trials:
        print(f'Found {len(invalid_trials)} unique invalid error trial(s):', file=sys.stderr)
        for etype, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f'  {etype}: {count}', file=sys.stderr)

except Exception as e:
    print(f'Error parsing result.json: {e}', file=sys.stderr)
    print('0')
" 2>&1 | tail -n 1)

  echo "Invalid error count: ${INVALID_ERROR_COUNT}"


  # If too many invalid errors, log and skip upload
  if [ "${INVALID_ERROR_COUNT:-0}" -gt "$ERROR_THRESHOLD" ]; then
    echo "Job has ${INVALID_ERROR_COUNT} invalid errors (> ${ERROR_THRESHOLD}), skipping upload"

    # Log to error file
    {
      echo "==============================================="
      echo "Timestamp: $(date)"
      echo "Job: ${RUN_TAG}"
      echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
      echo "Model: ${MODEL}"
      echo "Repo: ${REPO_ID}"
      echo "Invalid Errors: ${INVALID_ERROR_COUNT}"
      echo "Error Threshold: ${ERROR_THRESHOLD}"
      echo "Result file: ${RESULT_FILE}"

      # Extract error details with breakdown by type
      python3 -c "
import json

VALID_ERROR_TYPES = {'AgentTimeoutError', 'ContextLengthExceededError', 'SummarizationTimeout', 'SummarizationTimeoutError'}

with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)
if 'stats' in data and 'evals' in data['stats']:
    for eval_key, eval_data in data['stats']['evals'].items():
        if 'exception_stats' in eval_data:
            eval_has_errors = False
            for error_type, trial_names in eval_data['exception_stats'].items():
                if error_type not in VALID_ERROR_TYPES and isinstance(trial_names, list) and trial_names:
                    if not eval_has_errors:
                        print(f'Eval: {eval_key}')
                        eval_has_errors = True
                    print(f'  {error_type} ({len(trial_names)}):')
                    for i, tid in enumerate(trial_names[:5], 1):
                        print(f'    {i}. {tid}')
                    if len(trial_names) > 5:
                        print(f'    ... and {len(trial_names) - 5} more')
"
      echo "==============================================="
    } >> "$ERROR_LOG"

    echo "Error details logged to: $ERROR_LOG"
    echo "Job completed but not uploaded due to excessive invalid errors"
    exit 0  # Exit successfully but skip upload
  fi
else
  echo "Warning: result.json not found, continuing with upload"
fi



# ---- Upload results to DB ----

export RUN_DIR="$RUN_DIR"
export UPLOAD_USERNAME="${EVAL_UPLOAD_USERNAME:-$USER}"
export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
export RUN_TAG="$RUN_TAG"
UPLOAD_LOG="experiments/logs/upload_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$UPLOAD_LOG")"

echo "Uploading results from: $RUN_DIR" | tee -a "$UPLOAD_LOG"
echo "Using username=${UPLOAD_USERNAME}, mode=${UPLOAD_MODE}" | tee -a "$UPLOAD_LOG"

# Run the uploader (uses database.unified_db from the repo)
python - <<'PY' 2>&1 | tee -a "$UPLOAD_LOG"
import os, sys
from database.unified_db.utils import upload_eval_results
import re
import hashlib


def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    """
    Sanitize a Hugging Face repo_id to comply with naming rules.
    Keeps org prefix (e.g. 'mlfoundations-dev/') and cleans up the rest.
    No extra '-' before hash suffix.
    """
    def collapse(s: str) -> str:
        prev = None
        while s != prev:
            prev = s
            s = s.replace("--", "-").replace("..", ".")
        return s

    org, name = repo_id.split("/", 1) if "/" in repo_id else (None, repo_id)

    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = collapse(name).strip("-.")

    if not name:
        name = "repo"

    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest))
        base = name[:keep].rstrip("-.")
        if not base:
            base = "r"
        name = f"{base}{digest}"  # no '-' before hash
        name = collapse(name).strip("-.")

    # final cleanup
    name = collapse(name).strip("-.")
    if name[0] in "-.":
        name = "r" + name[1:]
    if name[-1] in "-.":
        name = name[:-1] + "0"

    return f"{org}/{name}" if org else name


run_dir   = os.environ["RUN_DIR"]
run_tag   = os.environ["RUN_TAG"]
username  = os.environ.get("UPLOAD_USERNAME", os.environ.get("USER", "unknown"))
error_mode= os.environ.get("UPLOAD_MODE", "skip_on_error")
hf_repo_id = sanitize_hf_repo_id(f"DCAgent2/{run_tag}")
hf_token = os.environ["HF_TOKEN"]
print(f"[uploader] upload_eval_results(path={run_dir!r}, username={username!r}, error_mode={error_mode!r}, hf_repo_id={hf_repo_id!r})")
upload_eval_results(run_dir, username=username, error_mode=error_mode, hf_token=hf_token, hf_repo_id=hf_repo_id, register_benchmark=True)
print("[uploader] done.")
PY
UPLOAD_EXIT=${PIPESTATUS[0]}

if [ $UPLOAD_EXIT -ne 0 ]; then
  echo "Upload failed with exit code: $UPLOAD_EXIT"
  exit $UPLOAD_EXIT
fi

echo "Eval and upload finished successfully."
