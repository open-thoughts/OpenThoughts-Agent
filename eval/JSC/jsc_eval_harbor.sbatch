#!/bin/bash
#SBATCH -p dc-gpu-devel
#SBATCH --time=02:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --account=synthlaion
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=eval_harbor

# ============================================================================
# JSC Harbor Evaluation Script
# Based on TACC's tacc_eval_harbor.sbatch
# Adapted for JSC infrastructure (JURECA/JUWELS/JUPITER)
# 
# Usage: sbatch jsc_eval_harbor.sbatch MODEL DATASET [TASK_FILTER] [N_CONCURRENT] [N_ATTEMPTS] [ENABLE_UPLOAD]
#        sbatch jsc_eval_harbor.sbatch "Qwen/Qwen2.5-Coder-7B-Instruct" "swebench-verified@1.0" "" 8 1 "true"
# ============================================================================

# Workaround to avoid core dumps filling up space
ulimit -c 0

# Create timestamp and safe names
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

MODEL="${1:-Qwen/Qwen2.5-Coder-7B-Instruct}"
DATASET="${2:-swebench-verified@1.0}"
TASK_FILTER="${3:-}"  # Optional: specific task to run (e.g., "django__django-13406")
N_CONCURRENT="${4:-8}"
N_ATTEMPTS="${5:-1}"
ENABLE_UPLOAD="${6:-false}"  # Set to "true" to enable upload after evaluation

# Detect if using Harbor registry dataset (contains @version)
if [[ "$DATASET" == *"@"* ]]; then
    USE_REGISTRY=true
    DATASET_NAME="${DATASET%%@*}"  # Extract name before @
    DATASET_VERSION="${DATASET##*@}"  # Extract version after @
    echo "Using Harbor registry dataset"
else
    USE_REGISTRY=false
    REPO_ID="$DATASET"
    echo "Using HuggingFace dataset repository"
fi

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
SAFE_DATASET=$(echo "$DATASET" | tr '/:@' '_')

echo "============================================"
echo "JSC Harbor Evaluation"
echo "============================================"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
if [ -n "$TASK_FILTER" ]; then
    echo "Task filter: $TASK_FILTER"
fi
echo "Timestamp: $TIMESTAMP"
echo "Node: $HOSTNAME"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "============================================"

# ============================================================================
# Environment Setup
# ============================================================================

# Purge existing modules
module purge

# Load JSC modules (adjust based on your needs)
# Option 1: Use latest 2024 stack
ml Stages/2024
ml StdEnv/2024
ml Python/3.11.3
ml CUDA/12
ml cuDNN/8.9.5.29-CUDA-12
ml NCCL/default-CUDA-12

# Note: Using system git (/usr/bin/git) for Harbor dataset download

# Option 2: Or use a shared Mamba environment with Python 3.13
# SHARED_MAMBA_ENV=/p/project/ccstdl/shared/envs/mamba
# SHARED_ENV=/p/project/ccstdl/shared/envs/eval_py313
# source ${SHARED_MAMBA_ENV}/bin/activate ${SHARED_ENV}

# Activate your conda/mamba environment with Python 3.13
# IMPORTANT: Adjust this path to YOUR environment
MAMBA_ROOT="/p/project1/ccstdl/${USER}/mamba"
CONDA_ENV_PATH="/p/project1/ccstdl/${USER}/eval_py313"

# Initialize mamba and activate environment by path
source ${MAMBA_ROOT}/bin/activate ${CONDA_ENV_PATH}

echo "Python version: $(python --version)"
echo "Environment: $CONDA_ENV_PATH"
echo "Which python: $(which python)"
echo "Which harbor: $(which harbor)"

# ============================================================================
# Cache and Storage Configuration
# ============================================================================

# vLLM cache directories
export VLLM_USE_V1=1
export VLLM_CACHE_ROOT=/p/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm_cache
export VLLM_CONFIG_ROOT=/p/scratch/${SLURM_JOB_ACCOUNT}/${USER}/vllm_config
mkdir -p "$VLLM_CACHE_ROOT" "$VLLM_CONFIG_ROOT"

# Triton compiler cache
export TRITON_CACHE_DIR=/p/scratch/${SLURM_JOB_ACCOUNT}/${USER}/triton_cache
export TRITON_DUMP_DIR=/p/scratch/${SLURM_JOB_ACCOUNT}/${USER}/triton_dump
export TRITON_OVERRIDE_DIR=/p/scratch/${SLURM_JOB_ACCOUNT}/${USER}/triton_override
mkdir -p "$TRITON_CACHE_DIR" "$TRITON_DUMP_DIR" "$TRITON_OVERRIDE_DIR"

# UV cache
export UV_CACHE_DIR=/p/scratch/${SLURM_JOB_ACCOUNT}/${USER}/uv_cache
mkdir -p "$UV_CACHE_DIR"

# Hugging Face cache
export HF_CACHE_DIR=/p/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf_cache
export HF_HUB_CACHE=/p/scratch/${SLURM_JOB_ACCOUNT}/${USER}/hf_hub
export HF_HOME="$HF_CACHE_DIR"
mkdir -p "$HF_CACHE_DIR" "$HF_HUB_CACHE"

# Ray configuration
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Hydra (for debugging)
export HYDRA_FULL_ERROR=1

# ============================================================================
# Load Secrets (if needed)
# ============================================================================

# Load secrets from secret.env file (API keys, tokens, etc.)
# Use SLURM_SUBMIT_DIR which points to the directory where sbatch was run
SECRET_ENV_PATH="${SLURM_SUBMIT_DIR:-$(pwd)}/secret.env"

if [ -f "$SECRET_ENV_PATH" ]; then
    source "$SECRET_ENV_PATH"
    echo "Loaded secrets from $SECRET_ENV_PATH"
else
    echo "Warning: No secret.env found at $SECRET_ENV_PATH"
    echo "Create one with: export DAYTONA_API_KEY='your-key'"
fi

# ============================================================================
# Daytona Configuration
# ============================================================================

# Set Daytona API key
# IMPORTANT: Replace with your actual API key or load from a secure file
export DAYTONA_API_KEY="${DAYTONA_API_KEY:-}"

if [ -z "$DAYTONA_API_KEY" ]; then
    echo "ERROR: DAYTONA_API_KEY is not set!"
    echo "Please set it in your environment or in the secret.env file"
    echo "You can set it by running: export DAYTONA_API_KEY='your-api-key'"
    exit 1
fi

# Optional: Set Daytona API URL if using a custom instance
# export DAYTONA_API_URL="https://your-daytona-instance.com"

echo "Daytona API key configured: ${DAYTONA_API_KEY:+***set***}"

# ============================================================================
# Verify Dependencies
# ============================================================================

echo "Verifying dependencies..."

# Check if harbor (sandbox CLI) is installed
if ! command -v harbor &> /dev/null; then
    echo "ERROR: harbor command not found. Install with: uv pip install -e /path/to/sandboxes --group dev"
    exit 1
fi

# Check Python version (must be 3.13 for terminal-bench and sandboxes)
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" != "3.13" ]]; then
    echo "WARNING: Python 3.13 recommended for terminal-bench/sandboxes. Current: $PYTHON_VERSION"
fi

# Test harbor installation
harbor --help >/dev/null || {
    echo "ERROR: harbor command failed. Check installation."
    exit 1
}

echo "Dependencies verified."

# ============================================================================
# Start vLLM Server
# ============================================================================

mkdir -p experiments/logs
VLLM_LOG="experiments/logs/vllm_${SLURM_JOB_ID}.log"

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "Log: $VLLM_LOG"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$MODEL" \
  --max-model-len 49152 \
  --gpu-memory-utilization 0.90 \
  > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    if [ -n "${VLLM_PID:-}" ]; then
        kill $VLLM_PID 2>/dev/null || true
        echo "Killed vLLM server (PID $VLLM_PID)"
    fi
}
trap cleanup EXIT

# ============================================================================
# Wait for vLLM Server
# ============================================================================

echo "Waiting for vLLM server to be ready..."
MAX_RETRIES=60  # Increased for torch.compile on first run
RETRY_INTERVAL=10

for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "vLLM server is ready!"
        break
    fi
    
    # Check if vLLM process is still alive
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server died during startup"
        echo "Last 100 lines of log:"
        tail -n 100 "$VLLM_LOG"
        exit 1
    fi
    
    echo "Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
    
    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start within timeout"
        echo "Last 100 lines of log:"
        tail -n 100 "$VLLM_LOG"
        exit 1
    fi
done

# ============================================================================
# Harbor Registry Dataset (Hugging Face)
# ============================================================================

if [ "$USE_REGISTRY" = false ]; then
    echo "Downloading/locating HuggingFace dataset: $REPO_ID"

    # Use the snapshot_download.py script
    SNAPSHOT_SCRIPT="${SLURM_SUBMIT_DIR:-$(pwd)}/snapshot_download.py"
    if [ ! -f "$SNAPSHOT_SCRIPT" ]; then
        # Try parent directory (tacc/)
        SNAPSHOT_SCRIPT="${SLURM_SUBMIT_DIR:-$(pwd)}/../tacc/snapshot_download.py"
    fi

    if [ ! -f "$SNAPSHOT_SCRIPT" ]; then
        echo "ERROR: snapshot_download.py not found"
        exit 1
    fi

    # Use conda environment python (not system python)
    DATASET_PATH=$("${CONDA_ENV_PATH}/bin/python" "$SNAPSHOT_SCRIPT" "$REPO_ID" | grep DATASET_PATH | tail -n 1 | cut -d'=' -f2)

    if [ -z "${DATASET_PATH:-}" ]; then
        echo "ERROR: Failed to get dataset path"
        exit 1
    fi

    echo "Dataset path: $DATASET_PATH"

    # Verify dataset structure
    TASK_COUNT=$(find "$DATASET_PATH" -mindepth 1 -maxdepth 1 -type d -name "*" ! -name ".*" | wc -l)
    echo "Found $TASK_COUNT task directories"

    if [ "$TASK_COUNT" -eq 0 ]; then
        echo "ERROR: No task directories found in dataset"
        exit 1
    fi
else
    echo "Using Harbor registry dataset: $DATASET"
    echo "Dataset will be downloaded automatically by Harbor"
fi

# ============================================================================
# Run Harbor Evaluation
# ============================================================================

RUN_TAG="${SAFE_DATASET}_${SAFE_MODEL}_${TIMESTAMP}"
RUN_DIR="jobs/${RUN_TAG}"

echo "============================================"
echo "Starting Harbor evaluation"
echo "============================================"
echo "Run tag: $RUN_TAG"
echo "Output directory: $RUN_DIR"
echo "Concurrency: $N_CONCURRENT"
echo "Attempts per task: $N_ATTEMPTS"
if [ -n "$TASK_FILTER" ]; then
    echo "Task filter: $TASK_FILTER (running single task)"
fi
echo "============================================"

# Find config file (try multiple locations)
CONFIG_FILE="${SLURM_SUBMIT_DIR:-$(pwd)}/jsc_eval_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="$(dirname "$0")/jsc_eval_config.yaml"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found"
    exit 1
fi

echo "Using config: $CONFIG_FILE"

# Build Harbor command based on dataset type
# Note: 'harbor run' and 'harbor jobs start' are identical (run is an alias)
set +e

if [ "$USE_REGISTRY" = true ]; then
    # Use Harbor registry dataset (e.g., swebench-verified@1.0)
    echo "Running with Harbor registry dataset..."
    
    HARBOR_CMD=(
        harbor run
        -d "$DATASET"
        --n-concurrent $N_CONCURRENT
        --agent terminus-2
        --model "hosted_vllm/$MODEL"
        --env "daytona"
        --agent-kwarg "api_base=http://localhost:8000/v1"
        --agent-kwarg "key=fake_key"
        --n-attempts $N_ATTEMPTS
        --job-name "$RUN_TAG"
        --config "$CONFIG_FILE"
    )
    
    # Add task filter if specified
    if [ -n "$TASK_FILTER" ]; then
        HARBOR_CMD+=(-t "$TASK_FILTER")
    fi
    
else
    # Use local/HuggingFace dataset path
    echo "Running with local dataset path..."
    
    HARBOR_CMD=(
        harbor run
        -p "$DATASET_PATH"
        --n-concurrent $N_CONCURRENT
        --agent terminus-2
        --model "hosted_vllm/$MODEL"
        --env "daytona"
        --agent-kwarg "api_base=http://localhost:8000/v1"
        --agent-kwarg "key=fake_key"
        --n-attempts $N_ATTEMPTS
        --job-name "$RUN_TAG"
        --config "$CONFIG_FILE"
    )
    
    # Task filter not supported for local paths
    if [ -n "$TASK_FILTER" ]; then
        echo "WARNING: Task filter (-t) only works with Harbor registry datasets"
        echo "Ignoring task filter for local dataset path"
    fi
fi

# Execute Harbor command
echo "Executing: ${HARBOR_CMD[*]}"
"${HARBOR_CMD[@]}"

EVAL_EXIT=$?
set -e

echo "============================================"
echo "Evaluation completed with exit code: $EVAL_EXIT"
echo "============================================"

# ============================================================================
# Save Metadata
# ============================================================================

mkdir -p "$RUN_DIR"
{
  echo "MODEL=$MODEL"
  echo "DATASET=$DATASET"
  echo "USE_REGISTRY=$USE_REGISTRY"
  if [ -n "$TASK_FILTER" ]; then
    echo "TASK_FILTER=$TASK_FILTER"
  fi
  echo "TIMESTAMP=$TIMESTAMP"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "HOSTNAME=$HOSTNAME"
  echo "ACCOUNT=$SLURM_JOB_ACCOUNT"
  echo "PARTITION=$SLURM_JOB_PARTITION"
  echo "EVAL_EXIT=$EVAL_EXIT"
} > "$RUN_DIR/meta.env"

echo "Metadata saved to: $RUN_DIR/meta.env"

# ============================================================================
# Upload Results (Optional - enabled via ENABLE_UPLOAD parameter)
# ============================================================================

if [ ${EVAL_EXIT:-0} -ne 0 ]; then
    echo "WARNING: Evaluation failed with exit code $EVAL_EXIT"
    echo "Skipping upload."
    exit $EVAL_EXIT
fi

if [ "$ENABLE_UPLOAD" = "true" ]; then
    echo "Upload enabled, checking prerequisites..."

    # Verify HF_TOKEN is set
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: HF_TOKEN is not set. Please add it to secret.env"
        echo "Skipping upload."
    else
        # ---- Check for DaytonaErrors before upload ----
        RESULT_FILE="$RUN_DIR/result.json"
        ERROR_LOG="jobs/daytona_errors.log"
        DAYTONA_ERROR_THRESHOLD=3
        SKIP_UPLOAD=false

        if [ -f "$RESULT_FILE" ]; then
            echo "Checking for DaytonaErrors in $RESULT_FILE..."

            # Count DaytonaErrors using Python
            DAYTONA_COUNT=$(python3 -c "
import json
import sys

try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)

    total_errors = 0
    error_ids = []

    if 'stats' in data and 'evals' in data['stats']:
        for eval_key, eval_data in data['stats']['evals'].items():
            if 'exception_stats' in eval_data and 'DaytonaError' in eval_data['exception_stats']:
                daytona_errors = eval_data['exception_stats']['DaytonaError']
                if isinstance(daytona_errors, list):
                    total_errors += len(daytona_errors)
                    error_ids.extend(daytona_errors)

    print(total_errors)

    # Also write detailed info to stderr for logging
    if total_errors > 0:
        print(f'Found {total_errors} DaytonaError(s): {error_ids[:5]}...', file=sys.stderr)

except Exception as e:
    print(f'Error parsing result.json: {e}', file=sys.stderr)
    print('0')
" 2>&1 | tail -n 1)

            echo "DaytonaError count: ${DAYTONA_COUNT}"

            # If too many DaytonaErrors, log and skip upload
            if [ "${DAYTONA_COUNT:-0}" -gt "$DAYTONA_ERROR_THRESHOLD" ]; then
                echo "WARNING: Job has ${DAYTONA_COUNT} DaytonaErrors (> ${DAYTONA_ERROR_THRESHOLD}), skipping upload"

                # Log to error file
                {
                    echo "==============================================="
                    echo "Timestamp: $(date)"
                    echo "Job: ${RUN_TAG}"
                    echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
                    echo "Model: ${MODEL}"
                    echo "Dataset: ${DATASET}"
                    echo "DaytonaErrors: ${DAYTONA_COUNT}"
                    echo "Result file: ${RESULT_FILE}"

                    # Extract error details
                    python3 -c "
import json
with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)
if 'stats' in data and 'evals' in data['stats']:
    for eval_key, eval_data in data['stats']['evals'].items():
        if 'exception_stats' in eval_data and 'DaytonaError' in eval_data['exception_stats']:
            errors = eval_data['exception_stats']['DaytonaError']
            if errors:
                print(f'Eval: {eval_key}')
                for i, error_id in enumerate(errors[:10], 1):
                    print(f'  {i}. {error_id}')
                if len(errors) > 10:
                    print(f'  ... and {len(errors) - 10} more')
"
                    echo "==============================================="
                } >> "$ERROR_LOG"

                echo "Error details logged to: $ERROR_LOG"
                SKIP_UPLOAD=true
            fi
        else
            echo "Warning: result.json not found, continuing with upload"
        fi

        # ---- Upload results if not skipped ----
        if [ "$SKIP_UPLOAD" = "false" ]; then
            echo "Starting upload..."

            # Use project's internal database module
            OT_AGENT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}/../.."
            export PYTHONPATH="${OT_AGENT_ROOT}/database:${PYTHONPATH:-}"

            export RUN_DIR="$RUN_DIR"
            export UPLOAD_USERNAME="${UPLOAD_USERNAME:-${USER}}"
            export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
            export RUN_TAG="$RUN_TAG"
            UPLOAD_LOG="experiments/logs/upload_${SLURM_JOB_ID}.log"
            mkdir -p "$(dirname "$UPLOAD_LOG")"

            echo "Uploading results from: $RUN_DIR" | tee -a "$UPLOAD_LOG"
            echo "Using username=${UPLOAD_USERNAME}, mode=${UPLOAD_MODE}" | tee -a "$UPLOAD_LOG"

            # Run the uploader
            python - <<'PY' 2>&1 | tee -a "$UPLOAD_LOG"
import os, sys
from unified_db.utils import upload_eval_results
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
hf_repo_id = sanitize_hf_repo_id(f"mlfoundations-dev/{run_tag}")
hf_token = os.environ["HF_TOKEN"]
print(f"[uploader] upload_eval_results(path={run_dir!r}, username={username!r}, error_mode={error_mode!r}, hf_repo_id={hf_repo_id!r})")
upload_eval_results(run_dir, username=username, error_mode=error_mode, hf_token=hf_token, hf_repo_id=hf_repo_id, register_benchmark=True)
print("[uploader] done.")
PY
            UPLOAD_EXIT=${PIPESTATUS[0]}

            if [ $UPLOAD_EXIT -ne 0 ]; then
                echo "Upload failed with exit code: $UPLOAD_EXIT"
            else
                echo "Upload completed successfully"
            fi
        fi
    fi
else
    echo "Upload disabled (use ENABLE_UPLOAD=true to enable)"
fi

echo "============================================"
echo "Evaluation finished!"
echo "============================================"
echo "Results directory: $RUN_DIR"
echo "vLLM log: $VLLM_LOG"
echo "Job log: experiments/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "============================================"

exit $EVAL_EXIT

