#!/bin/bash
#SBATCH -p gh
#SBATCH --time=24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082,c637-091,c610-111
#SBATCH --account CCR24067
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=eval

# ==============================================================================
# Unified Eval Harbor Script
# ==============================================================================
# This script replaces all individual sbatch scripts with one parameterized version.
# Parameters are passed via environment variables (set by unified_eval_listener.py).
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
#   EVAL_DAYTONA_THRESHOLD  Max DaytonaErrors before abort (default: 3)
#   EVAL_VLLM_MAX_RETRIES   VLLM startup retries (default: 5)
#   EVAL_AGENT_PARSER       Agent parser type (default: "", use "xml" for swebench)
#   EVAL_SLURM_TIME         SLURM time limit (read at submit time)
#   EVAL_ENABLE_THINKING    Enable thinking blocks (default: false)
#   EVAL_AGENT_NAME         Agent name for harbor and DB entries (default: "terminus-2")
# ==============================================================================

# === CONFIGURABLE PARAMETERS (with defaults) ===
N_CONCURRENT="${EVAL_N_CONCURRENT:-64}"
N_ATTEMPTS="${EVAL_N_ATTEMPTS:-3}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.9}"
DAYTONA_ERROR_THRESHOLD="${EVAL_DAYTONA_THRESHOLD:-3}"
VLLM_MAX_RETRIES="${EVAL_VLLM_MAX_RETRIES:-5}"
AGENT_PARSER="${EVAL_AGENT_PARSER:-}"
ENABLE_THINKING="${EVAL_ENABLE_THINKING:-false}"
AGENT_NAME="${EVAL_AGENT_NAME:-terminus-2}"

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
echo "Unified Eval Harbor - Starting job"
echo "=============================================="
echo "Model: $MODEL"
echo "Repository: $REPO_ID"
echo "Benchmark ID: ${BENCHMARK_ID:-<auto>}"
echo ""
echo "Parameters:"
echo "  N_CONCURRENT: $N_CONCURRENT"
echo "  N_ATTEMPTS: $N_ATTEMPTS"
echo "  GPU_MEMORY_UTIL: $GPU_MEMORY_UTIL"
echo "  DAYTONA_ERROR_THRESHOLD: $DAYTONA_ERROR_THRESHOLD"
echo "  VLLM_MAX_RETRIES: $VLLM_MAX_RETRIES"
echo "  AGENT_PARSER: ${AGENT_PARSER:-<default>}"
echo "  ENABLE_THINKING: $ENABLE_THINKING"
echo "  AGENT_NAME: $AGENT_NAME"
echo "=============================================="

module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

# Set up environment variables
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_ADDRESS=${RAY_ADDRESS:-}
export VLLM_CACHE_ROOT=/scratch/10000/eguha3/vllm_cache
export VLLM_CONFIG_ROOT=/scratch/10000/eguha3/vllm_config
export TRITON_DUMP_DIR=/scratch/10000/eguha3/triton_dump_dir
export TRITON_OVERRIDE_DIR=/scratch/10000/eguha3/triton_override_dir
export TRITON_CACHE_DIR=/scratch/10000/eguha3/triton_cache_dir
export FLASHINFER_WORKSPACE_BASE=/scratch/08002/gsmyrnis/flashinfer_cache
export UV_CACHE_DIR=/scratch/10000/eguha3/uv_cache_dir
export HYDRA_FULL_ERROR=1
export HF_CACHE_DIR=/scratch/08134/negin/dc-agent-shared/.hf_cache
export HF_HUB_CACHE=$SCRATCH/hub
export LITELLM_LOCAL_MODEL_COST_MAP=True

# DB/API secrets etc.
source /scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/secret.env

# Toolchain fixes
ln -sf /home1/apps/gcc/15.1.0/lib64/libstdc++.so.6 /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Conda env
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5

# Start VLLM server
mkdir -p experiments/logs
vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name "$MODEL" \
  --enforce-eager \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
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

# Get the dataset path using the specified repo_id
echo "Downloading/locating dataset: $REPO_ID"
DATASET_PATH=$(python "./snapshot_download.py" "$REPO_ID" | grep DATASET_PATH | tail -n 1 | cut -d'=' -f2)
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

# Get harbor package version
HARBOR_VERSION=$(python -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION

python - <<'PY'
import os, sys, json
sys.path.insert(0, "/scratch/08134/negin/dc-agent-shared/dcagents-leaderboard")
from unified_db.utils import create_job_entry_started, update_job_status_to_started

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]
run_tag = os.environ["RUN_TAG"]
run_tag_arg = os.environ.get("RUN_TAG_ARG", "")
slurm_job_id = os.environ["SLURM_JOB_ID"]
harbor_version = os.environ.get("HARBOR_VERSION", "unknown")
n_concurrent = int(os.environ.get("N_CONCURRENT", "64"))
n_attempts = int(os.environ.get("N_ATTEMPTS", "3"))
agent_name = os.environ.get("AGENT_NAME", "terminus-2")

# If RUN_TAG was provided from listener, a Pending entry should exist - update it
# Otherwise, create a new Started entry (backward compatibility)
if run_tag_arg:
    print(f"Updating Pending job to Started: {run_tag}")
    result = update_job_status_to_started(
        job_name=run_tag,
        n_trials=n_concurrent,
        n_rep_eval=n_attempts,
        config={"agent": agent_name, "env": "daytona"},
        harbor_package_version=harbor_version
    )
else:
    print(f"Creating new Started job entry: {run_tag}")
    result = create_job_entry_started(
        model_hf_name=model_hf,
        benchmark_hf_name=dataset_hf,
        job_name=run_tag,
        username=os.environ.get("UPLOAD_USERNAME", os.environ.get("USER", "unknown")),
        slurm_job_id=slurm_job_id,
        harbor_package_version=harbor_version,
        agent_name=agent_name,
        config={"agent": agent_name, "env": "daytona"},
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

# Extract the job_id from Python output and save it
DB_JOB_ID=$(python - <<'PY'
import os, sys
sys.path.insert(0, "/scratch/08134/negin/dc-agent-shared/dcagents-leaderboard")
from unified_db.utils import get_latest_job_for_model_benchmark

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
HARBOR_CMD="$HARBOR_CMD --agent-kwarg \"api_base=http://localhost:8000/v1\""
HARBOR_CMD="$HARBOR_CMD --agent-kwarg \"key=fake_key\""
HARBOR_CMD="$HARBOR_CMD --agent-kwarg \"max_tokens=16384\""
HARBOR_CMD="$HARBOR_CMD --n-attempts $N_ATTEMPTS"
HARBOR_CMD="$HARBOR_CMD --export-traces"
HARBOR_CMD="$HARBOR_CMD --job-name \"$RUN_TAG\""
HARBOR_CMD="$HARBOR_CMD --config \"dcagent_eval_config.yaml\""

# Add parser arg if specified (e.g., for swebench)
if [ -n "$AGENT_PARSER" ]; then
    HARBOR_CMD="$HARBOR_CMD --agent-kwarg \"parser=$AGENT_PARSER\""
fi

# Add extra_body for thinking control (only if enable_thinking is true)
if [ "$ENABLE_THINKING" = "true" ]; then
    HARBOR_CMD="$HARBOR_CMD --agent-kwarg 'extra_body={\"chat_template_kwargs\":{\"enable_thinking\":true}}'"
    echo "Thinking blocks enabled via extra_body"
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
  echo "DAYTONA_ERROR_THRESHOLD=$DAYTONA_ERROR_THRESHOLD"
  echo "AGENT_PARSER=$AGENT_PARSER"
  echo "ENABLE_THINKING=$ENABLE_THINKING"
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


# ---- Check for DaytonaErrors before upload ----
RESULT_FILE="$RUN_DIR/result.json"
ERROR_LOG="jobs/daytona_errors.log"

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
    echo "Job has ${DAYTONA_COUNT} DaytonaErrors (> ${DAYTONA_ERROR_THRESHOLD}), skipping upload"

    # Log to error file
    {
      echo "==============================================="
      echo "Timestamp: $(date)"
      echo "Job: ${RUN_TAG}"
      echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
      echo "Model: ${MODEL}"
      echo "Repo: ${REPO_ID}"
      echo "DaytonaErrors: ${DAYTONA_COUNT}"
      echo "Threshold: ${DAYTONA_ERROR_THRESHOLD}"
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
    echo "Job completed but not uploaded due to excessive DaytonaErrors"
    exit 0  # Exit successfully but skip upload
  fi
else
  echo "Warning: result.json not found, continuing with upload"
fi



# ---- Upload results to DB ----

# Point PYTHONPATH at your uploader package
export PYTHONPATH="/scratch/08134/negin/dc-agent-shared/dcagents-leaderboard:${PYTHONPATH:-}"

export RUN_DIR="$RUN_DIR"
export UPLOAD_USERNAME="${EVAL_UPLOAD_USERNAME:-$USER}"
export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
export RUN_TAG="$RUN_TAG"
UPLOAD_LOG="experiments/logs/upload_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$UPLOAD_LOG")"

echo "Uploading results from: $RUN_DIR" | tee -a "$UPLOAD_LOG"
echo "Using username=${UPLOAD_USERNAME}, mode=${UPLOAD_MODE}" | tee -a "$UPLOAD_LOG"

# Run the uploader (from dcagents-leaderboard)
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
username  = os.environ.get("UPLOAD_USERNAME", "negin")
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
