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

# Create timestamp and safe names before setting job name/output
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

MODEL="${1:-mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1}"
REPO_ID="${2:-DCAgent2/swebench-verified-random-100-folders}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')

echo "Using model: $MODEL"
echo "Using repository: $REPO_ID"

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

# DB/API secrets etc.
source /scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/secret.env

# Toolchain fixes
ln -sf /home1/apps/gcc/15.1.0/lib64/libstdc++.so.6 /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Conda env
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5

# Verify sandbox installation
sb --help >/dev/null

# Start VLLM server
mkdir -p experiments/logs
vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name "$MODEL" \
  --device cuda --gpu-memory-utilization 0.95 \
  > "experiments/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Cleaning up..."
    kill $VLLM_PID 2>/dev/null || true
    conda deactivate || true
}
trap cleanup EXIT

# Wait for VLLM server to start with healthcheck
MAX_RETRIES=20
RETRY_INTERVAL=100
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "VLLM server is ready"
        break
    fi
    echo "Waiting for VLLM server to start (attempt $i/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
    if [ $i -eq $MAX_RETRIES ]; then
        echo "VLLM server failed to start"
        exit 1
    fi
done

# Get the dataset path using the specified repo_id
echo "Downloading/locating dataset: $REPO_ID"
DATASET_PATH=$(python "./snapshot_download.py"  "$REPO_ID" | grep DATASET_PATH | tail -n 1 | cut -d'=' -f2)
if [ -z "${DATASET_PATH:-}" ]; then
    echo "Failed to get dataset path"
    exit 1
fi
echo "Using dataset path: $DATASET_PATH"

# Construct run dir (sb honors --job-name in jobs/<job-name>)
RUN_TAG="${SAFE_REPO}_${SAFE_MODEL}_${TIMESTAMP}"
RUN_DIR="jobs/${RUN_TAG}"
echo "Job dir: ${RUN_DIR}"

# Create DB row with status='Started' BEFORE running eval
echo "Creating DB job entry..."
export MODEL="$MODEL"
export REPO_ID="$REPO_ID"
export RUN_TAG="$RUN_TAG"
export SLURM_JOB_ID="$SLURM_JOB_ID"

# Get harbor package version
HARBOR_VERSION=$(python -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION

python - <<'PY'
import os, sys, json
sys.path.insert(0, "/scratch/08134/negin/dc-agent-shared/dcagents-leaderboard")
from unified_db.utils import create_job_entry_started

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]
run_tag = os.environ["RUN_TAG"]
slurm_job_id = os.environ["SLURM_JOB_ID"]
harbor_version = os.environ.get("HARBOR_VERSION", "unknown")

# Create initial job row with status='Started'
result = create_job_entry_started(
    model_hf_name=model_hf,
    benchmark_hf_name=dataset_hf,
    job_name=run_tag,
    username="negin",
    slurm_job_id=slurm_job_id,
    harbor_package_version=harbor_version,
    agent_name="terminus-2",
    config={"agent": "terminus-2", "env": "daytona"},
    n_trials=128,
    n_rep_eval=3
)

if not result.get("success"):
    print(f"ERROR: {result.get('error')}", file=sys.stderr)
    sys.exit(1)

# IMPORTANT: Save the DB job_id for later update
db_job_id = result["job"]["id"]
print(f"DB job created with ID: {db_job_id}")

# Store in meta.env for upload script
PY

if [ $? -ne 0 ]; then
    echo "Failed to create DB job entry"
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

echo "DB job entry created: $DB_JOB_ID"

# Run sandbox evaluation
set +e
harbor jobs start \
  -p "$DATASET_PATH" \
  --n-concurrent 64 \
  --agent terminus-2 \
  --model "hosted_vllm/$MODEL" \
  --env "daytona" \
  --agent-kwarg "parser=xml" \
  --agent-kwarg "api_base=http://localhost:8000/v1" \
  --agent-kwarg "key=fake_key" \
  --k-attempts 3 \
  --job-name "$RUN_TAG" \
  --config "dcagent_eval_config.yaml"
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
} > "$RUN_DIR/meta.env"

# If eval failed, don't attempt upload
if [ ${SB_EXIT:-0} -ne 0 ]; then
  echo "sb run exited with non-zero status: ${SB_EXIT}. Skipping upload."
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
DAYTONA_ERROR_THRESHOLD=15

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
    echo "⚠️  Job has ${DAYTONA_COUNT} DaytonaErrors (> ${DAYTONA_ERROR_THRESHOLD}), skipping upload"
    
    # Log to error file
    {
      echo "==============================================="
      echo "Timestamp: $(date)"
      echo "Job: ${RUN_TAG}"
      echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
      echo "Model: ${MODEL}"
      echo "Repo: ${REPO_ID}"
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
export UPLOAD_USERNAME="${UPLOAD_USERNAME:-negin}"
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
