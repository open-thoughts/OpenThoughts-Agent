#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --account AIFAC_5C0_290
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=eval
#SBATCH --gres=gpu:4

# Create timestamp and safe names before setting job name/output
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

MODEL="${1:-mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1}"
DATASET_PATH="${2:-DCAgent2/terminal_bench_2}"

echo "Using model: $MODEL"
echo "Using model ID: ${MODEL_ID}"
echo "Using repo ID: ${REPO_ID}"
echo "Using dataset path: $DATASET_PATH"

# Set up environment variables
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook


# check if internet access is available
if ping -c 1 google.com &> /dev/null; then
    echo "Internet access is available."
    export INTERNET_AVAILABLE=1
else
    echo "Internet access is not available."
    export INTERNET_AVAILABLE=0
fi

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR="${MASTER_NODE}-interconnect-1.leonardo.local"
# MASTER_ADDR=$MASTER_NODE
# MASTER_NODE=$(nslookup $MASTER_ADDR | grep 'Address' | tail -n1 | awk '{print $2}')
export head_node=$MASTER_ADDR

echo "Master node: $MASTER_NODE"
echo "Master node IP: $MASTER_ADDR"
export head_node=$MASTER_ADDR
export head_node_ip=$MASTER_ADDR

CMD_PREFIX=""
# if internet is not available, set up proxy tunnel
if [ "$INTERNET_AVAILABLE" -eq 0 ]; then
    echo "Setting up proxy tunnel..."
    CMD_PREFIX=$(bash /leonardo/home/userexternal/mnezhuri/dcft/OpenThoughts-Agent/rl/hpc/sbatch/start_proxy_tunnel_lrdn.sh)
fi


# Verify sandbox installation
# sb --help >/dev/null

# Start VLLM server
mkdir -p experiments/logs
srun --overlap --gres=gpu:4 vllm serve "$MODEL" \
  --host $MASTER_ADDR --port 28000 \
  --served-model-name "$MODEL_ID" --gpu-memory-utilization 0.85 \
  > "experiments/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Cleaning up..."
    kill $VLLM_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait for VLLM server to start with healthcheck
MAX_RETRIES=20
RETRY_INTERVAL=100
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://$MASTER_ADDR:28000/v1/models > /dev/null; then
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


# Construct run dir (sb honors --job-name in jobs/<job-name>)
RUN_TAG="${SAFE_REPO}_${SAFE_MODEL}_${TIMESTAMP}"

RUN_DIR="jobs/${RUN_TAG}"
echo "Job dir: ${RUN_DIR}"

# Create DB row with status='Started' BEFORE running eval
echo "Creating DB job entry..."
export MODEL_ID="$MODEL_ID"
export RUN_TAG="$RUN_TAG"
export SLURM_JOB_ID="$SLURM_JOB_ID"

# Get harbor package version
HARBOR_VERSION=$(python -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION

$CMD_PREFIX python - <<'PY'
import os, sys, json
from unified_db.utils import create_job_entry_started

model_hf = os.environ["MODEL_ID"]
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
DB_JOB_ID=$($CMD_PREFIX python - <<'PY'
import os, sys
from unified_db.utils import get_latest_job_for_model_benchmark

model_hf = os.environ["MODEL_ID"]
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

TASK_NAME="sqlite-db-truncate"
$CMD_PREFIX harbor jobs start \
  -p "$DATASET_PATH" \
  --n-concurrent 4 \
  --task-name "$TASK_NAME" \
  --agent terminus-2 \
  --model "hosted_vllm/$MODEL_ID" \
  --env "daytona" \
  --agent-kwarg "api_base=http://$MASTER_ADDR:28000/v1" \
  --agent-kwarg "key=fake_key" \
  --agent-kwarg "model_info={\"max_input_tokens\": 24500, \"max_output_tokens\": 8192, \"input_cost_per_token\": 0.0, \"output_cost_per_token\": 0.0}" \
  --n-attempts 1 \
  --debug \
  --job-name "$RUN_TAG" \
  --config "eval/tacc/dcagent_eval_config.yaml"
SB_EXIT=$?
set -e

# Save originals for exact round-trip later
mkdir -p "$RUN_DIR"
{
  echo "MODEL_ID=$MODEL_ID"
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
DAYTONA_ERROR_THRESHOLD=10


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

export RUN_DIR="$RUN_DIR"
export UPLOAD_USERNAME="${UPLOAD_USERNAME:-negin}"
export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
export RUN_TAG="$RUN_TAG"
UPLOAD_LOG="experiments/logs/upload_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$UPLOAD_LOG")"

echo "Uploading results from: $RUN_DIR" | tee -a "$UPLOAD_LOG"
echo "Using username=${UPLOAD_USERNAME}, mode=${UPLOAD_MODE}" | tee -a "$UPLOAD_LOG"

# Run the uploader (from dcagents-leaderboard)
$CMD_PREFIX python - <<'PY' 2>&1 | tee -a "$UPLOAD_LOG"
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
