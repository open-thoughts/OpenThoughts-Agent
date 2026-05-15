#!/bin/bash
#SBATCH -p booster
#SBATCH --time=12:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --account reformo
#SBATCH --output=eval/jupiter/logs/%j.out
#SBATCH --job-name=eval

# ==============================================================================
# Unified Eval Harbor — Jupiter Cluster (JSC GH200)
#
# Starts vLLM, runs Harbor eval, checks errors, uploads results.
# Adapted from eval/tacc/tacc_eval_harbor.sbatch + Jupiter env patterns.
#
# Positional args:
#   $1 = MODEL (HF model name, e.g. mlfoundations-dev/some_model)
#   $2 = REPO_ID (HF dataset repo, e.g. DCAgent/dev_set_v2)
#   $3 = BENCHMARK_ID (optional, DB benchmark UUID)
#   $4 = RUN_TAG_ARG (optional, override run tag)
#
# Env vars from listener:
#   EVAL_N_CONCURRENT       (default: 128)
#   EVAL_GPU_MEMORY_UTIL    (default: 0.95)
#   EVAL_DAYTONA_THRESHOLD  (default: 3)
#   EVAL_HARBOR_CONFIG      (default: $DCFT/eval/jupiter/dcagent_eval_config.yaml)
#   EVAL_DB_JOB_ID          (optional, pre-created Pending job ID)
# ==============================================================================

set -eo pipefail
ulimit -c 0  # Disable core dumps
ulimit -n 65536 2>/dev/null || true

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# --- Parse positional args ---
MODEL="${1:-mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1}"
REPO_ID="${2:-DCAgent/dev_set_71_tasks}"
BENCHMARK_ID="${3:-}"
RUN_TAG_ARG="${4:-}"

# --- Read env vars from listener (with defaults) ---
N_CONCURRENT="${EVAL_N_CONCURRENT:-128}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.95}"
DAYTONA_ERROR_THRESHOLD="${EVAL_DAYTONA_THRESHOLD:-999999}"
HARBOR_CONFIG="${EVAL_HARBOR_CONFIG:-$DCFT/eval/jupiter/dcagent_eval_config.yaml}"
DB_JOB_ID="${EVAL_DB_JOB_ID:-}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
if [[ "$REPO_ID" == /* ]]; then
    # Local path: use basename as the safe repo name
    SAFE_REPO=$(basename "$REPO_ID")
else
    SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')
fi

# Update SLURM job name to include model short name (default "%j" is just job ID)
SHORT_MODEL=$(echo "$MODEL" | awk -F'/' '{print $NF}')
scontrol update JobId=$SLURM_JOB_ID Name="eval_${SHORT_MODEL}" 2>/dev/null || true

echo "=============================================="
echo "Jupiter Eval Harbor"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $REPO_ID"
echo "Benchmark ID: ${BENCHMARK_ID:-<auto>}"
echo "N concurrent: $N_CONCURRENT"
echo "GPU memory util: $GPU_MEMORY_UTIL"
echo "Harbor config: $HARBOR_CONFIG"
echo "Daytona threshold: $DAYTONA_ERROR_THRESHOLD"
echo "DB Job ID (pending): ${DB_JOB_ID:-<none>}"
echo "=============================================="

# ==============================================================================
# Jupiter Environment Setup
# (Matches patterns from data/sbatches/run_harbor_glm47_jupiter.sbatch)
# ==============================================================================

DCFT="${DCFT:-/e/scratch/jureap59/$USER/OpenThoughts-Agent}"

# Reset PYTHONPATH to avoid inheriting submitter's environment (e.g. numpy 2.4
# from guha1/site-packages which breaks numba/vLLM).
unset PYTHONPATH

# Source Jupiter environment
source "$DCFT/hpc/dotenv/jupiter.env"

# Source secrets
DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-$HOME/secrets.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

# --- Daytona API key for evals (3:1 weighted: 3/4 org2, 1/4 org1) ---
DAYTONA_KEY_ORG1="dtn_17868a1955b56a52cb367af6dd3c6e93ee531b2073df801784273435c0e0fc6c"
DAYTONA_KEY_ORG2="dtn_ecfb759d2abcbf0f044413182677b3cc7624af495cb48409e8873b5283bda313"
if (( RANDOM % 4 == 0 )); then
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG1"
    echo "Selected Daytona org: org1 (${DAYTONA_API_KEY:0:12}...)"
else
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG2"
    echo "Selected Daytona org: org2 (${DAYTONA_API_KEY:0:12}...)"
fi

# --- vLLM / Ray / Triton env vars ---
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export VLLM_CONFIG_ROOT="${VLLM_CACHE_ROOT:-$DCFT_SCRATCH/vllm_config}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_cache_${USER}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DCFT_SCRATCH/uv_cache_dir}"
export HYDRA_FULL_ERROR=1
# Pre-downloaded model cache — must match listener's HF_HUB_CACHE (compute nodes have no internet)
# Use DCFT_DATA from jupiter.env; fall back to user-owned path (avoids permission issues
# with shared /e/data1/datasets/playground/ot owned by other users).
export HF_HUB_CACHE="${DCFT_DATA}/hf_hub"
export HF_HOME="${DCFT_DATA}/hf_hub"
export HF_CACHE_DIR="$HF_HUB_CACHE"

# Harbor and DB operations
HARBOR_SRC="${HARBOR_SRC:-${SCRATCH}/harbor/src}"
export PYTHONPATH="${HARBOR_SRC}:${DCFT}:${PYTHONPATH:-}"

# ==============================================================================
# LD_LOADER Wrappers (Jupiter ARM GH200 — no execute perms on shared conda)
# (Uses feuer1's miniforge3/otagent env + system ld-linux loader,
#  matching run_harbor_glm47_jupiter.sbatch)
# ==============================================================================

OTAGENT_DIR="${OTAGENT_DIR:-/e/scratch/jureap59/feuer1/miniforge3/envs/otagent}"
export PATH="$OTAGENT_DIR/bin:$PATH"
export CONDA_PREFIX="$OTAGENT_DIR"

LD_LOADER="/lib/ld-linux-aarch64.so.1"
PYTHON_REAL="$OTAGENT_DIR/bin/python3.12"

# Create executable wrappers that use ld-linux loader
WRAPPER_DIR="/tmp/eval_wrappers_${SLURM_JOB_ID}"
mkdir -p "$WRAPPER_DIR"

cat > "$WRAPPER_DIR/python3" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL "\$@"
WEOF
cat > "$WRAPPER_DIR/python" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL "\$@"
WEOF
cat > "$WRAPPER_DIR/harbor" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL -c "import sys; from harbor.cli.main import app; sys.exit(app())" "\$@"
WEOF
chmod +x "$WRAPPER_DIR/python3" "$WRAPPER_DIR/python" "$WRAPPER_DIR/harbor"
export PATH="$WRAPPER_DIR:$PATH"

PYTHON_BIN="$WRAPPER_DIR/python3"

echo "Python: $($PYTHON_BIN --version)"
harbor --help >/dev/null
export RAY_DEDUP_LOGS=0

# ==============================================================================
# SSH Tunnel + Proxychains Setup (no internet on compute nodes)
# ==============================================================================

NODE_HOST=$(hostname -s)
TUNNEL_PORT=7003
LOGIN_NODE="jpbl-s01-02"
PROXYCHAINS_BIN="/e/scratch/jureap59/feuer1/proxychains-ng-aarch64/bin/proxychains4"
TUNNEL_PID=""

setup_proxy() {
    echo "[proxy] Setting up SSH tunnel to $LOGIN_NODE..."

    if [ -z "${SSH_KEY:-}" ]; then
        echo "[proxy] WARNING: SSH_KEY not set, skipping tunnel setup"
        echo "[proxy] External connectivity (Daytona, HF) will fail"
        return 1
    fi

    # Get node IP for multi-node access
    NODE_IP=$(nslookup "$NODE_HOST" 2>/dev/null | grep 'Address' | tail -n1 | awk '{print $2}')
    NODE_IP="${NODE_IP:-127.0.0.1}"

    # Create SSH tunnel with SOCKS5 proxy
    ssh -g -f -N -D ${TUNNEL_PORT} \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=1000 \
        -o ServerAliveInterval=10 \
        -o ServerAliveCountMax=30 \
        -o TCPKeepAlive=yes \
        -o ExitOnForwardFailure=yes \
        -o BatchMode=yes \
        -i "${SSH_KEY}" \
        "${USER}@${LOGIN_NODE}"

    sleep 5

    # Verify tunnel
    if pgrep -f "ssh.*-D.*${TUNNEL_PORT}" > /dev/null; then
        echo "[proxy] SSH tunnel started successfully"
        TUNNEL_PID=$(pgrep -f "ssh.*-D.*${TUNNEL_PORT}" | head -1)
    else
        echo "[proxy] ERROR: SSH tunnel failed to start"
        return 1
    fi

    # Generate proxychains config
    CFG_PATH="$HOME/.proxychains/proxychains_${SLURM_JOB_ID}.conf"
    mkdir -p "$HOME/.proxychains"

    cat > "$CFG_PATH" <<PCEOF
strict_chain
quiet_mode
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
localnet 169.254.0.0/255.255.0.0
[ProxyList]
socks5 ${NODE_IP} ${TUNNEL_PORT}
PCEOF

    export PROXYCHAINS_CONF_FILE="$CFG_PATH"
    export PROXYCHAINS_SOCKS5_HOST="${NODE_IP}"
    export PROXYCHAINS_SOCKS5_PORT="${TUNNEL_PORT}"

    echo "[proxy] Config at $CFG_PATH (socks5://${NODE_IP}:${TUNNEL_PORT})"

    # Daytona timeout/retry settings
    export DAYTONA_MAX_RETRIES=5
    export DAYTONA_RETRY_DELAY=30
    export DAYTONA_BACKOFF_FACTOR=2
    export DAYTONA_TIMEOUT=1800
    export AIOHTTP_CLIENT_TIMEOUT=900
    export AIOHTTP_CONNECTOR_TIMEOUT=900
    export AIOHTTP_SOCK_CONNECT_TIMEOUT=300
    export AIOHTTP_TOTAL_TIMEOUT=1800

    # Disable SSL verification (JSC cert issues)
    export PYTHONHTTPSVERIFY=0
    unset SSL_CERT_FILE
    unset CURL_CA_BUNDLE
    unset REQUESTS_CA_BUNDLE
    unset SSL_CERT_DIR

    # Test connectivity
    if [ -x "$PROXYCHAINS_BIN" ]; then
        if "$PROXYCHAINS_BIN" -f "$CFG_PATH" curl -s --connect-timeout 10 https://huggingface.co -o /dev/null 2>/dev/null; then
            echo "[proxy] Connectivity test passed"
        else
            echo "[proxy] WARNING: Connectivity test failed (may still work)"
        fi
    fi

    echo "[proxy] Setup complete"
    return 0
}

# Helper: run command through proxychains
proxied() {
    if [ -x "$PROXYCHAINS_BIN" ] && [ -n "${PROXYCHAINS_CONF_FILE:-}" ]; then
        "$PROXYCHAINS_BIN" -f "$PROXYCHAINS_CONF_FILE" "$@"
    else
        "$@"
    fi
}

# Setup proxy
setup_proxy || echo "[proxy] Continuing without proxy (some operations may fail)"

# ==============================================================================
# Cleanup trap
# ==============================================================================
cleanup() {
    echo "Cleaning up..."
    # Kill vLLM
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    # Kill SSH tunnel
    if [ -n "${TUNNEL_PID:-}" ]; then
        kill "$TUNNEL_PID" 2>/dev/null || true
    fi
    # Remove wrappers
    rm -rf "$WRAPPER_DIR" 2>/dev/null || true
    # Remove proxychains config
    rm -f "${PROXYCHAINS_CONF_FILE:-}" 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

# ==============================================================================
# Construct run tag and unified run directory (BEFORE vLLM start)
# ==============================================================================
EVAL_JOBS_DIR="${EVAL_JOBS_DIR:-${DCFT_DATA}/eval_jobs}"

if [ -n "$RUN_TAG_ARG" ]; then
    RUN_TAG="$RUN_TAG_ARG"
else
    RUN_TAG="eval-${SAFE_MODEL}_${SAFE_REPO}"
fi
RUN_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

# Symlink SLURM log into run dir for consolidated access
mkdir -p eval/jupiter/logs
SLURM_LOG="${DCFT}/eval/jupiter/logs/${SLURM_JOB_ID}.out"
ln -sf "$SLURM_LOG" "$RUN_DIR/slurm.log" 2>/dev/null || true

# Ensure Harbor writes directly to our unified run dir via symlink
mkdir -p "${DCFT}/trace_jobs"
rm -rf "${DCFT}/trace_jobs/${RUN_TAG}" 2>/dev/null || true
ln -sfn "$RUN_DIR" "${DCFT}/trace_jobs/${RUN_TAG}"

echo "Run tag: $RUN_TAG"
echo "Run dir: $RUN_DIR"

# ==============================================================================
# Start vLLM Server (single node, TP=4 for 4 GH200 GPUs)
# ==============================================================================

# Pre-download model via proxychains (compute nodes have no direct internet)
echo "Pre-downloading model: $MODEL"
proxied $PYTHON_BIN -c "
from huggingface_hub import snapshot_download
import os
cache = os.environ.get('HF_HUB_CACHE')
path = snapshot_download('$MODEL', cache_dir=cache)
print(f'Model cached at: {path}')
"
if [ $? -ne 0 ]; then
    echo "ERROR: Model pre-download failed for $MODEL. Exiting."
    exit 1
fi

echo "Starting vLLM server for model: $MODEL"
source "$DCFT/eval/build_vllm_cmd.sh"
build_vllm_cmd "$PYTHON_BIN" "$MODEL" "$GPU_MEMORY_UTIL"

env TORCHDYNAMO_DISABLE=1 \
    TMPDIR="/tmp" \
    TRITON_CACHE_DIR="/tmp/triton_cache_${USER}" \
    TORCH_COMPILE_CACHE_DIR="/tmp/torch_cache_${USER}" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    HF_HUB_OFFLINE=1 \
    VLLM_ALL2ALL_BACKEND=pplx \
"${VLLM_CMD[@]}" \
    > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

# Health check loop
MAX_RETRIES=20
RETRY_INTERVAL=100
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready"
        break
    fi
    echo "Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        # Dump last 50 lines of vLLM log
        tail -50 "$RUN_DIR/vllm.log" 2>/dev/null || true
        exit 1
    fi
done

# ==============================================================================
# Download dataset (via proxychains for HF access)
# Use --local-dir to get real files instead of symlinks (Daytona needs real files)
# Supports local paths: if REPO_ID starts with /, use it directly.
# ==============================================================================
if [[ "$REPO_ID" == /* ]]; then
    # Local dataset path
    echo "Using local dataset path: $REPO_ID"
    DATASET_PATH="$REPO_ID"
    if [ ! -d "$DATASET_PATH" ]; then
        echo "ERROR: Local dataset path does not exist: $DATASET_PATH"
        exit 1
    fi
    TASK_COUNT=$(ls -d "$DATASET_PATH"/*/instruction.md 2>/dev/null | wc -l)
    echo "[DEBUG] Found $TASK_COUNT tasks with instruction.md"
else
    echo "Downloading/locating dataset: $REPO_ID"
    DATASET_LOCAL_DIR="${DCFT_DATA}/datasets/${SAFE_REPO}"
    echo "[DEBUG] DATASET_LOCAL_DIR=$DATASET_LOCAL_DIR"
    echo "[DEBUG] Local dir exists: $([ -d "$DATASET_LOCAL_DIR" ] && echo yes || echo no)"
    if [ -d "$DATASET_LOCAL_DIR" ]; then
        TASK_COUNT=$(ls -d "$DATASET_LOCAL_DIR"/*/instruction.md 2>/dev/null | wc -l)
        echo "[DEBUG] Found $TASK_COUNT tasks with instruction.md in local dir"
        echo "[DEBUG] Sample files:"
        file "$DATASET_LOCAL_DIR"/$(ls "$DATASET_LOCAL_DIR" | head -1)/environment/Dockerfile 2>/dev/null || echo "[DEBUG] No Dockerfile found"
    fi
    DOWNLOAD_LOG=$(mktemp /tmp/download_XXXXXX.log)
    proxied $PYTHON_BIN "$DCFT/eval/jupiter/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_LOCAL_DIR" > "$DOWNLOAD_LOG" 2>&1
    DOWNLOAD_EXIT=$?
    cat "$DOWNLOAD_LOG"
    DATASET_PATH=$(grep DATASET_PATH "$DOWNLOAD_LOG" | tail -n 1 | cut -d'=' -f2)
    rm -f "$DOWNLOAD_LOG"
    if [ $DOWNLOAD_EXIT -ne 0 ] || [ -z "${DATASET_PATH:-}" ]; then
        echo "ERROR: Failed to get dataset path (exit code: $DOWNLOAD_EXIT)"
        exit 1
    fi
fi
echo "Using dataset path: $DATASET_PATH"
echo "[DEBUG] Dataset source: $([ -L "$DATASET_PATH/$(ls "$DATASET_PATH" | head -1)/environment/Dockerfile" ] && echo 'SYMLINK' || echo 'REAL FILE')"
echo "[DEBUG] Task count in dataset: $(ls -d "$DATASET_PATH"/*/instruction.md 2>/dev/null | wc -l)"

# --- Compute canonical benchmark name for DB uploads ---
# Map dataset names to their canonical DB benchmark names.
# No suffixes for timeout/memory — these are runtime settings, not benchmark identity.
declare -A BENCHMARK_NAME_MAP=(
    ["DCAgent2_aider_polyglot"]="aider_polyglot"
    ["DCAgent_dev_set_v2"]="dev_set_v2"
    ["DCAgent_dev_set_71_tasks"]="dev_set_71_tasks"
    ["DCAgent2_terminal_bench_2"]="terminal_bench_2"
    ["DCAgent_swebench_verified_eval_set"]="swebench-verified-random-100-folders"
    ["DCAgent2_bfcl-parity"]="bfcl-parity"
)
BENCHMARK_NAME="${BENCHMARK_NAME_MAP[$SAFE_REPO]:-$SAFE_REPO}"
export BENCHMARK_NAME
echo "Benchmark name: $BENCHMARK_NAME"

# ==============================================================================
# Update DB: Started (using database/unified_db)
# ==============================================================================
echo "Creating/updating DB job entry..."
export MODEL REPO_ID RUN_TAG SLURM_JOB_ID BENCHMARK_NAME

HARBOR_VERSION=$($PYTHON_BIN -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION

proxied $PYTHON_BIN - <<'PY' || true
import os, sys

from database.unified_db.utils import create_job_entry_started

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]
run_tag = os.environ["RUN_TAG"]
slurm_job_id = os.environ["SLURM_JOB_ID"]
harbor_version = os.environ.get("HARBOR_VERSION", "unknown")

result = create_job_entry_started(
    model_hf_name=model_hf,
    benchmark_hf_name=dataset_hf,
    job_name=run_tag,
    username=os.environ.get("USER", "jupiter"),
    slurm_job_id=slurm_job_id,
    harbor_package_version=harbor_version,
    agent_name="terminus-2",
    config={"agent": "terminus-2", "env": "daytona"},
    n_trials=128,
    n_rep_eval=3
)

if not result.get("success"):
    print(f"WARNING: DB create failed: {result.get('error')}", file=sys.stderr)
    sys.exit(0)

db_job_id = result["job"]["id"]
print(f"DB job created with ID: {db_job_id}")
PY

# Get the DB job ID
DB_JOB_ID=$(proxied $PYTHON_BIN - <<'PY' || true
import os, sys

from database.unified_db.utils import get_latest_job_for_model_benchmark

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]

try:
    result = get_latest_job_for_model_benchmark(model_hf, dataset_hf)
    if result and result.get("id"):
        print(result["id"])
except Exception as e:
    print(f"WARNING: DB lookup failed: {e}", file=sys.stderr)
PY
)

if [ -z "${DB_JOB_ID:-}" ]; then
    echo "WARNING: Failed to get DB job ID"
else
    echo "DB job entry: $DB_JOB_ID"
fi

# ==============================================================================
# Run Harbor Eval (via proxychains for Daytona access)
# ==============================================================================
set +e

# Build extra harbor args (e.g., snapshot for datasets with COPY . Dockerfiles)
EXTRA_HARBOR_ARGS=""
if [ -n "${EVAL_SNAPSHOT_NAME:-}" ]; then
    echo "Using Daytona snapshot: $EVAL_SNAPSHOT_NAME"
    EXTRA_HARBOR_ARGS="--environment-kwarg snapshot_template_name=$EVAL_SNAPSHOT_NAME --no-force-build"
fi
if [ -n "${EVAL_TIMEOUT_MULTIPLIER:-}" ]; then
    echo "Using timeout multiplier: $EVAL_TIMEOUT_MULTIPLIER"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --timeout-multiplier $EVAL_TIMEOUT_MULTIPLIER"
fi
if [ -n "${EVAL_OVERRIDE_MEMORY_MB:-}" ]; then
    echo "Using memory override: ${EVAL_OVERRIDE_MEMORY_MB}MB"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --override-memory-mb $EVAL_OVERRIDE_MEMORY_MB"
fi

echo "[DEBUG] EXTRA_HARBOR_ARGS=$EXTRA_HARBOR_ARGS"
echo "[DEBUG] DAYTONA_API_KEY=${DAYTONA_API_KEY:0:20}..."
echo "[DEBUG] Full harbor command:"
echo "  harbor jobs start -p $DATASET_PATH --n-concurrent $N_CONCURRENT --agent terminus-2 --model hosted_vllm/$MODEL --env daytona ... $EXTRA_HARBOR_ARGS"

# Check if a previous job dir exists for this run tag (for resume)
if [ -d "$RUN_DIR" ] && [ -f "$RUN_DIR/config.json" ]; then
    echo "Found existing job dir, resuming: $RUN_DIR"
    proxied $WRAPPER_DIR/harbor jobs resume \
        -p "$RUN_DIR" \
        --filter-error-type EnvironmentStartTimeoutError \
        --filter-error-type DaytonaError \
        --filter-error-type DaytonaRateLimitError
else
    echo "Starting new job"
    proxied $WRAPPER_DIR/harbor jobs start \
        -p "$DATASET_PATH" \
        --n-concurrent "$N_CONCURRENT" \
        --agent terminus-2 \
        --model "hosted_vllm/$MODEL" \
        --env "daytona" \
        --agent-kwarg "api_base=http://localhost:8000/v1" \
        --agent-kwarg "key=fake_key" \
        --n-attempts 3 \
        --job-name "$RUN_TAG" \
        --config "$HARBOR_CONFIG" \
        $EXTRA_HARBOR_ARGS
fi
SB_EXIT=$?
set -e

# ==============================================================================
# Save meta.env
# ==============================================================================
mkdir -p "$RUN_DIR"
{
    echo "MODEL=$MODEL"
    echo "REPO_ID=$REPO_ID"
    echo "TIMESTAMP=$TIMESTAMP"
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "DB_JOB_ID=${DB_JOB_ID:-}"
    echo "BENCHMARK_ID=${BENCHMARK_ID:-}"
    echo "N_CONCURRENT=$N_CONCURRENT"
    echo "GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
} > "$RUN_DIR/meta.env"

# Harbor writes trials, result.json, and config.json directly to RUN_DIR
# (via trace_jobs symlink → RUN_DIR)
TRIAL_COUNT=$(find "$RUN_DIR" -maxdepth 1 -type d -name '*__*' 2>/dev/null | wc -l)
echo "Found $TRIAL_COUNT trial dirs in $RUN_DIR"

# If eval failed, don't attempt upload
if [ ${SB_EXIT:-0} -ne 0 ]; then
    echo "Harbor eval exited with non-zero status: ${SB_EXIT}. Skipping upload."
    exit ${SB_EXIT}
fi

# Ensure run dir exists
if [ ! -d "$RUN_DIR" ]; then
    echo "Expected run directory not found: $RUN_DIR"
    exit 2
fi

# ==============================================================================
# Check for DaytonaErrors before upload
# ==============================================================================
RESULT_FILE="$RUN_DIR/result.json"
ERROR_LOG="$RUN_DIR/daytona_errors.log"

if [ -f "$RESULT_FILE" ]; then
    echo "Checking for DaytonaErrors in $RESULT_FILE..."

    DAYTONA_COUNT=$($PYTHON_BIN -c "
import json, sys
try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)

    total_errors = 0
    if 'stats' in data and 'evals' in data['stats']:
        for eval_key, eval_data in data['stats']['evals'].items():
            if 'exception_stats' in eval_data and 'DaytonaError' in eval_data['exception_stats']:
                daytona_errors = eval_data['exception_stats']['DaytonaError']
                if isinstance(daytona_errors, list):
                    total_errors += len(daytona_errors)
    print(total_errors)
except Exception as e:
    print(f'Error parsing result.json: {e}', file=sys.stderr)
    print('0')
" 2>&1 | tail -n 1)

    echo "DaytonaError count: ${DAYTONA_COUNT}"

    if [ "${DAYTONA_COUNT:-0}" -gt "$DAYTONA_ERROR_THRESHOLD" ]; then
        echo "Job has ${DAYTONA_COUNT} DaytonaErrors (> ${DAYTONA_ERROR_THRESHOLD}), skipping upload"

        {
            echo "==============================================="
            echo "Timestamp: $(date)"
            echo "Job: ${RUN_TAG}"
            echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
            echo "Model: ${MODEL}"
            echo "Repo: ${REPO_ID}"
            echo "DaytonaErrors: ${DAYTONA_COUNT}"
            echo "Result file: ${RESULT_FILE}"
            $PYTHON_BIN -c "
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
" 2>/dev/null || true
            echo "==============================================="
        } >> "$ERROR_LOG"

        echo "Error details logged to: $ERROR_LOG"
        echo "Job completed but not uploaded due to excessive DaytonaErrors"
        exit 0
    fi
else
    echo "Warning: result.json not found, continuing with upload"
fi

# ==============================================================================
# Upload results to DB (via proxychains for HF upload)
# ==============================================================================
# All data (trials, result.json, config.json, meta.env) is in RUN_DIR
export UPLOAD_DIR="$RUN_DIR"
export RUN_DIR
export UPLOAD_USERNAME="${UPLOAD_USERNAME:-$USER}"
export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
export RUN_TAG

UPLOAD_LOG="$RUN_DIR/upload.log"

echo "Uploading results from: $RUN_DIR" | tee -a "$UPLOAD_LOG"
echo "Using username=${UPLOAD_USERNAME}, mode=${UPLOAD_MODE}" | tee -a "$UPLOAD_LOG"

proxied $PYTHON_BIN - <<'PY' 2>&1 | tee -a "$UPLOAD_LOG"
import os, sys, re, hashlib

from database.unified_db.utils import upload_eval_results


def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
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
        name = f"{base}{digest}"
        name = collapse(name).strip("-.")

    name = collapse(name).strip("-.")
    if name[0] in "-.":
        name = "r" + name[1:]
    if name[-1] in "-.":
        name = name[:-1] + "0"

    return f"{org}/{name}" if org else name


upload_dir = os.environ.get("UPLOAD_DIR", os.environ["RUN_DIR"])
run_dir = os.environ["RUN_DIR"]
run_tag = os.environ["RUN_TAG"]
username = os.environ.get("UPLOAD_USERNAME", os.environ.get("USER", "jupiter"))
error_mode = os.environ.get("UPLOAD_MODE", "skip_on_error")
hf_repo_id = sanitize_hf_repo_id(f"DCAgent2/{run_tag}")
hf_token = os.environ["HF_TOKEN"]

print(f"[uploader] upload_eval_results(path={run_dir!r}, username={username!r}, "
      f"error_mode={error_mode!r}, hf_repo_id={hf_repo_id!r})")
dataset_hf = os.environ.get("REPO_ID", "")
# Use the canonical benchmark name computed in the shell section,
# which handles local paths and timeout/memory suffixes correctly.
benchmark_name = os.environ.get("BENCHMARK_NAME", "")
if not benchmark_name:
    # Fallback: derive from REPO_ID (legacy behavior)
    benchmark_name = dataset_hf.split("/")[-1] if "/" in dataset_hf else dataset_hf

# Compute a stable benchmark_version_hash from the benchmark name
import hashlib
benchmark_version_hash = hashlib.sha256(benchmark_name.encode()).hexdigest()
print(f"[uploader] benchmark_name={benchmark_name!r}, version_hash={benchmark_version_hash[:16]}...")

print(f"[uploader] Using upload dir: {upload_dir}")
upload_eval_results(
    upload_dir,
    username=username,
    error_mode=error_mode,
    hf_token=hf_token,
    hf_repo_id=hf_repo_id,
    register_benchmark=True,
    benchmark_name=benchmark_name,
    benchmark_version_hash=benchmark_version_hash,
)
print("[uploader] done.")
PY
UPLOAD_EXIT=${PIPESTATUS[0]}

if [ $UPLOAD_EXIT -ne 0 ]; then
    echo "Upload failed with exit code: $UPLOAD_EXIT"
    exit $UPLOAD_EXIT
fi

echo "=============================================="
echo "Eval and upload finished successfully."
echo "=============================================="
