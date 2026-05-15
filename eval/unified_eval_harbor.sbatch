#!/bin/bash
#SBATCH -p booster
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --output=eval/logs/%x_%j.out
#SBATCH --job-name=eval

# ==============================================================================
# Unified Eval Harbor v4 — Jupiter Cluster (JSC GH200)
#
# Starts vLLM, runs Harbor eval, checks errors, uploads results.
# Merged from Jupiter v1 env setup + TACC v4 features (configurable params,
# Pending→Started DB flow, unified error checking, thinking, export-traces).
#
# Positional args:
#   $1 = MODEL (HF model name, e.g. mlfoundations-dev/some_model)
#   $2 = REPO_ID (HF dataset repo, e.g. DCAgent/dev_set_v2)
#   $3 = BENCHMARK_ID (optional, DB benchmark UUID)
#   $4 = RUN_TAG_ARG (optional, override run tag)
#
# Env vars from listener (v4 SbatchParams.to_env()):
#   EVAL_N_CONCURRENT       (default: 128)
#   EVAL_N_ATTEMPTS         (default: 3)
#   EVAL_GPU_MEMORY_UTIL    (default: 0.95)
#   EVAL_DAYTONA_THRESHOLD  (default: 999999)
#   EVAL_VLLM_MAX_RETRIES   (default: 20)
#   EVAL_AGENT_PARSER       (default: "")
#   EVAL_ENABLE_THINKING    (default: "false")
#   EVAL_AGENT_NAME         (default: "terminus-2")
#   EVAL_STARTS_LOG         (optional, shared eval starts log)
#   EVAL_TIMEOUT_MULTIPLIER (default: 1.0)
#   EVAL_CONFIG_YAML        (default: dcagent_eval_config.yaml)
#   EVAL_DB_JOB_ID          (optional, pre-created Pending job ID)
#   EVAL_UPLOAD_USERNAME    (optional, override upload username)
#   EVAL_OVERRIDE_MEMORY_MB (optional, memory override for harbor)
#   EVAL_AUTO_SNAPSHOT      (optional, "true"/"false" override for auto_snapshot)
#   EVAL_SNAPSHOT_NAME      (optional, Daytona snapshot template)
#   EVAL_AGENT_ENVS         (optional, comma-separated KEY=VALUE pairs to pass via --ae into sandbox)
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
N_ATTEMPTS="${EVAL_N_ATTEMPTS:-3}"
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.95}"
ERROR_THRESHOLD="${EVAL_DAYTONA_THRESHOLD:-999999}"
VLLM_MAX_RETRIES="${EVAL_VLLM_MAX_RETRIES:-20}"
AGENT_PARSER="${EVAL_AGENT_PARSER:-}"
ENABLE_THINKING="${EVAL_ENABLE_THINKING:-false}"
AGENT_NAME="${EVAL_AGENT_NAME:-terminus-2}"
EVAL_STARTS_LOG="${EVAL_STARTS_LOG:-}"
TIMEOUT_MULTIPLIER="${EVAL_TIMEOUT_MULTIPLIER:-1.0}"
CONFIG_YAML="${EVAL_CONFIG_YAML:-dcagent_eval_config.yaml}"
DB_JOB_ID="${EVAL_DB_JOB_ID:-}"
UPLOAD_USERNAME="${EVAL_UPLOAD_USERNAME:-$USER}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
if [[ "$REPO_ID" == /* ]]; then
    # Local path: use basename as the safe repo name
    SAFE_REPO=$(basename "$REPO_ID")
else
    SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')
fi

# Derive benchmark shorthand for SLURM job name (for squeue readability)
# Listener passes --job-name to override, but for manual runs derive from REPO_ID.
declare -A BENCH_SHORT=(
    ["DCAgent_dev_set_v2"]="v2"
    ["DCAgent2_swebench-verified-random-100-folders"]="swe"
    ["DCAgent2_terminal_bench_2"]="tb2"
    ["DCAgent_swebench-verified"]="swefull"
    ["DCAgent2_aider_polyglot"]="aider"
    ["DCAgent2_bfcl-parity"]="bfcl"
    ["DCAgent_medagentbench"]="medagent"
    ["DCAgent_gaia_127"]="gaia"
    ["DCAgent_financeagent_terminal"]="finance"
    ["DCAgent_dev_set_71_tasks"]="v1"
)
BENCH_TAG="${BENCH_SHORT[$SAFE_REPO]:-${SAFE_REPO:0:12}}"
# Only rename if still the default "eval" (i.e. listener didn't override)
if [ "$SLURM_JOB_NAME" = "eval" ]; then
    scontrol update JobId="$SLURM_JOB_ID" JobName="eval_${BENCH_TAG}"
fi

echo "=============================================="
echo "Jupiter Eval Harbor v4"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $REPO_ID"
echo "Benchmark ID: ${BENCHMARK_ID:-<auto>}"
echo "N concurrent: $N_CONCURRENT"
echo "N attempts: $N_ATTEMPTS"
echo "GPU memory util: $GPU_MEMORY_UTIL"
echo "Error threshold: $ERROR_THRESHOLD"
echo "vLLM max retries: $VLLM_MAX_RETRIES"
echo "Agent: $AGENT_NAME"
echo "Agent parser: ${AGENT_PARSER:-<default>}"
echo "Thinking: $ENABLE_THINKING"
echo "Timeout multiplier: $TIMEOUT_MULTIPLIER"
echo "Config YAML: $CONFIG_YAML"
echo "DB Job ID (pending): ${DB_JOB_ID:-<none>}"
echo "Upload username: $UPLOAD_USERNAME"
echo "=============================================="

# ==============================================================================
# Cluster-Agnostic Environment Setup
# ==============================================================================

DCFT="${EVAL_PROJECT_ROOT:-${DCFT:?ERROR: EVAL_PROJECT_ROOT not set. Launch via the listener with --cluster-config.}}"
CLUSTER_NAME="${EVAL_CLUSTER_NAME:?ERROR: EVAL_CLUSTER_NAME not set. Launch via the listener with --cluster-config.}"

# Reset PYTHONPATH to avoid inheriting submitter's environment (e.g. numpy 2.4
# from guha1/site-packages which breaks numba/vLLM).
unset PYTHONPATH

# Preserve listener-provided overrides before dotenv clobbers them
_SAVED_SECRET_ENV="${DC_AGENT_SECRET_ENV:-}"

# Source cluster-specific dotenv (e.g. hpc/dotenv/jupiter.env or hpc/dotenv/mbz.env)
DOTENV_FILE="$DCFT/hpc/dotenv/${CLUSTER_NAME}.env"
if [ -f "$DOTENV_FILE" ]; then
    source "$DOTENV_FILE"
    echo "Sourced dotenv: $DOTENV_FILE"
else
    echo "WARNING: dotenv not found: $DOTENV_FILE"
fi

# Source secrets (listener override takes precedence over dotenv)
DC_AGENT_SECRET_ENV="${_SAVED_SECRET_ENV:-${DC_AGENT_SECRET_ENV:-$HOME/secrets.env}}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

# --- Daytona API key (sourced from secrets.env above) ---
if [ -z "${DAYTONA_API_KEY:-}" ]; then
    echo "WARNING: DAYTONA_API_KEY not set. Check $DC_AGENT_SECRET_ENV"
fi
echo "Daytona API key: ${DAYTONA_API_KEY:0:12}..."

# --- vLLM / Ray / Triton env vars ---
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export VLLM_CONFIG_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_config_${USER}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER}}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_CACHE_DIR:-/tmp/flashinfer_cache_${USER}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache_${USER}}"
export HYDRA_FULL_ERROR=1
# Use naive all2all backend for DP (pplx_kernels requires separate install)
export VLLM_ALL2ALL_BACKEND="${VLLM_ALL2ALL_BACKEND:-naive}"
# Model/dataset cache — overridden by EVAL_HF_CACHE from listener
export HF_HUB_CACHE="${EVAL_HF_CACHE:-${HF_HUB_CACHE:?ERROR: HF_HUB_CACHE not set. Source your cluster dotenv or use --cluster-config.}}"
export HF_HOME="${HF_HUB_CACHE}"
export HF_CACHE_DIR="$HF_HUB_CACHE"
# XET staging cache — must be user-writable
export HF_XET_CACHE="${HF_XET_CACHE:-/tmp/hf_xet_cache_${USER}}"
mkdir -p "$HF_XET_CACHE"

# Harbor and DB operations
HARBOR_SRC="${EVAL_HARBOR_SRC:-}"
export PYTHONPATH="${HARBOR_SRC}:${DCFT}:${PYTHONPATH:-}"

# ==============================================================================
# LD_LOADER Wrappers (needed on aarch64 Jupiter where shared conda lacks exec perms)
# ==============================================================================

OTAGENT_DIR="${OTAGENT_DIR:?ERROR: OTAGENT_DIR not set. Use --conda-env with the listener.}"
export PATH="$OTAGENT_DIR/bin:$PATH"
export CONDA_PREFIX="$OTAGENT_DIR"

PYTHON_REAL="$OTAGENT_DIR/bin/python3.12"

# LD_LOADER Wrappers: required on aarch64 (Jupiter GH200) where shared conda
# lacks execute permissions.  Skipped on x86_64 where the conda env is directly usable.
if [ "$(uname -m)" = "aarch64" ] && [ -f /lib/ld-linux-aarch64.so.1 ]; then
    LD_LOADER="/lib/ld-linux-aarch64.so.1"
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
    HARBOR_BIN="$WRAPPER_DIR/harbor"
else
    PYTHON_BIN="$PYTHON_REAL"
    WRAPPER_DIR=""
    HARBOR_BIN="harbor"
fi

# Set LD_LIBRARY_PATH so vLLM can find CUDA runtime libs (libcudart.so.12, etc.)
# The nvidia pip packages install libs under site-packages/nvidia/*/lib/
NVIDIA_LIB_DIR="$OTAGENT_DIR/lib/python3.12/site-packages/nvidia"
if [ -d "$NVIDIA_LIB_DIR" ]; then
    for libdir in "$NVIDIA_LIB_DIR"/*/lib; do
        [ -d "$libdir" ] && export LD_LIBRARY_PATH="${libdir}:${LD_LIBRARY_PATH:-}"
    done
fi

echo "Python: $($PYTHON_BIN --version)"
harbor --help >/dev/null
export RAY_DEDUP_LOGS=0

# Set CUDA_HOME for flashinfer JIT compilation (needs nvcc).
# Set EVAL_CUDA_HOME in your dotenv (see hpc/dotenv/example.env) to point
# at the CUDA module that ships nvcc on your cluster. If unset, we leave
# CUDA_HOME alone and trust whatever the conda env / system already has.
_CUDA_HOME="${EVAL_CUDA_HOME:-}"
if [ -n "${_CUDA_HOME}" ] && [ -d "${_CUDA_HOME}" ]; then
    export CUDA_HOME="${_CUDA_HOME}"
    export PATH="$CUDA_HOME/bin:$PATH"
fi


# ==============================================================================
# SSH Tunnel + Proxychains Setup (no internet on compute nodes)
# ==============================================================================

NODE_HOST=$(hostname -s)
TUNNEL_PORT=7003
# EVAL_LOGIN_NODE: hostname of the login node that compute nodes can SSH to
# for outbound connectivity (Daytona, HF). Required for no-internet clusters
# that go through proxy.enabled=true in the cluster config; ignored otherwise.
LOGIN_NODE="${EVAL_LOGIN_NODE:-}"
PROXYCHAINS_BIN="${EVAL_PROXYCHAINS_BIN:-}"
TUNNEL_PID=""

setup_proxy() {
    if [ -z "${LOGIN_NODE}" ]; then
        echo "[proxy] ERROR: EVAL_LOGIN_NODE not set."
        echo "[proxy] No-internet clusters must export EVAL_LOGIN_NODE in their dotenv."
        return 1
    fi
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

    # NOTE: Do NOT set HTTPS_PROXY/HTTP_PROXY env vars — `requests` library reads
    # them but fails without PySocks. Proxychains LD_PRELOAD handles all proxying.

    # Test connectivity (HuggingFace)
    if [ -x "$PROXYCHAINS_BIN" ]; then
        if "$PROXYCHAINS_BIN" -f "$CFG_PATH" curl -s --connect-timeout 10 https://huggingface.co -o /dev/null 2>/dev/null; then
            echo "[proxy] HuggingFace connectivity test passed"
        else
            echo "[proxy] WARNING: HuggingFace connectivity test failed"
        fi
    fi

    # Test Daytona API connectivity
    if [ -n "${DAYTONA_API_KEY:-}" ]; then
        echo "[daytona] Testing API connectivity..."
        DAYTONA_HTTP_CODE=$("$PROXYCHAINS_BIN" -f "$CFG_PATH" curl -s --connect-timeout 15 \
            -H "Authorization: Bearer $DAYTONA_API_KEY" \
            "${DAYTONA_API_URL:-https://app.daytona.io/api}/health" \
            -o /tmp/daytona_health_$$.json -w "%{http_code}" 2>/dev/null) || DAYTONA_HTTP_CODE="FAIL"
        echo "[daytona] Health check HTTP code: $DAYTONA_HTTP_CODE"
        if [ -f /tmp/daytona_health_$$.json ]; then
            echo "[daytona] Response: $(cat /tmp/daytona_health_$$.json | head -c 200)"
            rm -f /tmp/daytona_health_$$.json
        fi
        if [ "$DAYTONA_HTTP_CODE" = "200" ]; then
            echo "[daytona] API connectivity OK"
        else
            echo "[daytona] WARNING: API returned $DAYTONA_HTTP_CODE — sandbox creation may fail"
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

# Setup proxy (skip if EVAL_PROXY_ENABLED=false, e.g. clusters with direct internet)
if [ "${EVAL_PROXY_ENABLED:-true}" = "true" ]; then
    setup_proxy || echo "[proxy] Continuing without proxy (some operations may fail)"
else
    echo "[proxy] Proxy disabled (EVAL_PROXY_ENABLED=false), assuming direct internet access"
fi

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
    # Kill ngrok
    if [ -n "${NGROK_PID:-}" ]; then
        kill "$NGROK_PID" 2>/dev/null || true
    fi
    # Kill pinggy
    if [ -n "${PINGGY_PID:-}" ]; then
        kill "$PINGGY_PID" 2>/dev/null || true
    fi
    # Kill GLM-5 → terminus-2 JSON proxy
    if [ -n "${GLM5_PROXY_PID:-}" ]; then
        kill "$GLM5_PROXY_PID" 2>/dev/null || true
    fi
    # Remove wrappers and per-job cache
    [ -n "${WRAPPER_DIR:-}" ] && rm -rf "$WRAPPER_DIR" 2>/dev/null || true
    [ -n "${_JOB_CACHE:-}" ] && rm -rf "$_JOB_CACHE" 2>/dev/null || true
    # Remove proxychains config
    rm -f "${PROXYCHAINS_CONF_FILE:-}" 2>/dev/null || true
    echo "Cleanup done."
}
trap cleanup EXIT

# ==============================================================================
# Start vLLM Server
# ==============================================================================
EVAL_LOGS_DIR="${EVAL_LOGS_DIR:-eval/logs}"
mkdir -p "$EVAL_LOGS_DIR"

# Derive unique vLLM port. When the listener packs jobs (EVAL_VLLM_PORT is set),
# use the listener-assigned port directly — it centrally plans non-overlapping
# port ranges across all jobs on a node. Otherwise fall back to job-ID-based port.
# IMPORTANT: Do NOT export VLLM_PORT as an env var — vLLM's _get_open_port()
# reads the VLLM_PORT env var and all DP subprocess children would try to bind
# the same port, causing an infinite port-collision loop.
_PHYSICAL_GPUS="${SLURM_JOB_GPUS:-0}"
FIRST_GPU=$(echo "$_PHYSICAL_GPUS" | cut -d',' -f1)
if [ -n "${EVAL_VLLM_PORT:-}" ]; then
    VLLM_PORT="$EVAL_VLLM_PORT"
else
    VLLM_PORT=$(( 10000 + (SLURM_JOB_ID % 50000) ))
fi
unset -v VLLM_PORT_ENV  # ensure no VLLM_PORT env var leaks to subprocesses
echo "Physical GPUs: $_PHYSICAL_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES), vLLM port: $VLLM_PORT"

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

# Per-job cache dirs to avoid JIT compilation conflicts when packing
_JOB_CACHE="/tmp/${USER}_${SLURM_JOB_ID}"
mkdir -p "$_JOB_CACHE"

# With DP > 1, vLLM needs torch.compile (TORCHDYNAMO_DISABLE must be unset)
_DP_SIZE="${EVAL_VLLM_DATA_PARALLEL_SIZE:-1}"
_DYNAMO_DISABLE=1
if [ "$_DP_SIZE" -gt 1 ] 2>/dev/null; then
    _DYNAMO_DISABLE=0
fi

env TORCHDYNAMO_DISABLE=$_DYNAMO_DISABLE \
    TMPDIR="$_JOB_CACHE" \
    TRITON_CACHE_DIR="$_JOB_CACHE/triton" \
    TORCH_COMPILE_CACHE_DIR="$_JOB_CACHE/torch" \
    TORCHINDUCTOR_CACHE_DIR="$_JOB_CACHE/torchinductor" \
    RAY_TMPDIR="$_JOB_CACHE/ray" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    HF_HUB_OFFLINE=1 \
"${VLLM_CMD[@]}" \
    > "$EVAL_LOGS_DIR/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

# Health check loop (use configurable retries)
MAX_RETRIES=$VLLM_MAX_RETRIES
RETRY_INTERVAL=100
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null 2>&1; then
        echo "vLLM server is ready"
        break
    fi
    echo "Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        # Dump last 50 lines of vLLM log
        tail -50 "$EVAL_LOGS_DIR/vllm_${SLURM_JOB_ID}.log" 2>/dev/null || true
        exit 1
    fi
done

# ==============================================================================
# GLM-5 → terminus-2 JSON adapter sidecar (optional, gated by EVAL_USE_GLM5_PROXY)
#
# When enabled, harbor's terminus-2 agent talks to a local FastAPI proxy that
# rewrites GLM-5 native output into the JSON shape terminus-2 reads. The proxy
# is a no-op for non-GLM-5 models (pass-through fallback in the adapter), but
# we still gate launch on the env var to keep default fires byte-identical.
# Skip when EVAL_HARBOR_CONFIG is set (installed-agent / Cat 3 fires reach
# vLLM through pinggy — not through this api_base).
# ==============================================================================
HARBOR_API_PORT="$VLLM_PORT"
if [ "${EVAL_USE_GLM5_PROXY:-0}" = "1" ] && [ -z "${EVAL_HARBOR_CONFIG:-}" ]; then
    PROXY_PORT=$(( VLLM_PORT + 10000 ))
    PROXY_LOG="$EVAL_LOGS_DIR/glm5_proxy_${SLURM_JOB_ID}.log"
    echo "[glm5-proxy] starting on port $PROXY_PORT, backend=http://localhost:$VLLM_PORT"
    "$PYTHON_BIN" "$DCFT/eval/proxy/glm5_proxy.py" \
        --backend "http://localhost:$VLLM_PORT" \
        --port "$PROXY_PORT" > "$PROXY_LOG" 2>&1 &
    GLM5_PROXY_PID=$!
    for _i in $(seq 1 15); do
        if curl -sf "http://localhost:$PROXY_PORT/v1/models" > /dev/null 2>&1; then
            echo "[glm5-proxy] healthy on port $PROXY_PORT (pid=$GLM5_PROXY_PID)"
            HARBOR_API_PORT="$PROXY_PORT"
            break
        fi
        sleep 1
    done
    if [ "$HARBOR_API_PORT" = "$VLLM_PORT" ]; then
        echo "[glm5-proxy] FAILED to start; aborting"
        tail -50 "$PROXY_LOG" 2>/dev/null || true
        exit 1
    fi
fi

# Ngrok state (initialized here, started later if needed)
NGROK_PID=""
NGROK_URL=""

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
    # Check dataset directories (from EVAL_DATASETS_DIRS or fallback)
    _DS_DIRS="${EVAL_DATASETS_DIRS:?ERROR: EVAL_DATASETS_DIRS not set. Set datasets_dirs in cluster config YAML.}"
    DATASET_LOCAL_DIR=""
    IFS=':' read -ra _DS_DIRS_ARR <<< "$_DS_DIRS"
    for _ds_dir in "${_DS_DIRS_ARR[@]}"; do
        if [ -d "${_ds_dir}/${SAFE_REPO}" ]; then
            DATASET_LOCAL_DIR="${_ds_dir}/${SAFE_REPO}"
            break
        fi
    done
    # Default to first dir if nothing found (snapshot_download will create it)
    DATASET_LOCAL_DIR="${DATASET_LOCAL_DIR:-${_DS_DIRS_ARR[0]}/${SAFE_REPO}}"
    DATASET_LOCAL_DIR_ALT=""  # no longer needed; absorbed into EVAL_DATASETS_DIRS
    echo "[DEBUG] DATASET_LOCAL_DIR=$DATASET_LOCAL_DIR"
    echo "[DEBUG] Local dir exists: $([ -d "$DATASET_LOCAL_DIR" ] && echo yes || echo no)"
    if [ -d "$DATASET_LOCAL_DIR" ]; then
        TASK_COUNT=$(ls -d "$DATASET_LOCAL_DIR"/*/instruction.md 2>/dev/null | wc -l)
        echo "[DEBUG] Found $TASK_COUNT tasks with instruction.md in local dir"
        echo "[DEBUG] Sample files:"
        file "$DATASET_LOCAL_DIR"/$(ls "$DATASET_LOCAL_DIR" | head -1)/environment/Dockerfile 2>/dev/null || echo "[DEBUG] No Dockerfile found"
    fi
    DOWNLOAD_LOG=$(mktemp /tmp/download_XXXXXX.log)
    proxied $PYTHON_BIN "$DCFT/eval/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_LOCAL_DIR" > "$DOWNLOAD_LOG" 2>&1
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

# ==============================================================================
# Construct run tag and directory
# ==============================================================================
if [ -n "$RUN_TAG_ARG" ]; then
    RUN_TAG="$RUN_TAG_ARG"
else
    RUN_TAG="${SAFE_REPO}_${SAFE_MODEL}"
fi
EVAL_JOBS_DIR="${EVAL_JOBS_DIR:?ERROR: EVAL_JOBS_DIR not set. Set eval_jobs_dir in cluster config YAML.}"
RUN_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

echo "Run tag: $RUN_TAG"
echo "Run dir: $RUN_DIR"

# --- Compute canonical benchmark name for DB uploads ---
declare -A BENCHMARK_NAME_MAP=(
    ["DCAgent2_aider_polyglot"]="aider_polyglot"
    ["DCAgent_dev_set_v2"]="dev_set_v2"
    ["DCAgent_dev_set_71_tasks"]="dev_set_71_tasks"
    ["DCAgent2_terminal_bench_2"]="terminal_bench_2"
    ["DCAgent_swebench_verified_eval_set"]="swebench-verified-random-100-folders"
    ["DCAgent2_bfcl-parity"]="bfcl-parity"
    ["DCAgent_swebench-verified"]="swebench-verified"
    ["DCAgent_medagentbench"]="medagentbench"
    ["DCAgent_gaia_127"]="gaia_127"
    ["DCAgent_financeagent_terminal"]="financeagent_terminal"
)
BASE_NAME="${BENCHMARK_NAME_MAP[$SAFE_REPO]:-$SAFE_REPO}"

# Append suffix for non-default timeout/memory configs.
BENCHMARK_SUFFIX=""
if [ -n "${EVAL_OVERRIDE_MEMORY_MB:-}" ] && [ "$EVAL_OVERRIDE_MEMORY_MB" != "1024" ]; then
    mem_gb=$(( EVAL_OVERRIDE_MEMORY_MB / 1024 ))
    BENCHMARK_SUFFIX="${BENCHMARK_SUFFIX}_${mem_gb}gb"
fi
if [ "$TIMEOUT_MULTIPLIER" != "1" ] && [ "$TIMEOUT_MULTIPLIER" != "1.0" ]; then
    BENCHMARK_SUFFIX="${BENCHMARK_SUFFIX}_${TIMEOUT_MULTIPLIER}x"
fi
if [ -n "$BENCHMARK_SUFFIX" ]; then
    BENCHMARK_NAME="${BASE_NAME}${BENCHMARK_SUFFIX}"
else
    BENCHMARK_NAME="$BASE_NAME"
fi
export BENCHMARK_NAME
echo "Benchmark name: $BENCHMARK_NAME"

# ==============================================================================
# Update DB: Pending → Started (v4 flow)
# ==============================================================================
echo "Creating/updating DB job entry..."
export MODEL REPO_ID RUN_TAG SLURM_JOB_ID BENCHMARK_NAME

HARBOR_VERSION=$($PYTHON_BIN -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION

if [ -n "$DB_JOB_ID" ]; then
    # v4 flow: listener already created a Pending entry — transition to Started
    echo "Updating Pending job $DB_JOB_ID → Started (run_tag=$RUN_TAG)"
    _db_fallback_marker="/tmp/.db_fallback_$$"
    rm -f "$_db_fallback_marker"
    proxied $PYTHON_BIN - <<'PY' || true
import os, sys, json
from database.unified_db.utils import update_job_status_to_started

run_tag = os.environ["RUN_TAG"]
agent_name = os.environ.get("EVAL_AGENT_NAME", "terminus-2")
n_concurrent = int(os.environ.get("EVAL_N_CONCURRENT", "128"))
n_attempts = int(os.environ.get("EVAL_N_ATTEMPTS", "3"))
timeout_multiplier = float(os.environ.get("EVAL_TIMEOUT_MULTIPLIER", "1.0"))
harbor_version = os.environ.get("HARBOR_VERSION", "unknown")

config = {
    "agent": agent_name,
    "env": "daytona",
    "timeout_multiplier": timeout_multiplier,
}

result = update_job_status_to_started(
    job_name=run_tag,
    n_trials=n_concurrent,
    n_rep_eval=n_attempts,
    config=config,
    harbor_package_version=harbor_version,
)
if not result.get("success"):
    print(f"WARNING: Pending→Started update failed: {result.get('error')}", file=sys.stderr)
    print("Will create a new Started entry as fallback.", file=sys.stderr)
    # Signal fallback needed via marker file (can't use exit code due to set -e)
    import pathlib
    pathlib.Path(f"/tmp/.db_fallback_{os.getpid()}").parent  # ensure /tmp exists
    open(f"/tmp/.db_fallback_{os.environ.get('SLURM_JOB_ID', os.getpid())}", "w").close()
else:
    print(f"DB job {run_tag} → Started")
PY
    _db_update_rc=0
    [ -f "/tmp/.db_fallback_${SLURM_JOB_ID}" ] && _db_update_rc=1 && rm -f "/tmp/.db_fallback_${SLURM_JOB_ID}"
else
    _db_update_rc=1  # No DB_JOB_ID — force fallback
fi

if [ "$_db_update_rc" -ne 0 ]; then
    # Fallback: create Started entry directly (Pending entry missing/deleted, or no DB_JOB_ID)
    echo "Creating new Started entry (fallback)..."
    proxied $PYTHON_BIN - <<'PY' || true
import os, sys

from database.unified_db.utils import create_job_entry_started

model_hf = os.environ["MODEL"]
dataset_hf = os.environ["REPO_ID"]
run_tag = os.environ["RUN_TAG"]
slurm_job_id = os.environ["SLURM_JOB_ID"]
harbor_version = os.environ.get("HARBOR_VERSION", "unknown")
agent_name = os.environ.get("EVAL_AGENT_NAME", "terminus-2")
n_concurrent = int(os.environ.get("EVAL_N_CONCURRENT", "128"))
n_attempts = int(os.environ.get("EVAL_N_ATTEMPTS", "3"))

result = create_job_entry_started(
    model_hf_name=model_hf,
    benchmark_hf_name=dataset_hf,
    job_name=run_tag,
    username=os.environ.get("USER", "jupiter"),
    slurm_job_id=slurm_job_id,
    harbor_package_version=harbor_version,
    agent_name=agent_name,
    config={"agent": agent_name, "env": "daytona"},
    n_trials=n_concurrent,
    n_rep_eval=n_attempts
)

if not result.get("success"):
    print(f"WARNING: DB create failed: {result.get('error')}", file=sys.stderr)
    sys.exit(0)

db_job_id = result["job"]["id"]
print(f"DB job created with ID: {db_job_id}")
PY

    # Get the DB job ID for later use
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
fi

if [ -z "${DB_JOB_ID:-}" ]; then
    echo "WARNING: Failed to get DB job ID"
else
    echo "DB job entry: $DB_JOB_ID"
fi

# ==============================================================================
# Eval Starts Log (v4 — for model retry tracking)
# ==============================================================================
if [ -n "$EVAL_STARTS_LOG" ]; then
    echo "${TIMESTAMP} ${MODEL} ${REPO_ID} ${SLURM_JOB_ID} ${RUN_TAG}" >> "$EVAL_STARTS_LOG"
    echo "Logged eval start to: $EVAL_STARTS_LOG"
fi

# ==============================================================================
# Run Harbor Eval (via proxychains for Daytona access)
# ==============================================================================
set +e

# Resolve config path: check eval/configs/ (canonical), then legacy per-cluster dirs
if [[ "$CONFIG_YAML" != /* ]]; then
    if [ -f "$DCFT/eval/configs/$CONFIG_YAML" ]; then
        HARBOR_CONFIG="$DCFT/eval/configs/$CONFIG_YAML"
    elif [ -f "$DCFT/eval/${CLUSTER_NAME}/$CONFIG_YAML" ]; then
        HARBOR_CONFIG="$DCFT/eval/${CLUSTER_NAME}/$CONFIG_YAML"
    elif [ -f "$DCFT/eval/MBZ/$CONFIG_YAML" ]; then
        HARBOR_CONFIG="$DCFT/eval/MBZ/$CONFIG_YAML"
    else
        echo "ERROR: Harbor config not found: $CONFIG_YAML"
        exit 1
    fi
else
    HARBOR_CONFIG="$CONFIG_YAML"
fi
echo "Resolved harbor config: $HARBOR_CONFIG"

# Build extra harbor args
EXTRA_HARBOR_ARGS=""
if [ -n "${EVAL_SNAPSHOT_NAME:-}" ]; then
    echo "Using Daytona snapshot: $EVAL_SNAPSHOT_NAME"
    EXTRA_HARBOR_ARGS="--environment-kwarg snapshot_template_name=$EVAL_SNAPSHOT_NAME --no-force-build"
fi
if [ "$TIMEOUT_MULTIPLIER" != "1" ] && [ "$TIMEOUT_MULTIPLIER" != "1.0" ]; then
    echo "Using timeout multiplier: $TIMEOUT_MULTIPLIER"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --timeout-multiplier $TIMEOUT_MULTIPLIER"
fi
if [ -n "${EVAL_OVERRIDE_MEMORY_MB:-}" ]; then
    echo "Using memory override: ${EVAL_OVERRIDE_MEMORY_MB}MB"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --override-memory-mb $EVAL_OVERRIDE_MEMORY_MB"
fi
# auto_snapshot override from listener (overrides YAML config value)
if [ -n "${EVAL_AUTO_SNAPSHOT:-}" ]; then
    echo "Auto snapshot override: $EVAL_AUTO_SNAPSHOT"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --environment-kwarg auto_snapshot=$EVAL_AUTO_SNAPSHOT"
fi

# Build thinking args
THINKING_ARGS=""
if [ "$ENABLE_THINKING" = "true" ]; then
    echo "Thinking enabled"
    THINKING_ARGS='--agent-kwarg "enable_thinking=true"'
fi

# Build agent-parser args
PARSER_ARGS=""
if [ -n "$AGENT_PARSER" ]; then
    echo "Agent parser: $AGENT_PARSER"
    PARSER_ARGS="--agent-kwarg \"parser=$AGENT_PARSER\""
fi

echo "[DEBUG] EXTRA_HARBOR_ARGS=$EXTRA_HARBOR_ARGS"
echo "[DEBUG] DAYTONA_API_KEY=${DAYTONA_API_KEY:0:20}..."

# Check if a previous job dir exists for this run tag (for resume)
# Skip resume if EVAL_FORCE_FRESH=true
EXISTING_JOB_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
if [ "${EVAL_FORCE_FRESH:-false}" != "true" ] && [ -d "$EXISTING_JOB_DIR" ] && [ -f "$EXISTING_JOB_DIR/config.json" ]; then
    echo "Found existing job dir, resuming: $EXISTING_JOB_DIR"
    # Patch stale api_base port from prior SLURM run → current vLLM port. The
    # vLLM port formula is JID-derived (10000 + SLURM_JOB_ID%50000), so every
    # resume has a different port than the saved config.json. Without this
    # patch, terminus agents call the OLD port → "Connect call failed" → all
    # trials fail. (Confirmed empirically on JID 428153, 2026-05-07.)
    #
    # We must patch BOTH the job-level config.json AND every per-trial
    # config.json. Harbor's resume code (harbor/job.py:_init_remaining_trial_configs)
    # compares per-trial TrialConfig equality (excluding trial_name) to subtract
    # finished trials from the run queue. Without per-trial patches, every
    # existing trial's saved api_base mismatches the freshly generated config →
    # __eq__ fails → harbor logs "Existing trial X not found in generated configs;
    # skipping" → all n_attempts × n_tasks trials get re-run from scratch.
    # (Confirmed empirically on JID 428210, 2026-05-08.)
    echo "Patching api_base port in config.json: -> localhost:${VLLM_PORT}"
    sed -i.bak-port "s|\"api_base\": \"http://localhost:[0-9]\+/|\"api_base\": \"http://localhost:${VLLM_PORT}/|g" "$EXISTING_JOB_DIR/config.json"
    PATCHED_TRIAL_COUNT=$(find "$EXISTING_JOB_DIR" -mindepth 2 -maxdepth 2 -name config.json -print 2>/dev/null | wc -l)
    if [ "$PATCHED_TRIAL_COUNT" -gt 0 ]; then
        echo "Patching api_base port in $PATCHED_TRIAL_COUNT per-trial config.json files: -> localhost:${VLLM_PORT}"
        find "$EXISTING_JOB_DIR" -mindepth 2 -maxdepth 2 -name config.json -print0 2>/dev/null | \
            xargs -0 -r sed -i.bak-port "s|\"api_base\": \"http://localhost:[0-9]\+/|\"api_base\": \"http://localhost:${VLLM_PORT}/|g"
    fi
    proxied $HARBOR_BIN jobs resume \
        -p "$EXISTING_JOB_DIR" \
        --filter-error-type EnvironmentStartTimeoutError \
        --filter-error-type DaytonaError \
        --filter-error-type DaytonaAuthenticationError \
        --filter-error-type DaytonaAuthorizationError \
        --filter-error-type DaytonaNotFoundError \
        --filter-error-type DaytonaRateLimitError \
        --filter-error-type CancelledError \
        --filter-error-type AgentEnvironmentTimeoutError \
        --filter-error-type SandboxBuildFailedError
else
    echo "Starting new job"

    # Short-term fix: inject model_name into a temp copy of the YAML config so that
    # we do NOT pass --agent on the CLI. Passing --agent causes Harbor to wipe the
    # entire agents[] section from the YAML and rebuild from CLI flags only, losing
    # all YAML agent settings (max_timeout_sec, trajectory_config, kwargs like
    # proactive_summarization_threshold, etc.). By omitting --agent, Harbor uses the
    # YAML agents[] and --agent-kwarg values are MERGED into it.
    # See: harbor/src/harbor/cli/jobs.py lines 715-748
    # Long-term fix: Harbor PR to merge CLI --agent with YAML instead of replacing.
    RUNTIME_HARBOR_CONFIG="/tmp/harbor_config_${SLURM_JOB_ID}.yaml"
    python3 -c "
import yaml

with open('${HARBOR_CONFIG}') as f:
    cfg = yaml.safe_load(f)

agent = cfg['agents'][0]

# Model prefix routing for litellm:
# - hosted_vllm/: terminus family (in-process), oracle/nop (utility), mini-swe-agent
#   (installed agent but routes via hosted_vllm provider + auto-generates registry)
# - openai/: aider, openhands, and other installed agents using openai-compatible provider
_HOSTED_VLLM_PREFIX_AGENTS = {
    'terminus', 'terminus-1', 'terminus-2', 'terminus-kira',
    'oracle', 'nop',
    'mini-swe-agent',
    'swe-agent',
}
agent_name = agent.get('name', 'terminus-2')
if agent_name in _HOSTED_VLLM_PREFIX_AGENTS:
    agent['model_name'] = 'hosted_vllm/${MODEL}'
else:
    agent['model_name'] = 'openai/${MODEL}'

# Merge model_info and max_tokens: YAML values take priority, CLI defaults fill gaps.
kwargs = agent.setdefault('kwargs', {})

# Default model_info values (used only if YAML doesn't specify them)
defaults = {
    'max_output_tokens': ${EVAL_MAX_OUTPUT_TOKENS:-16384},
    'max_input_tokens': ${EVAL_MAX_INPUT_TOKENS:-32768},
    'input_cost_per_token': 0.0,
    'output_cost_per_token': 0.0,
}
yaml_model_info = kwargs.get('model_info', {})
# Fill in any missing keys from defaults
for k, v in defaults.items():
    yaml_model_info.setdefault(k, v)
kwargs['model_info'] = yaml_model_info

# max_tokens defaults to max_output_tokens from the resolved model_info
kwargs.setdefault('max_tokens', yaml_model_info['max_output_tokens'])

with open('${RUNTIME_HARBOR_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"
    echo "[DEBUG] Injected model_name + model_info into runtime config: $RUNTIME_HARBOR_CONFIG"

    # ==========================================================================
    # Tunnel for installed agents (Pinggy preferred, ngrok fallback)
    # Installed agents (aider, openhands, etc.) execute their CLI inside the
    # remote Daytona container and cannot reach localhost on the SLURM node.
    # A public tunnel exposes the vLLM server via a URL the sandbox can reach.
    # ==========================================================================
    _CFG_AGENT_NAME=$($PYTHON_BIN -c "
import yaml
with open('$RUNTIME_HARBOR_CONFIG') as f:
    cfg = yaml.safe_load(f)
agents = cfg.get('agents', [])
print(agents[0].get('name', 'terminus-2') if agents else 'terminus-2')
" 2>/dev/null || echo "terminus-2")

    _NEEDS_TUNNEL=false
    case "$_CFG_AGENT_NAME" in
        terminus|terminus-1|terminus-2|terminus-kira|oracle|nop)
            echo "[tunnel] Agent '$_CFG_AGENT_NAME' runs on host — tunnel not needed."
            ;;
        *)
            _NEEDS_TUNNEL=true
            echo "[tunnel] Installed agent '$_CFG_AGENT_NAME' runs in sandbox — need tunnel to vLLM."
            ;;
    esac

    if [ "$_NEEDS_TUNNEL" = true ]; then
        TUNNEL_URL=""

        # --- Try Pinggy first (SSH-based, persistent URL, no bandwidth limit) ---
        if [ -n "${EVAL_PINGGY_URL:-}" ] && [ -n "${EVAL_PINGGY_TOKEN:-}" ]; then
            echo "[pinggy] Starting SSH tunnel to ${EVAL_PINGGY_URL} (port $VLLM_PORT)..."
            PINGGY_LOG="$EVAL_LOGS_DIR/pinggy_${SLURM_JOB_ID}.log"

            bash -c "while true; do ssh -p 443 \
                -R0:localhost:${VLLM_PORT} \
                -o StrictHostKeyChecking=no \
                -o ServerAliveInterval=30 \
                -o ExitOnForwardFailure=yes \
                ${EVAL_PINGGY_TOKEN}@pro.pinggy.io; \
                sleep 10; done" > "$PINGGY_LOG" 2>&1 &
            PINGGY_PID=$!
            echo "[pinggy] Started with PID $PINGGY_PID"

            # Wait for SSH to establish (5s stabilization)
            _PINGGY_OK=true
            for _pg_i in $(seq 1 5); do
                sleep 1
                if ! kill -0 "$PINGGY_PID" 2>/dev/null; then
                    echo "[pinggy] ERROR: tunnel process died during startup"
                    cat "$PINGGY_LOG" 2>/dev/null || true
                    PINGGY_PID=""
                    _PINGGY_OK=false
                    break
                fi
            done

            if [ "$_PINGGY_OK" = true ]; then
                TUNNEL_URL="https://${EVAL_PINGGY_URL}"
                # Verify connectivity
                if curl -sk "${TUNNEL_URL}/v1/models" 2>/dev/null | grep -q "object"; then
                    echo "[pinggy] Connectivity verified: $TUNNEL_URL"
                else
                    echo "[pinggy] WARNING: connectivity test failed, continuing anyway..."
                fi
            fi
        fi

        # --- Fallback to ngrok ---
        if [ -z "$TUNNEL_URL" ]; then
            echo "[tunnel] Pinggy not available, trying ngrok..."
            NGROK_PATH="${EVAL_NGROK_PATH:-$HOME/ngrok}"
            NGROK_LOG="$EVAL_LOGS_DIR/ngrok_${SLURM_JOB_ID}.log"

            if [ -x "$NGROK_PATH" ]; then
                # Check if an existing ngrok tunnel is available (from another job on same node)
                NGROK_URL=$(curl -s localhost:4040/api/tunnels 2>/dev/null \
                    | grep -o '"public_url":"https://[^"]*"' | head -1 | cut -d'"' -f4)
                if [ -n "$NGROK_URL" ]; then
                    echo "[ngrok] Reusing existing tunnel: $NGROK_URL"
                    TUNNEL_URL="$NGROK_URL"
                else
                    "$NGROK_PATH" http "$VLLM_PORT" --log=stdout > "$NGROK_LOG" 2>&1 &
                    NGROK_PID=$!
                    echo "[ngrok] Started with PID $NGROK_PID, tunneling port $VLLM_PORT"

                    for _ng_i in $(seq 1 15); do
                        NGROK_URL=$(curl -s localhost:4040/api/tunnels 2>/dev/null \
                            | grep -o '"public_url":"https://[^"]*"' | head -1 | cut -d'"' -f4)
                        [ -n "$NGROK_URL" ] && break
                        echo "[ngrok] Waiting for tunnel (attempt $_ng_i/15)..."
                        sleep 2
                    done

                    if [ -n "$NGROK_URL" ]; then
                        TUNNEL_URL="$NGROK_URL"
                        echo "[ngrok] Tunnel established: $NGROK_URL"
                    else
                        echo "[ngrok] ERROR: Failed to establish ngrok tunnel"
                        cat "$NGROK_LOG" 2>/dev/null || true
                    fi
                fi
            else
                echo "[ngrok] Binary not found at $NGROK_PATH, skipping."
            fi
        fi

        # --- Final check ---
        if [ -z "$TUNNEL_URL" ]; then
            echo "ERROR: No tunnel available for installed agent '$_CFG_AGENT_NAME'."
            echo "  Set EVAL_PINGGY_URL + EVAL_PINGGY_TOKEN, or install ngrok at $HOME/ngrok"
            exit 1
        fi

        export OPENAI_API_KEY="fake_key"
        export OPENAI_API_BASE="${TUNNEL_URL}/v1"
        echo "[tunnel] OPENAI_API_BASE=${TUNNEL_URL}/v1"
        # OpenHands-specific: it reads LLM_BASE_URL which takes priority over
        # --agent-kwarg api_base=localhost. Export explicitly to override.
        if [ "$_CFG_AGENT_NAME" = "openhands" ]; then
            export LLM_BASE_URL="${TUNNEL_URL}/v1"
            export LLM_API_KEY="fake_key"
            echo "[tunnel] LLM_BASE_URL=${TUNNEL_URL}/v1 (openhands)"
        fi
        # mini-swe-agent reads MSWEA_MODEL_API_KEY (not LLM_API_KEY) and MSWEA_MODEL_API_BASE.
        # Gate on agent name so other installed agents' env is untouched.
        if [ "$_CFG_AGENT_NAME" = "mini-swe-agent" ]; then
            export MSWEA_MODEL_API_KEY="fake_key"
            export MSWEA_MODEL_API_BASE="${TUNNEL_URL}/v1"
            # litellm's hosted_vllm provider reads HOSTED_VLLM_API_BASE inside the
            # sandbox (NOT OPENAI_API_BASE or MSWEA_MODEL_API_BASE). Without this,
            # litellm falls back to https://api.openai.com and every call 404s.
            # The harbor agent forwards this env var into the sandbox via setdefault.
            export HOSTED_VLLM_API_BASE="${TUNNEL_URL}/v1"
            # Token limits used by mini-swe-agent to auto-generate litellm cost-registry in sandbox.
            # Match our default max_input/max_output so the registry reflects real vLLM capacity.
            export MSWEA_MODEL_MAX_TOKENS="${EVAL_MAX_INPUT_TOKENS:-32768}"
            export MSWEA_MODEL_MAX_OUTPUT_TOKENS="${EVAL_MAX_OUTPUT_TOKENS:-16384}"
            # Tiny non-zero per-token costs so the auto-generated registry's cost
            # calculation returns > 0 (mini-swe-agent crashes with "Cost must be > 0.0"
            # when cost == 0). Local vLLM has no real cost; this is a placeholder.
            export MSWEA_MODEL_INPUT_COST_PER_TOKEN="${MSWEA_MODEL_INPUT_COST_PER_TOKEN:-1e-10}"
            export MSWEA_MODEL_OUTPUT_COST_PER_TOKEN="${MSWEA_MODEL_OUTPUT_COST_PER_TOKEN:-1e-10}"
            echo "[tunnel] MSWEA_MODEL_API_BASE=${TUNNEL_URL}/v1 (for mini-swe-agent)"
            echo "[tunnel] HOSTED_VLLM_API_BASE=${TUNNEL_URL}/v1 (for mini-swe-agent)"
        fi
        # swe-agent needs SWEAGENT_CONFIG (URL or local path) and reads OPENAI_BASE_URL
        # with priority over the --agent-kwarg api_base=localhost (which would fail in sandbox).
        if [ "$_CFG_AGENT_NAME" = "swe-agent" ]; then
            # Upstream config on harbor/main. The legacy URL pointed to Negin's branch
            # which has since been deleted (404s silently via curl -sSL, causing sweagent
            # to parse a "404: Not Found" body as YAML and fail with NonZeroAgentExitCodeError).
            export SWEAGENT_CONFIG="${SWEAGENT_CONFIG:-https://raw.githubusercontent.com/laude-institute/harbor/refs/heads/main/examples/configs/swe-agent-infer-single.yaml}"
            export OPENAI_BASE_URL="${TUNNEL_URL}/v1"
            echo "[tunnel] SWEAGENT_CONFIG=$SWEAGENT_CONFIG (for swe-agent)"
            echo "[tunnel] OPENAI_BASE_URL=${TUNNEL_URL}/v1 (for swe-agent)"
        fi
    fi

    # Build command as array to handle quoting properly
    # NOTE: --agent and --model are NOT passed here. Agent name and model come from
    # the YAML config. model_info and max_tokens are set in the Python config injection
    # above (YAML values take priority, CLI env var defaults fill gaps).
    HARBOR_CMD=(
        proxied $HARBOR_BIN jobs start
        -p "$DATASET_PATH"
        --n-concurrent "$N_CONCURRENT"
        --env "daytona"
        --agent-kwarg "api_base=http://localhost:${HARBOR_API_PORT}/v1"
        --agent-kwarg "key=fake_key"
        --n-attempts "$N_ATTEMPTS"
        --job-name "$RUN_TAG"
        --export-traces
        --config "$RUNTIME_HARBOR_CONFIG"
        --jobs-dir "$EVAL_JOBS_DIR"
        --debug
    )
    # Add optional args
    if [ "$ENABLE_THINKING" = "true" ]; then
        HARBOR_CMD+=(--agent-kwarg "enable_thinking=true")
    fi
    if [ -n "$AGENT_PARSER" ]; then
        HARBOR_CMD+=(--agent-kwarg "parser=$AGENT_PARSER")
    fi

    # Forward env vars into the Daytona sandbox (e.g. API keys for tools).
    # EVAL_AGENT_ENVS is a comma-separated list of KEY=VALUE pairs.
    if [ -n "${EVAL_AGENT_ENVS:-}" ]; then
        IFS=',' read -ra _AE_PAIRS <<< "$EVAL_AGENT_ENVS"
        for _ae in "${_AE_PAIRS[@]}"; do
            _ae=$(echo "$_ae" | xargs)  # trim whitespace
            [ -n "$_ae" ] && HARBOR_CMD+=(--ae "$_ae")
        done
        echo "[DEBUG] Agent env vars: ${#_AE_PAIRS[@]} pairs forwarded via --ae"
    fi

    # Print command with secrets redacted
    _DEBUG_CMD="${HARBOR_CMD[*]} $EXTRA_HARBOR_ARGS"
    _DEBUG_CMD=$(echo "$_DEBUG_CMD" | sed -E 's/(--ae [A-Z_]+=)[^ ]*/\1<redacted>/g')
    echo "[DEBUG] Full harbor command:"
    echo "  $_DEBUG_CMD"

    "${HARBOR_CMD[@]}" $EXTRA_HARBOR_ARGS
fi
SB_EXIT=$?
set -e

# ==============================================================================
# Save meta.env (v4 — extended fields)
# ==============================================================================
# Track resume count
RESUME_COUNT=0
if [ -f "$RUN_DIR/meta.env" ]; then
    OLD_RESUME_COUNT=$(grep -oP 'RESUME_COUNT=\K[0-9]+' "$RUN_DIR/meta.env" 2>/dev/null || echo "0")
    RESUME_COUNT=$((OLD_RESUME_COUNT + 1))
    echo "Resume count: $RESUME_COUNT (was $OLD_RESUME_COUNT)"
fi

mkdir -p "$RUN_DIR"
{
    echo "MODEL=$MODEL"
    echo "REPO_ID=$REPO_ID"
    echo "TIMESTAMP=$TIMESTAMP"
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "DB_JOB_ID=${DB_JOB_ID:-}"
    echo "BENCHMARK_ID=${BENCHMARK_ID:-}"
    echo "N_CONCURRENT=$N_CONCURRENT"
    echo "N_ATTEMPTS=$N_ATTEMPTS"
    echo "GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL"
    echo "AGENT_NAME=$AGENT_NAME"
    echo "AGENT_PARSER=${AGENT_PARSER:-}"
    echo "ENABLE_THINKING=$ENABLE_THINKING"
    echo "TIMEOUT_MULTIPLIER=$TIMEOUT_MULTIPLIER"
    echo "BENCHMARK_NAME=$BENCHMARK_NAME"
    echo "RESUME_COUNT=$RESUME_COUNT"
} > "$RUN_DIR/meta.env"

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
# Check for errors before upload (v4 unified error checking)
# Counts ALL errors except known-benign types:
#   AgentTimeoutError, BadRequestError, ContextLengthExceededError,
#   SummarizationTimeout, SummarizationTimeoutError
# NOTE: AgentEnvironmentTimeoutError is NOT benign (infra issue, not model's fault)
#   but is excluded from retry (retrying won't help)
# ==============================================================================
RESULT_FILE="$RUN_DIR/result.json"
ERROR_LOG="${EVAL_JOBS_DIR}/invalid_errors_${SLURM_JOB_ID}.log"

if [ -f "$RESULT_FILE" ]; then
    echo "Checking for invalid errors in $RESULT_FILE..."

    INVALID_COUNT=$($PYTHON_BIN -c "
import json, sys
try:
    with open('$RESULT_FILE', 'r') as f:
        data = json.load(f)

    BENIGN_ERRORS = {
        'AgentTimeoutError',
        'BadRequestError',
        'ContextLengthExceededError',
        'SummarizationTimeout',
        'SummarizationTimeoutError',
    }

    total_invalid = 0
    if 'stats' in data and 'evals' in data['stats']:
        for eval_key, eval_data in data['stats']['evals'].items():
            if 'exception_stats' in eval_data:
                for err_type, err_ids in eval_data['exception_stats'].items():
                    if err_type not in BENIGN_ERRORS:
                        if isinstance(err_ids, list):
                            total_invalid += len(err_ids)
    print(total_invalid)
except Exception as e:
    print(f'Error parsing result.json: {e}', file=sys.stderr)
    print('0')
" 2>&1 | tail -n 1)

    echo "Invalid error count: ${INVALID_COUNT} (threshold: ${ERROR_THRESHOLD})"

    if [ "${INVALID_COUNT:-0}" -gt "$ERROR_THRESHOLD" ]; then
        echo "Job has ${INVALID_COUNT} invalid errors (> ${ERROR_THRESHOLD}), skipping upload"

        {
            echo "==============================================="
            echo "Timestamp: $(date)"
            echo "Job: ${RUN_TAG}"
            echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
            echo "Model: ${MODEL}"
            echo "Repo: ${REPO_ID}"
            echo "Invalid errors: ${INVALID_COUNT}"
            echo "Result file: ${RESULT_FILE}"
            $PYTHON_BIN -c "
import json
BENIGN = {'AgentTimeoutError', 'BadRequestError', 'ContextLengthExceededError', 'SummarizationTimeout', 'SummarizationTimeoutError'}
with open('$RESULT_FILE', 'r') as f:
    data = json.load(f)
if 'stats' in data and 'evals' in data['stats']:
    for eval_key, eval_data in data['stats']['evals'].items():
        if 'exception_stats' in eval_data:
            for err_type, err_ids in eval_data['exception_stats'].items():
                if err_type not in BENIGN and err_ids:
                    print(f'Eval: {eval_key} / {err_type}')
                    for i, error_id in enumerate(err_ids[:10], 1):
                        print(f'  {i}. {error_id}')
                    if len(err_ids) > 10:
                        print(f'  ... and {len(err_ids) - 10} more')
" 2>/dev/null || true
            echo "==============================================="
        } >> "$ERROR_LOG"

        echo "Error details logged to: $ERROR_LOG"
        echo "Job completed but not uploaded due to excessive invalid errors"
        exit 0
    fi
else
    echo "Warning: result.json not found, continuing with upload"
fi

# ==============================================================================
# Upload results to DB (via proxychains for HF upload)
# ==============================================================================
export RUN_DIR
export UPLOAD_USERNAME
export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
export RUN_TAG

UPLOAD_LOG="${EVAL_LOGS_DIR:-eval/logs}/upload_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$UPLOAD_LOG")"

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


run_dir = os.environ["RUN_DIR"]
run_tag = os.environ["RUN_TAG"]
username = os.environ.get("UPLOAD_USERNAME", os.environ.get("USER", "jupiter"))
error_mode = os.environ.get("UPLOAD_MODE", "skip_on_error")
hf_repo_id = sanitize_hf_repo_id(f"DCAgent3/{run_tag}")
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

upload_eval_results(
    run_dir,
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
