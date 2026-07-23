#!/bin/bash
#SBATCH -p gh
#SBATCH --time=23:59:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --account CCR24067
#SBATCH --output=eval/tacc/logs/%x_%j.out
#SBATCH --job-name=data

# ==============================================================================
# Unified Eval Harbor — TACC Vista (GH200 96GB, aarch64, sm_90)
#
# Starts vLLM (the from-source fork wheel in the otagent env), runs Harbor eval
# against Daytona sandboxes, checks errors, uploads results.
# Adapted from eval/leonardo/eval_harbor.sbatch.
#
# KEY DIFFERENCES vs Leonardo (all because Vista compute nodes have FULL DIRECT
# internet egress — verified job 787310):
#   - NO proxychains / SSH SOCKS5 tunnel / step-ca cert. Daytona API + HF are
#     reached directly. No `proxied()` wrapper — every command runs unwrapped.
#   - NO model pre-download step (compute node downloads from HF directly).
#   - NO --gres / --mem (Vista doesn't gres-track GPUs + misreports RealMemory);
#     the node is allocated whole. TP from the per-model config (TP=1 default for
#     the 32B family — fits the 96GB GH200).
#   - aarch64: load the build toolchain (gcc/13.2.0 cuda/12.8) that the fork vLLM
#     wheel was built against, and put conda's libstdc++ ahead on LD_LIBRARY_PATH.
#
# Positional args:
#   $1 = MODEL (HF model name)
#   $2 = REPO_ID (HF dataset repo, e.g. DCAgent2/terminal_bench_2)
#   $3 = BENCHMARK_ID (optional, DB benchmark UUID)
#   $4 = RUN_TAG_ARG (optional, override run tag)
#
# Env vars from listener: EVAL_N_CONCURRENT, EVAL_GPU_MEMORY_UTIL,
#   EVAL_DAYTONA_THRESHOLD, EVAL_CONFIG_YAML, EVAL_AGENT_KWARGS,
#   EVAL_AGENT_PARSER, EVAL_DB_JOB_ID, EVAL_VLLM_* (serve overrides).
# ==============================================================================

set -eo pipefail
ulimit -c 0
ulimit -n 65536 2>/dev/null || true

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

# --- Parse positional args ---
MODEL="${1:-Qwen/Qwen3-32B}"
REPO_ID="${2:-DCAgent2/terminal_bench_2}"
BENCHMARK_ID="${3:-}"
RUN_TAG_ARG="${4:-}"

# --- Read env vars from listener (with defaults) ---
N_CONCURRENT="${EVAL_N_CONCURRENT:-32}"
# Vista-only concurrency ceiling (FIX B, 2026-06-29). The shared eval presets set
# n_concurrent=32 (eval/presets/{swebench,tb2,v2}.yaml), forwarded here as
# EVAL_N_CONCURRENT and passed to `harbor jobs start` via --n-concurrent (which
# OVERRIDES any harbor-YAML n_concurrent_trials — see harbor cli/jobs.py:1095).
# On Vista that drives too many simultaneous Daytona create_sandbox/start calls →
# bursty env-build contention → EnvironmentStartTimeoutError (1800s) → trials
# finalize as VerificationNotCompletedError with 0 reward (diagnosed in
# agent_logs/2026-06-29_tacc_swebench_verifier_vnc.md). Clamp the START-path
# concurrency to <= TACC_N_CONCURRENT_CEILING here — this file is Vista-EXCLUSIVE
# (only eval/clusters/tacc.yaml references it; Leonardo/Jupiter use their own
# eval_harbor.sbatch), so the shared presets / harbor YAMLs are UNTOUCHED and
# Leonardo's concurrency is unaffected. (The RESUME path reads n_concurrent_trials
# from the run's stored config.json, not --n-concurrent, so it is clamped
# separately in the resume block below.)
#
# RELAXED 8 -> 16 (2026-06-29): the original ceiling of 8 over-indexed on a single
# transient Lustre/Daytona burst. 8 is very slow and Vista should handle far more
# concurrent env-builds. Raising to 16 and TRACKING whether the VNC storm recurs:
# watch the per-run VerificationNotCompletedError / EnvironmentStartTimeoutError
# rate (count in result.json outcomes; >~10% of attempted trials = the storm is
# back). If 16 stays clean, push higher; revert toward 8 ONLY if the storm
# demonstrably recurs at 16. Override per-run via EVAL_TACC_N_CONCURRENT_CEILING.
#
# REVERTED 16 -> 8 (2026-07-16): the throughput diagnosis on the axolotl-32B
# ID-eval legs showed conc-16 is GPU-BOUND, not env-build-bound — a single GH200
# serving the 32B at TP=1 saturates at ~4-5 concurrent in-flight generations, so 16
# concurrent Daytona trials just queue behind the vLLM engine (oversubscribed, no
# throughput gain, wasted sandbox lifetime). Back to 8 to match the serve ceiling.
# Override per-run via EVAL_TACC_N_CONCURRENT_CEILING.
TACC_N_CONCURRENT_CEILING="${EVAL_TACC_N_CONCURRENT_CEILING:-8}"
if [ "$N_CONCURRENT" -gt "$TACC_N_CONCURRENT_CEILING" ]; then
    echo "Vista concurrency ceiling: capping N_CONCURRENT $N_CONCURRENT -> $TACC_N_CONCURRENT_CEILING"
    N_CONCURRENT="$TACC_N_CONCURRENT_CEILING"
fi
GPU_MEMORY_UTIL="${EVAL_GPU_MEMORY_UTIL:-0.90}"
DAYTONA_ERROR_THRESHOLD="${EVAL_DAYTONA_THRESHOLD:-999999}"
CONFIG_YAML="${EVAL_CONFIG_YAML:-dcagent_eval_config_no_override.yaml}"
AGENT_KWARGS="${EVAL_AGENT_KWARGS:-}"
AGENT_PARSER="${EVAL_AGENT_PARSER:-}"
DB_JOB_ID="${EVAL_DB_JOB_ID:-}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
if [[ "$REPO_ID" == /* ]]; then
    SAFE_REPO=$(basename "$REPO_ID")
else
    SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')
fi

SHORT_MODEL=$(echo "$MODEL" | awk -F'/' '{print $NF}')
scontrol update JobId=$SLURM_JOB_ID Name="eval_${SHORT_MODEL}" 2>/dev/null || true

echo "=============================================="
echo "TACC Vista Eval Harbor (GH200 aarch64)"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $REPO_ID"
echo "Benchmark ID: ${BENCHMARK_ID:-<auto>}"
echo "N concurrent: $N_CONCURRENT"
echo "GPU memory util: $GPU_MEMORY_UTIL"
echo "Config YAML: $CONFIG_YAML"
echo "Agent kwargs: ${AGENT_KWARGS:-<none>}"
echo "Agent parser: ${AGENT_PARSER:-<default>}"
echo "Daytona threshold: $DAYTONA_ERROR_THRESHOLD"
echo "DB Job ID (pending): ${DB_JOB_ID:-<none>}"
echo "=============================================="

# ==============================================================================
# TACC Vista Environment Setup
# ==============================================================================
SCRATCH="${SCRATCH:-/scratch/10635/penfever}"
DCFT="${EVAL_PROJECT_ROOT:-${SCRATCH}/OpenThoughts-Agent}"

# Resolve the harbor config path from CONFIG_YAML (mirrors Leonardo/Jupiter:
# hpc/harbor_yaml/eval/configs/ -> eval/tacc/ -> eval/MBZ/).
if [[ "$CONFIG_YAML" == /* ]]; then
    HARBOR_CONFIG="$CONFIG_YAML"
elif [ -f "$DCFT/hpc/harbor_yaml/eval/configs/$CONFIG_YAML" ]; then
    HARBOR_CONFIG="$DCFT/hpc/harbor_yaml/eval/configs/$CONFIG_YAML"
elif [ -f "$DCFT/eval/tacc/$CONFIG_YAML" ]; then
    HARBOR_CONFIG="$DCFT/eval/tacc/$CONFIG_YAML"
elif [ -f "$DCFT/eval/MBZ/$CONFIG_YAML" ]; then
    HARBOR_CONFIG="$DCFT/eval/MBZ/$CONFIG_YAML"
else
    echo "ERROR: Harbor config not found: $CONFIG_YAML"
    exit 1
fi
echo "Resolved harbor config: $HARBOR_CONFIG"

unset PYTHONPATH

# --- aarch64 toolchain (matches the fork vLLM wheel build: gcc/13.2.0 cuda/12.8) ---
module purge 2>/dev/null || true
module load gcc/13.2.0 cuda/12.8 2>/dev/null || true

# --- Conda activation: otagent (harbor + fork vLLM + DB client + uploader) ---
source "${SCRATCH}/miniconda3/etc/profile.d/conda.sh"
conda activate otagent
echo "Conda env: otagent (prefix $CONDA_PREFIX)"

# Prepend conda's lib so its libstdc++ wins over the system/module versions.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Source TACC environment + secrets
source "$DCFT/hpc/dotenv/tacc.env" 2>/dev/null || true
DC_AGENT_SECRET_ENV="${DC_AGENT_SECRET_ENV:-${SCRATCH}/keys.env}"
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
fi

# --- Detect the harbor config's agent (host-run terminus-2 vs installed agent) ---
# terminus-2/oracle/nop run the LLM call ON the compute node (localhost vLLM);
# installed agents (opencode, openhands, ...) run their CLI INSIDE the Daytona
# sandbox and must reach the served model over a public tunnel. This flag gates
# every installed-agent-only addition below (strict-DATA Daytona key, pinggy
# tunnel, opencode routing) so the terminus-2 path stays byte-identical.
CFG_AGENT_NAME=$(python -c "
import yaml
with open('$HARBOR_CONFIG') as f:
    cfg = yaml.safe_load(f)
agents = cfg.get('agents', [])
print(agents[0].get('name', 'terminus-2') if agents else 'terminus-2')
" 2>/dev/null || echo "terminus-2")
case "$CFG_AGENT_NAME" in
    terminus|terminus-1|terminus-2|terminus-kira|oracle|nop) IS_INSTALLED_AGENT=false ;;
    *) IS_INSTALLED_AGENT=true ;;
esac
echo "Harbor config agent: $CFG_AGENT_NAME (installed_agent=$IS_INSTALLED_AGENT)"

# --- Daytona API key — same orgs as Leonardo (org1=DAYTONA_API_KEY, org2=DATA) ---
# Keys come from the secrets env (sourced above as $DC_AGENT_SECRET_ENV) — NEVER hardcode.
DAYTONA_KEY_ORG1="${DAYTONA_API_KEY:?DAYTONA_API_KEY unset — source the secrets env ($DC_AGENT_SECRET_ENV) before launch}"
DAYTONA_KEY_ORG2="${DAYTONA_DATA_API_KEY:?DAYTONA_DATA_API_KEY unset — source the secrets env ($DC_AGENT_SECRET_ENV) before launch}"
if [ -n "$EVAL_DAYTONA_ORG" ]; then
    # Explicit per-leg org override (org-spreading across the two eval orgs so no
    # single org's concurrency/snapshot cap is hit → fewer SandboxBuildFailedError).
    # Values: org1 (DAYTONA_API_KEY) | org2 (DAYTONA_DATA_API_KEY). Forwarded from the
    # listener submit env via sbatch --export=ALL. Overrides the installed-agent
    # strict-org2 default below; unset → default behavior unchanged.
    if [ "$EVAL_DAYTONA_ORG" = "org1" ]; then
        export DAYTONA_API_KEY="$DAYTONA_KEY_ORG1"
        echo "Selected Daytona org: org1 (explicit EVAL_DAYTONA_ORG) (${DAYTONA_API_KEY:0:12}...)"
    else
        export DAYTONA_API_KEY="$DAYTONA_KEY_ORG2"
        echo "Selected Daytona org: org2/DATA (explicit EVAL_DAYTONA_ORG) (${DAYTONA_API_KEY:0:12}...)"
    fi
elif [ "$IS_INSTALLED_AGENT" = true ]; then
    # Installed-agent evals (opencode) use STRICT 100% DATA org (org2) per the
    # per-experiment DATA policy. Gated on the installed-agent path so the running
    # terminus-2 evals keep their historical 3:1 org2:org1 load-spread unchanged.
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG2"
    echo "Selected Daytona org: org2/DATA (strict, installed-agent) (${DAYTONA_API_KEY:0:12}...)"
elif (( RANDOM % 4 == 0 )); then
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG1"
    echo "Selected Daytona org: org1 (${DAYTONA_API_KEY:0:12}...)"
else
    export DAYTONA_API_KEY="$DAYTONA_KEY_ORG2"
    echo "Selected Daytona org: org2 (${DAYTONA_API_KEY:0:12}...)"
fi

# Daytona timeout/retry settings
export DAYTONA_MAX_RETRIES=5
export DAYTONA_RETRY_DELAY=30
export DAYTONA_BACKOFF_FACTOR=2
export DAYTONA_TIMEOUT=1800
export AIOHTTP_CLIENT_TIMEOUT=900
export AIOHTTP_CONNECTOR_TIMEOUT=900
export AIOHTTP_SOCK_CONNECT_TIMEOUT=300
export AIOHTTP_TOTAL_TIMEOUT=1800

# --- Caches (home quota is tiny on Vista; use scratch) ---
export VLLM_CACHE_ROOT="${SCRATCH}/vllm"
export VLLM_CONFIG_ROOT="${SCRATCH}/vllm"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_${SLURM_JOB_ID}"
# FlashInfer: PERSISTENT cache on scratch (NOT /tmp — MoE kernel JIT compilation
# takes 20-40 min for 182 kernels and MUST persist across jobs).  The first job
# with a new model compiles; subsequent jobs hit the cache instantly.
export FLASHINFER_WORKSPACE_BASE="${SCRATCH}/flashinfer_cache"
# Limit ninja parallelism during flashinfer JIT — unlimited (72 cores) OOM-kills
# nvcc on MoE kernel compilation. 8 parallel × ~6GB/nvcc = ~48GB, safe on 480GB.
export MAX_JOBS=8
export TORCH_EXTENSIONS_DIR="${SCRATCH}/torch_extensions"
export TORCH_HOME="${SCRATCH}/torch_home"
export XDG_CACHE_HOME="${SCRATCH}/.cache"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/.cache/torchinductor"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${SCRATCH}/hub}"
# Pin NETRC to dodge the aiohttp Path.home() thread-unsafety segfault under
# high harbor concurrency (see Leonardo sbatch rationale).
export NETRC="${SCRATCH}/.netrc_eval_placeholder"
touch "$NETRC" 2>/dev/null || true
export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=0   # Vista compute nodes have direct egress

# PYTHONPATH: harbor src + project root
HARBOR_SRC="${EVAL_HARBOR_SRC:-${SCRATCH}/harbor/src}"
export PYTHONPATH="${HARBOR_SRC}:${DCFT}:${PYTHONPATH:-}"

PYTHON_BIN="$(which python)"
SERVE_PYTHON_BIN="$PYTHON_BIN"   # single otagent env serves everything on Vista
echo "Python: $($PYTHON_BIN --version)"
echo "PYTHONPATH: $PYTHONPATH"

# ==============================================================================
# Cleanup trap
# ==============================================================================
cleanup() {
    echo "Cleaning up..."
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    # Installed-agent pinggy tunnel (opencode) — kill the retry-loop + its ssh.
    if [ -n "${PINGGY_PID:-}" ]; then
        kill "$PINGGY_PID" 2>/dev/null || true
        pkill -P "$PINGGY_PID" 2>/dev/null || true
    fi
    echo "Cleanup done."
}
trap cleanup EXIT

# ==============================================================================
# Construct run tag and unified run directory (BEFORE vLLM start)
# ==============================================================================
EVAL_JOBS_DIR="${EVAL_JOBS_DIR:-${SCRATCH}/eval_jobs}"

if [ -n "$RUN_TAG_ARG" ]; then
    RUN_TAG="$RUN_TAG_ARG"
else
    RUN_TAG="eval-${SAFE_MODEL}_${SAFE_REPO}"
fi
RUN_DIR="${EVAL_JOBS_DIR}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

mkdir -p "$DCFT/eval/tacc/logs"
SLURM_LOG="${DCFT}/eval/tacc/logs/data_${SLURM_JOB_ID}.out"
ln -sf "$SLURM_LOG" "$RUN_DIR/slurm.log" 2>/dev/null || true

mkdir -p "${DCFT}/trace_jobs"
rm -rf "${DCFT}/trace_jobs/${RUN_TAG}" 2>/dev/null || true
ln -sfn "$RUN_DIR" "${DCFT}/trace_jobs/${RUN_TAG}"

echo "Run tag: $RUN_TAG"
echo "Run dir: $RUN_DIR"

# ==============================================================================
# Start vLLM Server (single node, TP from per-model config; default TP=1)
# ==============================================================================
export HF_HUB_OFFLINE=0

# --- TP=1 is a HARD Vista invariant (this sbatch is Vista-EXCLUSIVE) ---------
# Every Vista GH200 node has EXACTLY 1 GPU and this job hardcodes `#SBATCH --nodes 1`
# + serves vLLM directly in the single-node allocation (no multi-node vLLM), so the
# serve tensor-parallel size MUST be 1. But build_vllm_cmd.sh defaults
# EVAL_VLLM_TENSOR_PARALLEL_SIZE to 4 (a multi-GPU-cluster default). The listener
# path exports TP=1 via the gh200 registry profile, but a RAW resume submit
# (`sbatch eval/tacc/eval_harbor.sbatch <MODEL> <REPO> "" <RUN-TAG>`) never runs the
# listener -> the var is UNSET -> build_vllm_cmd falls back to TP=4 -> vLLM aborts
# with "World size (4) > available GPUs (1)", the server never comes up, 0 trials
# (the 2026-07-01 sweep-8 resume crash: jids 801464-801477 all cancelled). Pin the
# Vista hardware truth here so BOTH the listener path (unchanged: already 1) and the
# raw-resume path serve at TP=1. See agent_logs/2026-07-02_tacc-resume-tp4-bug.md.
export EVAL_VLLM_TENSOR_PARALLEL_SIZE="${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-1}"

# --- Installed-agent (opencode) tool-calling: serve WITH the tool parser --------
# opencode drives the OpenAI tool-calling API with tool_choice="auto", which vLLM
# REJECTS with HTTP 400 ("\"auto\" tool choice requires --enable-auto-tool-choice
# and --tool-call-parser to be set") unless the serve carries those flags. The eval
# baseline registry profile does NOT export a tool_call_parser for this model, so the
# 2026-07-16 opencode legs (836536/836538/836539) 400'd on EVERY trial -> all rewards
# 0.0 (agent_logs / STATE qwen3-30b-thinking-id-evals-opencode). build_vllm_cmd.sh
# already emits `--enable-auto-tool-choice --tool-call-parser <val>` when
# EVAL_VLLM_TOOL_CALL_PARSER is set; set it here, GATED to installed agents so the
# terminus-2 host-agent path stays byte-identical (terminus parses tool-calls
# client-side and must NOT get vLLM-side tool parsing). Default: hermes (the Qwen3
# family parser — every qwen3 model_config uses it). An explicit
# EVAL_VLLM_TOOL_CALL_PARSER from the listener wins (e.g. qwen3_coder).
if [ "$IS_INSTALLED_AGENT" = true ]; then
    export EVAL_VLLM_TOOL_CALL_PARSER="${EVAL_VLLM_TOOL_CALL_PARSER:-hermes}"
    echo "[opencode] vLLM serve tool-calling ENABLED: --enable-auto-tool-choice --tool-call-parser ${EVAL_VLLM_TOOL_CALL_PARSER}"
fi

echo "Starting vLLM server for model: $MODEL"
source "$DCFT/eval/build_vllm_cmd.sh"
build_vllm_cmd "$SERVE_PYTHON_BIN" "$MODEL" "$GPU_MEMORY_UTIL"

# ==============================================================================
# model_info for hosted_vllm (harbor #339 hard-requires it; see Jupiter sbatch).
# ==============================================================================
MI_MAX_MODEL_LEN="${EVAL_VLLM_MAX_MODEL_LEN:-32768}"
MI_MAX_OUTPUT="${EVAL_MAX_OUTPUT_TOKENS:-16384}"
if [ "$MI_MAX_OUTPUT" -ge "$MI_MAX_MODEL_LEN" ] 2>/dev/null; then
    MI_MAX_OUTPUT=$(( MI_MAX_MODEL_LEN / 2 ))
fi
MODEL_INFO_JSON="{\"max_input_tokens\":${MI_MAX_MODEL_LEN},\"max_output_tokens\":${MI_MAX_OUTPUT},\"input_cost_per_token\":0,\"output_cost_per_token\":0}"
echo "[model_info] $MODEL_INFO_JSON"

# Vista does NOT gres-track GPUs and the node is allocated whole, so the serve
# srun must NOT pass --gres (it would be rejected as "Invalid gres"). For a
# single-node (TP*DP <= 1 node) serve, run vLLM directly in this allocation.
"${VLLM_CMD[@]}" > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

# Health check loop
MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
VLLM_HEALTH_URL="http://${MASTER_ADDR}:8000/v1/models"
echo "Health check URL: $VLLM_HEALTH_URL"
MAX_RETRIES=30
RETRY_INTERVAL=100
for i in $(seq 1 $MAX_RETRIES); do
    if $PYTHON_BIN -c "
import urllib.request, sys
try:
    urllib.request.urlopen('$VLLM_HEALTH_URL', timeout=10)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        echo "vLLM server is ready"
        break
    fi
    echo "Waiting for vLLM server (attempt $i/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
    if [ $i -eq $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        echo "--- Diagnostic ---"
        $PYTHON_BIN -c "
import socket
try:
    s = socket.create_connection(('$MASTER_ADDR', 8000), timeout=5); s.close()
    print('TCP connect to port 8000: OK')
except Exception as e:
    print(f'TCP connect to port 8000: FAILED ({e})')
" 2>&1 || true
        tail -80 "$RUN_DIR/vllm.log" 2>/dev/null || true
        exit 1
    fi
done

# ==============================================================================
# Download / locate dataset (direct HF access — compute nodes have egress)
# ==============================================================================
if [[ "$REPO_ID" == /* ]]; then
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
    DATASET_LOCAL_DIR="${SCRATCH}/data/datasets/${SAFE_REPO}"

    if [ -d "$DATASET_LOCAL_DIR" ]; then
        TASK_COUNT=$(ls -d "$DATASET_LOCAL_DIR"/*/instruction.md 2>/dev/null | wc -l)
    fi
    if [ "${TASK_COUNT:-0}" -eq 0 ]; then
        HF_CACHE_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
        HF_REPO_CACHE_NAME="datasets--$(echo "$REPO_ID" | sed 's/\//--/g')"
        HF_DATASET_CACHE=$(find "$HF_CACHE_DIR" -maxdepth 3 -type d -name "snapshots" -path "*${HF_REPO_CACHE_NAME}*" 2>/dev/null | head -1)
        if [ -n "$HF_DATASET_CACHE" ]; then
            HF_SNAPSHOT=$(ls -dt "$HF_DATASET_CACHE"/*/ 2>/dev/null | head -1)
            if [ -n "$HF_SNAPSHOT" ]; then
                HF_TASK_COUNT=$(ls -d "$HF_SNAPSHOT"/*/instruction.md 2>/dev/null | wc -l)
                if [ "$HF_TASK_COUNT" -gt 0 ]; then
                    TASK_COUNT=$HF_TASK_COUNT
                    DATASET_LOCAL_DIR="$HF_SNAPSHOT"
                fi
            fi
        fi
    fi

    if [ "${TASK_COUNT:-0}" -gt 0 ]; then
        echo "Using pre-downloaded dataset: $DATASET_LOCAL_DIR ($TASK_COUNT tasks)"
        DATASET_PATH="$DATASET_LOCAL_DIR"
    else
        echo "Dataset not local, downloading directly..."
        DOWNLOAD_LOG=$(mktemp /tmp/download_XXXXXX.log)
        $PYTHON_BIN "$DCFT/eval/snapshot_download.py" "$REPO_ID" --local-dir "$DATASET_LOCAL_DIR" > "$DOWNLOAD_LOG" 2>&1
        DOWNLOAD_EXIT=$?
        cat "$DOWNLOAD_LOG"
        DATASET_PATH=$(grep DATASET_PATH "$DOWNLOAD_LOG" | tail -n 1 | cut -d'=' -f2)
        rm -f "$DOWNLOAD_LOG"
        if [ $DOWNLOAD_EXIT -ne 0 ] || [ -z "${DATASET_PATH:-}" ]; then
            echo "ERROR: Failed to get dataset path (exit code: $DOWNLOAD_EXIT)"
            exit 1
        fi
    fi
fi
echo "Using dataset path: $DATASET_PATH"

# --- Canonical benchmark name for DB uploads ---
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
# Update DB: Started
# ==============================================================================
echo "Creating/updating DB job entry..."
export MODEL REPO_ID RUN_TAG SLURM_JOB_ID BENCHMARK_NAME
HARBOR_VERSION=$($PYTHON_BIN -c "import harbor; print(harbor.__version__)" 2>/dev/null || echo "unknown")
export HARBOR_VERSION

$PYTHON_BIN - <<'PY' || true
import os, sys
from database.unified_db.utils import create_job_entry_started
result = create_job_entry_started(
    model_hf_name=os.environ["MODEL"],
    benchmark_hf_name=os.environ["REPO_ID"],
    job_name=os.environ["RUN_TAG"],
    username=os.environ.get("USER", "tacc"),
    slurm_job_id=os.environ["SLURM_JOB_ID"],
    harbor_package_version=os.environ.get("HARBOR_VERSION", "unknown"),
    agent_name="terminus-2",
    config={"agent": "terminus-2", "env": "daytona"},
    n_trials=128,
    n_rep_eval=3,
)
if not result.get("success"):
    print(f"WARNING: DB create failed: {result.get('error')}", file=sys.stderr)
    sys.exit(0)
print(f"DB job created with ID: {result['job']['id']}")
PY

DB_JOB_ID=$($PYTHON_BIN - <<'PY' || true
import os
from database.unified_db.utils import get_latest_job_for_model_benchmark
try:
    result = get_latest_job_for_model_benchmark(os.environ["MODEL"], os.environ["REPO_ID"])
    if result and result.get("id"):
        print(result["id"])
except Exception as e:
    import sys; print(f"WARNING: DB lookup failed: {e}", file=sys.stderr)
PY
)
[ -z "${DB_JOB_ID:-}" ] && echo "WARNING: Failed to get DB job ID" || echo "DB job entry: $DB_JOB_ID"

# ==============================================================================
# Run Harbor Eval (direct Daytona access — no proxychains)
# ==============================================================================
set +e

EXTRA_HARBOR_ARGS=""
if [ -n "${EVAL_SNAPSHOT_NAME:-}" ]; then
    EXTRA_HARBOR_ARGS="--environment-kwarg snapshot_template_name=$EVAL_SNAPSHOT_NAME --no-force-build"
fi
if [ -n "${EVAL_TIMEOUT_MULTIPLIER:-}" ]; then
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --timeout-multiplier $EVAL_TIMEOUT_MULTIPLIER"
fi
if [ -n "${EVAL_OVERRIDE_MEMORY_MB:-}" ]; then
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --override-memory-mb $EVAL_OVERRIDE_MEMORY_MB"
fi
# auto_snapshot override from the listener (mirrors eval/jupiter/eval_harbor.sbatch:811-814
# and eval/leonardo/eval_harbor.sbatch:616-618). WITHOUT this the TACC path never forwarded
# EVAL_AUTO_SNAPSHOT -> harbor defaulted auto_snapshot=False -> the from-scratch IMAGE-build
# sandbox path (non-ephemeral, auto_delete_interval=0, no server-side reaper) instead of the
# harbor__* SNAPSHOT path -> every eval sandbox LEAKED forever. Forwarding it routes evals onto
# the snapshot path -> ephemeral=True -> server-side auto-delete + restores the "snapshot"
# leaderboard field. (2026-06-28 leak fix; uses SHARED snapshots, no cap impact.)
if [ -n "${EVAL_AUTO_SNAPSHOT:-}" ]; then
    echo "Auto snapshot override: $EVAL_AUTO_SNAPSHOT"
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --environment-kwarg auto_snapshot=$EVAL_AUTO_SNAPSHOT"
fi
# EVAL_AGENT_KWARGS is newline-separated; each entry is a space-free key=value
# (e.g. the live nested extra_body thinking form — no enable_thinking flag), so
# it stays one token under the unquoted $EXTRA_HARBOR_ARGS word-split.
if [ -n "$AGENT_KWARGS" ]; then
    while IFS= read -r _kw; do
        [ -z "$_kw" ] && continue
        EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --agent-kwarg $_kw"
    done <<< "$AGENT_KWARGS"
fi
if [ -n "$AGENT_PARSER" ]; then
    EXTRA_HARBOR_ARGS="$EXTRA_HARBOR_ARGS --agent-kwarg parser=$AGENT_PARSER"
fi

echo "[DEBUG] EXTRA_HARBOR_ARGS=$EXTRA_HARBOR_ARGS"

# ==============================================================================
# Installed-agent (opencode) ingress: pinggy tunnel + model routing (ADDITIVE)
# ------------------------------------------------------------------------------
# opencode runs its CLI INSIDE the Daytona sandbox and calls the served model
# over a PUBLIC URL. Vista compute nodes have direct egress but no public
# INGRESS, so we expose the local vLLM (localhost:8000) via a pinggy SSH tunnel
# and route the agent at it (OPENAI_BASE_URL + OPENCODE_DUMMY_KEY; harbor
# registers the `vllm` provider via @ai-sdk/openai-compatible). Ported from
# eval/jupiter/eval_harbor.sbatch:1013-1188 minus proxychains (Vista = direct
# egress). Gated on IS_INSTALLED_AGENT so the terminus-2 host-agent path (used by
# the running axolotl evals) is byte-identical — this whole block is skipped for
# terminus-2. Runs for BOTH new-job and resume dispatches below.
# ==============================================================================
if [ "$IS_INSTALLED_AGENT" = true ]; then
    if [ -z "${EVAL_PINGGY_URL:-}" ] || [ -z "${EVAL_PINGGY_TOKEN:-}" ]; then
        echo "[pinggy] FATAL: installed agent '$CFG_AGENT_NAME' needs a public tunnel to vLLM, but EVAL_PINGGY_URL/EVAL_PINGGY_TOKEN are unset."
        echo "[pinggy] Pass --pinggy_persistent_url + --pinggy_token to the listener. Refusing to waste compute."
        exit 1
    fi
    echo "[pinggy] Starting SSH tunnel to ${EVAL_PINGGY_URL} (vLLM port 8000)..."
    PINGGY_LOG="$DCFT/eval/tacc/logs/pinggy_${SLURM_JOB_ID}.log"
    bash -c "while true; do ssh -p 443 \
        -R0:localhost:8000 \
        -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=30 \
        -o ExitOnForwardFailure=yes \
        ${EVAL_PINGGY_TOKEN}@pro.pinggy.io; \
        sleep 10; done" > "$PINGGY_LOG" 2>&1 &
    PINGGY_PID=$!
    echo "[pinggy] Started with PID $PINGGY_PID (log: $PINGGY_LOG)"
    # Stabilize (SSH establish)
    for _pg in $(seq 1 5); do
        sleep 1
        if ! kill -0 "$PINGGY_PID" 2>/dev/null; then
            echo "[pinggy] ERROR: tunnel process died during startup"
            cat "$PINGGY_LOG" 2>/dev/null || true
            exit 1
        fi
    done
    TUNNEL_URL="https://${EVAL_PINGGY_URL}"
    # Verify the sandbox-facing URL actually reaches OUR vLLM (direct egress on
    # Vista; no proxychains). Retry for pinggy DNS / vLLM-readiness lag.
    _verified=false
    for _vt in $(seq 1 12); do
        if curl -sk --connect-timeout 5 "${TUNNEL_URL}/v1/models" 2>/dev/null | grep -q "object"; then
            echo "[pinggy] Connectivity verified: $TUNNEL_URL (attempt $_vt)"
            _verified=true
            break
        fi
        sleep 5
    done
    if [ "$_verified" = false ]; then
        echo "[pinggy] FATAL: connectivity test failed after 12 attempts (60s)."
        echo "[pinggy] Installed-agent fires require a working tunnel; refusing to waste compute."
        echo "[pinggy] Last 20 lines of $PINGGY_LOG:"
        tail -20 "$PINGGY_LOG" 2>/dev/null || true
        exit 1
    fi
    # opencode routing: `vllm/<served_id>` provider + baseURL/key from env
    # (harbor agents/installed/opencode.py _build_register_config_command). The
    # served id is $MODEL (build_vllm_cmd --served-model-name defaults to $MODEL).
    export OPENAI_BASE_URL="${TUNNEL_URL}/v1"
    export OPENCODE_DUMMY_KEY="${OPENCODE_DUMMY_KEY:-capability-url-no-auth-header}"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-$OPENCODE_DUMMY_KEY}"
    echo "[opencode] routing: model=vllm/$MODEL OPENAI_BASE_URL=$OPENAI_BASE_URL"
fi

if [ -d "$RUN_DIR" ] && [ -f "$RUN_DIR/config.json" ]; then
    echo "Found existing job dir, resuming: $RUN_DIR"
    # --- Refresh api_base in config.json to the LIVE serve node before resume ---
    # `harbor jobs resume` has NO --agent-kwarg override (verified: cli/jobs.py
    # resume() loads config.json into JobConfig verbatim — no kwarg merge). A
    # same-run_tag job that resumes on a NEW SLURM node would otherwise reuse the
    # PRIOR run's api_base baked into config.json (agents[].kwargs.api_base),
    # dialing the DEAD predecessor node -> every LLM call fails "Cannot connect to
    # host" (jobs 790834/835/836, 2026-06-28: 600+ conn-fails, 0 scores). Rewrite
    # api_base to THIS allocation's MASTER_ADDR so the resume dials the live node.
    NEW_API_BASE="http://${MASTER_ADDR}:8000/v1"
    echo "Refreshing config.json api_base -> $NEW_API_BASE; clamping n_concurrent_trials -> $N_CONCURRENT"
    # Also clamp n_concurrent_trials in the stored config.json (FIX B, resume path).
    # `harbor jobs resume` does NOT accept --n-concurrent — it loads config.json
    # verbatim into JobConfig (cli/jobs.py:1492) and runs at that stored
    # n_concurrent_trials, which was baked at START time from the shared preset's
    # 32. Rewrite it down to the Vista ceiling ($N_CONCURRENT) so the VNC-resume
    # re-runs at the lower concurrency too (else the resume re-hits the same
    # Daytona env-start contention and re-VNCs).
    $PYTHON_BIN - "$RUN_DIR/config.json" "$NEW_API_BASE" "$N_CONCURRENT" <<'PY'
import json, sys
cfg_path, new_api_base, n_conc = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(cfg_path) as f:
    cfg = json.load(f)
changed = False
for agent in cfg.get("agents", []):
    kw = agent.get("kwargs")
    if isinstance(kw, dict) and kw.get("api_base") != new_api_base:
        print(f"  agent api_base {kw.get('api_base')!r} -> {new_api_base!r}")
        kw["api_base"] = new_api_base
        changed = True
# n_concurrent_trials is a top-level JobConfig field in current harbor
# (models/job/config.py); legacy configs nested it under `orchestrator`. Clamp
# whichever form the stored config.json uses.
cur = cfg.get("n_concurrent_trials")
if isinstance(cur, int) and cur > n_conc:
    print(f"  n_concurrent_trials {cur} -> {n_conc}")
    cfg["n_concurrent_trials"] = n_conc
    changed = True
orch = cfg.get("orchestrator")
if isinstance(orch, dict):
    ocur = orch.get("n_concurrent_trials")
    if isinstance(ocur, int) and ocur > n_conc:
        print(f"  orchestrator.n_concurrent_trials {ocur} -> {n_conc}")
        orch["n_concurrent_trials"] = n_conc
        changed = True
if changed:
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("  config.json updated")
else:
    print("  api_base + n_concurrent_trials already current; no change")
PY
    # Derive --filter-error-type flags from the canonical INFRA_ERROR_TYPES set
    # (database/unified_db/infra_errors.py) at runtime — single source of truth,
    # shared with the listener's resume manager, so the resume filter can never
    # drift from the set the listener counts. Reuses the same $PYTHON_BIN +
    # PYTHONPATH (incl. $DCFT repo root) the listener/uploads use.
    _RESUME_FILTER_STR="$($PYTHON_BIN -m database.unified_db.infra_errors --filter-flags)" || {
        echo "[resume] FATAL: failed to derive --filter-error-type flags from INFRA_ERROR_TYPES (bad PYTHONPATH?). Refusing to resume — a zero-filter resume would silently change scope." >&2
        exit 1
    }
    if [ -z "${_RESUME_FILTER_STR// /}" ]; then
        echo "[resume] FATAL: empty --filter-error-type flags from INFRA_ERROR_TYPES; refusing to resume with no infra filter." >&2
        exit 1
    fi
    read -ra RESUME_FILTER_FLAGS <<< "$_RESUME_FILTER_STR"
    echo "[resume] infra filter flags (from INFRA_ERROR_TYPES): ${RESUME_FILTER_FLAGS[*]}"
    harbor jobs resume \
        -p "$RUN_DIR" \
        "${RESUME_FILTER_FLAGS[@]}"
elif [ "$IS_INSTALLED_AGENT" = true ]; then
    echo "Starting new job (installed agent: $CFG_AGENT_NAME)"
    # harbor's `jobs start` IGNORES --model unless --agent is ALSO passed (and
    # --agent would WIPE the YAML agents[] — see cli/jobs.py:1213), so inject the
    # runtime model_name into a config copy and run WITHOUT --agent/--model. The
    # `vllm/<served>` provider + OPENAI_BASE_URL/OPENCODE_DUMMY_KEY (exported by
    # the tunnel block above) route opencode at our served vLLM.
    RUNTIME_HARBOR_CONFIG="/tmp/harbor_opencode_${SLURM_JOB_ID}.yaml"
    python - "$HARBOR_CONFIG" "$RUNTIME_HARBOR_CONFIG" "vllm/$MODEL" <<'PY'
import sys, yaml
src, dst, model = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src) as f:
    cfg = yaml.safe_load(f)
if not cfg.get("agents"):
    raise SystemExit(f"FATAL: no agents[] in harbor config {src}")
cfg["agents"][0]["model_name"] = model
with open(dst, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print(f"injected model_name={model} into {dst}")
PY
    echo "[opencode] runtime harbor config: $RUNTIME_HARBOR_CONFIG"
    harbor jobs start \
        -p "$DATASET_PATH" \
        --n-concurrent "$N_CONCURRENT" \
        --env "daytona" \
        --n-attempts 3 \
        --job-name "$RUN_TAG" \
        --jobs-dir "$EVAL_JOBS_DIR" \
        --config "$RUNTIME_HARBOR_CONFIG" \
        $EXTRA_HARBOR_ARGS
else
    echo "Starting new job"
    harbor jobs start \
        -p "$DATASET_PATH" \
        --n-concurrent "$N_CONCURRENT" \
        --agent terminus-2 \
        --model "hosted_vllm/$MODEL" \
        --env "daytona" \
        --agent-kwarg "api_base=http://${MASTER_ADDR}:8000/v1" \
        --agent-kwarg "key=fake_key" \
        --agent-kwarg "model_info=$MODEL_INFO_JSON" \
        --n-attempts 3 \
        --job-name "$RUN_TAG" \
        --jobs-dir "$EVAL_JOBS_DIR" \
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

TRIAL_COUNT=$(find "$RUN_DIR" -maxdepth 1 -type d -name '*__*' 2>/dev/null | wc -l)
echo "Found $TRIAL_COUNT trial dirs in $RUN_DIR"

if [ ${SB_EXIT:-0} -ne 0 ]; then
    echo "Harbor eval exited with non-zero status: ${SB_EXIT}. Skipping upload."
    exit ${SB_EXIT}
fi
if [ ! -d "$RUN_DIR" ]; then
    echo "Expected run directory not found: $RUN_DIR"
    exit 2
fi

# ==============================================================================
# Check for DaytonaErrors before upload
# ==============================================================================
RESULT_FILE="$RUN_DIR/result.json"
if [ -f "$RESULT_FILE" ]; then
    DAYTONA_COUNT=$($PYTHON_BIN -c "
import json, sys
try:
    with open('$RESULT_FILE') as f: data = json.load(f)
    total = 0
    if 'stats' in data and 'evals' in data['stats']:
        for k, v in data['stats']['evals'].items():
            es = v.get('exception_stats', {})
            de = es.get('DaytonaError')
            if isinstance(de, list): total += len(de)
    print(total)
except Exception as e:
    print(f'Error parsing result.json: {e}', file=sys.stderr); print('0')
" 2>&1 | tail -n 1)
    echo "DaytonaError count: ${DAYTONA_COUNT}"
    if [ "${DAYTONA_COUNT:-0}" -gt "$DAYTONA_ERROR_THRESHOLD" ]; then
        echo "Job has ${DAYTONA_COUNT} DaytonaErrors (> ${DAYTONA_ERROR_THRESHOLD}), skipping upload"
        exit 0
    fi
else
    echo "Warning: result.json not found, continuing with upload"
fi

# ==============================================================================
# Upload results to DB + HF (direct — no proxychains)
# ==============================================================================
export HF_HUB_OFFLINE=0
export UPLOAD_DIR="$RUN_DIR"
export RUN_DIR
export UPLOAD_USERNAME="${UPLOAD_USERNAME:-$USER}"
export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
export RUN_TAG

# Local-only smoke/canary guard (mirrors the leonardo sbatch): EVAL_SKIP_UPLOAD=1 preserves the
# results on disk and skips the HF/DB uploader entirely, so a subset/canary is NEVER registered as a
# full-benchmark leaderboard result. upload_eval_results() does HF upload AND Supabase registration in
# one call, so this single exit covers both. Also side-steps the bare os.environ["HF_TOKEN"] below when
# no token is configured (canary path).
case "${EVAL_SKIP_UPLOAD:-0}" in
    1|true|TRUE|yes|YES)
        echo "EVAL_SKIP_UPLOAD=${EVAL_SKIP_UPLOAD}: preserving local results at $RUN_DIR, skipping HF/DB upload"
        exit 0
        ;;
esac

UPLOAD_LOG="$RUN_DIR/upload.log"
echo "Uploading results from: $RUN_DIR" | tee -a "$UPLOAD_LOG"

$PYTHON_BIN - <<'PY' 2>&1 | tee -a "$UPLOAD_LOG"
import os, re, hashlib
from database.unified_db.utils import upload_eval_results

def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    def collapse(s):
        prev = None
        while s != prev:
            prev = s; s = s.replace("--", "-").replace("..", ".")
        return s
    org, name = repo_id.split("/", 1) if "/" in repo_id else (None, repo_id)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name); name = collapse(name).strip("-.")
    if not name: name = "repo"
    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest)); base = name[:keep].rstrip("-.") or "r"
        name = collapse(f"{base}{digest}").strip("-.")
    name = collapse(name).strip("-.")
    if name[0] in "-.": name = "r" + name[1:]
    if name[-1] in "-.": name = name[:-1] + "0"
    return f"{org}/{name}" if org else name

run_dir = os.environ["RUN_DIR"]; run_tag = os.environ["RUN_TAG"]
username = os.environ.get("UPLOAD_USERNAME", os.environ.get("USER", "tacc"))
error_mode = os.environ.get("UPLOAD_MODE", "skip_on_error")
hf_repo_id = sanitize_hf_repo_id(f"DCAgent2/{run_tag}")
hf_token = os.environ["HF_TOKEN"]
dataset_hf = os.environ.get("REPO_ID", "")
benchmark_name = os.environ.get("BENCHMARK_NAME", "") or (dataset_hf.split("/")[-1] if "/" in dataset_hf else dataset_hf)
benchmark_version_hash = hashlib.sha256(benchmark_name.encode()).hexdigest()
print(f"[uploader] upload_eval_results(path={run_dir!r}, hf_repo_id={hf_repo_id!r})")
upload_eval_results(
    os.environ.get("UPLOAD_DIR", run_dir),
    username=username, error_mode=error_mode, hf_token=hf_token,
    hf_repo_id=hf_repo_id, register_benchmark=True,
    benchmark_name=benchmark_name, benchmark_version_hash=benchmark_version_hash,
)
print("[uploader] done.")
PY
UPLOAD_EXIT=${PIPESTATUS[0]}
[ $UPLOAD_EXIT -ne 0 ] && { echo "Upload failed with exit code: $UPLOAD_EXIT"; exit $UPLOAD_EXIT; }

echo "=============================================="
echo "Eval and upload finished successfully."
echo "=============================================="
