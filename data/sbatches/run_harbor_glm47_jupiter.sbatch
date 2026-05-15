#!/bin/bash
#SBATCH -p booster
#SBATCH --time=12:00:00
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --account jureap59
#SBATCH --exclude=jpbo-031-[01-48]
#SBATCH --output=/e/scratch/jureap59/etash/OpenThoughts-Agent/data/sbatches/logs/harbor_glm47_jupiter_%j.out
#SBATCH --job-name=harbor_glm47_jupiter

set -eo pipefail

if [ -z "$1" ]; then
    echo "Usage: sbatch run_harbor_glm47_jupiter.sbatch <hf_dataset_repo_id>"
    exit 1
fi

INPUT_REPO_ID="$1"
DATASET_NAME="${INPUT_REPO_ID##*/}"
OUTPUT_REPO_ID="DCAgent/${DATASET_NAME}_glm_4.7_traces_jupiter"

# Jobs dir on /e/data1 for more inodes (agent retention trajectories only)
JOBS_DIR="${JOBS_DIR:-/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces}"

# Allow overriding shard configuration via environment
NUM_SHARDS="${NUM_SHARDS:-8}"
NODES_PER_SHARD="${NODES_PER_SHARD:-1}"

echo "============================================"
echo "vLLM + Harbor ${NUM_SHARDS}-Way Data Parallel (Jupiter)"
echo "============================================"
echo "Total Nodes: $SLURM_JOB_NUM_NODES"
echo "Shards: $NUM_SHARDS"
echo "Nodes per shard: $NODES_PER_SHARD"
echo "Job ID: $SLURM_JOB_ID"
echo "Input Dataset: $INPUT_REPO_ID"
echo "Output Traces: $OUTPUT_REPO_ID"
echo "Jobs Dir: $JOBS_DIR"
echo "============================================"

mkdir -p "$JOBS_DIR" 2>/dev/null || echo "WARNING: Could not create $JOBS_DIR (may already exist or need manual creation)"

# ============================================================================
# Step 1: Environment Setup
# ============================================================================
echo ""
echo "=== Step 1: Environment Setup ==="

REPO_DIR="/e/scratch/jureap59/feuer1/OpenThoughts-Agent"

# Source Jupiter environment
source "$REPO_DIR/hpc/dotenv/jupiter.env"

# Override secrets path to use data_gen_secrets.env
DC_AGENT_SECRET_ENV=~/data_gen_secrets.env
if [ -f "$DC_AGENT_SECRET_ENV" ]; then
    source "$DC_AGENT_SECRET_ENV"
else
    echo "WARNING: Secret env not found at $DC_AGENT_SECRET_ENV"
fi

# vLLM / Ray / Triton env vars
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_CGRAPH_get_timeout=900
export VLLM_CONFIG_ROOT="$DCFT_SCRATCH/vllm_config"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"
export FLASHINFER_WORKSPACE_BASE="/tmp/flashinfer_cache_${USER}"
export UV_CACHE_DIR="$DCFT_SCRATCH/uv_cache_dir"
export HYDRA_FULL_ERROR=1
export HF_CACHE_DIR="$HF_HUB_CACHE"
# Pre-downloaded model cache (compute nodes have no internet)
export HF_HUB_CACHE="/e/data1/datasets/playground/mmlaion/shared/guha1/hub"
export HF_HOME="/e/data1/datasets/playground/mmlaion/shared/guha1/hub"

# Set up otagent env directly (feuer1's miniforge3, binaries not +x for group)
# Use ld-linux loader to bypass execute permission requirement
OTAGENT_DIR="/e/scratch/jureap59/feuer1/miniforge3/envs/otagent"
export PATH="$OTAGENT_DIR/bin:$PATH"
export CONDA_PREFIX="$OTAGENT_DIR"
# Harbor is editable-installed to feuer1's home (inaccessible); use our clone instead
# Also add pip_extras for missing deps (e.g. boto3)
export PYTHONPATH="/e/scratch/jureap59/etash/harbor/src:/e/scratch/jureap59/etash/pip_extras${PYTHONPATH:+:$PYTHONPATH}"
LD_LOADER="/lib/ld-linux-aarch64.so.1"
PYTHON_REAL="$OTAGENT_DIR/bin/python3.12"

# Create executable wrappers that use ld-linux loader
# Must be on shared filesystem (not /tmp) so all nodes can access them
WRAPPER_DIR="/e/scratch/jureap59/etash/bin_wrappers_${SLURM_JOB_ID:-$$}"
mkdir -p "$WRAPPER_DIR"
cat > "$WRAPPER_DIR/python3" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL "\$@"
WEOF
cat > "$WRAPPER_DIR/python" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL "\$@"
WEOF
cat > "$WRAPPER_DIR/ray" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL -m ray.scripts.scripts "\$@"
WEOF
cat > "$WRAPPER_DIR/harbor" <<WEOF
#!/bin/bash
exec $LD_LOADER $PYTHON_REAL -c "import sys; from harbor.cli.main import app; sys.exit(app())" "\$@"
WEOF
chmod +x "$WRAPPER_DIR/python3" "$WRAPPER_DIR/python" "$WRAPPER_DIR/ray" "$WRAPPER_DIR/harbor"
export PATH="$WRAPPER_DIR:$PATH"

PYTHON_BIN="$WRAPPER_DIR/python3"
RAY_BIN="$WRAPPER_DIR/ray"

echo "Python: $($PYTHON_BIN --version)"
harbor --help >/dev/null
export RAY_DEDUP_LOGS=0

ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
ALL_NODES_ARRAY=($ALL_NODES)

echo "All nodes: ${ALL_NODES_ARRAY[@]}"

SBATCHES_DIR="/e/scratch/jureap59/etash/OpenThoughts-Agent/data/sbatches"
mkdir -p "$SBATCHES_DIR/logs"

export SRUN_EXPORT_ENV="ALL,LD_LIBRARY_PATH=$LD_LIBRARY_PATH,PATH=$PATH,HF_TOKEN=$HF_TOKEN"
RAY_ENV_VARS="LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH HF_TOKEN=$HF_TOKEN"

# vLLM model parameters
VLLM_MODEL_PATH="QuantTrio/GLM-4.7-AWQ"
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_MAX_NUM_SEQS=52
VLLM_MAX_MODEL_LEN=131072
VLLM_MAX_NUM_BATCHED_TOKENS=32768
VLLM_TENSOR_PARALLEL_SIZE=4
if [ "$NODES_PER_SHARD" -gt 1 ]; then
    VLLM_PIPELINE_PARALLEL_SIZE=$((NODES_PER_SHARD))
else
    VLLM_PIPELINE_PARALLEL_SIZE=1
fi

# ----------------------------------------------------------------------------
# SSH Tunnel + Proxychains Setup (Jupiter has no internet on compute nodes)
#
# Creates SOCKS5 proxy via SSH tunnel to login node, then uses proxychains
# binary wrapper to route external traffic through the tunnel.
# (LD_PRELOAD doesn't work reliably on ARM GH200)
# ----------------------------------------------------------------------------
echo ""
echo "--- Setting up SSH tunnel + proxychains ---"

LOGIN_NODE="jpbl-s01-02"
PROXYCHAINS_BIN="/e/scratch/jureap59/feuer1/proxychains-ng-aarch64/bin/proxychains4"
TUNNEL_PORT=7003

if [ ! -x "$PROXYCHAINS_BIN" ]; then
    echo "[proxy] WARNING: proxychains binary not found at $PROXYCHAINS_BIN"
    echo "[proxy] External connectivity (HF downloads, Daytona) will fail"
fi

if [ -z "${SSH_KEY:-}" ]; then
    echo "[proxy] SSH_KEY not set - skipping proxy setup"
    echo "[proxy] Set SSH_KEY in your environment to enable internet access"
else
    NODE_HOST=$(hostname -s)
    NODE_IP=$(nslookup "$NODE_HOST" | grep 'Address' | tail -n1 | awk '{print $2}')
    echo "[proxy] Setting up SSH tunnel to $LOGIN_NODE"
    echo "[proxy] SSH key: $SSH_KEY"
    echo "[proxy] Tunnel port: $TUNNEL_PORT"
    echo "[proxy] Node IP: $NODE_IP"

    if ssh -g -f -N -D ${TUNNEL_PORT} \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=1000 \
        -o ServerAliveInterval=10 \
        -o ServerAliveCountMax=30 \
        -o TCPKeepAlive=yes \
        -o ExitOnForwardFailure=yes \
        -o BatchMode=yes \
        -i "${SSH_KEY}" \
        "${USER}@${LOGIN_NODE}" 2>&1; then
        sleep 5
        echo "[proxy] SSH tunnel started successfully"
    else
        echo "[proxy] WARNING: SSH tunnel failed to start - continuing without proxy"
        echo "[proxy] Internet-dependent steps (dataset download, upload) will fail"
    fi

    # Generate proxychains config (even if tunnel failed - config is harmless)
    PROXYCHAINS_CONF_FILE=~/.proxychains/proxychains_${SLURM_JOB_ID}.conf
    mkdir -p ~/.proxychains

    cat > "$PROXYCHAINS_CONF_FILE" <<PCEOF
strict_chain
quiet_mode
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 127.0.0.1/255.255.255.255
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
[ProxyList]
socks5 ${NODE_IP} ${TUNNEL_PORT}
PCEOF

    export PROXYCHAINS_CONF_FILE
    echo "[proxy] Generated proxychains config at $PROXYCHAINS_CONF_FILE"

    # Daytona/aiohttp timeout settings
    export DAYTONA_MAX_RETRIES=5
    export DAYTONA_RETRY_DELAY=30
    export DAYTONA_BACKOFF_FACTOR=2
    export DAYTONA_TIMEOUT=1800
    export AIOHTTP_CLIENT_TIMEOUT=900
    export AIOHTTP_CONNECTOR_TIMEOUT=900
    export AIOHTTP_SOCK_CONNECT_TIMEOUT=300
    export AIOHTTP_TOTAL_TIMEOUT=1800

    # Disable SSL verification (JSC certificate issues)
    export PYTHONHTTPSVERIFY=0
    unset SSL_CERT_FILE
    unset CURL_CA_BUNDLE
    unset REQUESTS_CA_BUNDLE
    unset SSL_CERT_DIR

    # Test proxy connectivity
    echo "[proxy] Testing proxy connectivity..."
    if "$PROXYCHAINS_BIN" -f "$PROXYCHAINS_CONF_FILE" curl -s --connect-timeout 10 https://huggingface.co -o /dev/null; then
        echo "[proxy] Proxy connectivity test passed"
    else
        echo "[proxy] WARNING: Proxy connectivity test failed (may still work)"
    fi
fi

# ============================================================================
# Step 2: Downloading and Sharding Dataset
# ============================================================================
echo ""
echo "=== Step 2: Downloading and Sharding Dataset ==="

export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/scripts/harbor:$PYTHONPATH"
TEACHER_SCRIPTS_DIR="${REPO_DIR}/data/sbatches/teacher_scripts"

SHARD_OUTPUT_FILE=$(mktemp)

# Wrap with proxychains since dataset download needs internet
if [ -n "${PROXYCHAINS_CONF_FILE:-}" ] && [ -x "$PROXYCHAINS_BIN" ]; then
    "$PROXYCHAINS_BIN" -f "$PROXYCHAINS_CONF_FILE" \
        "$PYTHON_BIN" "$TEACHER_SCRIPTS_DIR/shard_dataset.py" "$INPUT_REPO_ID" "$NUM_SHARDS" "$SHARD_OUTPUT_FILE"
else
    "$PYTHON_BIN" "$TEACHER_SCRIPTS_DIR/shard_dataset.py" "$INPUT_REPO_ID" "$NUM_SHARDS" "$SHARD_OUTPUT_FILE"
fi

source "$SHARD_OUTPUT_FILE"
rm "$SHARD_OUTPUT_FILE"

if [ -z "$SHARD_DIR" ] || [ ! -d "$SHARD_DIR" ]; then
    echo "ERROR: Failed to create shards or path does not exist: $SHARD_DIR"
    exit 1
fi

echo "Shard directory: $SHARD_DIR"
echo "Dataset fingerprint: $DATASET_FINGERPRINT"

# ============================================================================
# Step 3: Starting Ray Clusters (one per shard)
# ============================================================================
echo ""
echo "=== Step 3: Starting $NUM_SHARDS Ray Clusters ==="

declare -a VLLM_PIDS
declare -a RAY_PORTS
declare -a API_PORTS
declare -a HEAD_IPS

BASE_RAY_PORT=6379
BASE_API_PORT=8000

start_ray_cluster_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local ray_port=$((BASE_RAY_PORT + shard_idx * 100))
    local api_port=$((BASE_API_PORT + shard_idx))

    local shard_nodes=()
    for ((i = 0; i < NODES_PER_SHARD; i++)); do
        shard_nodes+=(${ALL_NODES_ARRAY[$((start_node_idx + i))]})
    done

    local head_node=${shard_nodes[0]}
    local ray_temp_dir="/tmp/ray_${USER}_shard${shard_idx}"

    echo "  Shard $shard_idx: Nodes ${shard_nodes[@]}"
    echo "    Head: $head_node, Ray port: $ray_port, API port: $api_port"

    # Clean up any existing Ray processes
    for node in "${shard_nodes[@]}"; do
        srun --nodes=1 --ntasks=1 --overlap -w "$node" "$RAY_BIN" stop --force 2>/dev/null || true
        srun --nodes=1 --ntasks=1 --overlap -w "$node" rm -rf "/tmp/ray_${USER}_shard${shard_idx}" 2>/dev/null || true
    done
    sleep 2

    # Get head node IP via InfiniBand interface
    local head_iface="ib0"
    local head_ip
    if srun --nodes=1 --ntasks=1 --overlap -w "$head_node" ip -o -4 addr show "$head_iface" >/dev/null 2>&1; then
        head_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" ip -o -4 addr show "$head_iface" | awk '{print $4}' | cut -d/ -f1)
    else
        head_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" hostname --ip-address)
        head_ip=${head_ip%% *}
    fi

    HEAD_IPS[$shard_idx]=$head_ip
    RAY_PORTS[$shard_idx]=$ray_port
    API_PORTS[$shard_idx]=$api_port

    echo "    Head IP: $head_ip"

    # Start Ray head (4 GPUs per node on Jupiter GH200)
    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" bash -c \
        "env $RAY_ENV_VARS $RAY_BIN start --head --node-ip-address=${head_ip} --port=${ray_port} --num-gpus=4 --num-cpus=72 --temp-dir=${ray_temp_dir}" &
    sleep 5

    # Start Ray workers (if multi-node per shard)
    for ((i = 1; i < NODES_PER_SHARD; i++)); do
        local worker_node=${shard_nodes[$i]}
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$worker_node" bash -c \
            "env $RAY_ENV_VARS $RAY_BIN start --address ${head_ip}:${ray_port} --num-gpus=4 --num-cpus=72 --temp-dir=${ray_temp_dir}" &
        sleep 2
    done
}

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    start_ray_cluster_for_shard $shard_idx
done

echo "Waiting for Ray clusters to stabilize..."
sleep 30

# ============================================================================
# Step 4: Starting vLLM Servers
# ============================================================================
echo ""
echo "=== Step 4: Starting $NUM_SHARDS vLLM Servers ==="

start_vllm_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    local head_ip=${HEAD_IPS[$shard_idx]}
    local ray_port=${RAY_PORTS[$shard_idx]}
    local api_port=${API_PORTS[$shard_idx]}
    local ray_address="${head_ip}:${ray_port}"

    local vllm_log="${SBATCHES_DIR}/logs/vllm_${DATASET_NAME}_shard${shard_idx}_${SLURM_JOB_ID}.log"

    echo "  Starting vLLM for shard $shard_idx on $head_node (port $api_port)..."

    # Pre-seed torchinductor cache from shared storage to avoid recompilation
    local shared_cache="/e/scratch/jureap59/etash/torchinductor_cache/guha1"
    local local_cache="/tmp/torchinductor_${USER}"
    if [ -d "$shared_cache" ]; then
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
            bash -c "if [ ! -d '$local_cache' ] || [ \$(ls '$local_cache' 2>/dev/null | wc -l) -lt 100 ]; then cp -rn '$shared_cache' '$local_cache' 2>/dev/null; echo 'Seeded torchinductor cache on $head_node'; fi" &
    fi

    # Launch vLLM directly (bypasses ray.serve.llm version mismatch)
    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
        env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
            PATH="$PATH" \
            PYTHONPATH="$PYTHONPATH" \
            HF_HOME="/e/data1/datasets/playground/mmlaion/shared/guha1/hub" \
            HF_HUB_CACHE="/e/data1/datasets/playground/mmlaion/shared/guha1/hub" \
            HF_HUB_OFFLINE="1" \
            HF_TOKEN="$HF_TOKEN" \
            RAY_ADDRESS="$ray_address" \
            VLLM_USE_V1="$VLLM_USE_V1" \
            VLLM_ALL2ALL_BACKEND="pplx" \
            TORCHDYNAMO_DISABLE="1" \
            TMPDIR="/tmp" \
            TRITON_CACHE_DIR="/tmp/triton_cache_${USER}" \
            TORCH_COMPILE_CACHE_DIR="/tmp/torch_cache_${USER}" \
            TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}" \
        "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
            --model "$VLLM_MODEL_PATH" \
            --served-model-name "glm" \
            --host "0.0.0.0" \
            --port "$api_port" \
            --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
            --pipeline-parallel-size "$VLLM_PIPELINE_PARALLEL_SIZE" \
            --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            --max-model-len "$VLLM_MAX_MODEL_LEN" \
            --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
            --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
            --dtype auto \
            --kv-cache-dtype auto \
            --swap-space 4 \
            --block-size 16 \
            --enable-expert-parallel \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --trust-remote-code \
        >> "$vllm_log" 2>&1 &

    VLLM_PIDS[$shard_idx]=$!
    echo "    PID: ${VLLM_PIDS[$shard_idx]}, Log: $vllm_log"
}

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    start_vllm_for_shard $shard_idx
done

echo ""
echo "Waiting for all vLLM servers to become healthy..."

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    start_node_idx=$((shard_idx * NODES_PER_SHARD))
    head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    api_port=${API_PORTS[$shard_idx]}
    health_url="http://127.0.0.1:${api_port}/v1/models"

    echo "  Checking shard $shard_idx (port $api_port)..."
    for i in {1..180}; do
        if srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" curl -s "$health_url" > /dev/null 2>&1; then
            echo "    Shard $shard_idx is healthy!"
            break
        fi
        if [ "$i" -eq 180 ]; then
            echo "ERROR: Shard $shard_idx failed health check after 30 minutes"
            exit 1
        fi
        sleep 10
    done
done

echo "All vLLM servers are ready!"

# ============================================================================
# Step 5: Running Harbor Jobs in Parallel
# ============================================================================
echo ""
echo "=== Step 5: Running $NUM_SHARDS Harbor Jobs in Parallel ==="

HARBOR_CONFIG="${REPO_DIR}/eval/tacc/dcagent_eval_config.yaml"
CONFIG_NAME="${DATASET_NAME}_jupiter_${NUM_SHARDS}way_dp_${DATASET_FINGERPRINT}"

declare -a HARBOR_PIDS
declare -a HARBOR_LOGS

run_harbor_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    local api_port=${API_PORTS[$shard_idx]}
    local shard_dataset="$SHARD_DIR/shard_${shard_idx}"
    local run_tag="${CONFIG_NAME}_shard${shard_idx}"
    local harbor_log="${SBATCHES_DIR}/logs/harbor_${DATASET_NAME}_shard${shard_idx}_${SLURM_JOB_ID}.log"

    HARBOR_LOGS[$shard_idx]="$harbor_log"

    echo "  Starting Harbor for shard $shard_idx..."
    echo "    Dataset: $shard_dataset"
    echo "    API port: $api_port"
    echo "    Run tag: $run_tag"

    # Check if job directory already exists (resume directly instead of auto-resume
    # config matching, which breaks on config drift)
    local existing_job_dir="$JOBS_DIR/$run_tag"
    local harbor_cmd
    if [ -d "$existing_job_dir" ] && [ -f "$existing_job_dir/config.json" ]; then
        echo "    Resuming existing job at $existing_job_dir"
        harbor_cmd="harbor jobs resume \
            --job-path \"$existing_job_dir\" \
            --filter-error-type DaytonaRateLimitError \
            --filter-error-type EnvironmentStartTimeoutError \
            --filter-error-type DaytonaError"
    else
        # Fresh start
        # --jobs-dir on /e/data1 for more inodes; agent retention trajectories only
        harbor_cmd="harbor jobs start \
            -p \"$shard_dataset\" \
            --jobs-dir \"$JOBS_DIR\" \
            --n-concurrent 64 \
            --agent terminus-2 \
            --model \"hosted_vllm/glm\" \
            --env \"daytona\" \
            --agent-kwarg \"api_base=http://localhost:${api_port}/v1\" \
            --agent-kwarg \"key=fake_key\" \
            --n-attempts 1 \
            --max-retries 0 \
            --job-name \"$run_tag\" \
            --timeout-multiplier 2.0 \
            --config \"$HARBOR_CONFIG\" \
            --disable-verification \
            --ek auto_snapshot=${AUTO_SNAPSHOT:-true} \
            --agent-kwarg 'trajectory_config={\"raw_content\":true,\"linear_history\":true}' \
            --agent-kwarg 'extra_body={\"chat_template_kwargs\":{\"enable_thinking\":true}}'"
    fi

    # Wrap with proxychains since Daytona API needs internet
    if [ -n "${PROXYCHAINS_CONF_FILE:-}" ] && [ -x "$PROXYCHAINS_BIN" ]; then
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
            "$PROXYCHAINS_BIN" -f "$PROXYCHAINS_CONF_FILE" \
            bash -c "$harbor_cmd" \
            >> "$harbor_log" 2>&1 &
    else
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
            bash -c "$harbor_cmd" \
            >> "$harbor_log" 2>&1 &
    fi

    HARBOR_PIDS[$shard_idx]=$!
    echo "    PID: ${HARBOR_PIDS[$shard_idx]}, Log: $harbor_log"
}

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    run_harbor_for_shard $shard_idx
done

echo ""
echo "All Harbor jobs started. Waiting for completion..."

FAILED=0
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    wait ${HARBOR_PIDS[$shard_idx]} || FAILED=$((FAILED + 1))
    echo "  Shard $shard_idx completed (exit: $?)"
done

# ============================================================================
# Step 6: Exporting and Uploading Expert Traces
# ============================================================================
echo ""
echo "=== Step 6: Exporting and Uploading Expert Traces ==="

JOB_DIRS=""
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    harbor_log="${HARBOR_LOGS[$shard_idx]}"
    # Match job dir from "Results written to <path>/result.json" — extract parent directory
    job_dir=$(grep "Results written to" "$harbor_log" 2>/dev/null | sed 's/.*Results written to \([^ ]*\).*/\1/' | head -1 || true)
    if [ -n "$job_dir" ]; then
        job_dir=$(dirname "$job_dir")
        JOB_DIRS="$JOB_DIRS $job_dir"
    fi
done

echo "Job directories: $JOB_DIRS"
echo "Uploading traces to: $OUTPUT_REPO_ID"

# Wrap with proxychains since HF upload needs internet
if [ -n "${PROXYCHAINS_CONF_FILE:-}" ] && [ -x "$PROXYCHAINS_BIN" ]; then
    TRANSFORMERS_NO_TORCH=1 "$PROXYCHAINS_BIN" -f "$PROXYCHAINS_CONF_FILE" \
        "$PYTHON_BIN" "$TEACHER_SCRIPTS_DIR/upload_traces.py" "$OUTPUT_REPO_ID" $JOB_DIRS
else
    TRANSFORMERS_NO_TORCH=1 "$PYTHON_BIN" "$TEACHER_SCRIPTS_DIR/upload_traces.py" "$OUTPUT_REPO_ID" $JOB_DIRS
fi

EXPORT_EXIT=$?
if [ $EXPORT_EXIT -eq 0 ]; then
    echo "Expert traces uploaded successfully to: $OUTPUT_REPO_ID"
else
    echo "WARNING: Failed to upload expert traces (exit code: $EXPORT_EXIT)"
fi

# ============================================================================
# Step 7: Cleanup
# ============================================================================
echo ""
echo "=== Step 7: Cleanup ==="

# Kill vLLM processes
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    if [ -n "${VLLM_PIDS[$shard_idx]}" ]; then
        kill "${VLLM_PIDS[$shard_idx]}" 2>/dev/null || true
    fi
done

# Stop Ray on all nodes
for node in "${ALL_NODES_ARRAY[@]}"; do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" "$RAY_BIN" stop --force 2>/dev/null &
done
wait

# Kill SSH tunnel
pkill -f "ssh.*-D.*${TUNNEL_PORT}" 2>/dev/null || true

# Clean up wrapper scripts
rm -rf "$WRAPPER_DIR" 2>/dev/null || true

echo ""
echo "============================================"
if [ $FAILED -eq 0 ]; then
    echo "Experiment COMPLETED"
    echo "  Input: $INPUT_REPO_ID"
    echo "  Output: $OUTPUT_REPO_ID"
    echo "  All $NUM_SHARDS shards completed successfully!"
else
    echo "Experiment PARTIALLY FAILED"
    echo "  $FAILED out of $NUM_SHARDS shards failed"
fi
echo "============================================"

exit $FAILED
