#!/bin/bash
#SBATCH -p gh
#SBATCH --time=03:00:00
#SBATCH --nodes 16
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --account CCR24067
#SBATCH --output=${DCAGENT_DIR}/data/vllm_experiments/logs/harbor_staqc_10_16node_4way_dp_%j.out
#SBATCH --job-name=harbor_staqc_10_4way_dp

set -eo pipefail

EXPERIMENTS_DIR="${DCAGENT_DIR}/data/vllm_experiments"
CONFIG_NAME="staqc_10_16node_4way_dp"
NUM_SHARDS=4
NODES_PER_SHARD=4

echo "============================================"
echo "vLLM + Harbor 4-Way Data Parallel Experiment"
echo "============================================"
echo "Total Nodes: $SLURM_JOB_NUM_NODES"
echo "Shards: $NUM_SHARDS"
echo "Nodes per shard: $NODES_PER_SHARD"
echo "Job ID: $SLURM_JOB_ID"
echo "============================================"

mkdir -p "$EXPERIMENTS_DIR/results/$CONFIG_NAME"
mkdir -p "$EXPERIMENTS_DIR/logs"

# Environment Setup
echo ""
echo "=== Step 1: Environment Setup ==="

module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_CGRAPH_get_timeout=900
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

source /scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/secret.env

ln -sf /home1/apps/gcc/15.1.0/lib64/libstdc++.so.6 /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/10000/eguha3/vllm_sandboxes_backup/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5

harbor --help >/dev/null

# Paths
RAY_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/ray"
PYTHON_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3"
export RAY_DEDUP_LOGS=0
export TRITON_CC=$(which gcc)

# Get all nodes
ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
ALL_NODES_ARRAY=($ALL_NODES)

echo "All nodes: ${ALL_NODES_ARRAY[@]}"

# Environment to forward to workers
export SRUN_EXPORT_ENV="ALL,TRITON_CC=$TRITON_CC,LD_LIBRARY_PATH=$LD_LIBRARY_PATH,PATH=$PATH,HF_TOKEN=$HF_TOKEN"
RAY_ENV_VARS="TRITON_CC=$TRITON_CC LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH HF_TOKEN=$HF_TOKEN"

# vLLM configuration (TP=4, PP=1 - best performing config)
VLLM_MODEL_PATH="QuantTrio/GLM-4.6-AWQ"
VLLM_GPU_MEMORY_UTILIZATION=0.92
VLLM_MAX_NUM_SEQS=64
VLLM_MAX_MODEL_LEN=113000
VLLM_MAX_NUM_BATCHED_TOKENS=32768
VLLM_TENSOR_PARALLEL_SIZE=4
VLLM_PIPELINE_PARALLEL_SIZE=1

# === Step 2: Shard the dataset ===
echo ""
echo "=== Step 2: Sharding Dataset ==="

DATASET_PATH="${DCAGENT_DIR}/data/vllm_experiments/staqc_10_tasks"
SHARD_DIR="$EXPERIMENTS_DIR/dataset_shards_${SLURM_JOB_ID}"
mkdir -p "$SHARD_DIR"

# Create a Python script to shard the dataset
cat > "$SHARD_DIR/shard_dataset.py" << 'SHARD_SCRIPT'
import os
import sys
import json
import shutil
from pathlib import Path

dataset_path = sys.argv[1]
shard_dir = sys.argv[2]
num_shards = int(sys.argv[3])

# Find all task directories
task_dirs = sorted([d for d in Path(dataset_path).iterdir() if d.is_dir()])
print(f"Found {len(task_dirs)} tasks to shard into {num_shards} parts")

# Distribute tasks across shards
for shard_idx in range(num_shards):
    shard_path = Path(shard_dir) / f"shard_{shard_idx}"
    shard_path.mkdir(exist_ok=True)

    # Get tasks for this shard
    shard_tasks = task_dirs[shard_idx::num_shards]
    print(f"Shard {shard_idx}: {len(shard_tasks)} tasks")

    for task_dir in shard_tasks:
        dest = shard_path / task_dir.name
        if not dest.exists():
            # Copy task directory (symlinks cause dirhash SymlinkRecursionError)
            shutil.copytree(task_dir, dest)

print("Dataset sharding complete!")
SHARD_SCRIPT

python3 "$SHARD_DIR/shard_dataset.py" "$DATASET_PATH" "$SHARD_DIR" "$NUM_SHARDS"

# === Step 3: Start 4 Ray clusters and vLLM servers ===
echo ""
echo "=== Step 3: Starting 4 Ray Clusters ==="

# Arrays to store PIDs and ports
declare -a VLLM_PIDS
declare -a RAY_PORTS
declare -a API_PORTS
declare -a HEAD_IPS

# Base ports
BASE_RAY_PORT=6379
BASE_API_PORT=8000

# Function to start a Ray cluster for a shard
start_ray_cluster_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local ray_port=$((BASE_RAY_PORT + shard_idx * 100))
    local api_port=$((BASE_API_PORT + shard_idx))

    # Get nodes for this shard
    local shard_nodes=()
    for ((i = 0; i < NODES_PER_SHARD; i++)); do
        shard_nodes+=(${ALL_NODES_ARRAY[$((start_node_idx + i))]})
    done

    local head_node=${shard_nodes[0]}
    local ray_temp_dir="/tmp/ray_${USER}_shard${shard_idx}"

    echo "  Shard $shard_idx: Nodes ${shard_nodes[@]}"
    echo "    Head: $head_node, Ray port: $ray_port, API port: $api_port"

    # Cleanup existing Ray on these nodes
    for node in "${shard_nodes[@]}"; do
        srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null || true
        srun --nodes=1 --ntasks=1 --overlap -w "$node" rm -rf "/tmp/ray_${USER}_shard${shard_idx}" 2>/dev/null || true
    done
    sleep 2

    # Get head node IP via InfiniBand
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

    # Start Ray head
    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" bash -c \
        "env $RAY_ENV_VARS $RAY_BIN start --head --node-ip-address=${head_ip} --port=${ray_port} --num-gpus=1 --num-cpus=64 --temp-dir=${ray_temp_dir}" &
    sleep 5

    # Start workers
    for ((i = 1; i < NODES_PER_SHARD; i++)); do
        local worker_node=${shard_nodes[$i]}
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$worker_node" bash -c \
            "env $RAY_ENV_VARS $RAY_BIN start --address ${head_ip}:${ray_port} --num-gpus=1 --num-cpus=64 --temp-dir=${ray_temp_dir}" &
        sleep 2
    done
}

# Start all 4 Ray clusters
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    start_ray_cluster_for_shard $shard_idx
done

echo "Waiting for Ray clusters to stabilize..."
sleep 30

# === Step 4: Start vLLM servers ===
echo ""
echo "=== Step 4: Starting 4 vLLM Servers ==="

start_vllm_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    local head_ip=${HEAD_IPS[$shard_idx]}
    local ray_port=${RAY_PORTS[$shard_idx]}
    local api_port=${API_PORTS[$shard_idx]}
    local ray_address="${head_ip}:${ray_port}"

    local vllm_log="${EXPERIMENTS_DIR}/logs/vllm_serve_${SLURM_JOB_ID}_shard${shard_idx}.log"

    echo "  Starting vLLM for shard $shard_idx on $head_node (port $api_port)..."

    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
        env TRITON_CC="$TRITON_CC" \
            LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
            PATH="$PATH" \
            HF_HOME="/tmp/hf_home" \
            HF_TOKEN="$HF_TOKEN" \
            RAY_ADDRESS="$ray_address" \
            VLLM_MODEL_PATH="$VLLM_MODEL_PATH" \
            VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
            VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
            VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
            VLLM_MAX_NUM_BATCHED_TOKENS="$VLLM_MAX_NUM_BATCHED_TOKENS" \
            VLLM_TENSOR_PARALLEL_SIZE="$VLLM_TENSOR_PARALLEL_SIZE" \
            VLLM_PIPELINE_PARALLEL_SIZE="$VLLM_PIPELINE_PARALLEL_SIZE" \
            VLLM_ENABLE_EXPERT_PARALLEL="true" \
            VLLM_KV_CACHE_DTYPE="auto" \
            VLLM_ENABLE_PREFIX_CACHING="true" \
            VLLM_SWAP_SPACE="4" \
            VLLM_BLOCK_SIZE="16" \
            VLLM_ENABLE_CHUNKED_PREFILL="true" \
            VLLM_ALL2ALL_BACKEND="pplx" \
            VLLM_ENABLE_EPLB="false" \
            SERVE_HTTP_PORT="$api_port" \
        $PYTHON_BIN ${DCAGENT_DIR}/scripts/vllm/dp_debug.py \
        >> "$vllm_log" 2>&1 &

    VLLM_PIDS[$shard_idx]=$!
    echo "    PID: ${VLLM_PIDS[$shard_idx]}, Log: $vllm_log"
}

# Start all vLLM servers
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    start_vllm_for_shard $shard_idx
done

# Wait for all servers to become healthy
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
            echo "ERROR: Shard $shard_idx failed health check"
            exit 1
        fi
        sleep 10
    done
done

echo "All vLLM servers are ready!"

# === Step 5: Run Harbor jobs in parallel ===
echo ""
echo "=== Step 5: Running 4 Harbor Jobs in Parallel ==="

HARBOR_CONFIG="${DCAGENT_DIR}/eval/tacc/dcagent_eval_config.yaml"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

declare -a HARBOR_PIDS

run_harbor_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    local api_port=${API_PORTS[$shard_idx]}
    local shard_dataset="$SHARD_DIR/shard_${shard_idx}"
    local run_tag="${CONFIG_NAME}_shard${shard_idx}_${TIMESTAMP}"
    local harbor_log="${EXPERIMENTS_DIR}/logs/harbor_${CONFIG_NAME}_shard${shard_idx}_${SLURM_JOB_ID}.log"

    echo "  Starting Harbor for shard $shard_idx..."
    echo "    Dataset: $shard_dataset"
    echo "    API port: $api_port"
    echo "    Run tag: $run_tag"

    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
        harbor jobs start \
            -p "$shard_dataset" \
            --n-concurrent 64 \
            --agent terminus-2 \
            --model "hosted_vllm/glm" \
            --env "daytona" \
            --agent-kwarg "api_base=http://localhost:${api_port}/v1" \
            --agent-kwarg "key=fake_key" \
            --n-attempts 1 \
            --max-retries 0 \
            --job-name "$run_tag" \
            --config "$HARBOR_CONFIG" \
            --disable-verification \
        >> "$harbor_log" 2>&1 &

    HARBOR_PIDS[$shard_idx]=$!
    echo "    PID: ${HARBOR_PIDS[$shard_idx]}, Log: $harbor_log"
}

# Start all Harbor jobs
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    run_harbor_for_shard $shard_idx
done

echo ""
echo "All Harbor jobs started. Waiting for completion..."

# Wait for all Harbor jobs to complete
FAILED=0
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    wait ${HARBOR_PIDS[$shard_idx]} || FAILED=$((FAILED + 1))
    echo "  Shard $shard_idx completed (exit: $?)"
done

# === Step 6: Export and Upload Expert Traces ===
echo ""
echo "=== Step 6: Exporting and Uploading Expert Traces ==="

HF_REPO_ID="DCAgent/staqc_10_expert_traces_4way_dp"

# Source HF token from old-dc-agent
source /scratch/10000/eguha3/old-dc-agent/secret.env

# Clear LD_LIBRARY_PATH to avoid torch CUDA init issues on GH200
# TRANSFORMERS_NO_TORCH=1 bypasses torch import issues
LD_LIBRARY_PATH="" TRANSFORMERS_NO_TORCH=1 /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5/bin/python3 -c "
import sys
import os
from pathlib import Path

from harbor.utils.traces_utils import (
    iter_trial_dirs, load_run_metadata, collect_conversations_from_trial
)
from datasets import Dataset

def sanitize_surrogates(obj):
    '''Remove invalid surrogate characters from strings recursively.'''
    if isinstance(obj, str):
        return obj.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
    elif isinstance(obj, dict):
        return {k: sanitize_surrogates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_surrogates(v) for v in obj]
    return obj

# Find all shard job directories
jobs_dir = Path('jobs')
shard_dirs = sorted(jobs_dir.glob('${CONFIG_NAME}_shard*_${TIMESTAMP}'))
print(f'Found {len(shard_dirs)} shard job directories')

all_rows = []
for shard_dir in shard_dirs:
    print(f'Processing {shard_dir.name}...')
    trial_dirs = list(iter_trial_dirs(shard_dir, recursive=True))
    print(f'  Found {len(trial_dirs)} trial directories')

    for trial_dir in trial_dirs:
        try:
            run_meta = load_run_metadata(trial_dir)
            convs = collect_conversations_from_trial(
                trial_dir,
                run_meta=run_meta,
                episodes='last',
                verbose=False
            )
            for conv in convs:
                all_rows.append(sanitize_surrogates(conv))
        except Exception as e:
            print(f'  Skipping {trial_dir.name}: {e}')
            continue

print(f'[traces] Collected {len(all_rows)} total rows from all shards')

if all_rows:
    ds = Dataset.from_list(all_rows)
    print(f'Pushing {len(ds)} traces to ${HF_REPO_ID}...')
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    ds.push_to_hub('${HF_REPO_ID}', token=token)
    print(f'Exported {len(ds)} traces to ${HF_REPO_ID}')
else:
    print('No traces to export')
"

EXPORT_EXIT=$?
if [ $EXPORT_EXIT -eq 0 ]; then
    echo "Expert traces uploaded successfully!"
else
    echo "WARNING: Failed to upload expert traces (exit code: $EXPORT_EXIT)"
fi

# === Step 7: Cleanup ===
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
    srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null &
done
wait

# Clean up shard symlinks
rm -rf "$SHARD_DIR"

conda deactivate || true

echo ""
echo "============================================"
if [ $FAILED -eq 0 ]; then
    echo "Experiment COMPLETED: $CONFIG_NAME"
    echo "All $NUM_SHARDS shards completed successfully!"
else
    echo "Experiment PARTIALLY FAILED: $CONFIG_NAME"
    echo "$FAILED out of $NUM_SHARDS shards failed"
fi
echo "============================================"

exit $FAILED
