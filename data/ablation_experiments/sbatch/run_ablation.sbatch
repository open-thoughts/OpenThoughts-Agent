#!/bin/bash
#SBATCH -p gh
#SBATCH --time=24:00:00
#SBATCH --nodes 32
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --account CCR24067
#SBATCH --output=${DCAGENT_DIR}/data/ablation_experiments/logs/ablation_%j.out
#SBATCH --job-name=ablation

set -eo pipefail

# Check for required argument
if [ -z "$1" ]; then
    echo "Usage: sbatch run_ablation.sbatch <config_yaml_path>"
    echo "Example: sbatch run_ablation.sbatch configs/baseline.yaml"
    exit 1
fi

CONFIG_PATH="$1"
ABLATION_DIR="${DCAGENT_DIR}/data/ablation_experiments"

if [ ! -f "$CONFIG_PATH" ]; then
    # Try relative to ablation dir
    CONFIG_PATH="$ABLATION_DIR/$1"
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $1"
    exit 1
fi

# Parse YAML config using Python
parse_config() {
    python3 -c "
import yaml
import sys

with open('$CONFIG_PATH') as f:
    config = yaml.safe_load(f)

key = sys.argv[1]
parts = key.split('.')
val = config
for p in parts:
    if val is None:
        break
    val = val.get(p)

if val is not None:
    if isinstance(val, bool):
        print('true' if val else 'false')
    elif isinstance(val, dict):
        import json
        print(json.dumps(val))
    else:
        print(val)
" "$1"
}

EXPERIMENT_NAME=$(parse_config "experiment_name")
INPUT_REPO_ID=$(parse_config "input_repo_id")
OUTPUT_REPO_SUFFIX=$(parse_config "output_repo_suffix")
TEMPERATURE=$(parse_config "agent.kwargs.temperature")
MAX_EPISODES=$(parse_config "agent.kwargs.max_episodes")
ENABLE_SUMMARIZE=$(parse_config "agent.kwargs.enable_summarize")
PROACTIVE_THRESHOLD=$(parse_config "agent.kwargs.proactive_summarization_threshold")
INTERLEAVED_THINKING=$(parse_config "agent.kwargs.interleaved_thinking")
TRAJECTORY_CONFIG=$(parse_config "agent.kwargs.trajectory_config")
EXTRA_BODY=$(parse_config "agent.kwargs.extra_body")
EXTRA_COMPLETION_PARAMS=$(parse_config "agent.kwargs.extra_completion_params")
PARSER_NAME=$(parse_config "agent.kwargs.parser_name")
TMUX_PANE_WIDTH=$(parse_config "agent.kwargs.tmux_pane_width")
TMUX_PANE_HEIGHT=$(parse_config "agent.kwargs.tmux_pane_height")
N_CONCURRENT=$(parse_config "job.n_concurrent")
TIMEOUT_MULTIPLIER=$(parse_config "job.timeout_multiplier")

DATASET_NAME="${INPUT_REPO_ID##*/}"
OUTPUT_REPO_ID="DCAgent/${OUTPUT_REPO_SUFFIX}_traces"

EXPERIMENTS_DIR="${DCAGENT_DIR}/data/ablation_experiments"
NUM_SHARDS=8
NODES_PER_SHARD=4

echo "============================================"
echo "Ablation Experiment: $EXPERIMENT_NAME"
echo "============================================"
echo "Config: $CONFIG_PATH"
echo "Total Nodes: $SLURM_JOB_NUM_NODES"
echo "Shards: $NUM_SHARDS"
echo "Job ID: $SLURM_JOB_ID"
echo "Input Dataset: $INPUT_REPO_ID"
echo "Output Traces: $OUTPUT_REPO_ID"
echo ""
echo "Parameters:"
echo "  temperature: $TEMPERATURE"
echo "  max_episodes: $MAX_EPISODES"
echo "  enable_summarize: $ENABLE_SUMMARIZE"
echo "  proactive_threshold: $PROACTIVE_THRESHOLD"
echo "  interleaved_thinking: $INTERLEAVED_THINKING"
echo "  parser_name: $PARSER_NAME"
echo "  tmux_pane: ${TMUX_PANE_WIDTH}x${TMUX_PANE_HEIGHT}"
echo "  trajectory_config: $TRAJECTORY_CONFIG"
echo "  extra_body: $EXTRA_BODY"
echo "  extra_completion_params: $EXTRA_COMPLETION_PARAMS"
echo "============================================"

mkdir -p "$EXPERIMENTS_DIR/logs"
mkdir -p "$EXPERIMENTS_DIR/results/$EXPERIMENT_NAME"

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

source /scratch/10000/eguha3/old-dc-agent/secret.env

ln -sf /home1/apps/gcc/15.1.0/lib64/libstdc++.so.6 /scratch/10000/eguha3/vllm_sandboxes_backup/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/10000/eguha3/vllm_sandboxes_backup/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/vllm_sandboxes_backup

harbor --help >/dev/null

RAY_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/ray"
PYTHON_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3"
export RAY_DEDUP_LOGS=0
export TRITON_CC=$(which gcc)

ALL_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
ALL_NODES_ARRAY=($ALL_NODES)

echo "All nodes: ${ALL_NODES_ARRAY[@]}"

export SRUN_EXPORT_ENV="ALL,TRITON_CC=$TRITON_CC,LD_LIBRARY_PATH=$LD_LIBRARY_PATH,PATH=$PATH,HF_TOKEN=$HF_TOKEN"
RAY_ENV_VARS="TRITON_CC=$TRITON_CC LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH HF_TOKEN=$HF_TOKEN"

VLLM_MODEL_PATH="QuantTrio/GLM-4.7-AWQ"
VLLM_GPU_MEMORY_UTILIZATION=0.92
VLLM_MAX_NUM_SEQS=64
VLLM_MAX_MODEL_LEN=113000
VLLM_MAX_NUM_BATCHED_TOKENS=32768
VLLM_TENSOR_PARALLEL_SIZE=4
VLLM_PIPELINE_PARALLEL_SIZE=1

echo ""
echo "=== Step 2: Downloading Dataset ==="

export PYTHONPATH="${DCAGENT_DIR}:${DCAGENT_DIR}/scripts/harbor:$PYTHONPATH"

DATASET_PATH=$(LD_LIBRARY_PATH="" /scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3 -c "
from tasks_parquet_converter import from_hf_dataset
path = from_hf_dataset('$INPUT_REPO_ID')
print(path)
")

if [ -z "$DATASET_PATH" ] || [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Failed to download dataset or path does not exist: $DATASET_PATH"
    exit 1
fi
echo "Dataset downloaded to: $DATASET_PATH"

echo ""
echo "=== Step 3: Sharding Dataset ==="

SHARD_DIR="$EXPERIMENTS_DIR/results/$EXPERIMENT_NAME/shards_${SLURM_JOB_ID}"
mkdir -p "$SHARD_DIR"

cat > "$SHARD_DIR/shard_dataset.py" << 'SHARD_SCRIPT'
import os
import sys
import shutil
from pathlib import Path

dataset_path = sys.argv[1]
shard_dir = sys.argv[2]
num_shards = int(sys.argv[3])

task_dirs = sorted([d for d in Path(dataset_path).iterdir() if d.is_dir()])
print(f"Found {len(task_dirs)} tasks to shard into {num_shards} parts")

for shard_idx in range(num_shards):
    shard_path = Path(shard_dir) / f"shard_{shard_idx}"
    shard_path.mkdir(exist_ok=True)
    shard_tasks = task_dirs[shard_idx::num_shards]
    print(f"Shard {shard_idx}: {len(shard_tasks)} tasks")
    for task_dir in shard_tasks:
        dest = shard_path / task_dir.name
        if not dest.exists():
            shutil.copytree(task_dir, dest)

print("Dataset sharding complete!")
SHARD_SCRIPT

python3 "$SHARD_DIR/shard_dataset.py" "$DATASET_PATH" "$SHARD_DIR" "$NUM_SHARDS"

echo ""
echo "=== Step 4: Starting $NUM_SHARDS Ray Clusters ==="

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

    for node in "${shard_nodes[@]}"; do
        srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null || true
        srun --nodes=1 --ntasks=1 --overlap -w "$node" rm -rf "/tmp/ray_${USER}_shard${shard_idx}" 2>/dev/null || true
    done
    sleep 2

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

    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" bash -c \
        "env $RAY_ENV_VARS $RAY_BIN start --head --node-ip-address=${head_ip} --port=${ray_port} --num-gpus=1 --num-cpus=64 --temp-dir=${ray_temp_dir}" &
    sleep 5

    for ((i = 1; i < NODES_PER_SHARD; i++)); do
        local worker_node=${shard_nodes[$i]}
        srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$worker_node" bash -c \
            "env $RAY_ENV_VARS $RAY_BIN start --address ${head_ip}:${ray_port} --num-gpus=1 --num-cpus=64 --temp-dir=${ray_temp_dir}" &
        sleep 2
    done
}

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    start_ray_cluster_for_shard $shard_idx
done

echo "Waiting for Ray clusters to stabilize..."
sleep 30

echo ""
echo "=== Step 5: Starting $NUM_SHARDS vLLM Servers ==="

start_vllm_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    local head_ip=${HEAD_IPS[$shard_idx]}
    local ray_port=${RAY_PORTS[$shard_idx]}
    local api_port=${API_PORTS[$shard_idx]}
    local ray_address="${head_ip}:${ray_port}"

    local vllm_log="${EXPERIMENTS_DIR}/logs/vllm_${EXPERIMENT_NAME}_shard${shard_idx}_${SLURM_JOB_ID}.log"

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
            echo "ERROR: Shard $shard_idx failed health check"
            exit 1
        fi
        sleep 10
    done
done

echo "All vLLM servers are ready!"

echo ""
echo "=== Step 6: Running $NUM_SHARDS Harbor Jobs in Parallel ==="

HARBOR_CONFIG="${DCAGENT_DIR}/eval/tacc/dcagent_eval_config.yaml"
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
CONFIG_NAME="${EXPERIMENT_NAME}_${TIMESTAMP}"

declare -a HARBOR_PIDS
declare -a HARBOR_LOGS

run_harbor_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    local api_port=${API_PORTS[$shard_idx]}
    local shard_dataset="$SHARD_DIR/shard_${shard_idx}"
    local run_tag="${CONFIG_NAME}_shard${shard_idx}"
    local harbor_log="${EXPERIMENTS_DIR}/logs/harbor_${EXPERIMENT_NAME}_shard${shard_idx}_${SLURM_JOB_ID}.log"

    HARBOR_LOGS[$shard_idx]="$harbor_log"

    echo "  Starting Harbor for shard $shard_idx..."
    echo "    Dataset: $shard_dataset"
    echo "    API port: $api_port"

    # Build harbor command in a temp script to handle JSON quoting properly
    local harbor_script="$SHARD_DIR/harbor_shard${shard_idx}.sh"

    # Build extra_completion_params line if needed
    local extra_comp_line=""
    if [ -n "$EXTRA_COMPLETION_PARAMS" ] && [ "$EXTRA_COMPLETION_PARAMS" != "None" ]; then
        extra_comp_line="    --agent-kwarg 'extra_completion_params=${EXTRA_COMPLETION_PARAMS}'"
    fi

    cat > "$harbor_script" <<EOF
#!/bin/bash
harbor jobs start \\
    -p "${shard_dataset}" \\
    --n-concurrent "${N_CONCURRENT}" \\
    --agent terminus-2 \\
    --model "hosted_vllm/glm" \\
    --env "daytona" \\
    --n-attempts 1 \\
    --max-retries 0 \\
    --job-name "${run_tag}" \\
    --timeout-multiplier "${TIMEOUT_MULTIPLIER}" \\
    --config "${HARBOR_CONFIG}" \\
    --disable-verification \\
    --agent-kwarg "api_base=http://localhost:${api_port}/v1" \\
    --agent-kwarg "key=fake_key" \\
    --agent-kwarg "temperature=${TEMPERATURE}" \\
    --agent-kwarg "max_episodes=${MAX_EPISODES}" \\
    --agent-kwarg "enable_summarize=${ENABLE_SUMMARIZE}" \\
    --agent-kwarg "proactive_summarization_threshold=${PROACTIVE_THRESHOLD}" \\
    --agent-kwarg "interleaved_thinking=${INTERLEAVED_THINKING}" \\
    --agent-kwarg "parser_name=${PARSER_NAME}" \\
    --agent-kwarg "tmux_pane_width=${TMUX_PANE_WIDTH}" \\
    --agent-kwarg "tmux_pane_height=${TMUX_PANE_HEIGHT}" \\
    --agent-kwarg 'trajectory_config=${TRAJECTORY_CONFIG}' \\
    --agent-kwarg 'extra_body=${EXTRA_BODY}' ${extra_comp_line:+\\}
${extra_comp_line}
EOF

    chmod +x "$harbor_script"
    echo "    Script: $harbor_script"

    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \
        bash "$harbor_script" \
        >> "$harbor_log" 2>&1 &

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

echo ""
echo "=== Step 7: Exporting and Uploading Expert Traces ==="

JOB_DIRS=""
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    harbor_log="${HARBOR_LOGS[$shard_idx]}"
    job_dir=$(grep "Results written to" "$harbor_log" 2>/dev/null | sed 's/.*Results written to \(jobs\/[^/]*\).*/\1/' | head -1)
    if [ -n "$job_dir" ]; then
        JOB_DIRS="$JOB_DIRS $job_dir"
    fi
done

export JOB_DIRS
export HF_REPO_ID="$OUTPUT_REPO_ID"

echo "Job directories: $JOB_DIRS"
echo "Uploading traces to: $HF_REPO_ID"

LD_LIBRARY_PATH="" TRANSFORMERS_NO_TORCH=1 /scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3 -c "
import sys
import os
from pathlib import Path

from harbor.utils.traces_utils import export_traces, push_dataset
from datasets import concatenate_datasets

job_dirs_str = os.environ.get('JOB_DIRS', '').strip()
if not job_dirs_str:
    print('ERROR: No job directories found')
    sys.exit(1)

job_dirs = [Path(d) for d in job_dirs_str.split() if d]
repo_id = os.environ.get('HF_REPO_ID', '')

print(f'Found {len(job_dirs)} job directories to export')
print(f'Uploading to: {repo_id}')

all_datasets = []
for job_dir in job_dirs:
    if job_dir.exists():
        print(f'Exporting traces from {job_dir}...')
        ds = export_traces(
            root=job_dir,
            recursive=True,
            episodes='last',
            push=False,
            verbose=True,
        )
        if len(ds) > 0:
            all_datasets.append(ds)
            print(f'  Collected {len(ds)} rows from {job_dir}')
    else:
        print(f'WARNING: Job directory does not exist: {job_dir}')

if all_datasets:
    print(f'Merging {len(all_datasets)} datasets...')
    merged_ds = concatenate_datasets(all_datasets)
    print(f'Total merged rows: {len(merged_ds)}')
    print(f'Pushing to {repo_id}...')
    push_dataset(merged_ds, repo_id)
    print(f'Successfully uploaded {len(merged_ds)} traces to {repo_id}')
else:
    print('ERROR: No datasets collected from any shard')
    sys.exit(1)
"

EXPORT_EXIT=$?
if [ $EXPORT_EXIT -eq 0 ]; then
    echo "Expert traces uploaded successfully to: $OUTPUT_REPO_ID"
else
    echo "WARNING: Failed to upload expert traces (exit code: $EXPORT_EXIT)"
fi

echo ""
echo "=== Step 8: Cleanup ==="

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    if [ -n "${VLLM_PIDS[$shard_idx]}" ]; then
        kill "${VLLM_PIDS[$shard_idx]}" 2>/dev/null || true
    fi
done

for node in "${ALL_NODES_ARRAY[@]}"; do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null &
done
wait

# Keep shard data for debugging - uncomment to auto-delete
# rm -rf "$SHARD_DIR"

conda deactivate || true

echo ""
echo "============================================"
if [ $FAILED -eq 0 ]; then
    echo "Experiment COMPLETED: $EXPERIMENT_NAME"
    echo "  Input: $INPUT_REPO_ID"
    echo "  Output: $OUTPUT_REPO_ID"
    echo "  All $NUM_SHARDS shards completed successfully!"
else
    echo "Experiment PARTIALLY FAILED: $EXPERIMENT_NAME"
    echo "  $FAILED out of $NUM_SHARDS shards failed"
fi
echo "============================================"

exit $FAILED
