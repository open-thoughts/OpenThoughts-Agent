#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --output=data/sbatches/logs/curator_sharded_%j.out
#SBATCH --job-name=curator_sharded
#SBATCH --account=reformo

# NOTE: set -eo pipefail is deferred until after environment setup

# ============================================
# Curator Sharded Datagen (Multi-Node DP)
# ============================================
# Runs N independent vLLM servers (one per node), each processing
# a disjoint slice of the input dataset. Results are merged at the end.
#
# Supports auto-resume: each shard's output dir is stable across restarts,
# so SLURM timeout kills + afterany restart chains work seamlessly.
#
# Usage:
#   # Simple (no restarts):
#   sbatch run_curator_datagen_sharded.sbatch \
#     <model> <input_dataset> <output_repo> [limit] [save_every]
#
#   # With auto-restart (recommended for long datasets):
#   MAX_RESTARTS=8 sbatch run_curator_datagen_sharded.sbatch \
#     <model> <input_dataset> <output_repo> [limit] [save_every]
#
# Pre-requisites (run on login node before submitting):
#   conda activate curator
#   python -c "from datasets import load_dataset; ds = load_dataset('<dataset>', split='train'); print(f'Cached {len(ds)} rows')"
# ============================================

MODEL="${1:?Usage: sbatch run_curator_datagen_sharded.sbatch <model> <input_dataset> <output_repo> [limit] [save_every]}"
INPUT_DATASET="${2:?Missing input dataset}"
OUTPUT_REPO="${3:?Missing output repo}"
LIMIT="${4:-}"
SAVE_EVERY="${5:-200}"

NUM_SHARDS="${SLURM_JOB_NUM_NODES}"

# Stable output directory — must NOT contain timestamps or job IDs so that
# restarts write to the same place. Based on model + dataset + shard count.
MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||; s|[^a-zA-Z0-9._-]|_|g')
DATASET_SHORT=$(echo "$INPUT_DATASET" | sed 's|.*/||; s|[^a-zA-Z0-9._-]|_|g')
OUTPUT_BASE="data/sbatches/curator_runs/${MODEL_SHORT}__${DATASET_SHORT}__${NUM_SHARDS}shards"

echo "============================================"
echo "Curator Sharded Datagen (${NUM_SHARDS}-Way DP)"
echo "============================================"
echo "Model:          $MODEL"
echo "Input dataset:  $INPUT_DATASET"
echo "Output repo:    $OUTPUT_REPO"
echo "Limit:          ${LIMIT:-all}"
echo "Save every:     $SAVE_EVERY"
echo "Shards:         $NUM_SHARDS"
echo "Output base:    $OUTPUT_BASE"
echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "============================================"

# Check for existing progress (resume indicator)
EXISTING_ROWS=0
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_dir="${OUTPUT_BASE}/shard_${shard_idx}"
    if [ -d "$shard_dir" ]; then
        latest_ckpt=$(ls -1 "$shard_dir"/checkpoint_*.parquet 2>/dev/null | sort | tail -1)
        if [ -n "$latest_ckpt" ]; then
            echo "  Shard $shard_idx: found checkpoint $latest_ckpt"
            EXISTING_ROWS=$((EXISTING_ROWS + 1))
        fi
    fi
done
if [ "$EXISTING_ROWS" -gt 0 ]; then
    echo "RESUMING: Found checkpoints in $EXISTING_ROWS shards"
else
    echo "Starting fresh run"
fi

# -----------------------------------------------
# Step 1: Environment setup
# -----------------------------------------------
echo ""
echo "=== Step 1: Environment Setup ==="

source /e/scratch/jureap59/feuer1/miniforge3/etc/profile.d/conda.sh
conda activate curator

if [ -f ~/secrets.env ]; then
    source ~/secrets.env
fi

REPO_DIR="${DCFT:-/e/scratch/jureap59/feuer1/OpenThoughts-Agent}"
if [ ! -d "$REPO_DIR/scripts/datagen" ]; then
    REPO_DIR="$(cd "$(dirname "$(scontrol show job $SLURM_JOB_ID | grep Command | awk -F= '{print $2}')")/../.." 2>/dev/null && pwd)" || true
fi

if [ -f "$REPO_DIR/hpc/dotenv/jupiter.env" ]; then
    source "$REPO_DIR/hpc/dotenv/jupiter.env"
fi

export HF_HUB_CACHE="${HF_HUB_CACHE:-/e/data1/datasets/playground/ot/hf_hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge 2>/dev/null || true
module load nvidia-compilers/25.9-CUDA-13 2>&1 || true

export VLLM_USE_V1=0

cd "$REPO_DIR" || { echo "ERROR: Cannot cd to $REPO_DIR"; exit 1; }
mkdir -p data/sbatches/logs "$OUTPUT_BASE"

echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "Python: $(python3 --version 2>&1)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Now enable strict error handling
set -eo pipefail

# Get all node hostnames
ALL_NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

# -----------------------------------------------
# Step 2: Start vLLM servers on all nodes
# -----------------------------------------------
echo ""
echo "=== Step 2: Starting $NUM_SHARDS vLLM Servers ==="

VLLM_BASE_PORT=8000

declare -a VLLM_PIDS
declare -a VLLM_API_KEYS

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    node=${ALL_NODES[$shard_idx]}
    port=$((VLLM_BASE_PORT + shard_idx))
    api_key="token-curator-${SLURM_JOB_ID}-shard${shard_idx}"
    vllm_log="data/sbatches/logs/vllm_sharded_${SLURM_JOB_ID}_shard${shard_idx}.log"

    VLLM_API_KEYS[$shard_idx]="$api_key"

    echo "  Shard $shard_idx: node=$node, port=$port"

    srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none -w "$node" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --host "0.0.0.0" \
            --port "$port" \
            --api-key "$api_key" \
            --tensor-parallel-size 4 \
            --gpu-memory-utilization 0.90 \
            --max-model-len 32768 \
            --max-num-seqs 50 \
            --dtype bfloat16 \
            --kv-cache-dtype fp8 \
            --enable-chunked-prefill \
            --max-num-partial-prefills 1 \
            --max-num-batched-tokens 16384 \
            --enable-prefix-caching \
            --swap-space 32 \
            --enable-expert-parallel \
            --trust-remote-code \
        >> "$vllm_log" 2>&1 &

    VLLM_PIDS[$shard_idx]=$!
    echo "    PID: ${VLLM_PIDS[$shard_idx]}, Log: $vllm_log"
done

# Wait for all vLLM servers to become healthy
echo ""
echo "Waiting for all vLLM servers to start..."

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    node=${ALL_NODES[$shard_idx]}
    port=$((VLLM_BASE_PORT + shard_idx))
    api_key="${VLLM_API_KEYS[$shard_idx]}"

    echo "  Checking shard $shard_idx ($node:$port)..."
    for i in {1..120}; do
        if srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none -w "$node" \
            curl -s "http://localhost:${port}/v1/models" \
                -H "Authorization: Bearer ${api_key}" > /dev/null 2>&1; then
            echo "    Shard $shard_idx ready! (took ~$((i * 10))s)"
            break
        fi
        if ! kill -0 "${VLLM_PIDS[$shard_idx]}" 2>/dev/null; then
            echo "ERROR: vLLM shard $shard_idx died"
            tail -20 "data/sbatches/logs/vllm_sharded_${SLURM_JOB_ID}_shard${shard_idx}.log"
            exit 1
        fi
        if [ "$i" -eq 120 ]; then
            echo "ERROR: vLLM shard $shard_idx failed to start after 20 minutes"
            exit 1
        fi
        sleep 10
    done
done

echo "All $NUM_SHARDS vLLM servers are ready!"

# -----------------------------------------------
# Step 3: Run async datagen on each shard
# -----------------------------------------------
echo ""
echo "=== Step 3: Running $NUM_SHARDS Async Datagen Shards ==="

declare -a CURATOR_PIDS
declare -a SHARD_OUTPUTS

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    node=${ALL_NODES[$shard_idx]}
    port=$((VLLM_BASE_PORT + shard_idx))
    api_key="${VLLM_API_KEYS[$shard_idx]}"
    # Stable shard output dir — same across restarts
    shard_output="${OUTPUT_BASE}/shard_${shard_idx}"
    shard_log="data/sbatches/logs/curator_sharded_${SLURM_JOB_ID}_shard${shard_idx}.log"

    SHARD_OUTPUTS[$shard_idx]="$shard_output"

    DATAGEN_ARGS=(
        --input-dataset "$INPUT_DATASET"
        --model "$MODEL"
        --base-url "http://${node}:${port}/v1"
        --api-key "$api_key"
        --output-dir "$shard_output"
        --max-tokens 20480
        --temperature 0.7
        --top-p 0.95
        --enable-thinking
        --save-every "$SAVE_EVERY"
        --shard-index "$shard_idx"
        --num-shards "$NUM_SHARDS"
        --max-concurrent 45
        --request-timeout 1200
    )

    if [ -n "$LIMIT" ]; then
        DATAGEN_ARGS+=(--limit "$LIMIT")
    fi

    echo "  Shard $shard_idx: node=$node, output=$shard_output"

    srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none -w "$node" \
        python3 scripts/datagen/async_datagen.py "${DATAGEN_ARGS[@]}" \
        >> "$shard_log" 2>&1 &

    CURATOR_PIDS[$shard_idx]=$!
    echo "    PID: ${CURATOR_PIDS[$shard_idx]}, Log: $shard_log"
done

echo ""
echo "All $NUM_SHARDS curator shards started. Waiting for completion..."

FAILED=0
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    if ! wait "${CURATOR_PIDS[$shard_idx]}"; then
        echo "  Shard $shard_idx exited with error (may have saved checkpoint)"
        FAILED=$((FAILED + 1))
    else
        echo "  Shard $shard_idx completed"
    fi
done

# -----------------------------------------------
# Step 4: Merge shards and upload
# -----------------------------------------------
echo ""
echo "=== Step 4: Merging $NUM_SHARDS Shards ==="

SHARD_DIRS=""
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    SHARD_DIRS="$SHARD_DIRS ${SHARD_OUTPUTS[$shard_idx]}"
done

MERGED_OUTPUT="${OUTPUT_BASE}/merged"

python3 -c "
import sys, os, glob
from datasets import Dataset, load_from_disk, concatenate_datasets

shard_dirs = '''$SHARD_DIRS'''.split()
print(f'Merging {len(shard_dirs)} shards...')

datasets = []
for sd in shard_dirs:
    sd = sd.strip()
    if not sd or not os.path.exists(sd):
        print(f'  WARNING: Shard dir not found: {sd}')
        continue
    # Try loading from save_to_disk format first
    try:
        ds = load_from_disk(sd)
        datasets.append(ds)
        print(f'  Loaded {sd}: {len(ds)} rows')
        continue
    except Exception:
        pass
    # Fall back to latest checkpoint parquet
    parquets = sorted(glob.glob(os.path.join(sd, 'checkpoint_*.parquet')))
    if not parquets:
        parquets = sorted(glob.glob(os.path.join(sd, 'data.parquet')))
    if parquets:
        try:
            ds = Dataset.from_parquet(parquets[-1])
            datasets.append(ds)
            print(f'  Loaded checkpoint from {sd}: {len(ds)} rows')
        except Exception as e:
            print(f'  WARNING: Could not load {sd}: {e}')
    else:
        print(f'  WARNING: No data found in {sd}')

if not datasets:
    print('ERROR: No datasets to merge')
    sys.exit(1)

merged = concatenate_datasets(datasets)
print(f'Merged: {len(merged)} total rows')

out_path = '$MERGED_OUTPUT'
os.makedirs(out_path, exist_ok=True)
merged.save_to_disk(out_path)
merged.to_parquet(os.path.join(out_path, 'data.parquet'))
print(f'Saved to {out_path}')

repo = '$OUTPUT_REPO'
if repo:
    print(f'Pushing to HuggingFace: {repo}')
    try:
        merged.push_to_hub(repo, private=False)
        print(f'Uploaded to https://huggingface.co/datasets/{repo}')
    except Exception as e:
        print(f'WARNING: HF push failed (offline?): {e}')
        print('Merged data is available locally for manual upload.')
"

MERGE_EXIT=$?

# -----------------------------------------------
# Step 5: Cleanup
# -----------------------------------------------
echo ""
echo "=== Step 5: Cleanup ==="

for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    kill "${VLLM_PIDS[$shard_idx]}" 2>/dev/null || true
done
wait 2>/dev/null || true

# Report total progress across all shards
echo ""
echo "============================================"
TOTAL_ROWS=0
for shard_idx in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_dir="${SHARD_OUTPUTS[$shard_idx]}"
    latest=$(ls -1 "$shard_dir"/checkpoint_*.parquet 2>/dev/null | sort | tail -1)
    if [ -n "$latest" ]; then
        # Extract row count from filename (checkpoint_NNNNNNN.parquet)
        count=$(basename "$latest" .parquet | sed 's/checkpoint_//')
        echo "  Shard $shard_idx: ~$count rows (checkpoint: $(basename $latest))"
        TOTAL_ROWS=$((TOTAL_ROWS + count))
    fi
done
echo "  Total progress: ~$TOTAL_ROWS rows across $NUM_SHARDS shards"

if [ $MERGE_EXIT -eq 0 ]; then
    echo "Curator Sharded Datagen: merge OK"
    echo "  Input:  $INPUT_DATASET"
    echo "  Output: $OUTPUT_REPO"
    echo "  Local:  $MERGED_OUTPUT"
else
    echo "Curator Sharded Datagen: merge had issues (exit $MERGE_EXIT)"
fi
echo "============================================"

# Exit 0 even if some shards failed — they saved checkpoints and will
# resume on the next restart. Only fail hard on merge issues.
exit $MERGE_EXIT
