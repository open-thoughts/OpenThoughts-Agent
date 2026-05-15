#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --output=data/sbatches/logs/curator_datagen_%j.out
#SBATCH --job-name=curator_datagen
#SBATCH --account=reformo

# NOTE: set -eo pipefail is deferred until after environment setup
# because conda/module commands can trigger benign errors
# (e.g., mkdir /var/spool/data on Jupiter compute nodes)

# ============================================
# Curator Non-Agentic Datagen
# ============================================
# Starts a vLLM server on the node, then runs
# curator_datagen.py to generate completions
# from a HuggingFace dataset of prompts.
#
# Uses a dedicated 'curator' conda env to avoid
# dependency conflicts with other environments.
#
# Pre-requisites (run on login node before submitting):
#   conda create -n curator python=3.12 -y
#   conda activate curator
#   pip install 'bespokelabs-curator>=0.1.0' 'vllm>=0.11.2' \
#       'datasets>=2.0.0' 'huggingface_hub>=0.20.0'
#
# Usage:
#   sbatch run_curator_datagen.sbatch \
#     <model> <input_dataset> <output_repo> [limit]
#
# Example:
#   sbatch run_curator_datagen.sbatch \
#     Qwen/Qwen3-32B \
#     open-thoughts/OpenThoughts3-1.2M \
#     mlfoundations-dev/OT3-1.2M-Qwen3-32B-completions
# ============================================

MODEL="${1:?Usage: sbatch run_curator_datagen.sbatch <model> <input_dataset> <output_repo> [limit]}"
INPUT_DATASET="${2:?Missing input dataset}"
OUTPUT_REPO="${3:?Missing output repo}"
LIMIT="${4:-}"  # Optional: limit number of prompts

echo "============================================"
echo "Curator Datagen"
echo "============================================"
echo "Model:          $MODEL"
echo "Input dataset:  $INPUT_DATASET"
echo "Output repo:    $OUTPUT_REPO"
echo "Limit:          ${LIMIT:-all}"
echo "Job ID:         $SLURM_JOB_ID"
echo "Node:           $(hostname)"
echo "============================================"

# -----------------------------------------------
# Step 1: Environment setup
# -----------------------------------------------
echo ""
echo "=== Step 1: Environment Setup ==="

# Activate the dedicated curator conda env (source conda.sh directly
# instead of ~/.bashrc — compute nodes may have issues with /etc/bashrc)
source /e/scratch/jureap59/feuer1/miniforge3/etc/profile.d/conda.sh
conda activate curator

# Source secrets (HF_TOKEN, etc.)
if [ -f ~/secrets.env ]; then
    source ~/secrets.env
fi

# Set REPO_DIR explicitly (SLURM working dir may not be the repo)
REPO_DIR="${DCFT:-/e/scratch/jureap59/feuer1/OpenThoughts-Agent}"
if [ ! -d "$REPO_DIR/scripts/datagen" ]; then
    # Fallback: try to find it from the sbatch path
    REPO_DIR="$(cd "$(dirname "$(scontrol show job $SLURM_JOB_ID | grep Command | awk -F= '{print $2}')")/../.." 2>/dev/null && pwd)" || true
fi

# Source cluster dotenv for HF_HUB_CACHE, etc.
if [ -f "$REPO_DIR/hpc/dotenv/jupiter.env" ]; then
    source "$REPO_DIR/hpc/dotenv/jupiter.env"
fi

# Ensure HF_HUB_CACHE is set even if dotenv didn't resolve
export HF_HUB_CACHE="${HF_HUB_CACHE:-/e/data1/datasets/playground/ot/hf_hub}"

# No internet on compute nodes — force offline mode for HF
# (dataset must be pre-cached on login node before submitting, see Step 1b)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# -----------------------------------------------
# Step 1b: Verify dataset is cached locally
# -----------------------------------------------
echo ""
echo "=== Step 1b: Verifying Dataset Cache ==="

# Check if the input dataset is available in the HF cache.
# Compute nodes have no internet, so we must fail early with a clear message.
python3 -c "
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from datasets import load_dataset
ds_name = '$INPUT_DATASET'
split = 'train'
try:
    ds = load_dataset(ds_name, split=split, streaming=True)
    # Just check we can read one row
    next(iter(ds))
    print(f'Dataset {ds_name} is cached and accessible.')
except Exception as e:
    print(f'ERROR: Dataset {ds_name} is NOT cached in HF_HUB_CACHE={os.environ.get(\"HF_HUB_CACHE\", \"unset\")}')
    print(f'Pre-cache it on the login node before submitting:')
    print(f'  conda activate curator')
    print(f'  python -c \"from datasets import load_dataset; ds = load_dataset(\\\"{ds_name}\\\", split=\\\"{split}\\\"); print(f\\\"Cached {{len(ds)}} rows\\\")\"')
    print(f'Original error: {e}')
    sys.exit(1)
" || exit 1

# Load modules (Jupiter GH200 uses CUDA 13.0 only)
module purge 2>/dev/null || true
module load nvidia-compilers/25.9-CUDA-13 2>&1 || true
echo "CUDA_HOME: ${CUDA_HOME:-not set}"

# Disable V1 engine — crashes on GH200 with nightly cu130 build
export VLLM_USE_V1=0

cd "$REPO_DIR" || { echo "ERROR: Cannot cd to $REPO_DIR"; exit 1; }
mkdir -p data/sbatches/logs

echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "Python: $(python3 --version 2>&1)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Now enable strict error handling
set -eo pipefail

# -----------------------------------------------
# Step 2: Start vLLM server
# -----------------------------------------------
echo ""
echo "=== Step 2: Starting vLLM Server ==="

VLLM_PORT=8000
VLLM_API_KEY="token-curator-${SLURM_JOB_ID}"
VLLM_LOG="data/sbatches/logs/vllm_${SLURM_JOB_ID}.log"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port $VLLM_PORT \
    --api-key "$VLLM_API_KEY" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --trust-remote-code \
    >> "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
echo "vLLM log: $VLLM_LOG"

# Wait for vLLM to become healthy
echo "Waiting for vLLM server to start..."
for i in {1..120}; do
    if curl -s "http://localhost:${VLLM_PORT}/v1/models" -H "Authorization: Bearer ${VLLM_API_KEY}" > /dev/null 2>&1; then
        echo "vLLM server is ready! (took ~$((i * 10))s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Check $VLLM_LOG"
        tail -30 "$VLLM_LOG"
        exit 1
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: vLLM server failed to start after 20 minutes"
        tail -30 "$VLLM_LOG"
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 10
done

# -----------------------------------------------
# Step 3: Run Curator datagen
# -----------------------------------------------
echo ""
echo "=== Step 3: Running Curator Datagen ==="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="data/sbatches/logs/curator_output_${SLURM_JOB_ID}_${TIMESTAMP}"

SAVE_EVERY="${5:-10000}"  # Optional: checkpoint every N prompts (default: 10000)

CURATOR_ARGS=(
    --input-dataset "$INPUT_DATASET"
    --model "$MODEL"
    --base-url "http://localhost:${VLLM_PORT}/v1"
    --api-key "$VLLM_API_KEY"
    --output-repo "$OUTPUT_REPO"
    --output-dir "$OUTPUT_DIR"
    --max-tokens 16384
    --temperature 0.7
    --top-p 0.95
    --save-every "$SAVE_EVERY"
)

if [ -n "$LIMIT" ]; then
    CURATOR_ARGS+=(--limit "$LIMIT")
fi

echo "Running: python3 scripts/datagen/curator_datagen.py ${CURATOR_ARGS[*]}"

python3 scripts/datagen/curator_datagen.py "${CURATOR_ARGS[@]}"
CURATOR_EXIT=$?

# -----------------------------------------------
# Step 4: Cleanup
# -----------------------------------------------
echo ""
echo "=== Step 4: Cleanup ==="

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "============================================"
if [ $CURATOR_EXIT -eq 0 ]; then
    echo "Curator Datagen COMPLETED"
    echo "  Input:  $INPUT_DATASET"
    echo "  Output: $OUTPUT_REPO"
    echo "  Local:  $OUTPUT_DIR"
else
    echo "Curator Datagen FAILED (exit code: $CURATOR_EXIT)"
fi
echo "============================================"

exit $CURATOR_EXIT
