#!/bin/bash
#SBATCH -p dc-gpu-devel
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=synthlaion
#SBATCH --output=/p/scratch/synthlaion/dc-agent-shared/logs/harbor_merge_%j.out
#SBATCH --job-name=harbor_merge

# ============================================
# Merge traces from 8-shard Harbor job
# Submit with: sbatch --dependency=afterok:<array_job_id> harbor_merge_traces.sbatch
# ============================================

set -eo pipefail

# Get the array job ID from environment or argument
ARRAY_JOB_ID="${HARBOR_ARRAY_JOB_ID:-$1}"

if [ -z "$ARRAY_JOB_ID" ]; then
    echo "ERROR: HARBOR_ARRAY_JOB_ID not set. Usage:"
    echo "  HARBOR_ARRAY_JOB_ID=12345 sbatch harbor_merge_traces.sbatch"
    echo "  or: sbatch --export=HARBOR_ARRAY_JOB_ID=12345 harbor_merge_traces.sbatch"
    exit 1
fi

echo "============================================"
echo "Merging traces from array job: $ARRAY_JOB_ID"
echo "============================================"

# === 1. Source environment ===
source /p/project1/laionize/marianna/dc_agent/dc-agent/train/hpc/dotenv/jsc.env
source $SECRET_ENV_PATH

export PATH=$HOME/.local/bin:$PATH

# === 2. Activate conda ===
source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate /p/scratch/laionize/raj3/vllm_sandboxes

# === 3. Setup proxy ===
PROXY_HOST="10.14.0.53"
PROXY_PORT="1080"

export PROXYCHAINS_CONF_FILE="/tmp/proxychains_${SLURM_JOB_ID}.conf"
cat > "$PROXYCHAINS_CONF_FILE" << EOF
strict_chain
tcp_read_time_out 15000
tcp_connect_time_out 8000
localnet 10.0.0.0/255.0.0.0
localnet 127.0.0.0/255.0.0.0
localnet 192.168.0.0/255.255.0.0
[ProxyList]
socks5 ${PROXY_HOST} ${PROXY_PORT}
EOF

PROXYCHAINS_PRELOAD="/p/scratch/laionize/raj3/vllm_sandboxes/lib/libproxychains.so.4"

export HF_HUB_OFFLINE=0
export PYTHONPATH="/p/project1/laionize/eguha/OpenThoughts-Agent:/p/project1/laionize/eguha/OpenThoughts-Agent/scripts/harbor:$PYTHONPATH"

# === 4. Collect job directories from all shards ===
MARKER_DIR="/tmp/harbor_sharding_${ARRAY_JOB_ID}"
JOB_DIRS=""
COMPLETED=0
FAILED=0

echo "Checking shard completion markers in: $MARKER_DIR"

for SHARD_IDX in {0..7}; do
    MARKER_FILE="$MARKER_DIR/shard_${SHARD_IDX}_done"
    if [ -f "$MARKER_FILE" ]; then
        JOB_DIR=$(cat "$MARKER_FILE")
        if [ -d "$JOB_DIR" ]; then
            JOB_DIRS="$JOB_DIRS $JOB_DIR"
            echo "  Shard ${SHARD_IDX}: $JOB_DIR ✓"
            ((COMPLETED++))
        else
            echo "  Shard ${SHARD_IDX}: Directory not found: $JOB_DIR"
            ((FAILED++))
        fi
    else
        echo "  Shard ${SHARD_IDX}: No completion marker"
        ((FAILED++))
    fi
done

echo ""
echo "Completed shards: $COMPLETED/8"
echo "Failed shards: $FAILED/8"

if [ -z "$JOB_DIRS" ]; then
    echo "ERROR: No job directories found to merge"
    exit 1
fi

# === 5. Merge and upload traces ===
OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-DCAgent/harbor-glm47-8shard-traces}"
TEACHER_SCRIPTS_DIR="/p/project1/laionize/eguha/OpenThoughts-Agent/data/sbatches/teacher_scripts"

UPLOAD_HF_CACHE="$HOME/.cache/huggingface"
mkdir -p "$UPLOAD_HF_CACHE"

echo ""
echo "============================================"
echo "Uploading merged traces to: $OUTPUT_REPO_ID"
echo "Job directories:$JOB_DIRS"
echo "============================================"

HF_HOME="$UPLOAD_HF_CACHE" HF_HUB_CACHE="$UPLOAD_HF_CACHE" \
    LD_PRELOAD="$PROXYCHAINS_PRELOAD" LD_LIBRARY_PATH="" TRANSFORMERS_NO_TORCH=1 \
    python "$TEACHER_SCRIPTS_DIR/upload_traces.py" "$OUTPUT_REPO_ID" $JOB_DIRS

UPLOAD_EXIT=$?

if [ $UPLOAD_EXIT -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "SUCCESS: Merged traces uploaded to $OUTPUT_REPO_ID"
    echo "Shards merged: $COMPLETED"
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "FAILED: Upload failed with exit code: $UPLOAD_EXIT"
    echo "============================================"
    exit 1
fi
