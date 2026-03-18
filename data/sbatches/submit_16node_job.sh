#!/bin/bash
# Submit 16-node 8-shard Harbor job with automatic merge
#
# Usage: ./submit_16node_job.sh [INPUT_REPO_ID] [OUTPUT_REPO_ID]
#
# Defaults:
#   INPUT_REPO_ID:  DCAgent/harbor-devel-sandboxes
#   OUTPUT_REPO_ID: DCAgent/harbor-glm47-8shard-traces

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_REPO_ID="${1:-DCAgent/harbor-devel-sandboxes}"
OUTPUT_REPO_ID="${2:-DCAgent/harbor-glm47-8shard-traces}"

echo "============================================"
echo "Submitting 16-node 8-shard Harbor Job"
echo "============================================"
echo "Input dataset:  $INPUT_REPO_ID"
echo "Output traces:  $OUTPUT_REPO_ID"
echo "============================================"

# Submit the array job (8 tasks × 2 nodes = 16 nodes)
ARRAY_JOB=$(sbatch --parsable \
    --export=INPUT_REPO_ID="$INPUT_REPO_ID" \
    "$SCRIPT_DIR/harbor_16node_8shard.sbatch")

echo "Submitted array job: $ARRAY_JOB (8 shards × 2 nodes)"

# Submit merge job with dependency on array completion
MERGE_JOB=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB} \
    --export=HARBOR_ARRAY_JOB_ID="$ARRAY_JOB",OUTPUT_REPO_ID="$OUTPUT_REPO_ID" \
    "$SCRIPT_DIR/harbor_merge_traces.sbatch")

echo "Submitted merge job: $MERGE_JOB (depends on $ARRAY_JOB)"

echo ""
echo "============================================"
echo "Jobs submitted successfully!"
echo "============================================"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f /p/scratch/synthlaion/dc-agent-shared/logs/harbor_8shard_${ARRAY_JOB}_*.out"
echo ""
echo "After completion, traces will be at:"
echo "  https://huggingface.co/datasets/$OUTPUT_REPO_ID"
