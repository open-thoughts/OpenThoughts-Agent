#!/bin/bash
# Test on random subset of shards
set -e

NUM_SHARDS=${1:-10}
SAMPLE_SIZE=${2:-20}
THRESHOLD=${3:-4.0}
WORKERS=${4:-8}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DCLM_DIR="${DCLM_DIR:-/data/horse/ws/rehe951g-dcagent/dclm-baseline}"
TEMP_DIR="${SCRIPT_DIR}/test_shards"
OUTPUT_DIR="${SCRIPT_DIR}/test_results_$(date +%Y%m%d_%H%M%S)"

echo "=== DCLM Bash Finder - Test Pipeline ==="
echo "Selecting ${NUM_SHARDS} random shards from ${DCLM_DIR}"

# Create temp dir and copy random shards
mkdir -p "${TEMP_DIR}"
find "${DCLM_DIR}" -name "*.jsonl.zst" \
    | sort -R \
    | head -n "${NUM_SHARDS}" \
    | xargs -I {} cp {} "${TEMP_DIR}/"

echo "Running bash finder with ${WORKERS} workers..."
python3 "${SCRIPT_DIR}/bash_finder.py" \
    --shard-dir "${TEMP_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --threshold "${THRESHOLD}" \
    --sample-size "${SAMPLE_SIZE}" \
    --workers "${WORKERS}"

echo ""
echo "Results: ${OUTPUT_DIR}/bash_sequences_sample_${SAMPLE_SIZE}_readable.txt"
rm -rf "${TEMP_DIR}"
