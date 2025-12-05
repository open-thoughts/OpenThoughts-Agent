#!/bin/bash
# Process all shards in DCLM directory
set -e

NUM_WORKERS=${1:-$(python3 -c "import multiprocessing; print(multiprocessing.cpu_count())")}
THRESHOLD=${2:-8.0}
SAMPLE_SIZE=${3:-100}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DCLM_DIR="${DCLM_DIR:-/data/horse/ws/rehe951g-dcagent/dclm-baseline}"
OUTPUT_DIR="/data/horse/ws/rehe951g-dcagent/dclm-terminal-like/results_all_$(date +%Y%m%d_%H%M%S)"

TOTAL_FILES=$(find "${DCLM_DIR}" -name "*.jsonl.zst" 2>/dev/null | wc -l)

echo "=== DCLM Bash Finder - All Shards ==="
echo "Directory: ${DCLM_DIR}"
echo "Total files: ${TOTAL_FILES}"
echo "Workers: ${NUM_WORKERS}"
echo "Threshold: ${THRESHOLD}"
echo ""

python3 "${SCRIPT_DIR}/bash_finder.py" \
    --shard-dir "${DCLM_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --threshold "${THRESHOLD}" \
    --sample-size "${SAMPLE_SIZE}" \
    --workers "${NUM_WORKERS}" \
    --checkpoint-interval 100 \
    2>&1 | tee "${OUTPUT_DIR}/processing.log"

echo ""
echo "Results: ${OUTPUT_DIR}/bash_sequences_sample_${SAMPLE_SIZE}_readable.txt"
