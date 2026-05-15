#!/bin/bash
# Submit a sharded curator datagen job with optional auto-restart chaining.
#
# Usage:
#   ./scripts/datagen/submit_curator_sharded.sh \
#     --max-restarts 8 \
#     -- <model> <input_dataset> <output_repo> [limit] [save_every]
#
# Example:
#   ./scripts/datagen/submit_curator_sharded.sh \
#     --max-restarts 8 \
#     -- QuantTrio/GLM-4.7-AWQ \
#        open-thoughts/OpenThoughts3-1.2M \
#        laion/OT3-1.2M-GLM-4.7-AWQ-completions
#
# The --max-restarts flag pre-submits N+1 jobs chained with afterany
# dependencies so the run auto-resumes on SLURM timeout kills.

set -euo pipefail

MAX_RESTARTS=0
SBATCH_EXTRA_ARGS=""

# Parse our args (before --)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-restarts)
            MAX_RESTARTS="$2"
            shift 2
            ;;
        --nodes)
            SBATCH_EXTRA_ARGS="$SBATCH_EXTRA_ARGS --nodes=$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--max-restarts N] [--nodes N] -- <sbatch args...>"
            exit 1
            ;;
    esac
done

if [ $# -lt 3 ]; then
    echo "Usage: $0 [--max-restarts N] [--nodes N] -- <model> <input_dataset> <output_repo> [limit] [save_every]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/data/sbatches/run_curator_datagen_sharded.sbatch"

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "ERROR: sbatch script not found at $SBATCH_SCRIPT"
    exit 1
fi

echo "Submitting sharded curator datagen with $((MAX_RESTARTS + 1)) total runs"
echo "  Script: $SBATCH_SCRIPT"
echo "  Args: $*"
echo "  Max restarts: $MAX_RESTARTS"
echo ""

# Submit the first job (no dependency)
FIRST_JOB_ID=$(sbatch $SBATCH_EXTRA_ARGS "$SBATCH_SCRIPT" "$@" 2>&1 | awk '{print $NF}')
echo "Job 1/$((MAX_RESTARTS + 1)) submitted: $FIRST_JOB_ID"

PREV_JOB_ID="$FIRST_JOB_ID"

# Chain restart jobs with afterany dependencies
for i in $(seq 1 "$MAX_RESTARTS"); do
    JOB_ID=$(sbatch --dependency="afterany:${PREV_JOB_ID}" $SBATCH_EXTRA_ARGS "$SBATCH_SCRIPT" "$@" 2>&1 | awk '{print $NF}')
    echo "Job $((i + 1))/$((MAX_RESTARTS + 1)) submitted: $JOB_ID (afterany:$PREV_JOB_ID)"
    PREV_JOB_ID="$JOB_ID"
done

echo ""
echo "All $((MAX_RESTARTS + 1)) jobs submitted."
echo "  First: $FIRST_JOB_ID"
echo "  Last:  $PREV_JOB_ID"
echo "  Each run resumes from the previous run's checkpoints."
echo ""
echo "To cancel all: scancel $FIRST_JOB_ID $PREV_JOB_ID  (or scancel the full chain)"
