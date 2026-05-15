#!/usr/bin/env bash
# Launch GLM-4.7 trace generation with sliding-window concurrency.
#
# Each new job depends on the job N positions before it (where N = concurrency).
# As soon as one slot frees up, the next job starts immediately.
#
# Usage:
#   bash launch_glm47_pairs.sh                    # Submit all from launch_list.txt
#   bash launch_glm47_pairs.sh --dry-run           # Preview without submitting
#   bash launch_glm47_pairs.sh --after 123456      # Seed slot 0 with existing job
#   bash launch_glm47_pairs.sh --concurrency 2     # Max 2 concurrent jobs (default)
#   bash launch_glm47_pairs.sh --start-idx 3       # Skip first 3 datasets (already done)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_harbor_glm47_jupiter.sbatch"
LAUNCH_LIST="${SCRIPT_DIR}/launch_list.txt"
LAUNCH_LOG="${SCRIPT_DIR}/launch_log.txt"

DRY_RUN=false
AFTER_JOBS=()
CONCURRENCY=2
START_IDX=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --after)
            AFTER_JOBS+=("$2")
            shift 2
            ;;
        --concurrency|--batch-size)
            CONCURRENCY="$2"
            shift 2
            ;;
        --start-idx)
            START_IDX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--after JOBID]... [--concurrency N] [--start-idx N]"
            exit 1
            ;;
    esac
done

if [[ ! -f "$LAUNCH_LIST" ]]; then
    echo "ERROR: $LAUNCH_LIST not found. Run prepare_datasets.py first."
    exit 1
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: $SBATCH_SCRIPT not found."
    exit 1
fi

# Read datasets into array
mapfile -t DATASETS < <(grep -v '^\s*$' "$LAUNCH_LIST")
TOTAL=${#DATASETS[@]}

echo "============================================================"
echo "GLM-4.7 Trace Generation Launch (sliding window)"
echo "============================================================"
echo "Datasets: $TOTAL (starting from index $START_IDX)"
echo "Concurrency: $CONCURRENCY"
echo "Sbatch script: $SBATCH_SCRIPT"
echo "Dry run: $DRY_RUN"
[[ ${#AFTER_JOBS[@]} -gt 0 ]] && echo "Seeding with: ${AFTER_JOBS[*]}"
echo "============================================================"

if [[ "$DRY_RUN" == "false" ]]; then
    echo "# GLM-4.7 Launch Log - $(date)" > "$LAUNCH_LOG"
    echo "# concurrency=$CONCURRENCY total=$TOTAL start_idx=$START_IDX" >> "$LAUNCH_LOG"
fi

# Sliding window: JOB_IDS[i] holds the job ID for dataset i.
# Job i depends on job (i - CONCURRENCY), so at most CONCURRENCY run at once.
declare -a JOB_IDS=()

# Seed slots with --after jobs
for j in "${!AFTER_JOBS[@]}"; do
    JOB_IDS[$j]="${AFTER_JOBS[$j]}"
done

submitted=0
for (( i=START_IDX; i<TOTAL; i++ )); do
    ds="${DATASETS[$i]}"

    # Determine dependency: the job CONCURRENCY positions back
    dep_idx=$((i - CONCURRENCY))
    DEP_ARG=""
    if [[ $dep_idx -ge 0 ]] && [[ -n "${JOB_IDS[$dep_idx]:-}" ]]; then
        DEP_ARG="--dependency=afterany:${JOB_IDS[$dep_idx]}"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "  [$((i+1))/$TOTAL] [DRY-RUN] sbatch $DEP_ARG $SBATCH_SCRIPT $ds"
        JOB_IDS[$i]="DRY${i}"
    else
        if [[ -n "$DEP_ARG" ]]; then
            OUTPUT=$(sbatch $DEP_ARG "$SBATCH_SCRIPT" "$ds" 2>&1)
        else
            OUTPUT=$(sbatch "$SBATCH_SCRIPT" "$ds" 2>&1)
        fi

        JOB_ID=$(echo "$OUTPUT" | grep -oP 'Submitted batch job \K[0-9]+')
        if [[ -z "$JOB_ID" ]]; then
            echo "  [$((i+1))/$TOTAL] ERROR submitting $ds: $OUTPUT"
            echo "ERROR $ds $OUTPUT" >> "$LAUNCH_LOG"
            JOB_IDS[$i]=""
            continue
        fi

        dep_info=""
        [[ -n "$DEP_ARG" ]] && dep_info=" (after ${JOB_IDS[$dep_idx]})"
        echo "  [$((i+1))/$TOTAL] $JOB_ID -> $ds$dep_info"
        echo "$JOB_ID $ds" >> "$LAUNCH_LOG"
        JOB_IDS[$i]="$JOB_ID"
        submitted=$((submitted + 1))
        sleep 0.5
    fi
done

echo ""
echo "============================================================"
echo "Done! Submitted $submitted jobs with max $CONCURRENCY concurrent."
if [[ "$DRY_RUN" == "false" ]]; then
    echo "Launch log: $LAUNCH_LOG"
    echo ""
    echo "Monitor with: squeue -u $(whoami)"
fi
echo "============================================================"
