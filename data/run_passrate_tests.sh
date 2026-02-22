#!/bin/bash
# Run pass rate tests for all Stack-based datasets
# Using the documented Harbor testing methodology

set -e

# Setup environment
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6
source /scratch/10000/eguha3/old-dc-agent/secret.env

cd /scratch/10000/eguha3/dc-agent

export PYTHONPATH="/scratch/10000/eguha3/dc-agent/scripts/harbor:$PYTHONPATH"

# Configuration
MODEL="openai/gpt-5-nano-2025-08-07"
N_TASKS=25
N_CONCURRENT=10

test_dataset() {
    local DATASET="$1"
    local DATASET_NAME="${DATASET##*/}"
    local LOG_FILE="data/passrate_logs/${DATASET_NAME}.log"

    echo "============================================" | tee "$LOG_FILE"
    echo "Testing: $DATASET" | tee -a "$LOG_FILE"
    echo "Model: $MODEL" | tee -a "$LOG_FILE"
    echo "Tasks: $N_TASKS" | tee -a "$LOG_FILE"
    echo "============================================" | tee -a "$LOG_FILE"

    # Download and extract dataset
    echo "=== Downloading dataset ===" | tee -a "$LOG_FILE"
    DATASET_PATH=$(LD_LIBRARY_PATH="" python3 -c "
import sys
import os
os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'
sys.path.insert(0, '/scratch/10000/eguha3/dc-agent/scripts/harbor')
from tasks_parquet_converter import from_hf_dataset
path = from_hf_dataset('$DATASET')
print(path)
" 2>&1 | tail -1)

    echo "Dataset path: $DATASET_PATH" | tee -a "$LOG_FILE"

    if [ -z "$DATASET_PATH" ] || [ ! -d "$DATASET_PATH" ]; then
        echo "ERROR: Failed to download dataset" | tee -a "$LOG_FILE"
        return 1
    fi

    # Count tasks
    TOTAL_TASKS=$(find "$DATASET_PATH" -name "instruction.md" | wc -l)
    echo "Total tasks available: $TOTAL_TASKS" | tee -a "$LOG_FILE"

    # Create subset with first N tasks - COPY instead of symlink
    SUBSET_DIR=$(mktemp -d -t "${DATASET_NAME}_subset_XXXXXX")
    TASK_COUNT=0
    for task_dir in $(find "$DATASET_PATH" -name "instruction.md" -printf '%h\n' | sort | head -n $N_TASKS); do
        cp -r "$task_dir" "$SUBSET_DIR/$(basename $task_dir)" 2>/dev/null || true
        TASK_COUNT=$((TASK_COUNT + 1))
    done

    echo "Created subset with $TASK_COUNT tasks at: $SUBSET_DIR" | tee -a "$LOG_FILE"

    # Run Harbor
    echo "" | tee -a "$LOG_FILE"
    echo "=== Running Harbor ===" | tee -a "$LOG_FILE"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="passrate_${DATASET_NAME}_${TIMESTAMP}"

    # Delete old job directory if exists
    rm -rf "jobs/passrate_${DATASET_NAME}"* 2>/dev/null || true

    harbor jobs start \
        -p "$SUBSET_DIR" \
        --n-concurrent $N_CONCURRENT \
        --agent terminus-2 \
        --model "$MODEL" \
        --env daytona \
        --n-attempts 1 \
        --job-name "$JOB_NAME" \
        2>&1 | tee -a "$LOG_FILE"

    # Parse results
    echo "" | tee -a "$LOG_FILE"
    echo "=== Results ===" | tee -a "$LOG_FILE"

    # Find the result file (handle timestamp in name)
    RESULT_FILE=$(ls -t jobs/passrate_${DATASET_NAME}_*/result.json 2>/dev/null | head -1)
    if [ -n "$RESULT_FILE" ] && [ -f "$RESULT_FILE" ]; then
        python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)

# Try new format first
if 'stats' in data:
    stats = data['stats']
    evals = stats.get('evals', {})
    if evals:
        eval_key = list(evals.keys())[0]
        metrics = evals[eval_key].get('metrics', [])
        if metrics:
            pass_rate = metrics[0].get('mean', 0) * 100
            print(f'Dataset: $DATASET')
            print(f'Pass rate: {pass_rate:.1f}%')
        else:
            print('No metrics found')
    else:
        print('No evals found')
# Try old format
elif 'results' in data:
    results = data['results']
    passed = sum(1 for r in results if r.get('reward', 0) > 0)
    total = len(results)
    rate = (passed / total * 100) if total > 0 else 0
    print(f'Dataset: $DATASET')
    print(f'Tasks tested: {total}')
    print(f'Passed: {passed}')
    print(f'Pass rate: {rate:.1f}%')
" | tee -a "$LOG_FILE"
    else
        echo "No results file found" | tee -a "$LOG_FILE"
    fi

    # Cleanup
    rm -rf "$SUBSET_DIR"

    echo "============================================" | tee -a "$LOG_FILE"
}

# Run single dataset if provided as argument
if [ -n "$1" ]; then
    test_dataset "$1"
    exit 0
fi

echo "Usage: $0 <dataset_repo_id>"
