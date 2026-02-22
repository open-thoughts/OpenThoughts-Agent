#!/bin/bash
# Test pass rate for a Harbor dataset
# Usage: ./test_passrate.sh <dataset_repo_id> [n_tasks] [model]

set -e

DATASET="${1:?Usage: $0 <dataset_repo_id> [n_tasks] [model]}"
N_TASKS="${2:-25}"
MODEL="${3:-openai/gpt-4.1-nano}"

DATASET_NAME="${DATASET##*/}"
echo "============================================"
echo "Testing: $DATASET"
echo "Tasks: $N_TASKS"
echo "Model: $MODEL"
echo "============================================"

# Setup environment
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6
source /scratch/10000/eguha3/old-dc-agent/secret.env

export PYTHONPATH="/scratch/10000/eguha3/dc-agent/scripts/harbor:$PYTHONPATH"

# Download dataset - capture only the last line (the path)
echo "=== Downloading dataset ==="
DATASET_PATH=$(LD_LIBRARY_PATH="" python3 -c "
import sys
sys.stderr = sys.stdout  # Redirect progress to stderr
from tasks_parquet_converter import from_hf_dataset
path = from_hf_dataset('$DATASET')
# Print only the path to a separate file descriptor
import os
os.write(3, path.encode())
" 3>&1 1>&2 2>&1)

# If that didn't work, try simpler approach
if [ -z "$DATASET_PATH" ] || [ ! -d "$DATASET_PATH" ]; then
    echo "Trying alternative download method..."
    DATASET_PATH=$(LD_LIBRARY_PATH="" python3 << 'PYEOF'
import sys
# Suppress all output during import/download
import io
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

import os
os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'

sys.path.insert(0, '/scratch/10000/eguha3/dc-agent/scripts/harbor')
from tasks_parquet_converter import from_hf_dataset
path = from_hf_dataset('DATASET_PLACEHOLDER')

# Restore and print only the path
sys.stdout = old_stdout
sys.stderr = old_stderr
print(path)
PYEOF
)
    DATASET_PATH=$(echo "$DATASET_PATH" | sed "s/DATASET_PLACEHOLDER/$DATASET/")
fi

# Last resort - look for existing extracted dataset
if [ -z "$DATASET_PATH" ] || [ ! -d "$DATASET_PATH" ]; then
    echo "Looking for cached dataset..."
    CACHE_DIR="/tmp/harbor_tasks"
    DATASET_PATH=$(find "$CACHE_DIR" -maxdepth 1 -type d -name "*${DATASET_NAME}*" 2>/dev/null | head -1)
fi

if [ -z "$DATASET_PATH" ] || [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Failed to download dataset"
    echo "Trying direct approach..."

    # Direct download and extract
    cd /scratch/10000/eguha3/dc-agent
    LD_LIBRARY_PATH="" HF_DATASETS_DISABLE_PROGRESS_BARS=1 python3 << PYEOF
import sys
sys.path.insert(0, '/scratch/10000/eguha3/dc-agent/scripts/harbor')
from tasks_parquet_converter import from_hf_dataset
path = from_hf_dataset('$DATASET')
print(f"DATASET_PATH={path}")
PYEOF
    exit 1
fi

echo "Dataset path: $DATASET_PATH"

# Count tasks and create subset
TOTAL_TASKS=$(find "$DATASET_PATH" -name "instruction.md" 2>/dev/null | wc -l)
echo "Total tasks available: $TOTAL_TASKS"

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "ERROR: No tasks found!"
    exit 1
fi

# Create subset with first N tasks
SUBSET_DIR=$(mktemp -d -t "${DATASET_NAME}_subset_XXXXXX")
TASK_COUNT=0
for task_dir in $(find "$DATASET_PATH" -name "instruction.md" -printf '%h\n' 2>/dev/null | sort | head -n $N_TASKS); do
    ln -s "$task_dir" "$SUBSET_DIR/$(basename $task_dir)" 2>/dev/null || true
    TASK_COUNT=$((TASK_COUNT + 1))
done
echo "Created subset with $TASK_COUNT tasks at: $SUBSET_DIR"

# Run Harbor
echo ""
echo "=== Running Harbor ==="
JOB_NAME="passrate_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)"

cd /scratch/10000/eguha3/dc-agent

harbor jobs start \
    -p "$SUBSET_DIR" \
    --n-concurrent 8 \
    --agent terminus-2 \
    --model "$MODEL" \
    --env "daytona" \
    --n-attempts 1 \
    --max-retries 0 \
    --job-name "$JOB_NAME" \
    --config "/scratch/10000/eguha3/dc-agent/eval/tacc/dcagent_eval_config.yaml" \
    2>&1

# Parse results
echo ""
echo "=== Results ==="
RESULT_FILE="jobs/$JOB_NAME/result.json"
if [ -f "$RESULT_FILE" ]; then
    python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
results = data.get('results', [])
passed = sum(1 for r in results if r.get('reward', 0) > 0)
total = len(results)
rate = (passed / total * 100) if total > 0 else 0
print(f'Dataset: $DATASET')
print(f'Tasks tested: {total}')
print(f'Passed: {passed}')
print(f'Pass rate: {rate:.1f}%')
"
else
    echo "Could not find results at $RESULT_FILE"
fi

# Cleanup
rm -rf "$SUBSET_DIR"

echo "============================================"
