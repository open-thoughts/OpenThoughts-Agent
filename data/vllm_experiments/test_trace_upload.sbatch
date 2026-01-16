#!/bin/bash
#SBATCH -p gh
#SBATCH --time=00:10:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=16
#SBATCH --account CCR24067
#SBATCH --output=${DCAGENT_DIR}/data/vllm_experiments/logs/test_trace_upload_%j.out
#SBATCH --job-name=test_upload

set -eo pipefail

echo "============================================"
echo "Testing Trace Export/Upload"
echo "============================================"

# Environment Setup
module purge
module load gcc/15.1.0

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5

# Source HF token
source /scratch/10000/eguha3/old-dc-agent/secret.env

cd ${DCAGENT_DIR}/data/vllm_experiments

# Use an existing job directory for testing
JOB_DIR="jobs/staqc_10_4node_tp4_pp1_20251214_214233"
HF_REPO_ID="DCAgent/staqc_10_expert_traces_test"

echo "Job directory: $JOB_DIR"
echo "HF Repo: $HF_REPO_ID"
echo ""

if [ -d "$JOB_DIR" ]; then
    echo "Found job directory, starting trace export..."

    # TRANSFORMERS_NO_TORCH=1 bypasses torch import issues on GH200 nodes
    TRANSFORMERS_NO_TORCH=1 /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5/bin/python3 -c "
import sys
import os
from pathlib import Path

from harbor.utils.traces_utils import (
    iter_trial_dirs, load_run_metadata, collect_conversations_from_trial
)
from datasets import Dataset

def sanitize_surrogates(obj):
    if isinstance(obj, str):
        return obj.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
    elif isinstance(obj, dict):
        return {k: sanitize_surrogates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_surrogates(v) for v in obj]
    return obj

job_dir = Path('${JOB_DIR}')
print(f'Job directory: {job_dir}')

rows = []
trial_dirs = list(iter_trial_dirs(job_dir, recursive=True))
print(f'Found {len(trial_dirs)} trial directories')

for trial_dir in trial_dirs:
    try:
        run_meta = load_run_metadata(trial_dir)
        convs = collect_conversations_from_trial(
            trial_dir,
            run_meta=run_meta,
            episodes='last',
            verbose=False
        )
        for conv in convs:
            rows.append(sanitize_surrogates(conv))
    except Exception as e:
        print(f'Skipping {trial_dir.name}: {e}')
        continue

print(f'Collected {len(rows)} rows')

if rows:
    ds = Dataset.from_list(rows)
    print(f'Pushing {len(ds)} traces to ${HF_REPO_ID}...')
    token = os.getenv('HF_TOKEN')
    ds.push_to_hub('${HF_REPO_ID}', token=token)
    print(f'SUCCESS: Exported {len(ds)} traces to ${HF_REPO_ID}')
else:
    print('No traces to export')
"
    EXPORT_EXIT=$?
    if [ $EXPORT_EXIT -eq 0 ]; then
        echo ""
        echo "============================================"
        echo "TRACE UPLOAD TEST PASSED!"
        echo "============================================"
    else
        echo ""
        echo "============================================"
        echo "TRACE UPLOAD TEST FAILED (exit code: $EXPORT_EXIT)"
        echo "============================================"
    fi
else
    echo "ERROR: Job directory not found: $JOB_DIR"
    exit 1
fi
