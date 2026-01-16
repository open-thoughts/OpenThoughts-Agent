#!/bin/bash
# Export and upload expert traces from a harbor job directory
#
# Usage: ./export_traces.sh <job_dir> [repo_id] [success_filter]
#
# Arguments:
#   job_dir        - Path to the job directory (e.g., jobs/staqc_1000_4node_tp4_pp1_20251214_050122)
#   repo_id        - HuggingFace repo ID (default: DCAgent/staqc_1000_expert_traces)
#   success_filter - Filter: "success", "failure", or empty for all (default: success)
#
# Examples:
#   ./export_traces.sh jobs/staqc_1000_4node_tp4_pp1_20251214_050122
#   ./export_traces.sh jobs/staqc_1000_4node_tp4_pp1_20251214_050122 DCAgent/my_traces
#   ./export_traces.sh jobs/staqc_1000_4node_tp4_pp1_20251214_050122 DCAgent/my_traces ""

set -eo pipefail

JOB_DIR="${1:-}"
HF_REPO_ID="${2:-DCAgent/staqc_1000_expert_traces}"
SUCCESS_FILTER="${3:-None}"

if [ -z "$JOB_DIR" ]; then
    echo "Usage: $0 <job_dir> [repo_id] [success_filter]"
    echo ""
    echo "Arguments:"
    echo "  job_dir        - Path to the job directory"
    echo "  repo_id        - HuggingFace repo ID (default: DCAgent/staqc_1000_expert_traces)"
    echo "  success_filter - Filter: 'success', 'failure', or empty for all (default: success)"
    exit 1
fi

if [ ! -d "$JOB_DIR" ]; then
    echo "ERROR: Job directory not found: $JOB_DIR"
    exit 1
fi

echo "============================================"
echo "Exporting Expert Traces"
echo "============================================"
echo "Job directory: $JOB_DIR"
echo "HuggingFace repo: $HF_REPO_ID"
echo "Success filter: ${SUCCESS_FILTER:-None (all)}"
echo "============================================"

# Load environment
source /scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/secret.env 2>/dev/null || true

/scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5/bin/python3 -c "
import sys
import os
from pathlib import Path
sys.path.insert(0, '${DCAGENT_DIR}')

# Import harbor utilities directly to collect rows before dataset creation
from harbor.utils.traces_utils import (
    iter_trial_dirs, load_run_metadata, collect_conversations_from_trial,
    _trial_is_success
)
from datasets import Dataset

def sanitize_surrogates(obj):
    '''Remove invalid surrogate characters from strings recursively.'''
    if isinstance(obj, str):
        return obj.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
    elif isinstance(obj, dict):
        return {k: sanitize_surrogates(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_surrogates(v) for v in obj]
    return obj

success_filter = None if '${SUCCESS_FILTER}' in ('None', 'none', '') else '${SUCCESS_FILTER}'
job_dir = Path('${JOB_DIR}')

print('Starting export...')
print(f'Job directory: {job_dir}')

# Collect rows manually so we can sanitize before Dataset creation
rows = []
trial_dirs = list(iter_trial_dirs(job_dir, recursive=True))
print(f'[traces] Found {len(trial_dirs)} trial directories')

for trial_dir in trial_dirs:
    try:
        run_meta = load_run_metadata(trial_dir)

        # Apply success filter if specified
        if success_filter in ('success', 'failure'):
            succ = _trial_is_success(trial_dir, run_meta)
            if succ is None:
                continue
            if success_filter == 'success' and not succ:
                continue
            if success_filter == 'failure' and succ:
                continue

        convs = collect_conversations_from_trial(
            trial_dir,
            run_meta=run_meta,
            episodes='last',
            verbose=False
        )
        # Sanitize each conversation immediately
        for conv in convs:
            rows.append(sanitize_surrogates(conv))
    except Exception as e:
        print(f'[traces] Skipping {trial_dir.name}: {e}')
        continue

print(f'[traces] Collected {len(rows)} rows')

if rows:
    # Create dataset from sanitized rows
    ds = Dataset.from_list(rows)

    # Push to hub
    print(f'Pushing {len(ds)} traces to ${HF_REPO_ID}...')
    token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
    ds.push_to_hub('${HF_REPO_ID}', token=token)
    print(f'Exported {len(ds)} traces to ${HF_REPO_ID}')
else:
    print('No traces to export')
"

echo ""
echo "============================================"
echo "Export complete!"
echo "============================================"
