#!/bin/bash
# Test trace export script

JOB_DIR="${1:-jobs/staqc_1000_4node_tp4_pp1_20251214_050122}"
HF_REPO_ID="${2:-DCAgent/staqc_1000_expert_traces}"
SUCCESS_FILTER="${3:-}"  # Empty = None, or "success" or "failure"

echo "Testing trace export..."
echo "Job dir: $JOB_DIR"
echo "Repo ID: $HF_REPO_ID"
echo "Success filter: ${SUCCESS_FILTER:-None}"

/scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5/bin/python3 -c "
import sys
sys.path.insert(0, '${DCAGENT_DIR}')
from scripts.harbor.run_and_export_traces import only_export_traces

success_filter = '${SUCCESS_FILTER}' if '${SUCCESS_FILTER}' else None

ds = only_export_traces(
    job_dir='${JOB_DIR}',
    push=False,
    repo_id='${HF_REPO_ID}',
    success_filter=success_filter,
    verbose=True
)
print(f'Result: {ds}')
if hasattr(ds, 'num_rows'):
    print(f'Rows: {ds.num_rows}')
"
