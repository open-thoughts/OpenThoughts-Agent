#!/bin/bash
# Run all remaining exp8 harbor tests in parallel
set -e
source /scratch/10000/eguha3/old-dc-agent/secret.env
cd /scratch/10000/eguha3/dc-agent

MODEL="openai/gpt-5-nano-2025-08-07"
CONFIG="eval/tacc/dcagent_eval_config.yaml"
NCONCURRENT=8
PID=$$

# Already completed (from previous sequential run)
DONE="8.1_github_issue 8.1_slack_message 8.1_stackoverflow 8.1_code_review 8.1_error_report"

# Read manifest and launch remaining experiments
python3 -c "
import json
manifest = json.load(open('data/exp8_test_manifest.json'))
done = set('$DONE'.split())
for name, path in sorted(manifest.items()):
    if name not in done:
        print(f'{name}|{path}')
" | while IFS='|' read name task_dir; do
    job_name="exp8_${name}_${PID}"
    echo "Launching: $name -> $job_name"
    harbor jobs start \
        -p "$task_dir" \
        --n-concurrent "$NCONCURRENT" \
        --agent terminus-2 \
        --model "$MODEL" \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$job_name" \
        --config "$CONFIG" &
done

echo "All experiments launched. Waiting for completion..."
wait
echo "All done!"
