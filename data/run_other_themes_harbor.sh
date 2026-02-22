#!/bin/bash
# Run harbor pass rate tests for all other-theme experiments in parallel
set -e
source /scratch/10000/eguha3/old-dc-agent/secret.env
cd /scratch/10000/eguha3/dc-agent

MODEL="openai/gpt-5-nano-2025-08-07"
CONFIG="eval/tacc/dcagent_eval_config.yaml"
NCONCURRENT=8
PID=$$

echo "Launching harbor tests for other themes (PID=$PID)"

python3 -c "
import json
manifest = json.load(open('data/other_themes_manifest.json'))
for name, path in sorted(manifest.items()):
    print(f'{name}|{path}')
" | while IFS='|' read name task_dir; do
    job_name="other_${name}_${PID}"
    echo "Launching: $name -> $job_name ($task_dir)"
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
echo "All done! (PID=$PID)"
