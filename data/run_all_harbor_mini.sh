#!/bin/bash
# Run ALL experiments with gpt-5-mini for higher pass rates
set -e
source /scratch/10000/eguha3/old-dc-agent/secret.env
cd /scratch/10000/eguha3/dc-agent

MODEL="openai/gpt-5-mini-2025-08-07"
CONFIG="eval/tacc/dcagent_eval_config.yaml"
NCONCURRENT=8
PID=$$

echo "Launching ALL harbor tests with $MODEL (PID=$PID)"

# Theme 8 experiments (use latest manifest)
python3 -c "
import json
m8 = json.load(open('data/exp8_test_manifest.json'))
for name, path in sorted(m8.items()):
    print(f'mini8_{name}_{PID}|{path}')
mo = json.load(open('data/other_themes_manifest.json'))
for name, path in sorted(mo.items()):
    print(f'mini_{name}_{PID}|{path}')
" | while IFS='|' read job_name task_dir; do
    echo "Launching: $job_name"
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

echo "All experiments launched. Waiting..."
wait
echo "All done! (PID=$PID)"
