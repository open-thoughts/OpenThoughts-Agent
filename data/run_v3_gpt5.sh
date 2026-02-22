#!/bin/bash
set -u
source /scratch/10000/eguha3/old-dc-agent/secret.env
cd /scratch/10000/eguha3/dc-agent

MODEL_GPT5="openai/gpt-5-2025-08-07"
CONFIG="eval/tacc/dcagent_eval_config.yaml"
PID=$$
BATCH_SIZE=4

echo "PID=$PID, BATCH_SIZE=$BATCH_SIZE, MODEL=$MODEL_GPT5"

launch() {
    local name="$1"
    local task_dir="$2"
    local job_name="v3gpt5_${name}_${PID}"
    harbor jobs start \
        -p "$task_dir" \
        --n-concurrent 5 \
        --agent terminus-2 \
        --model "$MODEL_GPT5" \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$job_name" \
        --config "$CONFIG"
}

# Read full manifest (all 46 experiments)
EXPERIMENTS=$(python3 -c "
import json
manifest = json.load(open('/tmp/all_experiments_manifest.json'))
for name, task_dir in sorted(manifest.items()):
    print(f'{name}\t{task_dir}')
")

batch_count=0
total=0
while IFS=$'	' read -r name task_dir; do
    echo "$(date +%H:%M:%S) Launching: $name"
    launch "$name" "$task_dir" &
    batch_count=$((batch_count + 1))
    total=$((total + 1))

    if [ "$batch_count" -ge "$BATCH_SIZE" ]; then
        echo "$(date +%H:%M:%S) Waiting for batch of $batch_count experiments ($total total)..."
        wait
        echo "$(date +%H:%M:%S) Batch done"
        batch_count=0
    fi
done <<< "$EXPERIMENTS"

if [ "$batch_count" -gt 0 ]; then
    echo "$(date +%H:%M:%S) Waiting for final batch ($batch_count experiments)..."
    wait
fi

echo "$(date +%H:%M:%S) ALL DONE - $total experiments completed"
