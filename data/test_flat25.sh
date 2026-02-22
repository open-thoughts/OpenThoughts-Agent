#!/bin/bash
# Test all flat-60% experiments at 25 tasks with nano, mini, and GPT-5
set -u

source /scratch/10000/eguha3/old-dc-agent/secret.env 2>/dev/null
cd /scratch/10000/eguha3/dc-agent

CONFIG="eval/tacc/dcagent_eval_config.yaml"
MANIFEST="data/flat25_manifest.json"

# Read manifest
EXPERIMENTS=$(python3 -c "import json; d=json.load(open('$MANIFEST')); [print(f'{k}|{v}') for k,v in sorted(d.items())]")

MODEL="$1"  # nano, mini, or gpt5
case "$MODEL" in
    nano) MODEL_FULL="openai/gpt-5-nano-2025-08-07" ;;
    mini) MODEL_FULL="openai/gpt-5-mini-2025-08-07" ;;
    gpt5) MODEL_FULL="openai/gpt-5-2025-08-07" ;;
    *) echo "Usage: $0 {nano|mini|gpt5}"; exit 1 ;;
esac

echo "Running $MODEL on all flat25 experiments..."

while IFS='|' read -r name path; do
    JOB_NAME="flat25_${MODEL}_${name}_$$"
    echo "=== $name ($MODEL) ==="
    harbor jobs start \
        -p "$path" \
        --n-concurrent 8 \
        --agent terminus-2 \
        --model "$MODEL_FULL" \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$JOB_NAME" \
        --config "$CONFIG" 2>&1 | tail -3
    echo ""
done <<< "$EXPERIMENTS"

echo "All jobs complete for $MODEL"
