#!/bin/bash
# V4: Re-test experiments with infrastructure fixes
# - 8.8_devops, 8.8_ecommerce: regenerated (test syntax errors fixed)
# - 8.4_1constraint, 8.4_2constraints, 8.4_3constraints, 8.4_4constraints: pytest import fixed
# - 5.4_2skill: regenerated (test syntax error fixed)
# - All: requests_mock added to whitelist

set -euo pipefail

source /scratch/10000/eguha3/old-dc-agent/secret.env

PROJECT_ROOT="/scratch/10000/eguha3/dc-agent"
CONFIG="$PROJECT_ROOT/eval/tacc/dcagent_eval_config.yaml"

MODEL_NANO="openai/gpt-5-nano-2025-08-07"
MODEL_MINI="openai/gpt-5-mini-2025-08-07"
MODEL_GPT5="openai/gpt-5-2025-08-07"
PID=$$

MANIFEST="/tmp/all_experiments_manifest.json"

# Experiments to retest
EXPERIMENTS=(
    "8.8_devops"
    "8.8_ecommerce"
    "8.4_1constraint"
    "8.4_2constraints"
    "8.4_3constraints"
    "8.4_4constraints"
    "5.4_2skill"
)

run_harbor() {
    local exp_name="$1"
    local model="$2"
    local model_short="$3"
    local task_dir="$4"
    local job_name="v4${model_short}_${exp_name}_${PID}"

    echo "[$(date +%H:%M:%S)] Starting $job_name"
    cd "$PROJECT_ROOT"
    harbor jobs start \
        -p "$task_dir" \
        --n-concurrent 8 \
        --agent terminus-2 \
        --model "$model" \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$job_name" \
        --config "$CONFIG" \
        2>&1 | tail -3
    echo "[$(date +%H:%M:%S)] Finished $job_name"
}

BATCH_SIZE=4
count=0

for exp_name in "${EXPERIMENTS[@]}"; do
    task_dir=$(python3 -c "import json; m=json.load(open('$MANIFEST')); print(m.get('$exp_name', ''))")
    if [ -z "$task_dir" ] || [ ! -d "$task_dir" ]; then
        echo "SKIP $exp_name: no task directory"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "EXPERIMENT: $exp_name ($task_dir)"
    echo "=========================================="

    # Run all 3 models in parallel for this experiment
    run_harbor "$exp_name" "$MODEL_NANO" "nano" "$task_dir" &
    run_harbor "$exp_name" "$MODEL_MINI" "mini" "$task_dir" &
    run_harbor "$exp_name" "$MODEL_GPT5" "gpt5" "$task_dir" &
    count=$((count + 3))

    # Wait if we've launched enough
    if [ $count -ge $((BATCH_SIZE * 3)) ]; then
        echo "Waiting for batch to complete..."
        wait
        count=0
    fi
done

echo ""
echo "Waiting for remaining jobs..."
wait

echo ""
echo "=========================================="
echo "ALL V4 JOBS COMPLETE"
echo "=========================================="
echo "Job prefix: v4{nano,mini,gpt5}_*_${PID}"
echo "Check results in: $PROJECT_ROOT/jobs/"
