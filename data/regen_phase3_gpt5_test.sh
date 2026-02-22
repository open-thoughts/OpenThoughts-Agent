#!/bin/bash
# Phase 3: Test remaining low-pass-rate datasets with gpt-5 (stronger model)
# to determine if low rates are model weakness vs generator bugs

set -e
source /scratch/10000/eguha3/old-dc-agent/secret.env
export SKIP_HF_UPLOAD=1

cd /scratch/10000/eguha3/dc-agent

DATASETS=(
    "codenet_python:/scratch/10000/eguha3/data_cache/codenet-python_tasks_n4mlvq3n"
    "codereval_python:/scratch/10000/eguha3/data_cache/codereval-python_tasks_9sfene37"
    "e2egit:/scratch/10000/eguha3/data_cache/e2egit_tasks_1i2pmycr"
    "ghactions:/scratch/10000/eguha3/data_cache/ghactions_tasks_pkodddcv"
    "pymethods2test:/scratch/10000/eguha3/data_cache/pymethods2test_tasks_w3rpt4n7"
    "unitsyn_python:/scratch/10000/eguha3/data_cache/unitsyn-python_tasks_irq83dz9"
    "taco:/scratch/10000/eguha3/data_cache/taco_tasks_fnyabxiz"
    "exercism_python:/scratch/10000/eguha3/data_cache/exercism-python_tasks_6m4c0da1"
    "defects4j:/scratch/10000/eguha3/data_cache/defects4j_tasks_p02t2u2g"
    "manybugs:/scratch/10000/eguha3/data_cache/manybugs_tasks_gcsoiimi"
    "bugsinpy:/scratch/10000/eguha3/data_cache/bugsinpy_tasks_ah6llkei"
)

MODEL="openai/gpt-5-2025-08-07"

for entry in "${DATASETS[@]}"; do
    name="${entry%%:*}"
    task_dir="${entry##*:}"
    job_name="gpt5_regen_${name}"

    # Skip if task dir doesn't exist
    if [ ! -d "$task_dir" ]; then
        echo "SKIP: $name - task dir not found: $task_dir"
        continue
    fi

    # Remove old job dir if exists
    if [ -d "jobs/$job_name" ]; then
        rm -rf "jobs/$job_name"
        echo "Cleaned up old job dir: jobs/$job_name"
    fi

    task_count=$(ls -d "$task_dir"/*/ 2>/dev/null | wc -l)
    echo "Launching $name ($task_count tasks) with $MODEL..."

    harbor jobs start \
        -p "$task_dir" \
        --n-concurrent 5 \
        --agent terminus-2 \
        --model "$MODEL" \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$job_name" \
        --config eval/tacc/dcagent_eval_config.yaml &

    echo "  Started: $job_name (PID: $!)"
done

echo ""
echo "All gpt-5 harbor jobs launched in parallel. Waiting..."
wait
echo "All jobs completed!"

# Collect results
echo ""
echo "=== GPT-5 Results ==="
for entry in "${DATASETS[@]}"; do
    name="${entry%%:*}"
    job_name="gpt5_regen_${name}"
    result_file="jobs/$job_name/result.json"
    if [ -f "$result_file" ]; then
        mean=$(python3 -c "
import json
with open('$result_file') as f:
    d = json.load(f)
for k,v in d['stats']['evals'].items():
    print(f'{v[\"metrics\"][0][\"mean\"]:.0%}')
    break
" 2>/dev/null || echo "ERROR")
        echo "  $name: $mean"
    else
        echo "  $name: NO RESULT"
    fi
done
