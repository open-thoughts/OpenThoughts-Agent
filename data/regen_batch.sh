#!/bin/bash
# Regenerate datasets and test with harbor (batch runner)
# Usage: bash data/regen_batch.sh

set -e
source /scratch/10000/eguha3/old-dc-agent/secret.env

LIMIT=${1:-25}
RESULTS_DIR="data/regen_results"
mkdir -p "$RESULTS_DIR"

# Function to regenerate a dataset and run harbor
regen_and_test() {
    local name="$1"
    local generator="$2"
    shift 2
    local args=("$@")

    echo "============================================"
    echo "[$(date +%H:%M:%S)] Starting: $name"
    echo "============================================"

    # Run generator
    echo "  Generating $LIMIT tasks..."
    local output
    output=$(python3 "$generator" --limit "$LIMIT" "${args[@]}" 2>&1)
    echo "$output" > "$RESULTS_DIR/${name}_gen.log"

    # Extract task directory from output
    local task_dir
    task_dir=$(echo "$output" | grep -oP '(?:Task directory|Output directory|Generating harbor tasks in): \K\S+' | tail -1)
    if [ -z "$task_dir" ]; then
        task_dir=$(echo "$output" | grep -oP '/\S*tasks_\S+' | tail -1)
    fi

    if [ -z "$task_dir" ] || [ ! -d "$task_dir" ]; then
        echo "  FAILED: Could not find task directory"
        echo '{"status":"GENERATE_FAILED"}' > "$RESULTS_DIR/${name}.json"
        return 1
    fi

    local task_count
    task_count=$(find "$task_dir" -maxdepth 1 -mindepth 1 -type d | wc -l)
    echo "  Generated $task_count tasks in $task_dir"

    # Run harbor
    local job_name="regen_${name}"
    echo "  Running harbor (job: $job_name)..."
    harbor jobs start -p "$task_dir" \
        --n-concurrent 5 \
        --agent terminus-2 \
        --model openai/gpt-5-nano-2025-08-07 \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$job_name" \
        --config eval/tacc/dcagent_eval_config.yaml \
        > "$RESULTS_DIR/${name}_harbor.log" 2>&1

    # Parse results
    local result_file="jobs/${job_name}/result.json"
    if [ -f "$result_file" ]; then
        local pass_rate
        pass_rate=$(python3 -c "
import json
with open('$result_file') as f:
    data = json.load(f)
stats = data.get('stats', {})
for k, v in stats.get('evals', {}).items():
    metrics = v.get('metrics', [])
    if metrics:
        print(f\"{metrics[0]['mean']*100:.0f}\")
        break
" 2>/dev/null)
        local n_errors
        n_errors=$(python3 -c "
import json
with open('$result_file') as f:
    data = json.load(f)
print(data.get('stats', {}).get('n_errors', '?'))
" 2>/dev/null)
        echo "  RESULT: pass_rate=${pass_rate}%, errors=${n_errors}"
        echo "{\"status\":\"done\",\"pass_rate\":${pass_rate},\"errors\":${n_errors},\"task_dir\":\"${task_dir}\",\"task_count\":${task_count}}" > "$RESULTS_DIR/${name}.json"
    else
        echo "  FAILED: No result.json"
        echo '{"status":"HARBOR_FAILED","task_dir":"'"$task_dir"'"}' > "$RESULTS_DIR/${name}.json"
    fi

    echo "[$(date +%H:%M:%S)] Done: $name"
    echo ""
}

# ===== NO-LLM DATASETS (fast) =====
echo "====== Phase 1: No-LLM datasets ======"

regen_and_test "bigcodebench" "data/dclm-mine/generate_bigcodebench_tasks.py" --no-upload
regen_and_test "codeelo" "data/dclm-mine/generate_codeelo_tasks.py" --no-upload
regen_and_test "exercism_python" "data/dclm-mine/generate_exercism_tasks.py" --language python
regen_and_test "swebench" "data/dclm-mine/generate_swebench_tasks.py" --no-upload
regen_and_test "manybugs" "data/dclm-mine/generate_manybugs_tasks.py" --no-upload
regen_and_test "crosscodeeval_python" "data/dclm-mine/generate_crosscodeeval_tasks.py" --no-upload --language python
regen_and_test "bugsinpy_mf" "data/dclm-mine/generate_bugsinpy_mf_tasks.py" --no-upload
regen_and_test "bugswarm" "data/dclm-mine/generate_bugswarm_tasks.py" --no-upload

# ===== LLM DATASETS =====
echo "====== Phase 2: LLM datasets ======"

regen_and_test "codenet_python" "data/dclm-mine/generate_codenet_tasks.py" --no-upload --language python
regen_and_test "codereval_python" "data/dclm-mine/generate_codereval_tasks.py" --language python
regen_and_test "defects4j" "data/dclm-mine/generate_defects4j_tasks.py"
regen_and_test "methods2test" "data/dclm-mine/generate_methods2test_tasks.py"
regen_and_test "bugsinpy" "data/dclm-mine/generate_bugsinpy_tasks.py"
regen_and_test "taco" "data/dclm-mine/generate_taco_tasks.py"
regen_and_test "unitsyn_python" "data/dclm-mine/generate_unitsyn_tasks.py" --no-upload --language python
regen_and_test "softwareheritage" "data/dclm-mine/generate_softwareheritage_tasks.py"
regen_and_test "e2egit" "data/dclm-mine/generate_e2egit_tasks.py" --no-upload
regen_and_test "ghactions" "data/dclm-mine/generate_ghactions_tasks.py" --skip-clone
regen_and_test "pymethods2test" "data/dclm-mine/generate_pymethods2test_tasks.py" --no-upload

# ===== SUMMARY =====
echo "====== FINAL SUMMARY ======"
for f in "$RESULTS_DIR"/*.json; do
    name=$(basename "$f" .json)
    if [ -f "$f" ]; then
        echo "  $name: $(cat "$f")"
    fi
done
echo "====== DONE ======"
