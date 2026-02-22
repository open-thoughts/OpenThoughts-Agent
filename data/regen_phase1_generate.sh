#!/bin/bash
# Phase 1: Generate 25 tasks for each dataset (no harbor)
set -e
source /scratch/10000/eguha3/old-dc-agent/secret.env
export SKIP_HF_UPLOAD=1

LIMIT=${1:-25}
MANIFEST="data/regen_results/manifest.json"
mkdir -p data/regen_results

echo "{" > "$MANIFEST"
FIRST=true

generate() {
    local name="$1"
    local generator="$2"
    shift 2
    local args=("$@")

    echo "[$(date +%H:%M:%S)] Generating: $name"
    local output
    output=$(python3 "$generator" --limit "$LIMIT" "${args[@]}" 2>&1)
    echo "$output" > "data/regen_results/${name}_gen.log"

    # Extract task directory
    local task_dir
    task_dir=$(echo "$output" | grep -oP '(?:Task directory|Output directory|Generating harbor tasks in): \K\S+' | tail -1)
    if [ -z "$task_dir" ]; then
        task_dir=$(echo "$output" | grep -oP '/\S*tasks_\S+' | tail -1)
    fi

    local task_count=0
    if [ -n "$task_dir" ] && [ -d "$task_dir" ]; then
        task_count=$(find "$task_dir" -maxdepth 1 -mindepth 1 -type d | wc -l)
        echo "  -> $task_count tasks in $task_dir"
    else
        echo "  -> FAILED: no task dir found"
        task_dir="FAILED"
    fi

    # Write to manifest
    if [ "$FIRST" = true ]; then
        FIRST=false
    else
        echo "," >> "$MANIFEST"
    fi
    echo "  \"$name\": {\"task_dir\": \"$task_dir\", \"task_count\": $task_count}" >> "$MANIFEST"
}

# No-LLM datasets (fast)
generate "bigcodebench" "data/dclm-mine/generate_bigcodebench_tasks.py" --no-upload
generate "codeelo" "data/dclm-mine/generate_codeelo_tasks.py" --no-upload
generate "exercism_python" "data/dclm-mine/generate_exercism_tasks.py" --language python
generate "swebench" "data/dclm-mine/generate_swebench_tasks.py" --no-upload
generate "manybugs" "data/dclm-mine/generate_manybugs_tasks.py" --no-upload
generate "crosscodeeval_python" "data/dclm-mine/generate_crosscodeeval_tasks.py" --no-upload --language python
generate "bugsinpy_mf" "data/dclm-mine/generate_bugsinpy_mf_tasks.py" --no-upload
generate "bugswarm" "data/dclm-mine/generate_bugswarm_tasks.py" --no-upload

# LLM datasets
generate "codenet_python" "data/dclm-mine/generate_codenet_tasks.py" --no-upload --language python
generate "codereval_python" "data/dclm-mine/generate_codereval_tasks.py" --language python
generate "defects4j" "data/dclm-mine/generate_defects4j_tasks.py"
generate "methods2test" "data/dclm-mine/generate_methods2test_tasks.py"
generate "bugsinpy" "data/dclm-mine/generate_bugsinpy_tasks.py"
generate "taco" "data/dclm-mine/generate_taco_tasks.py"
generate "unitsyn_python" "data/dclm-mine/generate_unitsyn_tasks.py" --no-upload --language python
generate "softwareheritage" "data/dclm-mine/generate_softwareheritage_tasks.py"
generate "e2egit" "data/dclm-mine/generate_e2egit_tasks.py" --no-upload
generate "ghactions" "data/dclm-mine/generate_ghactions_tasks.py" --skip-clone
generate "pymethods2test" "data/dclm-mine/generate_pymethods2test_tasks.py" --no-upload

echo "" >> "$MANIFEST"
echo "}" >> "$MANIFEST"

echo ""
echo "====== Generation Complete ======"
echo "Manifest: $MANIFEST"
cat "$MANIFEST"
