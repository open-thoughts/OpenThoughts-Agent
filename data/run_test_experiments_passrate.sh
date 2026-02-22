#!/bin/bash
# Run test-generating experiments: generate 25 tasks each, then run through harbor
# to check pass rate with gpt-5-nano.
set -e
cd "$(dirname "$0")/.."

MODEL="${1:-openai/gpt-5-nano-2025-08-07}"
N_TASKS=25

echo "============================================="
echo "Test-generating experiments: pass rate check"
echo "Model: $MODEL | Tasks per experiment: $N_TASKS"
echo "============================================="

# All test-generating experiment scripts (one representative per experiment)
declare -A SCRIPTS
SCRIPTS["8.4_constraint_ladder"]="data/exp_8_4_constraint_ladder/generate_1_constraint.py"
SCRIPTS["8.7_ambiguity"]="data/exp_8_7_ambiguity/generate_partial.py"
SCRIPTS["8.8_domain_shift"]="data/exp_8_8_domain_shift/generate_medical.py"
SCRIPTS["8.9_task_stacking"]="data/exp_8_9_task_stacking/generate_pairs.py"
SCRIPTS["5.2_adversarial_tests"]="data/exp_5_2_adversarial_tests/generate_adversarial.py"
SCRIPTS["5.3_test_synthesis"]="data/exp_5_3_test_synthesis/generate_synthesized_tests.py"
SCRIPTS["5.4_compositional_2skill"]="data/exp_5_4_compositional_tasks/generate_2skill.py"
SCRIPTS["7.3_cross_lang_rust"]="data/exp_7_3_cross_language/generate_rust.py"
SCRIPTS["7.4_test_first"]="data/exp_7_4_test_first/generate_pytest_reverse.py"

CONFIG_PATH="eval/tacc/dcagent_eval_config.yaml"
RESULTS_FILE="data/test_gen_passrate_results.json"

echo "[]" > "$RESULTS_FILE"

for name in $(echo "${!SCRIPTS[@]}" | tr ' ' '\n' | sort); do
    script="${SCRIPTS[$name]}"
    echo ""
    echo "============================================="
    echo "Experiment: $name"
    echo "Script: $script"
    echo "============================================="

    # Step 1: Generate 25 tasks
    # We override LIMIT by setting it as env var or monkey-patching
    # For now, generate and then take first 25 task dirs
    echo "--- Step 1: Generating tasks ---"
    TASK_OUTPUT=$(python -c "
import sys, os, tempfile, json, shutil
from pathlib import Path

# Override limit to 25 for validation
os.environ['EXP_LIMIT'] = '30'  # generate a few extra

# Run the script's main() with reduced limit
sys.path.insert(0, '.')
script_path = '$script'
# We can't easily override LIMIT in the script, so we'll generate
# and take first $N_TASKS task dirs from the output
print('Generating...')
" 2>&1 || echo "SKIP")

    # Actually run the script - it will generate to a temp dir and upload
    # For validation, we run and check the local output
    echo "--- Running generate script ---"
    python "$script" 2>&1 | tee "/tmp/exp_${name}_log.txt" &
    GEN_PID=$!

    # Wait for generation (with timeout)
    TIMEOUT=1800  # 30 min
    if ! timeout $TIMEOUT wait $GEN_PID 2>/dev/null; then
        echo "TIMEOUT after ${TIMEOUT}s for $name"
        kill $GEN_PID 2>/dev/null || true
        continue
    fi

    # Find the output directory (last line with temp dir path)
    OUTPUT_DIR=$(grep -oP '/tmp/exp-[^\s]+' "/tmp/exp_${name}_log.txt" | head -1)
    if [ -z "$OUTPUT_DIR" ]; then
        echo "Could not find output directory for $name"
        continue
    fi

    # Count generated tasks
    TOTAL_GENERATED=$(find "$OUTPUT_DIR" -name "instruction.md" | wc -l)
    echo "Generated $TOTAL_GENERATED tasks"

    if [ "$TOTAL_GENERATED" -eq 0 ]; then
        echo "No tasks generated for $name — skipping harbor"
        continue
    fi

    # Create subset of first N_TASKS
    SUBSET_DIR=$(mktemp -d -t "harbor_${name}_XXXXX")
    TASK_DIRS=($(find "$OUTPUT_DIR" -name "instruction.md" -exec dirname {} \; | sort | head -$N_TASKS))
    for td in "${TASK_DIRS[@]}"; do
        cp -r "$td" "$SUBSET_DIR/"
    done
    SUBSET_COUNT=$(find "$SUBSET_DIR" -name "instruction.md" | wc -l)
    echo "Subset: $SUBSET_COUNT tasks in $SUBSET_DIR"

    # Step 2: Run through harbor
    echo "--- Step 2: Running harbor evaluation ---"
    JOB_NAME="passrate_${name}_$$"

    harbor jobs start \
        -p "$SUBSET_DIR" \
        --n-concurrent 8 \
        --agent terminus-2 \
        --model "$MODEL" \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$JOB_NAME" \
        --config "$CONFIG_PATH" \
        2>&1 | tee "/tmp/harbor_${name}_log.txt" || true

    # Step 3: Parse results
    echo "--- Step 3: Results ---"
    RESULT_JSON="jobs/${JOB_NAME}/result.json"
    if [ -f "$RESULT_JSON" ]; then
        python3 -c "
import json
with open('$RESULT_JSON') as f:
    data = json.load(f)
results = data.get('results', [])
passed = sum(1 for r in results if r.get('reward', 0) > 0)
total = len(results)
started = sum(1 for r in results if 'reward' in r)
errors = data.get('exception_stats', {})
n_errors = sum(len(v) for v in errors.values()) if isinstance(errors, dict) else 0
rate = (passed / total * 100) if total > 0 else 0
print(f'  Experiment: $name')
print(f'  Started: {started}/{total}')
print(f'  Passed: {passed}/{total}')
print(f'  Pass rate: {rate:.1f}%')
print(f'  Errors: {n_errors}')

# Append to results file
import os
results_file = '$RESULTS_FILE'
with open(results_file) as f:
    all_results = json.load(f)
all_results.append({
    'experiment': '$name',
    'started': started,
    'passed': passed,
    'total': total,
    'pass_rate': rate,
    'errors': n_errors,
})
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)
"
    else
        echo "  No result.json found at $RESULT_JSON"
    fi

    # Cleanup
    rm -rf "$SUBSET_DIR"
done

echo ""
echo "============================================="
echo "FINAL SUMMARY"
echo "============================================="
python3 -c "
import json
with open('$RESULTS_FILE') as f:
    results = json.load(f)
print(f'{'Experiment':<35} {'Started':>8} {'Passed':>8} {'Rate':>8} {'Errors':>8}')
print('-' * 75)
for r in results:
    print(f'{r[\"experiment\"]:<35} {r[\"started\"]:>8} {r[\"passed\"]:>8} {r[\"pass_rate\"]:>7.1f}% {r[\"errors\"]:>8}')
"
