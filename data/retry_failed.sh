#!/bin/bash
set -o pipefail
source data/secrets.env 2>/dev/null
source ../old-dc-agent/secret.env 2>/dev/null
export PYTHONPATH=/scratch/10000/eguha3/dc-agent:${PYTHONPATH:-}

FAILED_SCRIPTS=(
    "data/exp_1_1_difficulty_mixing/generate_goldilocks.py"
    "data/exp_1_1_difficulty_mixing/generate_70_20_10.py"
    "data/exp_1_1_difficulty_mixing/generate_50_30_20.py"
    "data/exp_1_1_difficulty_mixing/generate_equal_bands.py"
    "data/exp_1_3_test_quality/generate_top50.py"
    "data/exp_1_3_test_quality/generate_top25.py"
    "data/exp_4_3_skill_distillation/generate_curated_1k.py"
    "data/exp_5_1_consistency_verification/generate_filtered_solvable.py"
    "data/exp_7_1_dockerfile_curriculum/generate_d1_full.py"
    "data/exp_7_1_dockerfile_curriculum/generate_d3_some.py"
    "data/exp_7_5_staged_rewards/generate_staged.py"
)

LOG_DIR="data/nontest_logs"
mkdir -p "$LOG_DIR"

SUCCESS=0
FAIL=0

for script in "${FAILED_SCRIPTS[@]}"; do
    name=$(basename "$script" .py)
    echo "$(date +%H:%M:%S) Running SEQUENTIALLY: $script"
    if python "$script" > "$LOG_DIR/${name}_retry.log" 2>&1; then
        echo "  OK: $script"
        ((SUCCESS++))
    else
        echo "  FAIL: $script (see $LOG_DIR/${name}_retry.log)"
        ((FAIL++))
    fi
done

echo ""
echo "Retry results: $SUCCESS succeeded, $FAIL failed out of ${#FAILED_SCRIPTS[@]}"
