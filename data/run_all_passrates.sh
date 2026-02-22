#!/bin/bash
# Run pass rate tests for all Stack-based datasets

set -e

cd /scratch/10000/eguha3/dc-agent

# Setup environment
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6
source /scratch/10000/eguha3/old-dc-agent/secret.env

export PYTHONPATH="/scratch/10000/eguha3/dc-agent/scripts/harbor:$PYTHONPATH"

# All Stack-based datasets (17 total)
DATASETS=(
    "DCAgent/exp_rpt_stack-bash"
    "DCAgent/exp_rpt_stack-bash-withtests"
    "DCAgent/exp_rpt_stack-dockerfile"
    "DCAgent/exp_rpt_stack-ruby"
    "DCAgent/exp_rpt_stack-go"
    "DCAgent/exp_rpt_stack-php"
    "DCAgent/exp_rpt_stack-pytest"
    "DCAgent/exp_rpt_stack-pytest-withtests"
    "DCAgent/exp_rpt_stack-junit"
    "DCAgent/exp_rpt_stack-selfdoc"
    "DCAgent/exp_rpt_stack-csharp"
    "DCAgent/exp_rpt_stack-rust"
    "DCAgent/exp_rpt_stack-selfdoc-gpt4o"
    "DCAgent/exp_rpt_stack-pytest-synthetic-gpt5nano"
    "DCAgent/exp_rpt_stack-dockerfile-gpt4o"
    "DCAgent/exp_rpt_stack-cpp"
    "DCAgent/exp_rpt_stack-jest"
)

for ds in "${DATASETS[@]}"; do
    DATASET_NAME="${ds##*/}"
    LOG_FILE="data/passrate_logs/${DATASET_NAME}.log"

    # Skip if already has pass rate result
    if grep -q "Pass rate:" "$LOG_FILE" 2>/dev/null; then
        rate=$(grep "Pass rate:" "$LOG_FILE" | tail -1)
        echo "SKIP: $DATASET_NAME already tested - $rate"
        continue
    fi

    echo ""
    echo "================================================"
    echo "Testing: $ds"
    echo "================================================"
    bash data/run_passrate_tests.sh "$ds"
done

echo ""
echo "================================================"
echo "ALL TESTS COMPLETE - Summary:"
echo "================================================"
for log in data/passrate_logs/*.log; do
    name=$(basename "$log" .log)
    if grep -q "Pass rate:" "$log" 2>/dev/null; then
        rate=$(grep "Pass rate:" "$log" | tail -1)
        echo "$name: $rate"
    else
        echo "$name: INCOMPLETE"
    fi
done
