#!/bin/bash
#SBATCH -p gg
#SBATCH --time=4:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --account CCR24067
#SBATCH --output=${DCAGENT_DIR}/data/sbatches/logs/benchmark_pass_rates_%j.out
#SBATCH --job-name=benchmark_pass_rates

set -eo pipefail

MODEL="${1:-openai/gpt-4o-mini}"
BENCHMARK_DIR="${DCAGENT_DIR}/data/benchmark_tasks"

LOGS_DIR="${DCAGENT_DIR}/data/sbatches/logs"
HARBOR_CONFIG="${DCAGENT_DIR}/eval/tacc/dcagent_eval_config.yaml"
RESULTS_FILE="${LOGS_DIR}/benchmark_results_${SLURM_JOB_ID}.json"

echo "============================================"
echo "Benchmark Pass Rate Evaluation"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $MODEL"
echo "Benchmark Dir: $BENCHMARK_DIR"
echo "============================================"

mkdir -p "$LOGS_DIR"

# Environment Setup
echo ""
echo "=== Step 1: Environment Setup ==="

module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

export HYDRA_FULL_ERROR=1
source /scratch/10000/eguha3/old-dc-agent/secret.env

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6

harbor --help >/dev/null
echo "Harbor is ready"

# Initialize results
echo "{}" > "$RESULTS_FILE"

# Run each tier
TIERS=("tier_d_exercism" "tier_c_algorithms" "tier_b_bugfix" "tier_a_hard" "tier_s_research")

for TIER in "${TIERS[@]}"; do
    echo ""
    echo "============================================"
    echo "Running: $TIER"
    echo "============================================"

    TIER_PATH="$BENCHMARK_DIR/$TIER"
    if [ ! -d "$TIER_PATH" ]; then
        echo "Warning: $TIER_PATH does not exist, skipping"
        continue
    fi

    TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
    RUN_TAG="${TIER}_${TIMESTAMP}"
    TIER_LOG="${LOGS_DIR}/harbor_${TIER}_${SLURM_JOB_ID}.log"

    echo "  Path: $TIER_PATH"
    echo "  Model: $MODEL"
    echo "  Log: $TIER_LOG"

    # Run harbor on this tier
    harbor jobs start \
        -p "$TIER_PATH" \
        --n-concurrent 32 \
        --agent terminus-2 \
        --model "$MODEL" \
        --env "daytona" \
        --n-attempts 1 \
        --max-retries 1 \
        --job-name "$RUN_TAG" \
        --config "$HARBOR_CONFIG" \
        2>&1 | tee "$TIER_LOG" || true

    # Parse results
    JOB_DIR=$(grep "Results written to" "$TIER_LOG" | sed 's/.*Results written to \(jobs\/[^/]*\).*/\1/' | head -1)

    if [ -n "$JOB_DIR" ] && [ -f "$JOB_DIR/result.json" ]; then
        echo "Parsing results from: $JOB_DIR/result.json"

        # Count pass/fail
        PASSED=$(jq '[.[] | select(.reward == 1)] | length' "$JOB_DIR/result.json" 2>/dev/null || echo "0")
        TOTAL=$(jq 'length' "$JOB_DIR/result.json" 2>/dev/null || echo "0")

        if [ "$TOTAL" -gt 0 ]; then
            PASS_RATE=$(echo "scale=2; $PASSED * 100 / $TOTAL" | bc)
        else
            PASS_RATE="N/A"
        fi

        echo ""
        echo "  $TIER Results:"
        echo "    Passed: $PASSED / $TOTAL"
        echo "    Pass Rate: ${PASS_RATE}%"

        # Update results JSON
        jq --arg tier "$TIER" --arg passed "$PASSED" --arg total "$TOTAL" --arg rate "$PASS_RATE" \
           '. + {($tier): {"passed": ($passed|tonumber), "total": ($total|tonumber), "pass_rate": $rate}}' \
           "$RESULTS_FILE" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
    else
        echo "  Warning: Could not find results for $TIER"
    fi
done

conda deactivate || true

# Print summary
echo ""
echo "============================================"
echo "FINAL SUMMARY"
echo "============================================"
echo "Model: $MODEL"
echo ""
cat "$RESULTS_FILE" | jq '.'
echo ""
echo "Results saved to: $RESULTS_FILE"
echo "============================================"
