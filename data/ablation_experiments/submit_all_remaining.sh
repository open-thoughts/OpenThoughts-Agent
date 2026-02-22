#!/bin/bash
# Submit all remaining experiments as fresh jobs
# Generated 2026-01-03
# Criteria: not running/queued, not found on HF, or <9000 datapoints
# Currently running/queued (SKIP): temp_2.0, temp_4.0, top_p_0.8

SBATCH_DIR="/scratch/10000/eguha3/dc-agent/data/ablation_experiments/sbatch"

echo "=== Submitting All Remaining Experiments (Fresh) ==="
echo ""

# Experiments to submit fresh
# - Not found on HF: top_p_0.9, trajectory_minimal
# - Empty on HF: repetition_penalty_1.2
# - <9000 rows: temp_0.25 (540), temp_0.5 (225), tmux_large (3070), top_k_32 (790), summarize_threshold_4096 (1610)
FRESH_EXPERIMENTS=(
    "temp_0.25"                # 540 rows
    "temp_0.5"                 # 225 rows
    "tmux_large"               # 3,070 rows
    "top_k_32"                 # 790 rows
    "summarize_threshold_4096" # 1,610 rows
    "top_p_0.9"                # not found on HF
    "trajectory_minimal"       # not found on HF
    "repetition_penalty_1.2"   # empty dataset on HF
)

echo "Fresh experiments to submit (${#FRESH_EXPERIMENTS[@]}):"
for exp in "${FRESH_EXPERIMENTS[@]}"; do
    echo "  - $exp"
done
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "=== Submitting Fresh Jobs ==="
for exp in "${FRESH_EXPERIMENTS[@]}"; do
    sbatch_file="${SBATCH_DIR}/${exp}.sbatch"
    if [ -f "$sbatch_file" ]; then
        echo "Submitting fresh: $exp"
        sbatch "$sbatch_file"
        sleep 2
    else
        echo "MISSING: $sbatch_file"
    fi
done

echo ""
echo "=== All remaining experiments submitted ==="
