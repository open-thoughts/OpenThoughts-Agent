#!/bin/bash
# Submit new experiments that have never been started
# Generated 2025-12-31

SBATCH_DIR="/scratch/10000/eguha3/dc-agent/data/ablation_experiments/sbatch"

echo "=== Submitting New Experiments (never started) ==="
echo ""

# Experiments that have sbatches but no job directories
NEW_EXPERIMENTS=(
    # Summarization
    "summarize_threshold_4096"

    # Temperature experiments
    "temp_0.25"
    "temp_0.5"
    "temp_2.0"
    "temp_4.0"

    # Top-k experiments
    "top_k_16"
    "top_k_32"
    "top_k_64"
    "top_k_128"

    # Top-p experiments
    "top_p_0.8"
    "top_p_0.9"
    "top_p_0.95"

    # Other (uncomment if needed)
    # "tmux_large"
    # "trajectory_minimal"
    # "baseline"  # Already has 10K traces on HF
)

echo "Will submit ${#NEW_EXPERIMENTS[@]} experiments:"
for exp in "${NEW_EXPERIMENTS[@]}"; do
    echo "  - $exp"
done
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

for exp in "${NEW_EXPERIMENTS[@]}"; do
    sbatch_file="${SBATCH_DIR}/${exp}.sbatch"
    if [ -f "$sbatch_file" ]; then
        echo "Submitting: $exp"
        sbatch "$sbatch_file"
        sleep 2
    else
        echo "MISSING: $sbatch_file"
    fi
done

echo ""
echo "=== All new experiments submitted ==="
