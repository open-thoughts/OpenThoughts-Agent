#!/bin/bash
# Submit all resume sbatches for partial/incomplete experiments
# Generated 2025-12-31

RESUME_DIR="/scratch/10000/eguha3/dc-agent/data/ablation_experiments/resume_sbatch"

echo "=== Submitting Resume Jobs ==="
echo ""

# Experiments that need resume (have job dirs, partial traces)
RESUME_EXPERIMENTS=(
    # Very low traces - high priority
    "frequency_penalty_1.0"      # 127 traces
    "max_tokens_1024"            # 36 traces
    "high_diversity"             # 2,214 traces

    # Medium traces
    "max_tokens_2048"            # 4,879 traces
    "max_episodes_64"            # 5,070 traces
    "max_episodes_512"           # 5,084 traces
    "max_episodes_32"            # 5,633 traces
    "min_p_0.05"                 # 6,009 traces
    "min_p_0.01"                 # 6,437 traces
    "max_tokens_4096"            # 6,444 traces
    # "max_tokens_8192"          # 6,570 traces - ALREADY RUNNING (job 514651)
    "max_episodes_256"           # 6,995 traces

    # High traces (close to done)
    "low_diversity"              # 9,293 traces
    "repetition_penalty_1.05"    # 9,598 traces
    "linear_history_off"         # 9,647 traces

    # Not uploaded yet (need to check if they finished)
    "raw_content_off"
    "repetition_penalty_1.1"
    "repetition_penalty_1.2"
    "summarize_off"
    "summarize_threshold_16384"
    "summarize_threshold_2048"
    "summarize_threshold_32768"
)

for exp in "${RESUME_EXPERIMENTS[@]}"; do
    sbatch_file="${RESUME_DIR}/${exp}_resume.sbatch"
    if [ -f "$sbatch_file" ]; then
        echo "Submitting: $exp"
        sbatch "$sbatch_file"
        sleep 2
    else
        echo "MISSING: $sbatch_file"
    fi
done

echo ""
echo "=== All resume jobs submitted ==="
echo "Note: max_tokens_8192 is already running (job 514651)"
