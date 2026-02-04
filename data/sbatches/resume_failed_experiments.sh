#!/bin/bash
# Resume failed/partial experiment jobs
# These all have resumable DaytonaError/EnvironmentStartTimeoutError failures

SBATCHES_DIR="/scratch/10000/eguha3/dc-agent/data/sbatches"

echo "Resuming experiments with --auto-resume..."

# R2EGYM datasets (partial success - 55-98% failed)
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-2_1x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-33_6x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-4_2x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-8_4x

# Tezos datasets (99% failed)
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-1unique
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-128unique

# GFI dataset (100% failed)
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-gfi-swesmith-askllm-filtered-10K

echo ""
echo "7 resume jobs submitted! Use 'squeue -u \$USER' to monitor."
