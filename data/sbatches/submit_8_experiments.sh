#!/bin/bash
# Submit 8 experiment jobs

SBATCHES_DIR="/scratch/10000/eguha3/dc-agent/data/sbatches"

echo "Submitting 8 Harbor jobs..."

# R2EGYM datasets (4)
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-2_1x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-33_6x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-4_2x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-8_4x

# Tezos datasets (2)
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-1unique
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-128unique

# GFI datasets (2)
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-gfi-swesmith-askllm-filtered-10K
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-gfi-staqc-askllm-filtered-10K

echo ""
echo "8 jobs submitted! Use 'squeue -u \$USER' to monitor."
