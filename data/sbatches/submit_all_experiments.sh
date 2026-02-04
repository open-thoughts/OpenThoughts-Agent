#!/bin/bash
# Submit all experiment jobs
# Generated for batch submission of 18 datasets

SBATCHES_DIR="/scratch/10000/eguha3/dc-agent/data/sbatches"

echo "Submitting Harbor jobs (terminus-2 agent)..."

# R2EGYM datasets
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-2_1x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-33_6x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-4_2x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-r2egym-8_4x

# Tezos datasets
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-10x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-128unique
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-160x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-1unique
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-20x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-40x
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-uns-tezos-80x

# GFI datasets (SweSmith)
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-gfi-swesmith-random-filtered-10K
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-gfi-swesmith-embedding-mean-filtered-10K
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-gfi-swesmith-short-response-filtered-10K
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-gfi-swesmith-askllm-filtered-10K

# GFI datasets (StaQC)
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-gfi-staqc-embedding-mean-filtered-10K
sbatch "$SBATCHES_DIR/run_harbor_32node_8way_dp_glm47.sbatch" DCAgent/exp-gfi-staqc-askllm-filtered-10K

echo ""
echo "Submitting OpenHands job..."
sbatch "$SBATCHES_DIR/openhands.sbatch" mlfoundations-dev/stackexchange-tezos-sandboxes

echo ""
echo "All jobs submitted! Use 'squeue -u \$USER' to monitor."
