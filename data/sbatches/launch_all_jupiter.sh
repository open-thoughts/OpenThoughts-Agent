#!/bin/bash
# Launch GLM-4.7 trace generation on Jupiter for all datasets
# Usage: bash launch_all_jupiter.sh

set -e

SBATCH_SCRIPT="/e/scratch/jureap59/etash/OpenThoughts-Agent/data/sbatches/run_harbor_glm47_jupiter.sbatch"

DATASETS=(
  "DCAgent/swesmith-datascience-skorch-sandboxes"
  "DCAgent/exp-swd-swesmith-wo-docker"
  "DCAgent/perturbed-docker-exp-magicoder-tasks-1"
  "DCAgent/exp-uns-r2egym-128unique"
  "DCAgent/exp-uns-r2egym-16_8x"
  "DCAgent/exp-uns-r2egym-1unique"
  "DCAgent/exp-uns-r2egym-2_1x"
  "DCAgent/exp-uns-r2egym-33_6x"
  "DCAgent/exp-uns-r2egym-4_2x"
  "DCAgent/exp-uns-r2egym-8_4x"
  "DCAgent/exp-uns-tezos-10x"
  "DCAgent/exp-uns-tezos-128unique"
  "DCAgent/exp-uns-tezos-160x"
  "DCAgent/exp-uns-tezos-1unique"
  "DCAgent/exp-uns-tezos-20x"
  "DCAgent/exp-uns-tezos-40x"
  "DCAgent/exp-uns-tezos-80x"
  "DCAgent/exp-gfi-swesmith-random-filtered-10K"
  "DCAgent/exp-gfi-swesmith-embedding-mean-filtered-10K"
  "DCAgent/exp-gfi-swesmith-short-response-filtered-10K"
  "DCAgent/exp-gfi-staqc-embedding-mean-filtered-10K"
  "DCAgent/exp-gfi-swesmith-askllm-filtered-10K"
  "DCAgent/exp-gfi-staqc-askllm-filtered-10K"
  "DCAgent/exp-syh-r2egym-askllm-hardened"
  "DCAgent/exp-syh-tezos-askllm-hardened"
  "DCAgent/exp-gfi-staqc-short-response-filtered-10K"
  "DCAgent/dev_set_part1_10k"
  "DCAgent/exp-syh-r2egym-swesmith-mixed"
  "DCAgent/exp-syh-tezos-stackoverflow-mixed"
  "DCAgent/exp-syh-r2egym-askllm-constrained"
  "DCAgent/exp-syh-tezos-askllm-constrained"
  "DCAgent/perturbed-docker-exp-magicoder-tasks-2"
)

echo "Submitting ${#DATASETS[@]} datasets to Jupiter GLM-4.7 generation"
echo "Jobs dir: /e/data1/datasets/playground/mmlaion"
echo "Sbatch: $SBATCH_SCRIPT"
echo ""

for ds in "${DATASETS[@]}"; do
  echo "Submitting: $ds"
  sbatch "$SBATCH_SCRIPT" "$ds"
  sleep 1  # small delay between submissions
done

echo ""
echo "Done. Submitted ${#DATASETS[@]} jobs."
echo "Monitor with: squeue -u $USER"
