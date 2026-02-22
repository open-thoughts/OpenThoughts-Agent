#!/bin/bash

for dataset in \
    "DCAgent/exp-gfi-staqc-short-response-filtered-10K" \
    "DCAgent/dev_set_part1_10k" \
    "DCAgent/exp-syh-r2egym-swesmith-mixed" \
    "DCAgent/exp-syh-tezos-stackoverflow-mixed" \
    "DCAgent/exp-syh-r2egym-askllm-constrained" \
    "DCAgent/exp-syh-tezos-askllm-constrained" \
    "DCAgent/perturbed-docker-exp-magicoder-tasks-2"
do
    echo "Submitting: $dataset"
    sbatch /scratch/10000/eguha3/dc-agent/data/sbatches/run_harbor_32node_8way_dp_glm47.sbatch "$dataset"
    sleep 2
done
