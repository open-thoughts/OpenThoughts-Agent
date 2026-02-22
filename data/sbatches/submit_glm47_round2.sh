#!/bin/bash

for dataset in \
    "DCAgent/perturbed-docker-exp-magicoder-tasks-2" \
    "DCAgent/exp-syh-tezos-stackoverflow-mixed" \
    "DCAgent/exp-syh-r2egym-askllm-constrained" \
    "DCAgent/exp-syh-tezos-askllm-constrained" \
    "DCAgent/exp-gfi-swesmith-short-response-filtered-10K" \
    "DCAgent/exp-gfi-staqc-embedding-mean-filtered-10K" \
    "DCAgent/exp-gfi-swesmith-askllm-filtered-10K" \
    "DCAgent/exp-gfi-staqc-askllm-filtered-10K"
do
    echo "Submitting: $dataset"
    sbatch /scratch/10000/eguha3/dc-agent/data/sbatches/run_harbor_32node_8way_dp_glm47.sbatch "$dataset"
    sleep 2
done
