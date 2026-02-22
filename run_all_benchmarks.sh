#!/bin/bash
set -e

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6
source /scratch/10000/eguha3/old-dc-agent/secret.env

DATASETS="e2egit manybugs bugswarm bugsinpy_mf softwareheritage defects4j"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for dataset in $DATASETS; do
    echo "============================================"
    echo "Running: $dataset"
    echo "============================================"

    harbor jobs start \
        -p /scratch/10000/eguha3/dc-agent/data/benchmark_tasks_by_dataset/$dataset \
        --n-concurrent 10 \
        --agent terminus-2 \
        --model openai/gpt-5-nano-2025-08-07 \
        --env daytona \
        --n-attempts 1 \
        --job-name dataset_${dataset}_${TIMESTAMP}

    echo ""
done

echo "All benchmarks completed!"
