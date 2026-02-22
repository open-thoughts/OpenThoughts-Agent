#!/bin/bash
# Run all resume sbatch files
cd /scratch/10000/eguha3/dc-agent/data/ablation_experiments/resume_sbatch

for sbatch_file in *.sbatch; do
    echo "Submitting $sbatch_file..."
    sbatch "$sbatch_file"
    sleep 2
done

echo "All resume jobs submitted!"
