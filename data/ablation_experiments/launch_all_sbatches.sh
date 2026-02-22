#!/bin/bash
# Launch all ablation sbatch files (excluding baseline)
cd /scratch/10000/eguha3/dc-agent/data/ablation_experiments/sbatch

for sbatch_file in *.sbatch; do
    if [ "$sbatch_file" != "baseline.sbatch" ]; then
        echo "Submitting $sbatch_file..."
        sbatch "$sbatch_file"
        sleep 2
    fi
done

echo "All sbatch jobs submitted (excluding baseline)!"
