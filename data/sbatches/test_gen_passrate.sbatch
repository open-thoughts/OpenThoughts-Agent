#!/bin/bash
#SBATCH -p gh
#SBATCH --time=6:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=16
#SBATCH --account CCR24067
#SBATCH --output=/scratch/10000/eguha3/dc-agent/data/sbatches/logs/test_gen_passrate_%j.out
#SBATCH --job-name=test_gen_passrate

# Run all test-generating experiments with 25 samples through harbor
# to check pass rates. Uses gpt-5-nano for both generation and solving.
# Usage: sbatch test_gen_passrate.sbatch [experiment_name]
# If no experiment name given, runs all 9 test-generating experiments.

set -eo pipefail

DCAGENT_DIR="/scratch/10000/eguha3/dc-agent"
cd "$DCAGENT_DIR"

# Source secrets
source "$DCAGENT_DIR/rl/hpc/dotenv/secret.env"

# Setup conda environment
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/vllm_sandboxes_backup

export PYTHONPATH="$DCAGENT_DIR:$DCAGENT_DIR/scripts/harbor:$PYTHONPATH"

echo "============================================="
echo "Test-Generating Experiments Pass Rate Check"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "============================================="

if [ -n "$1" ]; then
    echo "Running specific experiment: $1"
    python data/generate_25_and_test.py "$1"
else
    echo "Running all test-generating experiments..."
    python data/generate_25_and_test.py
fi

echo ""
echo "============================================="
echo "Done! Results saved to data/test_gen_passrate_results.json"
echo "============================================="
