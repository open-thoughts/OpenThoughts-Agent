#!/bin/bash
# Test all Nemotron generators with Harbor (25 tasks each)
#
# Usage:
#   bash data/nemotron-mine/run_test_all.sh                     # all generators
#   bash data/nemotron-mine/run_test_all.sh bash junit rust      # specific generators
#   bash data/nemotron-mine/run_test_all.sh --generate-only      # skip harbor
#
# Run from repo root: /scratch/10000/eguha3/dc-agent

set -e

# Environment setup
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6
source /scratch/10000/eguha3/old-dc-agent/secret.env

cd /scratch/10000/eguha3/dc-agent

export PYTHONPATH="/scratch/10000/eguha3/dc-agent/scripts/harbor:/scratch/10000/eguha3/dc-agent/data/nemotron-mine:$PYTHONPATH"

echo "============================================"
echo "Nemotron Generator Pass Rate Tests"
echo "Limit: 25 tasks per generator"
echo "Model: openai/gpt-5-nano-2025-08-07"
echo "============================================"
echo ""

python data/nemotron-mine/test_all_nemotron.py "$@"
