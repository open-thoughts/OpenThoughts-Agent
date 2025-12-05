#!/bin/bash
#SBATCH -p gh
#SBATCH --time=24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082
#SBATCH --account CCR24067
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=reset_hf_repos
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu
# module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/miniconda3/envs/dcagent

source $DCFT/hpc/dotenv/tacc.env

set -euo pipefail

cd "$DCFT"

ORG="${HF_ORG:-DCAgent2}"
mkdir -p "${SCRATCH}/dc-agent/experiments"
MODEL_LIST_FILE="$(mktemp "${SCRATCH}/dc-agent/experiments/${ORG}_public_models.XXXXXX.txt")"
trap 'rm -f "$MODEL_LIST_FILE"' EXIT

echo "Listing public models for organization: ${ORG}"
python scripts/database/list_public_models.py --org "${ORG}" --output "${MODEL_LIST_FILE}"

if [ ! -s "${MODEL_LIST_FILE}" ]; then
  echo "No public models found for organization ${ORG}; exiting."
  exit 0
fi

while IFS= read -r model_id; do
  if [ -z "${model_id}" ]; then
    continue
  fi
  echo "Resetting Hugging Face model repo: ${model_id}"
  python scripts/database/reset_hf_repo.py "${model_id}" --repo-type model
done < "${MODEL_LIST_FILE}"
