#!/bin/bash
# Reset all public Hugging Face repos for an organization on ZIH Capella.
# Uses the same helper scripts as the TACC variant but runs on the Capella
# Slurm partition with the appropriate modules and environment hooks.
# Optionally pass a newline-delimited list of repos as the first argument
# (see scripts/huggingface/sample_model_list.txt for the expected format).
#SBATCH --account=p_agents_finetuning
#SBATCH --partition=capella
#SBATCH --time=47:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=reset_hf_repos
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -euo pipefail

# Ensure DCFT is defined so relative paths resolve consistently.
if [ -z "${DCFT:-}" ]; then
  if [ -n "${DC_AGENT:-}" ]; then
    export DCFT="$DC_AGENT"
  else
    export DCFT="$PWD"
  fi
fi

# Load Capella-specific dotenv to pick up credentials and environment hooks.
if [ -f "$DCFT/hpc/dotenv/zih_capella.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/zih_capella.env"
elif [ -n "${DC_AGENT:-}" ] && [ -f "$DC_AGENT/hpc/dotenv/zih_capella.env" ]; then
  # shellcheck disable=SC1090
  source "$DC_AGENT/hpc/dotenv/zih_capella.env"
fi

# CUDA is not strictly required, but the shared environment expects this module.
module load CUDA/12.8.0

# Activate the requested conda environment if DCFT_ACTIVATE_ENV is provided.
if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
else
  echo "Warning: DCFT_ACTIVATE_ENV is not set; falling back to default conda environment" >&2
  if [ -f "$SCRATCH/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$SCRATCH/miniconda3/etc/profile.d/conda.sh"
    conda activate "$SCRATCH/miniconda3/envs/dcagent"
  else
    echo "ERROR: Unable to locate conda environment on Capella." >&2
    exit 1
  fi
fi

# Optional secrets file (same pattern used elsewhere in the repo).
SECRET_FILE="${DC_AGENT_SECRET_ENV:-${KEYS:-}}"
if [[ -n "${SECRET_FILE}" && -f "${SECRET_FILE}" ]]; then
  echo "Sourcing secrets from ${SECRET_FILE}"
  set -a
  # shellcheck disable=SC1090
  source "${SECRET_FILE}"
  set +a
fi

cd "$DCFT"

ORG="${HF_ORG:-DCAgent2}"
USER_MODEL_LIST="${1:-}"
SCRATCH_ROOT="${SCRATCH:-$PWD}"

mkdir -p "${SCRATCH_ROOT}/OpenThoughts-Agent/experiments"
if [ -n "$USER_MODEL_LIST" ]; then
  if [ ! -f "$USER_MODEL_LIST" ]; then
    echo "ERROR: Provided model list file '$USER_MODEL_LIST' does not exist." >&2
    exit 1
  fi
  MODEL_LIST_FILE="$USER_MODEL_LIST"
  echo "Using user-supplied model list: $MODEL_LIST_FILE"
else
  MODEL_LIST_FILE="$(mktemp "${SCRATCH_ROOT}/OpenThoughts-Agent/experiments/${ORG}_public_models.XXXXXX.txt")"
  trap 'rm -f "$MODEL_LIST_FILE"' EXIT

  echo "Listing public models for organization: ${ORG}"
  python scripts/database/list_public_models.py --org "${ORG}" --output "${MODEL_LIST_FILE}"
fi

if [ ! -s "${MODEL_LIST_FILE}" ]; then
  echo "No public models found for organization ${ORG}; exiting."
  exit 0
fi

while IFS= read -r model_id; do
  if [ -z "${model_id}" ] || [[ "${model_id}" =~ ^# ]]; then
    continue
  fi
  echo "Resetting Hugging Face model repo: ${model_id}"
  python scripts/huggingface/reset_hf_repo.py "${model_id}" --repo-type model
done < "${MODEL_LIST_FILE}"
