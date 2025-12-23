#!/bin/bash
#SBATCH --job-name=nyu-upload-traces
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_40_tandon_advanced

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <job_dir> <hf_repo_id>" >&2
    exit 1
fi

JOB_DIR=$1
HF_REPO_ID=$2

SCRATCH_ROOT="/scratch/bf996"
PROJECT_DIR="${SCRATCH_ROOT}/OpenThoughts-Agent"
CONDA_ENV_PATH="${SCRATCH_ROOT}/miniconda3/envs/dcagent312"
CONDA_BIN="${CONDA_ENV_PATH}/bin/python"

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory not found: $PROJECT_DIR" >&2
    exit 1
fi

if [[ ! -x "$CONDA_BIN" ]]; then
    echo "Python interpreter not found at $CONDA_BIN" >&2
    exit 1
fi

cd "$PROJECT_DIR"

set +u
source "${SCRATCH_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate dcagent312
source hpc/dotenv/nyutorch.env
source "${HOME}/secrets.env"
set -u

python -m scripts.harbor.make_and_upload_trace_dataset \
    --job_dir "$JOB_DIR" \
    --repo_id "$HF_REPO_ID" \
    --episodes last
