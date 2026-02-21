


ml purge 
# DB/API secrets etc.
source /leonardo/home/userexternal/mnezhuri/dcft/OpenThoughts-Agent/eval/leonardo/secret.env
source /leonardo/home/userexternal/mnezhuri/dcft/OpenThoughts-Agent/rl/hpc/dotenv/leonardo.env

export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_PROJECT=$WANDB_PROJECT
export HF_TOKEN=$HF_TOKEN
export SUPABASE_ANON_KEY=$SUPABASE_ANON_KEY
export SUPABASE_SERVICE_ROLE_KEY=$SUPABASE_SERVICE_ROLE_KEY
export SUPABASE_URL=$SUPABASE_URL
export DAYTONA_API_KEY=$DAYTONA_API_KEY



export PATH=$PATH:~/.local/bin
export SSH_KEY=~/.ssh/leonardo_daytona

# Conda env
export MINICONDA_PATH="/leonardo_work/AIFAC_5C0_290/dc-agent-shared/envs/mamba"
export CONDA_ENV="/leonardo_work/AIFAC_5C0_290/dc-agent-shared/envs/py3.12_new"
source ${MINICONDA_PATH}/bin/activate ${CONDA_ENV}

export HARBOR_PATCHED=/leonardo/home/userexternal/mnezhuri/dcft/harbor_patched/harbor_patched/src
export DATABASE_PATH=/leonardo/home/userexternal/mnezhuri/dcft/OpenThoughts-Agent/database
export PYTHONPATH=${HARBOR_PATCHED}:${DATABASE_PATH}:$PYTHONPATH



MODEL="${1:-Qwen/Qwen3-8B}"
REPO_ID="${2:-DCAgent2/terminal_bench_2}"


export MODEL_ID=$MODEL
export REPO_ID=$REPO_ID

# Strip slashes and special chars for file-safe names
export SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
export SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')


# Get the dataset path using the specified repo_id
echo "Downloading/locating dataset: $REPO_ID"
DATASET_PATH=$(python "eval/tacc/snapshot_download.py"  "$REPO_ID" | grep DATASET_PATH | tail -n 1 | cut -d'=' -f2)
if [ -z "${DATASET_PATH:-}" ]; then
    echo "Failed to get dataset path"
    exit 1
fi
echo "Using dataset path: $DATASET_PATH"

echo "Downloading/locating model: $MODEL"
MODEL_PATH=$(python "eval/leonardo/download_model.py"  "$MODEL" | grep MODEL_PATH | tail -n 1 | cut -d'=' -f2)

if [ -z "${MODEL_PATH:-}" ]; then
    echo "Failed to get model path"
    exit 1
fi
echo "Using model path: $MODEL_PATH"

sbatch --job-name=eval_"$SAFE_MODEL"_"$SAFE_REPO" eval/leonardo/tb2_eval_harbor.sbatch "$MODEL_PATH" "$DATASET_PATH"

