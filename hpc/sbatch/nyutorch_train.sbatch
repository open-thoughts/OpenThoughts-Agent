#!/bin/bash
#SBATCH --constraint="h200"
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --account=torch_pr_40_tandon_advanced
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --mem={mem_per_node}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=${USER}@nyu.edu
set -euo pipefail

export DCFT=/scratch/$USER/dcagent_scratch

# Explicit dotenv path to avoid brittle expansions
export DCFT_DOTENV=${DCFT_DOTENV:-/scratch/$USER/dc-agent/hpc/dotenv/nyutorch.env}
source "$DCFT_DOTENV"

if [ -f "$DCFT_PRIVATE/database/access.env" ]; then
    source "$DCFT_PRIVATE/database/access.env"
else
    echo "[nyutorch_train] database/access.env not found; continuing without DB creds" >&2
fi

SECRET_FILE="${DC_AGENT_SECRET_ENV:-${KEYS:-}}"
if [[ -n "${SECRET_FILE}" ]]; then
    if [[ -f "${SECRET_FILE}" ]]; then
        echo "Sourcing secrets from ${SECRET_FILE}"
        set -a
        # shellcheck disable=SC1090
        source "${SECRET_FILE}"
        set +a
    else
        echo "Warning: secrets file ${SECRET_FILE} not found; database registration may fail." >&2
    fi
else
    echo "Warning: DC_AGENT_SECRET_ENV is not set; database registration may fail." >&2
fi

for _supabase_var in SUPABASE_URL SUPABASE_ANON_KEY SUPABASE_SERVICE_ROLE_KEY; do
    if [[ -n "${!_supabase_var:-}" ]]; then
        export "${_supabase_var}=${!_supabase_var}"
    else
        echo "Warning: ${_supabase_var} is not set; Supabase registration may fail." >&2
    fi
done

mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"
TMP_DIR="{experiments_dir}/tmp"
mkdir -p "$TMP_DIR"

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# Basic NCCL networking configuration (adjust if needed)
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=120
export OMP_NUM_THREADS=1

# Rendezvous details
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=${MASTER_ADDR}
export MASTER_PORT=20043
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS_PER_NODE={gpus_per_node}
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))

export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
WANDB_DIR_DEFAULT="{experiments_dir}/wandb"
export WANDB_DIR=${DCFT_WANDB_DIR:-$WANDB_DIR_DEFAULT}
export RUN_SINGULARITY=/scratch/$USER/dc-agent/hpc/scripts/run-singularity-torch.bash

if [ ! -f "$RUN_SINGULARITY" ]; then
    echo "Missing singularity launcher at $RUN_SINGULARITY" >&2
    exit 1
fi

CONFIG={train_config_path_out}
OUTPUT_DIR={experiments_dir}
export CONFIG
export OUTPUT_DIR
echo -e "CONFIG: $CONFIG\nOUTPUT_DIR: $OUTPUT_DIR"

# Accelerate configuration
{accelerate_config_block}

srun --cpu_bind=v --accel-bind=v bash -c '
  set -euo pipefail

  TRAIN_LAUNCH_CMD=$(cat <<EOC
cd "/scratch/$USER/dc-agent"
export PYTHONPATH="/scratch/$USER/dc-agent:\$PYTHONPATH"
export HF_HOME="/scratch/$USER/.cache/huggingface"
export CHECKPOINTS_DIR="/scratch/$USER/dcagent_scratch/checkpoints"
export WANDB_DIR="$OUTPUT_DIR/wandb"
export TRITON_CACHE_DIR="$OUTPUT_DIR/triton_cache"
export OUTLINES_CACHE_DIR="/tmp/.outlines"
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
python -u -m accelerate.commands.launch \
  --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
  --config_file "$ACCELERATE_CONFIG_FILE" \
  --machine_rank $SLURM_PROCID \
  --role $(hostname -s): --tee 3 \
  sft/llamafactory/src/train.py "$CONFIG"
EOC
  )

  /bin/bash "$RUN_SINGULARITY" /bin/bash -c "$TRAIN_LAUNCH_CMD"
'

# Cleanup
rm -f "$ACCELERATE_CONFIG_FILE"
