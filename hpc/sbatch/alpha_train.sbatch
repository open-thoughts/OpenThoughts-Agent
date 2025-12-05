#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --mem={mem_per_node}
#SBATCH --time={time_limit}
#SBATCH --exclusive
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}

set -euo pipefail

export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CONDA_BACKUP_CC="${CONDA_BACKUP_CC:-}"
export CONDA_BACKUP_FC="${CONDA_BACKUP_FC:-}"

if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/alpha.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/alpha.env"
fi

module load release/24.04 GCCcore/12.3.0
module load CUDA/12.1.1
module load NCCL/2.18.3-CUDA-12.1.1

if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
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

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_PROTO=simple
export NCCL_IB_TIMEOUT=23
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export OMP_NUM_THREADS=1

if [ -z "${DCFT:-}" ]; then
  export DCFT="$PWD"
fi

mkdir -p "$DCFT/{experiments_dir}"
mkdir -p "$DCFT/{experiments_dir}/logs"
TMP_DIR="$DCFT/{experiments_dir}/tmp"
mkdir -p "$TMP_DIR"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-12802}
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS_PER_NODE={gpus_per_node}
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))
export HF_HOME=${HF_HOME:-$HF_HUB_CACHE}
export WANDB_DIR=${DCFT_WANDB_DIR:-$DCFT/{experiments_dir}/wandb}

CONFIG="$DCFT/{train_config_path_out}"
OUTPUT_DIR="$DCFT/{experiments_dir}"
echo -e "CONFIG: $CONFIG\nOUTPUT_DIR: $OUTPUT_DIR"

DEEPSPEED_CONFIG_FILE={deepspeed}

CMD="torchrun \
    --nproc-per-node $NUM_GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    sft/llamafactory/src/train.py $CONFIG"

SRUN_ARGS="
    --nodes=$NUM_NODES \
    --gpus-per-node=$NUM_GPUS_PER_NODE \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --label \
    --jobid $SLURM_JOBID"

export PYTHONPATH="$DCFT:${PYTHONPATH:-}"

cd "$DCFT"
srun $SRUN_ARGS bash -c "$CMD"
