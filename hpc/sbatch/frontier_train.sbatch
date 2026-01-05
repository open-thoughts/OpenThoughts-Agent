#!/bin/bash
# Frontier training template for HPC launch.
#SBATCH -A {account}
#SBATCH -p batch
#SBATCH -q {qos}
#SBATCH --exclusive
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --time={time_limit}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -euo pipefail

# Guard conda deactivate scripts from set -u complaints.
export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"
export CONDA_BACKUP_CC="${CONDA_BACKUP_CC:-}"
export CONDA_BACKUP_FC="${CONDA_BACKUP_FC:-}"

# Ensure dotenv variables are available on the compute node.
frontier_env_loaded=0
if [ -n "${DCFT:-}" ] && [ -f "$DCFT/hpc/dotenv/frontier.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/frontier.env"
  frontier_env_loaded=1
fi

# Ensure DCFT points to a valid checkout even if SCRATCH is unavailable.
DEFAULT_OT_ROOT="${OPEN_THOUGHTS_AGENT_ROOT:-$HOME/OpenThoughts-Agent}"
if [ -z "${DCFT:-}" ] && [ -d "$DEFAULT_OT_ROOT" ]; then
  export DCFT="$DEFAULT_OT_ROOT"
elif [ -n "${DCFT:-}" ] && [ ! -d "$DCFT" ] && [ -d "$DEFAULT_OT_ROOT" ]; then
  echo "DCFT path $DCFT not found; falling back to $DEFAULT_OT_ROOT"
  export DCFT="$DEFAULT_OT_ROOT"
fi

if [ "$frontier_env_loaded" -eq 0 ] && [ -f "$DCFT/hpc/dotenv/frontier.env" ]; then
  # shellcheck disable=SC1090
  source "$DCFT/hpc/dotenv/frontier.env"
fi

# Frontier jobs should always hit NVMe-backed caches when available.
if [ -n "${FRONTIER_NVME_HF_HOME:-}" ]; then
  export HF_HOME="$FRONTIER_NVME_HF_HOME"
fi
if [ -n "${FRONTIER_NVME_HF_HUB_CACHE:-}" ]; then
  export HF_HUB_CACHE="$FRONTIER_NVME_HF_HUB_CACHE"
fi

# Activate requested environment; fall back to manual setup if unset.
if [ -n "${DCFT_ACTIVATE_ENV:-}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
else
  module load conda || true
  source "$HOME/.bashrc"
  set +u
  conda activate dcagent
  set -u
  if [ -f "$HOME/secrets.env" ]; then
    # shellcheck disable=SC1090
    source "$HOME/secrets.env"
  fi
fi

module load PrgEnv-amd
module load amd/6.4.0
module load rocm/6.4.0
module load craype-accel-amd-gfx90a
module load cray-mpich

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

if [ -n "${LIBRARY_PATH:-}" ]; then
  export LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LIBRARY_PATH"
else
  export LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64"
fi

if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
else
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64"
fi

export PATH="$ROCM_PATH/bin:$PATH"
export MPICH_GPU_SUPPORT_ENABLED=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-90a}

if [ -n "${CONDA_PREFIX:-}" ]; then
  export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
  export FC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gfortran"
  export PATH="$CONDA_PREFIX/bin:$PATH"
  if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
  else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
  fi
else
  echo "Warning: CONDA_PREFIX is not set; compiler environment variables were not configured." >&2
fi

echo "ROCM_PATH=${ROCM_PATH:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

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

mkdir -p "$DCFT/{experiments_dir}"
mkdir -p "$DCFT/{experiments_dir}/logs"
TMP_DIR="$DCFT/{experiments_dir}/tmp"
mkdir -p "$TMP_DIR"

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_PROTO=simple
export NCCL_IB_TIMEOUT=22
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
if [ -n "${MASTER_PORT:-}" ]; then
  export MASTER_PORT="$MASTER_PORT"
else
  export MASTER_PORT=12802
fi
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS_PER_NODE={gpus_per_node}
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))
if [ -n "${HF_HOME:-}" ]; then
  export HF_HOME="$HF_HOME"
else
  export HF_HOME="$HF_HUB_CACHE"
fi
if [ -n "${WANDB_DIR:-}" ]; then
  export WANDB_DIR="$WANDB_DIR"
else
  export WANDB_DIR="$DCFT/{experiments_dir}/wandb"
fi

CONFIG="$DCFT/{train_config_path_out}"
OUTPUT_DIR="$DCFT/{experiments_dir}"
echo -e "CONFIG: $CONFIG\nOUTPUT_DIR: $OUTPUT_DIR"

# Deepspeed configuration is passed via HF TrainingArguments
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
