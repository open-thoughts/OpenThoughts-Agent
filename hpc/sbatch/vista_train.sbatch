#!/bin/bash
#SBATCH -p {partition}
#SBATCH --time={time_limit}
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082
#SBATCH --account {account}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu
# module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/miniconda3/envs/dcagent

source $DCFT/hpc/dotenv/tacc.env

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

#echo "Moving HF models to /tmp..."
export HF_HOME="/tmp/hf_home"
#srun rsync -az $SCRATCH/hf_home /tmp

# export HF_TOKEN=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

export NCCL_PROTO=simple
export NCCL_DEBUG=INFO
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
# GPUDirect isn’t available on these nodes yet; let NCCL fall back to host staging
unset FI_EFA_USE_DEVICE_RDMA
unset NCCL_NET_GDR_LEVEL
unset NCCL_NET_GDR_READ

export PYTHONFAULTHANDLER=1
# export NCCL_SOCKET_IFNAME="eno1"

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export NCCL_TIMEOUT=1800  # 30 minutes
export NCCL_IB_TIMEOUT=23  # InfiniBand timeout

export OUTLINES_CACHE_DIR="/tmp/.outlines"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export TRITON_CACHE_DIR="/tmp/triton_cache"
export FORCE_TORCHRUN=1

cd $DCFT
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_HOME=/home1/apps/nvidia/Linux_aarch64/25.3/cuda/12.8
export LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/25.3/math_libs/12.8/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LIBRARY_PATH


CONFIG={train_config_path_out}
OUTPUT_DIR={experiments_dir}
echo -e "CONFIG: $CONFIG\nOUTPUT_DIR: $OUTPUT_DIR"

srun torchrun \
    --nproc-per-node 1 \
    --nnodes $SLURM_JOB_NUM_NODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    sft/llamafactory/src/train.py $CONFIG
