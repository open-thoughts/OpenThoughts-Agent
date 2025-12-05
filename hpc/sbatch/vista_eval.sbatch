#!/bin/bash
#SBATCH -p gh
#SBATCH --time={time_limit}
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --exclude=c610-021,c611-011,c640-041
#SBATCH --account {account}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}

module load gcc/14.2.0
module load cuda/12.8
module load tacc-apptainer

source /scratch/10635/penfever/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10635/penfever/miniconda3/envs/dcagent

# $DCFT_PRIVATE_ACTIVATE_ENV
source $DCFT_PRIVATE/hpc/dotenv/tacc.env
source $DCFT_PRIVATE/secrets.env

#echo "Moving HF models to /tmp..."
export HF_HOME="/tmp/hf_home"
#srun rsync -az $SCRATCH/hf_home /tmp

export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_NET_GDR_LEVEL="SYS"
export NCCL_NET_GDR_READ=1

export PYTHONFAULTHANDLER=1
#export NCCL_SOCKET_IFNAME="eno1"

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
export TRITON_CACHE_DIR="$SCRATCH/triton_cache"
export FORCE_TORCHRUN=1

cd $DCFT_PRIVATE
export PYTHONPATH=$PWD:$PYTHONPATH

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

rsync -az /tmp/dcft_checkpoints $SCRATCH/dcft_checkpoints
