#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --nodes=__NODES__
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --job-name=cf-v6-__SIZE__-axolotl__Qwen3-8B
#SBATCH --output=/e/scratch/jureap59/feuer1/code/axolotl_sbatch/logs/%x_%j.out
#SBATCH -p booster
#SBATCH --account reformo
#SBATCH --gres=gpu:4
#SBATCH --exclude=jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]

set -eo pipefail
ml purge
ulimit -c 0

source /e/scratch/jureap59/feuer1/miniforge3/etc/profile.d/conda.sh
conda activate sera-axolotl

export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export GCC_HOME=/e/software/default/stages/2026/software/GCCcore/14.3.0
export CC=$GCC_HOME/bin/gcc
export CXX=$GCC_HOME/bin/g++
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$GCC_HOME/lib64:${LD_LIBRARY_PATH:-}
source /e/scratch/jureap59/feuer1/OpenThoughts-Agent/hpc/dotenv/jupiter.env
source ~/secrets.env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export AXOLOTL_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER=0

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_TIMEOUT=60

CONFIG=/e/scratch/jureap59/feuer1/code/axolotl_configs/qwen3_8b_cf_v6___SIZE__.yaml
NPROC_PER_NODE=4

echo "=== Launching axolotl on $SLURM_JOB_NODELIST ==="
echo "  CONFIG=$CONFIG  NNODES=$SLURM_JOB_NUM_NODES  NPROC=$NPROC_PER_NODE  MASTER=$MASTER_ADDR"
echo "  CUDA_HOME=$CUDA_HOME  torch=$(python -c 'import torch;print(torch.__version__)')"

srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --cpu-bind=none \
  bash -c '
    export NODE_RANK=$SLURM_NODEID
    accelerate launch \
      --num_machines="$SLURM_JOB_NUM_NODES" \
      --num_processes=$(( $SLURM_JOB_NUM_NODES * '$NPROC_PER_NODE' )) \
      --machine_rank="$NODE_RANK" \
      --main_process_ip="$MASTER_ADDR" --main_process_port="$MASTER_PORT" \
      --mixed_precision=bf16 --dynamo_backend=no \
      --use_deepspeed --deepspeed_config_file /e/scratch/jureap59/feuer1/code/axolotl/deepspeed_configs/zero3_bf16.json \
      -m axolotl.cli.train '"$CONFIG"'
  '
