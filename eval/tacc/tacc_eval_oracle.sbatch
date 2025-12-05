#!/bin/bash
#SBATCH -p gh-dev
#SBATCH --time=02:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082,c637-091,c610-111
#SBATCH --account CCR24067
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=eval_oracle_sandbox

module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer
# Set up environment variables
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_ADDRESS=$RAY_ADDRESS
export VLLM_CACHE_ROOT=/scratch/10000/eguha3/vllm_cache
export VLLM_CONFIG_ROOT=/scratch/10000/eguha3/vllm_config
export TRITON_DUMP_DIR=/scratch/10000/eguha3/triton_dump_dir
export TRITON_OVERRIDE_DIR=/scratch/10000/eguha3/triton_override_dir
export TRITON_CACHE_DIR=/scratch/10000/eguha3/triton_cache_dir
export FLASHINFER_WORKSPACE_BASE=/scratch/08002/gsmyrnis/flashinfer_cache
export UV_CACHE_DIR=/scratch/10000/eguha3/uv_cache_dir
export HYDRA_FULL_ERROR=1
export HF_CACHE_DIR=/scratch/08134/negin/dc-agent-shared/.hf_cache
export HF_HUB_CACHE=$SCRATCH/hub
source /scratch/08134/negin/dc-agent-shared/dc-agent/secret.env
ln -sf /home1/apps/gcc/15.1.0/lib64/libstdc++.so.6 /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v4/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v4/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v4

# Verify sandbox installation
sb --help

# Start VLLM server

# Add cleanup trap
cleanup() {
    echo "Cleaning up..."
    conda deactivate
}
trap cleanup EXIT

# Run the sandbox-based evaluation. Remove --task-name hello-world to run all tasks.
# Get the dataset path from the last line of output
DATASET_PATH=$(python "./snapshot_download.py" "mlfoundations-dev/clean-sandboxes-tasks-recleaned" | grep DATASET_PATH | tail -n 1 | cut -d'=' -f2)
if [ -z "$DATASET_PATH" ]; then
    echo "Failed to get dataset path" >&2
    exit 1
fi
echo "Using dataset path: $DATASET_PATH"

sb run \
  --dataset-path "$DATASET_PATH" \
  --task-name hello-world \
  --n-concurrent 32 \
  --agent oracle \
  --environment daytona \
# Script will run cleanup function on exit
