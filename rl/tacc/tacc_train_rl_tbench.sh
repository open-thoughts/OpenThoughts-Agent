#!/bin/bash
#SBATCH -p gh-dev
#SBATCH --time=02:00:00
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082,c637-091,c610-111
#SBATCH --account CCR24067
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=train_rl

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
source /scratch/08134/negin/ot-agent-shared/ot-agent/secret.env
ln -s /home1/apps/gcc/15.1.0/lib64/libstdc++.so.6 /scratch/08134/negin/ot-agent-shared/SkyRL/envs/tacc_rl_v4/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/08134/negin/ot-agent-shared/SkyRL/envs/tacc_rl_v4/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate  /scratch/08134/negin/ot-agent-shared/SkyRL/envs/tacc_rl_v4

SKYRL_HOME=/scratch/10000/eguha3/SkyRL
cd $SKYRL_HOME
SKYRL_OUTPUT_DIR="/scratch/08134/negin/ot-agent-shared/SkyRL/skyrl-train"
export DATA_DIR="${SKYRL_OUTPUT_DIR}/data_rl/gsm8k"
CHECKPOINT_PATH="${SKYRL_OUTPUT_DIR}/checkpoints/gsm8k"
mkdir -p $DATA_DIR
mkdir -p $CHECKPOINT_PATH

python /scratch/08134/negin/ot-agent-shared/SkyRL/skyrl-train/examples/gsm8k/gsm8k_dataset.py --output_dir $DATA_DIR

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
NUM_GPUS_PER_NODE=1
NUM_GPUS=$(($NUM_GPUS_PER_NODE * $SLURM_JOB_NUM_NODES))
echo "Number of GPUs: $NUM_GPUS"
LOGGER="console"
echo "CUDA_HOME: $CUDA_HOME"
bash /scratch/08134/negin/ot-agent-shared/ot-agent/eval/tacc/start_ray_server.sh

sleep 30
# Run training
python -m skyrl_train.entrypoints.main_base \
  data.train_data=[mlfoundations-dev/sandboxes-tasks] \
  data.val_data=[mlfoundations-dev/sandboxes-tasks] \
  +data.cache_dir=/scratch/08134/negin/ot-agent-shared/data_cache \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path=$MODEL_PATH \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=$SLURM_JOB_NUM_NODES \
  trainer.placement.ref_num_nodes=$SLURM_JOB_NUM_NODES \
  trainer.placement.reward_num_nodes=$SLURM_JOB_NUM_NODES \
  trainer.placement.critic_num_nodes=$SLURM_JOB_NUM_NODES \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.reward_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1  \
  trainer.eval_batch_size=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1 \
  trainer.policy_mini_batch_size=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=16000 \
  generator.sampling_params.max_generate_length=16000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  +generator.trial_runs_dir=/scratch/08134/negin/ot-agent-shared/sandboxes/runs \
  +generator.agent_name=terminus \
  generator.use_http_server_inference_engine_client=true \
  generator.http_server_inference_engine_client_host=127.0.0.1 \
  generator.http_server_inference_engine_client_port=8000 \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  +generator.sandboxes_dir=/scratch/08134/negin/ot-agent-shared/sandboxes \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger=$LOGGER \
  trainer.project_name=gsm8k \
  trainer.run_name=gsm8k_test \
  trainer.resume_mode=null \
  trainer.ckpt_path=$CHECKPOINT_PATH \
  trainer.export_path=/scratch/08134/negin/ot-agent-shared/sandboxes/exports
      