#!/bin/bash
#SBATCH -p gh-dev
#SBATCH --time=2:00:00
#SBATCH --nodes 5
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --exclude=
#SBATCH --account CCR24067
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=code-contests-nl2bash-bugsseq_Qwen3-8B-maxEps32-accThink-disableSummarize-4nodes-4inf

module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

# Set up environment variables
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
# RAY_ADDRESS is set later after starting the Ray cluster
export VLLM_CACHE_ROOT=/scratch/11151/charlieruan/vllm_cache
export VLLM_CONFIG_ROOT=/scratch/11151/charlieruan/vllm_config
export TRITON_DUMP_DIR=/scratch/11151/charlieruan/triton_dump_dir
export TRITON_OVERRIDE_DIR=/scratch/11151/charlieruan/triton_override_dir
export TRITON_CACHE_DIR=/scratch/11151/charlieruan/triton_cache_dir
export FLASHINFER_WORKSPACE_BASE=/scratch/11151/charlieruan/flashinfer_cache
export UV_CACHE_DIR=/scratch/11151/charlieruan/uv_cache_dir
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load secret environment variables
source /scratch/11151/charlieruan/dc-agent/train/hpc/dotenv/secret.env

# Fix library path issues
ln -sf /home1/apps/gcc/15.1.0/lib64/libstdc++.so.6 /scratch/10000/eguha3/tacc_rl_v6/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/10000/eguha3/tacc_rl_v6/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Activate conda environment
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6
echo "CONDA ENV PATH: "
echo /scratch/10000/eguha3/tacc_rl_v6

# Custom PYTHONPATH (e.g. Harbor)
# export PYTHONPATH=/scratch/11151/charlieruan/harbor/src:$PYTHONPATH

# SkyRL paths
export SKYRL_HOME=/scratch/10000/eguha3/SkyRL
cd "$SKYRL_HOME/skyrl-train"

# Clean up any leftover Ray processes on all nodes
echo "Stopping any existing Ray processes on all nodes..."
srun --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 ray stop --force 2>/dev/null || true
sleep 5

# Set up Ray server (using fixed script with reliable IP detection)
bash /scratch/10000/eguha3/dc-agent/data/sbatches/start_ray_server_fixed.sh

sleep 30

# Set RAY_ADDRESS=auto so Python connects to the local Ray instance
# (The script runs on head node where Ray head is running)
export RAY_ADDRESS=auto
echo "Set RAY_ADDRESS=auto (will connect to local Ray instance)"

# Run SkyRL command
python -m examples.terminal_bench.entrypoints.main_tbench \
  hydra.searchpath=['file://examples/terminal_bench'] \
  hydra.run.dir=/scratch/10000/eguha3/hydra_outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S} \
  +terminal_bench_config=terminal_bench \
  +terminal_bench_config.agent_name=terminus \
  +terminal_bench_config.max_episodes=32 \
  +terminal_bench_config.trials_dir=/scratch/08134/negin/dc-agent-shared/sandboxes/run \
  +terminal_bench_config.override_memory_mb=1024 \
  +terminal_bench_config.override_storage_mb=1024 \
  +terminal_bench_config.override_cpus=1 \
  +terminal_bench_config.enable_summarize=false \
  trainer.export_path=/scratch/11151/charlieruan/skyrl_exports/code-contests-async-4nodes-2inf-0107-8stale \
  trainer.ckpt_path=/scratch/11151/charlieruan/skyrl_exports/code-contests-async-4nodes-2inf-0107-8stale \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=2 \
  trainer.placement.ref_num_nodes=2 \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  trainer.fully_async.max_staleness_steps=8 \
  trainer.fully_async.num_parallel_generation_workers=256 \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=1 \
  +generator.engine_init_kwargs.custom_chat_template_chat_completion_path=/scratch/11151/charlieruan/dc-agent/train/hpc/qwen3_acc_thinking.jinja2 \
  trainer.epochs=3 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=20 \
  trainer.max_prompt_length=2048 \
  trainer.gradient_checkpointing=true \
  trainer.gradient_checkpointing_use_reentrant=false \
  generator.sampling_params.max_generate_length=30720 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  generator.enforce_eager=true \
  trainer.logger=wandb \
  trainer.project_name=dc-agent \
  trainer.run_name=code-contests-async-4nodes-2inf-0107-8stale \
  trainer.resume_mode=null \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host=127.0.0.1 \
  generator.http_endpoint_port=8000 \
  data.train_data="['/scratch/09749/atutej/hub/DCAgent_code_contests_sandboxes_with_tests_tasks']" \
  data.val_data="['/scratch/11151/charlieruan/new_hub/datasets--DCAgent--dev_set_71_tasks/snapshots/da9bcc03f95a264602be912dcdc6db9371a3e4e3']" \
  trainer.policy.model.path=/scratch/11151/charlieruan/new_hub/models--laion--nl2bash-bugsseq_Qwen3-8B/snapshots/b2f5a18c2219ff1d95b23f03673599cb1f36b0f6
