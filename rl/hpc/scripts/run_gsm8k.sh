# This is an example script for running a SkyRL GSM8K training job on TACC.
# It is equivalent to the usual skyrl GSM8K training script: https://skyrl.readthedocs.io/en/latest/examples/ppo.html
# All the -S will be passed directly to SkyRL, while thoes without `-S` will be used for HPC-specific logics.
# Note that `train_data`, `val_data`, and `model_path`, after post-processing, will be passed to SkyRL as well.
# Mainly need pre-downloading and hence mutating the arguments in certain HPC without internet access.

NUM_NODES=2  # In TACC, number of nodes = number of GPUs
LOGGER="wandb"  # wandb for logging into WANDB, console for printing to stdout
RUN_NAME="gsm8k_test"  # The name of this run (fed to `-S trainer.run_name` for WANDB logging and various other places)
DATA_DIR="/scratch/08134/negin/dc-agent-shared/SkyRL/skyrl-train/data_rl/gsm8k"  # can also be HF dataset name
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct"  # base model to start the RL from

# We will dump the following in the export path: SkyRL dumped eval output, and model checkpoints
# Note that this differs from the EXPERIMENTS_DIR specified in {hpc_name}.env, which is where we
# dump the sbatch scripts and SLURM `.out` logs.
SKYRL_EXPORT_PATH="/scratch/08134/negin/dc-agent-shared/sandboxes/exports/$RUN_NAME"

MAX_RESTARTS=2

python3 -m hpc.launch \
  --job_name $RUN_NAME \
  --final_model_name $RUN_NAME \
  --enable_hf_upload \
  --time_limit 01:00:00 \
  --num_nodes $NUM_NODES \
  --train_data $DATA_DIR \
  --val_data $DATA_DIR \
  --model_path $MODEL_PATH \
  --max_restarts $MAX_RESTARTS \
  --partition gh-dev \
  --skyrl_entrypoint skyrl_train.entrypoints.main_base \
  -S trainer.export_path=$SKYRL_EXPORT_PATH \
  -S trainer.ckpt_path=$SKYRL_EXPORT_PATH \
  -S trainer.algorithm.advantage_estimator="grpo" \
  -S trainer.placement.colocate_all=true \
  -S trainer.strategy=fsdp2 \
  -S trainer.placement.policy_num_nodes=$NUM_NODES \
  -S trainer.placement.ref_num_nodes=$NUM_NODES \
  -S trainer.placement.policy_num_gpus_per_node=1 \
  -S trainer.placement.ref_num_gpus_per_node=1 \
  -S generator.num_inference_engines=$NUM_NODES \
  -S generator.inference_engine_tensor_parallel_size=1 \
  -S trainer.epochs=20 \
  -S trainer.eval_batch_size=1024 \
  -S trainer.eval_before_train=true \
  -S trainer.eval_interval=5 \
  -S trainer.update_epochs_per_batch=1 \
  -S trainer.train_batch_size=1024 \
  -S trainer.policy_mini_batch_size=256 \
  -S trainer.micro_forward_batch_size_per_gpu=64 \
  -S trainer.micro_train_batch_size_per_gpu=64 \
  -S trainer.ckpt_interval=10 \
  -S trainer.hf_save_interval=10 \
  -S trainer.max_prompt_length=512 \
  -S generator.sampling_params.max_generate_length=1024 \
  -S trainer.policy.optimizer_config.lr=1.0e-6 \
  -S trainer.algorithm.use_kl_loss=true \
  -S generator.n_samples_per_prompt=5 \
  -S generator.gpu_memory_utilization=0.8 \
  -S trainer.logger=$LOGGER \
  -S trainer.project_name=OpenThoughts-Agent \
  -S trainer.run_name=$RUN_NAME \
  -S trainer.resume_mode=null \
  -S environment.env_class=gsm8k \
  -S generator.backend=vllm \
  -S generator.run_engines_locally=true \
  -S generator.weight_sync_backend=nccl \
  -S generator.async_engine=true \
  -S generator.batched=true \
  $@
