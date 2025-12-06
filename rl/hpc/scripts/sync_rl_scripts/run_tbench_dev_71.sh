#!/bin/bash
NUM_NODES=8  # In TACC, number of nodes = number of GPUs. Need to tune this for JSC.
LOGGER="wandb"  # wandb for logging into WANDB, console for printing to stdout
RUN_NAME="tbench-dev-71-qwen3-8b-8nodes-sync"  # The name of this run (fed to `-S trainer.run_name` for WANDB logging and various other places)
TRAIN_DATA_DIR="DCAgent/dev_set_71_tasks"
EVAL_DATA_DIR="DCAgent/dev_set_71_tasks"
MODEL_PATH="Qwen/Qwen3-8B"  # base model to start the RL from

# We will dump the following in the export path: SkyRL dumped eval output, and model checkpoints
# Note that this differs from the EXPERIMENTS_DIR specified in {hpc_name}.env, which is where we
# dump the sbatch scripts and SLURM `.out` logs.
SKYRL_CKPT_PATH=$SCRATCH/skyrl_exports/$RUN_NAME/ckpts
SKYRL_EXPORT_PATH=$SCRATCH/skyrl_exports/$RUN_NAME/exports
SANDBOXES_DIR="/scratch/08134/negin/dc-agent-shared/sandboxes/run"

# We set train_batch_size and mini_batch_size to the same value for on-policy
TRAIN_AND_MINI_BATCH_SIZE=64
EVAL_BATCH_SIZE=128  # we can afford more concurrency because there is no n_samples_per_prompt for eval
EVAL_INTERVAL=200  # since tbench dev itself is the eval set, we don't run eval

MAX_RESTARTS=2

EPOCHS=100  # since tbench dev only has 1 step per epoch, we run 100 steps

source $(dirname "$0")/run_common.sh
