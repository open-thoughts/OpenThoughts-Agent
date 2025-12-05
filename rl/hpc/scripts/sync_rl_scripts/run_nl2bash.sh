#!/bin/bash
NUM_NODES=8  # In TACC, number of nodes = number of GPUs. Need to tune this for JSC.
LOGGER="wandb"  # wandb for logging into WANDB, console for printing to stdout
RUN_NAME="nl2bash-qwen3-8b-8nodes-sync"  # The name of this run (fed to `-S trainer.run_name` for WANDB logging and various other places)
TRAIN_DATA_DIR="DCAgent2/nl2bash-tasks-cleaned-oracle"  # can also be a local path
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
EVAL_BATCH_SIZE=128  # we can afford more concurrency because there are only 71 tasks in the eval set
EVAL_INTERVAL=20

MAX_RESTARTS=2

EPOCHS=3  # since nl2bash is 1.57k samples -- about 24 steps

source $(dirname "$0")/run_common.sh
