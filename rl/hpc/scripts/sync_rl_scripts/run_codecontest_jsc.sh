#!/bin/bash

NUM_NODES=2 # JSC-specific number of nodes
NUM_GPUS_PER_NODE=4
LOGGER="wandb"
RUN_NAME="code-contests-qwen3-8b-2nodes-sync"  # The name of this run
TRAIN_DATA_DIR="DCAgent/code-contests-sandboxes-with-tests" 
#TRAIN_DATA_DIR="/p/scratch/laionize/dc-agent-experiments/code-contests-parquet/tasks_extracted"
#TRAIN_DATA_DIR="DCAgent2/nl2bash-gpt-5-codex-passed"
EVAL_DATA_DIR="DCAgent/dev_set_71_tasks"
MODEL_PATH="Qwen/Qwen3-8B"  # base model to start the RL from

# Use environment variables from jsc.env if available, otherwise fallback to reasonable defaults relative to user scratch
: "${DCFT_SCRATCH:=/p/scratch/laionize/${USER}/dc-agent-shared}"
SKYRL_EXPORT_PATH="${DCFT_SCRATCH}/exports/${RUN_NAME}"
SANDBOXES_DIR="${DCFT_SCRATCH}/sandboxes/run"
# Use a separate subdir for checkpoints under the export path
SKYRL_CKPT_PATH="${SKYRL_EXPORT_PATH}/ckpts"

# We set train_batch_size and mini_batch_size to the same value for on-policy
TRAIN_AND_MINI_BATCH_SIZE=4
EVAL_BATCH_SIZE=128  # we can afford more concurrency because there are only 71 tasks in the eval set
EVAL_INTERVAL=20
MAX_RESTARTS=2
EPOCHS=1  # since code contest is 13k samples -- about 200 steps
# Set partition and QOS (JSC-specific)
PARTITION="booster"
QOS="normal"
TIME_LIMIT="06:00:00"

source "$(dirname "$0")/run_common.sh"


