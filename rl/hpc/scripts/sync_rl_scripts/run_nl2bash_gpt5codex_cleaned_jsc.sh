#!/bin/bash

NUM_NODES=2  # JSC-specific number of nodes (tune as needed)
NUM_GPUS_PER_NODE=4
LOGGER="wandb"
RUN_NAME="nl2bashGPT5CodexPassed-qwen3-8b-8nodes-sync"
TRAIN_DATA_DIR="DCAgent2/nl2bash-gpt-5-codex-passed"
EVAL_DATA_DIR="DCAgent/dev_set_71_tasks"
MODEL_PATH="Qwen/Qwen3-8B"

# Use environment variables from jsc.env if available, otherwise fallback to reasonable defaults
: "${DCFT_SCRATCH:=/p/scratch/laionize/${USER}/dc-agent-shared}"
SKYRL_EXPORT_PATH="${DCFT_SCRATCH}/exports/${RUN_NAME}"
SKYRL_CKPT_PATH="${SKYRL_EXPORT_PATH}/ckpts"
SANDBOXES_DIR="${DCFT_SCRATCH}/sandboxes/run"

# We set train_batch_size and mini_batch_size to the same value for on-policy
TRAIN_AND_MINI_BATCH_SIZE=64
EVAL_BATCH_SIZE=128
EVAL_INTERVAL=20
MAX_RESTARTS=2
EPOCHS=3  # cleaned nl2bash is ~700 samples -- about 11 steps

# Set partition and QOS (JSC-specific)
PARTITION="booster"
QOS="normal"
TIME_LIMIT="24:00:00"

source "$(dirname "$0")/run_common.sh"


