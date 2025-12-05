#!/bin/bash

# Script to launch train and eval for specified pipeline experiment 
# Usage: ./go.sh <experiment_name>

if [ -z "$1" ]; then
    echo "Error: Experiment name is required (e.g. a1_math_numina_math)"
    echo "Usage: ./go.sh <experiment_name>"
    exit 1
fi
experiment=$1

cd $DCFT_PRIVATE
python3 -m hpc.launch \
    --train_config_path sft/hp_settings/paper/reasoning_medium.yaml \
    --time_limit $DEFAULT_TIME_LIMIT \
    --num_nodes $NUM_NODES_FAST \
    --dataset mlfoundations-dev/${experiment} \
    --eval_tasks AIME24,AMC23,MATH500,MMLUPro,JEEBench,GPQADiamond,LiveCodeBench,CodeElo,CodeForces \
    --eval_num_nodes $NUM_NODES_FAST