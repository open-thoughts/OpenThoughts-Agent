#!/bin/bash
# Usage: ./train_pipeline.sh <experiment>
experiment=$1
cd $EVALCHEMY
cd $DCFT_PRIVATE
python3 -m hpc.launch --max_restarts 3 --train_config_path sft/hp_settings/paper/reasoning_medium.yaml --time_limit $DEFAULT_TIME_LIMIT --num_nodes $NUM_NODES_FAST --dataset mlfoundations-dev/${experiment}
