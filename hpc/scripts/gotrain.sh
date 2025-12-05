#!/bin/bash
# Usage: ./train_pipeline.sh <experiment>
experiment=$1
cd $DCFT_PRIVATE
python3 -m hpc.launch --train_config_path sft/hp_settings/paper/reasoning_medium.yaml --time_limit $DEFAULT_TIME_LIMIT --num_nodes $NUM_NODES_DEFAULT --dataset mlfoundations-dev/${experiment}
