#!/bin/bash
# Usage: ./evaluate_pipeline.sh <experiment>
experiment=$1
cd $EVALCHEMY
python eval/distributed/launch_simple.py --tasks AIME24,AMC23,MATH500,MMLUPro,JEEBench,GPQADiamond,LiveCodeBench,CodeElo,CodeForces --num_shards 16 --max-job-duration 4 --model_name "mlfoundations-dev/${experiment}"