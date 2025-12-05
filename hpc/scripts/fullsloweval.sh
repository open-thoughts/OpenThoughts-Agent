#!/bin/bash
# Usage: ./evaluate_pipeline.sh <experiment>
experiment=$1
cd $EVALCHEMY
python eval/distributed/launch_simple.py --tasks AIME24,AMC23,MATH500,MMLUPro,JEEBench,GPQADiamond,LiveCodeBench,CodeElo,CodeForces,AIME25,HLE,LiveCodeBenchv5 --num_shards 4 --max-job-duration 16 --model_name "mlfoundations-dev/${experiment}"