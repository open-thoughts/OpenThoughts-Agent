#!/bin/bash

# super_go.sh - Combines all "go" scripts with configurable size, speed, and evaluation type
#
# Basic usage:
#   go experiment_name                  # train + pipeline eval with medium HP
#   go small experiment_name            # train + pipeline eval with small HP
#   go large experiment_name            # train + pipeline eval with large HP
#   go micro experiment_name            # train + pipeline eval with micro HP
#
# Mode options:
#   go small train experiment_name      # only train with small HP, no eval
#   go eval experiment_name             # only run pipeline eval (no train)
#
# Evaluation types:
#   go full experiment_name             # train with medium HP + FULL evaluation (all benchmarks)
#   go small full experiment_name       # train with small HP + FULL evaluation (all benchmarks)
#   go eval full experiment_name        # only run FULL evaluation (no train, size ignored)
#
# Speed options:
#   go fast experiment_name             # fast training with medium HP (more nodes)
#
# Combined examples:
#   go small train fast experiment_name # fast training with small HP, no eval
#   go small full fast experiment_name  # fast training with small HP + FULL evaluation
#   go eval full experiment_name        # FULL evaluation only (no train)

cd $DCFT_PRIVATE

# Default values
SIZE="medium"     # medium is default
MODE="traineval"  # traineval = train + standard eval
SPEED="normal"    # normal or fast
EVAL_TYPE="pipeline"  # pipeline or full
EXPERIMENT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        small|micro|medium|large)
            SIZE="$1"
            shift
            ;;
        train)
            MODE="train"
            shift
            ;;
        eval)
            MODE="eval"
            shift
            ;;
        fast)
            SPEED="fast"
            shift
            ;;
        slow)
            SPEED="slow"
            shift
            ;;
        full)
            EVAL_TYPE="full"
            shift
            ;;
        *)
            EXPERIMENT="$1"
            shift
            ;;
    esac
done

# Check if experiment name is provided
if [ -z "$EXPERIMENT" ]; then
    echo "Error: Experiment name is required"
    echo "Usage: super_go.sh [size] [mode] [speed] [eval_type] experiment_name"
    echo "  size: small, micro, medium, large (default: medium)"
    echo "  mode: train, eval (default: both)"
    echo "  speed: fast, slow (default: normal)"
    echo "  eval_type: full (default: pipeline)"
    echo "  experiment_name: required parameter"
    exit 1
fi

# Set HP config based on size
HP_CONFIG="sft/hp_settings/paper/reasoning_${SIZE}.yaml"
if [ ! -f "$HP_CONFIG" ]; then
    echo "Error: HP config $HP_CONFIG not found. Please choose a valid size (small, micro, medium, large)."
    exit 1
fi

# Set nodes based on speed
if [ "$SPEED" == "fast" ]; then
    NODES="$NUM_NODES_FAST"
elif [ "$SPEED" == "slow" ]; then
    NODES="$NUM_NODES_SLOW"
else  # normal
    NODES="$NUM_NODES_DEFAULT"
fi

# Standard evaluation tasks
STANDARD_EVAL_TASKS="AIME24,AMC23,MATH500,MMLUPro,JEEBench,GPQADiamond,LiveCodeBench,CodeElo,CodeForces"

# Full evaluation tasks
FULL_EVAL_TASKS="AIME24,AMC23,MATH500,MMLUPro,JEEBench,GPQADiamond,LiveCodeBench,CodeElo,CodeForces,AIME25,HLE,LiveCodeBenchv5"

# Set the eval tasks based on eval type
if [ "$EVAL_TYPE" == "full" ]; then
    EVAL_TASKS="$FULL_EVAL_TASKS"
    echo "Using FULL evaluation tasks: $EVAL_TASKS"
else
    EVAL_TASKS="$STANDARD_EVAL_TASKS"
    echo "Using standard pipeline evaluation tasks: $EVAL_TASKS"
fi

# Function to calculate number of shards and max duration based on node count
# Returns: "num_shards max_duration"
calculate_shards_and_duration() {
    local nodes=$1
    local num_shards=$(($nodes * $NUM_GPUS_PER_NODE))
    local max_duration=$((64 / $num_shards))
    if [ $max_duration -lt 1 ]; then
        max_duration=1
    fi
    echo "$num_shards $max_duration"
}

# Execute based on mode
if [ "$MODE" == "traineval" ] || [ "$MODE" == "train" ]; then
    echo "Starting training job for $EXPERIMENT with ${SIZE} HP config using ${NODES} nodes"
    
    # Determine if we should set up evaluation as part of the train command
    if [ "$MODE" == "traineval" ]; then
        # Calculate shards and duration using the shared function
        read NUM_EVAL_SHARDS EVAL_DURATION <<< $(calculate_shards_and_duration $NODES)
        
        # Format as HH:00:00 for eval_time_limit
        EVAL_TIME_LIMIT="${EVAL_DURATION}:00:00"
        
        EVAL_CMD="python3 -m hpc.launch \
            --train_config_path \"$HP_CONFIG\" \
            --time_limit $DEFAULT_TIME_LIMIT \
            --num_nodes $NODES \
            --dataset \"mlfoundations-dev/${EXPERIMENT}\" \
            --eval_tasks \"$EVAL_TASKS\" \
            --eval_num_nodes $NODES \
            --eval_time_limit \"$EVAL_TIME_LIMIT\""
    else
        EVAL_CMD="python3 -m hpc.launch \
            --train_config_path \"$HP_CONFIG\" \
            --time_limit $DEFAULT_TIME_LIMIT \
            --num_nodes $NODES \
            --dataset \"mlfoundations-dev/${EXPERIMENT}\""
    fi

    echo "Running: $EVAL_CMD"
    
    # Execute the command
    eval $EVAL_CMD
fi

# Handle separate evaluation if needed
if [ "$MODE" == "eval" ]; then
    cd $EVALCHEMY
    
    # Print message about size being ignored in eval-only mode if specified
    if [ "$SIZE" != "medium" ]; then
        echo "Note: Size parameter ($SIZE) is ignored in eval-only mode"
    fi
    
    echo "Starting evaluation job for $EXPERIMENT with tasks: $EVAL_TASKS"
    
    # Set shard count and job duration using the shared function
    read NUM_SHARDS MAX_JOB_DURATION <<< $(calculate_shards_and_duration $NODES)
    echo "Using ${NUM_SHARDS} shards with ${MAX_JOB_DURATION} hour jobs (64/$NUM_SHARDS)"

    # Execute evaluation
    python eval/distributed/launch_simple.py \
        --tasks $EVAL_TASKS \
        --num_shards $NUM_SHARDS \
        --max-job-duration $MAX_JOB_DURATION \
        --model_name "mlfoundations-dev/${EXPERIMENT}"
fi

# Final completion message - only mention HP size if training was involved
if [ "$MODE" == "traineval" ] || [ "$MODE" == "train" ]; then
    echo "Completed operations for $EXPERIMENT with ${SIZE} HP config"
else
    echo "Completed operations for $EXPERIMENT"
fi