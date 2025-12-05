#!/bin/bash

# Parse arguments: first arg is the number of lines to tail
TAIL_LINES=1
if [[ "$1" =~ ^[0-9]+$ ]]; then
  TAIL_LINES=$1
  shift
fi

# Check if stdin is a pipe or terminal
if [ -t 0 ]; then
  # No input piped, default to sqteam command
  queue_output=$(squeue --format="%.10i %.9P %.40j %.8u %.1T %.8M %.9l %.6D %.15r" -A $(sacctmgr -n show associations user=$USER format=account | awk 'NR==1{print $1}' | xargs))
  is_squeue=true
else
  # Input is piped, read from stdin
  queue_output=$(cat)
  # Check if output is from squeue or sacct (sfail/swin)
  if echo "$queue_output" | grep -q "JOBID.*PARTITION.*NAME"; then
    is_squeue=false
  else
    is_squeue=true
  fi
fi

process_job() {
  local jobname=$1
  local jobid=$2
  
  echo ""  
  echo "Job: $jobname"
  echo "JobID: $jobid"
  
  # First check checkpoint logs (always show only the last line)
  found_logs=false
  find $CHECKPOINTS_DIR -name "*${jobname}*" -type d -exec test -f "{}/trainer_log.jsonl" \; -print | while read dir; do
    LOG_PATH="$dir/trainer_log.jsonl"
    echo "=== Checkpoint Log: $LOG_PATH ==="
    found_logs=true
    tail -n 1 "$LOG_PATH"
  done
  
  if [ "$found_logs" = false ]; then
    echo "No checkpoint logs found for this job"
  fi
  
  # Then get SLURM log output (using user-specified line count)
  LOG_PATH="experiments/logs/${jobname}_${jobid}.out"
  if [ -f "$LOG_PATH" ]; then
    echo "=== SLURM Log: $LOG_PATH ==="
    tail -n $TAIL_LINES "$LOG_PATH"
  else
    echo "Log file not found: $LOG_PATH"
  fi
}

if [ "$is_squeue" = true ]; then
  # Processing squeue output (sqme, sqteam)
  echo "$queue_output" | awk '{print $3}' | grep -v "NAME" | grep -v "_eval" | while read jobname; do
    jobid=$(echo "$queue_output" | grep "$jobname" | awk '{print $1}')
    process_job "$jobname" "$jobid"
  done
else
  # Processing sacct output (sfail, swin)
  echo "$queue_output" | grep -v "JOBID" | grep -v "_eval" | while read -r line; do
    # Parse job ID and name from sacct output
    jobid=$(echo "$line" | awk '{print $1}')
    jobname=$(echo "$line" | awk '{print $3}')
    process_job "$jobname" "$jobid"
  done
fi
