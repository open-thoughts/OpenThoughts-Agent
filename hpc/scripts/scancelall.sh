#!/bin/bash

# Check if stdin is a pipe or terminal
if [ -t 0 ]; then
  echo "Error: This script expects input piped from sqme or sqteam (possibly filtered with grep)."
  echo "Example usage: sqteam | grep mydesc | cancelall"
  exit 1
fi

# Read input from stdin and save it
job_list=$(cat)

# Extract job IDs
job_ids=$(echo "$job_list" | awk '{print $1}' | grep -v "JOBID")

# Check if there are jobs to cancel
if [ -z "$job_ids" ]; then
  echo "No jobs found to cancel."
  exit 0
fi

# Count jobs to cancel
job_count=$(echo "$job_ids" | wc -l | tr -d ' ')

# Display jobs that will be cancelled
echo "The following $job_count jobs will be cancelled:"
echo "$job_list" | column -t

# Ask for confirmation using /dev/tty to read user input directly from the terminal
read -p "Are you sure you want to cancel these jobs? (y/n): " confirm </dev/tty

if [[ "$confirm" == [Yy]* ]]; then
  echo "Cancelling jobs..."
  for job_id in $job_ids; do
    echo "Cancelling job $job_id"
    scancel "$job_id"
  done
  echo "Done. All $job_count jobs have been cancelled."
else
  echo "Operation cancelled by user."
fi