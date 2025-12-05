#!/bin/bash

# Script to remove log files with numbers below a specified threshold
# Usage: ./rmlogs.sh [threshold_number]

# Function to display usage
show_usage() {
  echo "Usage: $0 [threshold_number]"
  echo "If no threshold is provided, you will be prompted to enter one."
}

# Get the threshold number
if [ $# -eq 1 ]; then
  # Check if the argument is a valid number
  if [[ $1 =~ ^[0-9]+$ ]]; then
    threshold=$1
  else
    echo "Error: Threshold must be a positive integer."
    show_usage
    exit 1
  fi
else
  # If no argument provided, prompt the user
  echo -n "Enter the threshold number (files with numbers below this will be listed): "
  read threshold
  
  # Validate input
  if ! [[ $threshold =~ ^[0-9]+$ ]]; then
    echo "Error: Threshold must be a positive integer."
    exit 1
  fi
fi

echo "Searching for log files with numbers below $threshold..."

# Find and list files that would be removed
count=0
declare -a files_to_remove

while read file; do
  num=$(echo "$file" | grep -o '[0-9]\+\.out$' | grep -o '[0-9]\+')
  if [ "$num" -lt "$threshold" ]; then
    echo "Would remove: $file"
    files_to_remove+=("$file")
    ((count++))
  fi
done < <(find $DCFT_PRIVATE/experiments/logs/ -type f -name "*.out" | grep -E '[0-9]+\.out$')

# Summary
echo "Found $count files to remove."

# Ask for confirmation before actually removing
if [ $count -gt 0 ]; then
  echo -n "Do you want to proceed with deletion? (y/n): "
  read confirm
  
  if [[ $confirm == [Yy]* ]]; then
    for file in "${files_to_remove[@]}"; do
      rm "$file"
      echo "Removed: $file"
    done
    echo "Deletion complete. Removed $count files."
  else
    echo "Operation cancelled. No files were removed."
  fi
else
  echo "No files found below the threshold. Nothing to remove."
fi