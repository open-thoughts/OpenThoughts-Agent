#!/bin/bash

# Script that reads directory patterns from stdin or command line arguments
# Usage 1: ./upload.sh << EOF
#        pattern1
#        pattern2
#        EOF
# Usage 2: ./upload.sh pattern1 pattern2

# Function to process a pattern
process_pattern() {
  local pattern="$1"
  
  # Skip empty patterns
  [[ -z "$pattern" ]] && return
  
  echo "Processing pattern: $pattern"
  
  # List matching directories
  echo "Listing directories matching pattern '$pattern':"
  find "$CHECKPOINTS_DIR" -name "$pattern" -type d | xargs ls
  
  # Display last line of trainer_log.jsonl
  echo "Displaying last line of trainer_log.jsonl for pattern '$pattern':"
  find "$CHECKPOINTS_DIR" -name "$pattern" -type d | xargs -I{} tail -n 1 {}/trainer_log.jsonl
  
  # Replace path in README.md using MODELS_DIR variable
  echo "Updating README.md for pattern '$pattern':"
  find "$CHECKPOINTS_DIR" -name "$pattern" -type d | xargs -I{} sed -i "s|$MODELS_DIR/models--Qwen--Qwen2.5-7B-Instruct/snapshots/[a-f0-9]*|Qwen/Qwen2.5-7B-Instruct|g" {}/README.md
  
  # Upload to Hugging Face
  echo "Uploading directories matching pattern '$pattern' to Hugging Face:"
  find "$CHECKPOINTS_DIR" -name "$pattern" -type d | xargs -I{} bash -c 'MODEL_NAME=$(basename {}) && echo "Uploading $MODEL_NAME" && huggingface-cli upload mlfoundations-dev/$MODEL_NAME {} --exclude="checkpoint*" --repo-type=model --commit-message="Upload model"'
  
  # Display upload completion message
  echo "Displaying upload completion messages for pattern '$pattern':"
  find "$CHECKPOINTS_DIR" -name "$pattern" -type d | xargs -I{} bash -c 'MODEL_NAME=$(basename {}) && echo "Uploaded $MODEL_NAME to https://huggingface.co/mlfoundations-dev/$MODEL_NAME"'
  
  echo "Finished processing pattern: $pattern"
  echo "----------------------------------------"
}

# Check if CHECKPOINTS_DIR is set
if [ -z "$CHECKPOINTS_DIR" ]; then
  echo "Error: CHECKPOINTS_DIR environment variable is not set"
  echo "Please set it with: export CHECKPOINTS_DIR=/path/to/checkpoints"
  exit 1
fi

# Check if MODELS_DIR is set
if [ -z "$MODELS_DIR" ]; then
  echo "Error: MODELS_DIR environment variable is not set"
  echo "Please set it with: export MODELS_DIR=/path/to/models"
  exit 1
fi

# Check if command line arguments are provided
if [ $# -gt 0 ]; then
  # Use command line arguments
  for pattern in "$@"; do
    process_pattern "$pattern"
  done
else
  # Read from stdin
  while IFS= read -r pattern; do
    process_pattern "$pattern"
  done
fi
