#!/bin/bash

# Create temp directory if it doesn't exist
mkdir -p /tmp/dcft_uploads

# Initialize the last check time to one hour ago (first run)
LAST_CHECK_TIME=$(date -d "-1 hour" +"%F-%H:%M:%S")

while true; do
  # Get current date for logging
  CURRENT_DATE=$(date +"%Y-%m-%d_%H:%M:%S")
  
  # Create temp file for job names
  TEMP_FILE="/tmp/dcft_uploads/completed_jobs_$CURRENT_DATE.txt"
  
  # Log the check start
  echo "=== $CURRENT_DATE - Starting check ===" >> $DCFT_PRIVATE/hpc/scripts/auto_upload.log
  echo "Checking jobs completed since $LAST_CHECK_TIME" >> $DCFT_PRIVATE/hpc/scripts/auto_upload.log
  
  # Get completed job names from the last check time and save to temp file
  sacct -X -S "$LAST_CHECK_TIME" -o JobID%10,JobName%80,State | grep -v -E "_eval|_pretokenize" | awk '$3=="COMPLETED" {print $2}' > $TEMP_FILE
  
  # Log the completed jobs to the log file
  echo "Found completed jobs: " >> $DCFT_PRIVATE/hpc/scripts/auto_upload.log
  cat $TEMP_FILE >> $DCFT_PRIVATE/hpc/scripts/auto_upload.log
  
  # Store the current time as the next last check time BEFORE uploading
  # This ensures we don't miss jobs that complete during the upload
  LAST_CHECK_TIME=$(date +"%F-%H:%M:%S")

  # Upload the temp file and capture the output
  echo "Uploading completed jobs: " >> $DCFT_PRIVATE/hpc/scripts/auto_upload.log
  UPLOAD_OUTPUT=$(cat $TEMP_FILE | $DCFT_PRIVATE/hpc/scripts/upload.sh 2>&1)
  
  # Log results
  echo "Found $(wc -l < $TEMP_FILE) completed jobs" >> $DCFT_PRIVATE/hpc/scripts/auto_upload.log
  echo "Upload output:" >> $DCFT_PRIVATE/hpc/scripts/auto_upload.log
  echo "$UPLOAD_OUTPUT" >> $DCFT_PRIVATE/hpc/scripts/auto_upload.log
  echo "Next check will be from $LAST_CHECK_TIME" >> $DCFT_PRIVATE/hpc/scripts/auto_upload.log
  echo "" >> $DCFT_PRIVATE/hpc/scripts/auto_upload.log
  
  # Wait for one hour (3600 seconds) before the next iteration
  sleep 3600
done