#!/bin/bash
# Remove old log files

THRESHOLD=${1:-""}

if [ -z "$THRESHOLD" ]; then
    echo "Usage: rmlogs <threshold>"
    echo "Example: rmlogs 100 (removes logs with job ID < 100)"
    echo "Current log files:"
    ls -la $DC_AGENT_TRAIN/experiments/logs/*.out 2>/dev/null | head -10
    echo ""
    echo "Enter threshold (job ID): "
    read THRESHOLD
fi

if [ -z "$THRESHOLD" ]; then
    echo "No threshold provided. Exiting."
    exit 1
fi

echo "Removing log files with job ID < $THRESHOLD..."

if [ -d "$DC_AGENT_TRAIN/experiments/logs" ]; then
    for file in $DC_AGENT_TRAIN/experiments/logs/*.out; do
        if [ -f "$file" ]; then
            job_id=$(basename "$file" | grep -o '[0-9]\+' | tail -1)
            if [ -n "$job_id" ] && [ "$job_id" -lt "$THRESHOLD" ]; then
                echo "Removing $file (job ID: $job_id)"
                rm "$file"
            fi
        fi
    done
    echo "Done."
else
    echo "No logs directory found at $DC_AGENT_TRAIN/experiments/logs"
fi
