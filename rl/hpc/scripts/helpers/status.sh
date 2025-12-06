#!/bin/bash
# Status monitoring script for OpenThoughts-Agent jobs

LINES=${1:-1}

echo "=== Recent Job Status ==="
echo "Recent jobs:"
squeue -u $USER --format="%.10i %.9P %.40j %.8u %.1T %.8M %.9l %.6D %.15r" | head -10

echo ""
echo "=== Recent Logs ==="
if [ -d "$OT_AGENT_RL/experiments/logs" ]; then
    echo "Recent log files:"
    ls -lt $OT_AGENT_RL/experiments/logs/*.out | head -5
    
    echo ""
    echo "Latest log content (last $LINES lines):"
    LATEST_LOG=$(ls -t $OT_AGENT_RL/experiments/logs/*.out | head -1)
    if [ -n "$LATEST_LOG" ]; then
        tail -n $LINES "$LATEST_LOG"
    fi
else
    echo "No logs directory found at $OT_AGENT_RL/experiments/logs"
fi

echo ""
echo "=== Checkpoint Status ==="
if [ -d "$CHECKPOINTS_DIR" ]; then
    echo "Recent checkpoints:"
    find $CHECKPOINTS_DIR -name "*.pt" -o -name "*.pth" -o -name "*.safetensors" | head -5
else
    echo "No checkpoints directory found at $CHECKPOINTS_DIR"
fi
