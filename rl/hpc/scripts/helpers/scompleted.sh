#!/bin/bash
# Show completed jobs in the last N hours

HOURS=${1:-1}

echo "=== Completed Jobs in the last $HOURS hour(s) ==="
sacct -S $(date -d "$HOURS hours ago" +%Y-%m-%d) -u $USER --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS,ReqMem,ReqCPUS,ReqNodes,Partition,Account | grep "COMPLETED"
