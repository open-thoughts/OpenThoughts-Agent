#!/bin/bash
# Show cancelled jobs in the last N hours

HOURS=${1:-1}

echo "=== Cancelled Jobs in the last $HOURS hour(s) ==="
sacct -S $(date -d "$HOURS hours ago" +%Y-%m-%d) -u $USER --format=JobID,JobName,State,ExitCode,Start,End,Elapsed,MaxRSS,ReqMem,ReqCPUS,ReqNodes,Partition,Account | grep "CANCELLED"
