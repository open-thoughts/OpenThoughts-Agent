#!/bin/bash
# Usage: ./scancelled.sh <hours>
# Set default value of 1 hour if no parameter is provided
if [ -z "$1" ]; then
    hours=1
else
    hours=$1
fi
sacct -X -S "$(date -d "-$hours hours" +%F-%H:%M)" -o JobID%10,Partition%9,JobName%40,State,Start,Elapsed,Node%9 | grep "CANCELLED"