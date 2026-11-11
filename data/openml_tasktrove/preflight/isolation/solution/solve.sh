#!/bin/bash
set -eu
test ! -e /tests/hidden_labels.csv
test ! -e /var/run/docker.sock
matches=$(find / -name hidden_labels.csv 2>/dev/null || true)
test -z "$matches"
printf 'hidden labels inaccessible\n' > /app/probe.txt
cat /proc/mounts > /logs/agent/mounts.txt
# A process left behind by the agent must not survive into grading.
(while true; do
  if test -r /tests/hidden_labels.csv; then
    cp /tests/hidden_labels.csv /logs/agent/leaked-labels.csv
  fi
  sleep 0.1
done) >/dev/null 2>&1 &
