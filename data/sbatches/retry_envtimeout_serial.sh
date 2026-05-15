#!/bin/bash
# Retry all datasets that had EnvironmentStartTimeoutError, submitted serially
# Harbor auto-resumes: detects existing job dir and retries only failed trials
# Ordered by env timeout count (worst first)
#
# Usage: bash retry_envtimeout_serial.sh [--after JOBID]
#   --after JOBID: first job depends on this job finishing (default: 225009)

set -e

SBATCH_SCRIPT="/e/scratch/jureap59/etash/OpenThoughts-Agent/data/sbatches/run_harbor_glm47_jupiter.sbatch"

# Datasets with >1% EnvironmentStartTimeoutError, ordered worst-first
DATASETS=(
  "DCAgent/exp-gfi-staqc-askllm-filtered-10K"          # 4634 (46.3%)
  "DCAgent/exp-syh-tezos-askllm-hardened"               # 3965 (39.6%)
  "DCAgent/exp-syh-tezos-stackoverflow-mixed"           # 3617 (36.2%)
  "DCAgent/exp-uns-tezos-10x"                           # 3295 (33.0%)
  "DCAgent/exp-syh-r2egym-swesmith-mixed"               # 2794 (27.9%)
  "DCAgent/dev_set_part1_10k"                           # 2689 (26.9%)
  "DCAgent/perturbed-docker-exp-magicoder-tasks-2"      # 2689 (26.9%)
  "DCAgent/exp-uns-r2egym-33_6x"                        # 2339 (23.4%)
  "DCAgent/exp-syh-r2egym-askllm-constrained"           # 2151 (21.5%)
  "DCAgent/exp-gfi-swesmith-random-filtered-10K"        # 1536 (15.4%)
  "DCAgent/exp-uns-r2egym-2_1x"                         # 1201 (12.0%)
  "DCAgent/exp-uns-tezos-128unique"                     #  960 (9.6%)
  "DCAgent/exp-uns-tezos-80x"                           #  695 (7.0%)
  "DCAgent/exp-uns-tezos-160x"                          #  520 (5.2%)
  "DCAgent/exp-uns-r2egym-16_8x"                        #  429 (4.3%)
)

# Parse --after flag
AFTER_JOB="${2:-225009}"
if [ "$1" = "--after" ] && [ -n "$2" ]; then
    AFTER_JOB="$2"
fi

echo "Submitting ${#DATASETS[@]} retry jobs (serial chain)"
echo "First job depends on: $AFTER_JOB"
echo ""

PREV_JOB="$AFTER_JOB"
for ds in "${DATASETS[@]}"; do
  echo "Submitting: $ds (after $PREV_JOB)"
  JOB_ID=$(sbatch --dependency=afterany:${PREV_JOB} "$SBATCH_SCRIPT" "$ds" | awk '{print $4}')
  echo "  -> Job $JOB_ID"
  PREV_JOB="$JOB_ID"
  sleep 1
done

echo ""
echo "Done. Submitted ${#DATASETS[@]} retry jobs in serial chain."
echo "First: depends on $AFTER_JOB"
echo "Last: $PREV_JOB"
echo "Monitor with: squeue -u $USER"
