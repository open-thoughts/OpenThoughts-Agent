#!/bin/bash
# ==============================================================================
# Test: vLLM native Data Parallel eval (DP=4, TP=1) on single node
#
# Uses vLLM's --data-parallel-size flag to run 4 model replicas (one per GPU),
# each with TP=1. vLLM load-balances requests across replicas internally.
# Harbor sees a single endpoint — no sharding needed.
#
# Usage:
#   bash eval/MBZ/test_dp_eval.sh
#
# To clean up DB entries from failed tests:
#   python -c "
#   from database.unified_db.utils import delete_sandbox_job_by_id, load_supabase_keys
#   load_supabase_keys()
#   delete_sandbox_job_by_id('<uuid>')
#   "
# ==============================================================================

set -euo pipefail

MODEL="DCAgent/a1-nl2bash"
DATASET="DCAgent/dev_set_v2"
BENCHMARK_ID="b94dfab2-c438-4c32-ba29-23e46d566763"

# DP config: 4 replicas, TP=1 each, 16 concurrent harbor trials per replica = 64 total
DP_SIZE=4
TP_SIZE=1
N_CONCURRENT=64  # 16 per DP replica × 4 replicas
GPU_MEMORY_UTIL=0.9
TIMEOUT_MULTIPLIER=2.0

echo "=== vLLM Native DP Test ==="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "DP=$DP_SIZE, TP=$TP_SIZE, N_CONCURRENT=$N_CONCURRENT"
echo ""

# Submit using the v6 sbatch with DP env vars
EVAL_VLLM_TENSOR_PARALLEL_SIZE=$TP_SIZE \
EVAL_VLLM_DATA_PARALLEL_SIZE=$DP_SIZE \
EVAL_N_CONCURRENT=$N_CONCURRENT \
EVAL_GPU_MEMORY_UTIL=$GPU_MEMORY_UTIL \
EVAL_ENABLE_THINKING=true \
EVAL_TIMEOUT_MULTIPLIER=$TIMEOUT_MULTIPLIER \
EVAL_VLLM_MAX_RETRIES=10 \
EVAL_DAYTONA_THRESHOLD=10 \
EVAL_CONDA_ENV=otagent2 \
EVAL_AUTO_SNAPSHOT=true \
EVAL_CONFIG_YAML=dcagent_eval_config_no_override.yaml \
sbatch \
  --time 24:00:00 \
  --partition main \
  --gres gpu:4 \
  --cpus-per-task=32 \
  --job-name data_dp_test \
  --output eval/MBZ/logs/data_dp_test_%j.out \
  eval/MBZ/unified_eval_harbor_v6.sbatch \
  "$MODEL" "$DATASET" "$BENCHMARK_ID" ""

echo "Submitted! Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f eval/MBZ/logs/data_dp_test_*.out"
echo "  tail -f experiments/logs/vllm_*.log"
