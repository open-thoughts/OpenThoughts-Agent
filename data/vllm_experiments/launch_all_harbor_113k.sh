#!/bin/bash
# Launch all harbor 113K experiments
# Generated automatically

set -e

EXPERIMENTS_DIR="${DCAGENT_DIR}/data/vllm_experiments"
cd "$EXPERIMENTS_DIR"

echo "Launching 31 harbor 113K experiments..."
echo ""

# Experiment 1/31
echo "Submitting: run_harbor_113k_baseline_conservative.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_baseline_conservative.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 2/31
echo "Submitting: run_harbor_113k_batch_8192_seqs_256.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_8192_seqs_256.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 3/31
echo "Submitting: run_harbor_113k_batch_8192_seqs_512.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_8192_seqs_512.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 4/31
echo "Submitting: run_harbor_113k_batch_16384_seqs_128.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_16384_seqs_128.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 5/31
echo "Submitting: run_harbor_113k_batch_16384_seqs_256.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_16384_seqs_256.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 6/31
echo "Submitting: run_harbor_113k_batch_16384_seqs_512.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_16384_seqs_512.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 7/31
echo "Submitting: run_harbor_113k_batch_32768_seqs_128.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_32768_seqs_128.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 8/31
echo "Submitting: run_harbor_113k_batch_32768_seqs_256.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_32768_seqs_256.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 9/31
echo "Submitting: run_harbor_113k_batch_32768_seqs_512.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_32768_seqs_512.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 10/31
echo "Submitting: run_harbor_113k_fp8_batch_16384.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_fp8_batch_16384.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 11/31
echo "Submitting: run_harbor_113k_fp8_batch_32768.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_fp8_batch_32768.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 12/31
echo "Submitting: run_harbor_113k_ep_pplx.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_ep_pplx.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 13/31
echo "Submitting: run_harbor_113k_ep_deepep_low_latency.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_ep_deepep_low_latency.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 14/31
echo "Submitting: run_harbor_113k_eplb_enabled.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_eplb_enabled.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 15/31
echo "Submitting: run_harbor_113k_prefix_caching.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_prefix_caching.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 16/31
echo "Submitting: run_harbor_113k_aggressive_max_throughput.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_aggressive_max_throughput.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 17/31
echo "Submitting: run_harbor_113k_tp4_pp2.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_tp4_pp2.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 18/31
echo "Submitting: run_harbor_113k_tp2_pp4.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_tp2_pp4.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 19/31
echo "Submitting: run_harbor_113k_batch_65536_seqs_256.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_65536_seqs_256.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 20/31
echo "Submitting: run_harbor_113k_batch_131072_seqs_256.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_131072_seqs_256.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 21/31
echo "Submitting: run_harbor_113k_batch_65536_seqs_512.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_batch_65536_seqs_512.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 22/31
echo "Submitting: run_harbor_113k_ep_deepep_high_throughput.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_ep_deepep_high_throughput.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 23/31
echo "Submitting: run_harbor_113k_optimal.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_optimal.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 24/31
echo "Submitting: run_harbor_113k_optimal_4node_tp4.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_optimal_4node_tp4.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 25/31
echo "Submitting: run_harbor_113k_optimal_2node_tp2.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_optimal_2node_tp2.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 26/31
echo "Submitting: run_harbor_113k_optimal_1node_tp1.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_optimal_1node_tp1.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 27/31
echo "Submitting: run_harbor_113k_4node_tp4_pp1.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_4node_tp4_pp1.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 28/31
echo "Submitting: run_harbor_113k_4node_tp2_pp2.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_4node_tp2_pp2.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 29/31
echo "Submitting: run_harbor_113k_4node_tp1_pp4.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_4node_tp1_pp4.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 30/31
echo "Submitting: run_harbor_113k_2node_pp2.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_2node_pp2.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 31/31
echo "Submitting: run_harbor_113k_optimal_4node_pp4.sbatch"
JOB_ID=$(sbatch --parsable "${DCAGENT_DIR}/data/vllm_experiments/run_harbor_113k_optimal_4node_pp4.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

echo ""
echo "All experiments submitted!"
echo "Check status with: squeue -u $USER"
