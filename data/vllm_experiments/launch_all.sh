#!/bin/bash
# Launch all vLLM experiments with Ray Serve data parallelism
# Generated automatically - do not edit

set -e

EXPERIMENTS_DIR="/scratch/10000/eguha3/ot-agent/data/vllm_experiments"
cd "$EXPERIMENTS_DIR"

echo "Launching 26 vLLM experiments..."
echo ""


# Experiment 1/26
echo "Submitting: run_8acd1260_baseline_conservative.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_8acd1260_baseline_conservative.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 2/26
echo "Submitting: run_377e753a_batch_8192_seqs_256.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_377e753a_batch_8192_seqs_256.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 3/26
echo "Submitting: run_d097391e_batch_8192_seqs_512.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_d097391e_batch_8192_seqs_512.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 4/26
echo "Submitting: run_b2181dc6_batch_16384_seqs_128.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_b2181dc6_batch_16384_seqs_128.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 5/26
echo "Submitting: run_7675bf4f_batch_16384_seqs_256.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_7675bf4f_batch_16384_seqs_256.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 6/26
echo "Submitting: run_8e787323_batch_16384_seqs_512.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_8e787323_batch_16384_seqs_512.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 7/26
echo "Submitting: run_1951b620_batch_32768_seqs_128.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_1951b620_batch_32768_seqs_128.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 8/26
echo "Submitting: run_b417fa28_batch_32768_seqs_256.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_b417fa28_batch_32768_seqs_256.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 9/26
echo "Submitting: run_5b8e1126_batch_32768_seqs_512.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_5b8e1126_batch_32768_seqs_512.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 10/26
echo "Submitting: run_d8321e9e_fp8_batch_16384.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_d8321e9e_fp8_batch_16384.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 11/26
echo "Submitting: run_50685f7b_fp8_batch_32768.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_50685f7b_fp8_batch_32768.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 12/26
echo "Submitting: run_fbd12323_ep_pplx.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_fbd12323_ep_pplx.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 13/26
echo "Submitting: run_cbca79ba_ep_deepep_low_latency.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_cbca79ba_ep_deepep_low_latency.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 14/26
echo "Submitting: run_9336f8f2_eplb_enabled.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_9336f8f2_eplb_enabled.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 15/26
echo "Submitting: run_73d2e4ba_prefix_caching.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_73d2e4ba_prefix_caching.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 16/26
echo "Submitting: run_65b957fb_aggressive_max_throughput.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_65b957fb_aggressive_max_throughput.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 17/26
echo "Submitting: run_32fc1967_tp4_pp2.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_32fc1967_tp4_pp2.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 18/26
echo "Submitting: run_a2dd0a48_tp2_pp4.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_a2dd0a48_tp2_pp4.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 19/26
echo "Submitting: run_f41fcbda_batch_65536_seqs_256.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_f41fcbda_batch_65536_seqs_256.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 20/26
echo "Submitting: run_2b0b55b8_batch_131072_seqs_256.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_2b0b55b8_batch_131072_seqs_256.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 21/26
echo "Submitting: run_fda9727f_batch_65536_seqs_512.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_fda9727f_batch_65536_seqs_512.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 22/26
echo "Submitting: run_d235f27e_ep_deepep_high_throughput.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_d235f27e_ep_deepep_high_throughput.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 23/26
echo "Submitting: run_b5f3dccc_optimal.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_b5f3dccc_optimal.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 24/26
echo "Submitting: run_cd612497_optimal_4node_tp4.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_cd612497_optimal_4node_tp4.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 25/26
echo "Submitting: run_ceb5defd_optimal_2node_tp2.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_ceb5defd_optimal_2node_tp2.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

# Experiment 26/26
echo "Submitting: run_aaecaf28_optimal_1node_tp1.sbatch"
JOB_ID=$(sbatch --parsable "/scratch/10000/eguha3/ot-agent/data/vllm_experiments/run_aaecaf28_optimal_1node_tp1.sbatch")
echo "  Job ID: $JOB_ID"
sleep 2

echo ""
echo "All experiments submitted!"
echo "Check status with: squeue -u $USER"
echo "View results in: $EXPERIMENTS_DIR/results/"
