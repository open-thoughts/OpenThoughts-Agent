#!/bin/bash
# RL Training Script for TACC - Code Contests with DCAgent sandboxes
# Adapted from JSC BenSkyRL script for TACC (Vista/Lonestar)
# Dataset: DCAgent/code-contests-sandboxes-with-tests
# Model: laion/nl2bash-bugsseq_Qwen3-8B
set -e

# ============================================================================
# CONFIGURATION - Edit these as needed
# ============================================================================
MODEL="laion/nl2bash-bugsseq_Qwen3-8B"
DATASET="DCAgent/code-contests-sandboxes-with-tests"
EVAL_DATASET="open-thoughts/OpenThoughts-TB-dev"

NUM_NODES=4
NUM_GPUS_PER_NODE=1  # TACC Vista has 1 GPU per node (GH200)

PARTITION="gh"   # Use gh-dev for testing, gh for production
ACCOUNT="BCS24021"  # Update with your TACC allocation
TIME_LIMIT="24:00:00"

# Training settings
TENSOR_PARALLEL_SIZE=1
NUM_INFERENCE_ENGINES=4
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=128
EPOCHS=1
MAX_EPISODES=128
EVAL_INTERVAL=20
N_CONCURRENT_TRIALS=256
NUM_PARALLEL_WORKERS=256
TIMEOUT_SEC=600
MAX_RESTARTS=2

# ============================================================================
# RUN NAME AND OUTPUT DIRECTORIES
# ============================================================================
MODEL_STRIPPED=$(echo ${MODEL} | tr '/' '_')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="benskyrl_codecontests_${MODEL_STRIPPED}_${NUM_NODES}nodes_${TIMESTAMP}"

SKYRL_EXPORT_PATH="${SCRATCH}/skyrl_exports/${RUN_NAME}/exports"
SKYRL_CKPT_PATH="${SCRATCH}/skyrl_exports/${RUN_NAME}/ckpts"
TRIAL_RUNS_DIR="${SCRATCH}/skyrl_exports/${RUN_NAME}/trials"
mkdir -p "$SKYRL_EXPORT_PATH" "$SKYRL_CKPT_PATH" "$TRIAL_RUNS_DIR"

SANDBOXES_DIR="/scratch/08134/negin/dc-agent-shared/sandboxes/run"

# ============================================================================
# CHAT TEMPLATE
# ============================================================================
CHAT_TEMPLATE_PATH="${SKYRL_HOME}/skyrl-train/examples/terminal_bench/qwen3_thinking_acc.jinja2"

# ============================================================================
# LOGGING
# ============================================================================
LOGGER="wandb"

# ============================================================================
# LAUNCH TRAINING
# ============================================================================
echo "=============================================="
echo "Starting RL Training - Code Contests (BenSkyRL) on TACC"
echo "=============================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Eval Dataset: $EVAL_DATASET"
echo "Nodes: $NUM_NODES"
echo "Run name: $RUN_NAME"
echo "Concurrent trials: $N_CONCURRENT_TRIALS"
echo "Export path: $SKYRL_EXPORT_PATH"
echo "=============================================="

python3 -m hpc.launch \
    --job_name "${RUN_NAME}" \
    --final_model_name "${RUN_NAME}" \
    --enable_hf_upload \
    --hf_repo_name "DCAgent2/${RUN_NAME}" \
    --time_limit "${TIME_LIMIT}" \
    --num_nodes "${NUM_NODES}" \
    --train_data "${DATASET}" \
    --val_data "${EVAL_DATASET}" \
    --model_path "${MODEL}" \
    --max_restarts "${MAX_RESTARTS}" \
    --partition "${PARTITION}" \
    --skyrl_entrypoint examples.terminal_bench.entrypoints.main_tbench \
    -S hydra.searchpath="['file://examples/terminal_bench']" \
    -S +terminal_bench_config=terminal_bench \
    -S +terminal_bench_config.trials_dir="$TRIAL_RUNS_DIR" \
    -S +terminal_bench_config.harbor.name=terminus-2 \
    -S +terminal_bench_config.harbor.max_episodes=$MAX_EPISODES \
    -S +terminal_bench_config.harbor.override_cpus=1 \
    -S +terminal_bench_config.harbor.override_memory_mb=1024 \
    -S +terminal_bench_config.harbor.override_storage_mb=1024 \
    -S +terminal_bench_config.harbor.enable_summarize=false \
    -S +terminal_bench_config.harbor.store_all_messages=true \
    -S +terminal_bench_config.harbor.collect_rollout_details=true \
    -S +terminal_bench_config.harbor.n_concurrent_trials=$N_CONCURRENT_TRIALS \
    -S +terminal_bench_config.harbor.override_timeout_sec=$TIMEOUT_SEC \
    -S +terminal_bench_config.harbor.enable_episode_logging=false \
    -S +terminal_bench_config.harbor.strict_json_parser=false \
    -S +terminal_bench_config.harbor.interleaved_thinking=true \
    -S +terminal_bench_config.harbor.log_level=DEBUG \
    -S +terminal_bench_config.harbor.auto_snapshot=true \
    -S '+terminal_bench_config.harbor.extra_body={chat_template_kwargs:{enable_thinking:true}}' \
    -S +terminal_bench_config.model_info.max_input_tokens=24576 \
    -S +terminal_bench_config.model_info.max_output_tokens=8192 \
    -S +terminal_bench_config.model_info.input_cost_per_token=0.0 \
    -S +terminal_bench_config.model_info.output_cost_per_token=0.0 \
    -S trainer.export_path="$SKYRL_EXPORT_PATH" \
    -S trainer.ckpt_path="$SKYRL_CKPT_PATH" \
    -S trainer.algorithm.advantage_estimator=grpo \
    -S trainer.algorithm.use_kl_loss=true \
    -S trainer.algorithm.kl_loss_coef=0.01 \
    -S trainer.placement.colocate_all=false \
    -S trainer.strategy=fsdp2 \
    -S trainer.policy.fsdp_config.cpu_offload=false \
    -S trainer.policy.fsdp_config.reshard_after_forward=true \
    -S trainer.policy.fsdp_config.fsdp_size=4 \
    -S trainer.ref.fsdp_config.cpu_offload=false \
    -S trainer.ref.fsdp_config.reshard_after_forward=true \
    -S trainer.ref.fsdp_config.fsdp_size=4 \
    -S trainer.placement.policy_num_nodes=1 \
    -S trainer.placement.ref_num_nodes=1 \
    -S trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
    -S trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
    -S trainer.fully_async.num_parallel_generation_workers=$NUM_PARALLEL_WORKERS \
    -S trainer.fully_async.max_staleness_steps=15 \
    -S generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
    -S generator.inference_engine_tensor_parallel_size=$TENSOR_PARALLEL_SIZE \
    -S generator.model_name=nl2bash-bugsseq_Qwen3-8B \
    -S generator.model_dtype=bfloat16 \
    -S generator.gpu_memory_utilization=0.92 \
    -S generator.max_num_seqs=128 \
    -S generator.max_num_batched_tokens=65536 \
    -S generator.enable_prefix_caching=true \
    -S generator.enable_chunked_prefill=true \
    -S generator.enforce_eager=false \
    -S trainer.epochs=$EPOCHS \
    -S trainer.eval_batch_size=$EVAL_BATCH_SIZE \
    -S trainer.eval_before_train=false \
    -S trainer.eval_interval=$EVAL_INTERVAL \
    -S trainer.update_epochs_per_batch=1 \
    -S trainer.train_batch_size=$TRAIN_BATCH_SIZE \
    -S trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
    -S trainer.micro_forward_batch_size_per_gpu=2 \
    -S trainer.micro_train_batch_size_per_gpu=1 \
    -S trainer.ckpt_interval=5 \
    -S trainer.hf_save_interval=5 \
    -S trainer.max_prompt_length=2048 \
    -S generator.sampling_params.max_generate_length=30720 \
    -S trainer.policy.optimizer_config.lr=1.0e-6 \
    -S trainer.policy.optimizer_config.weight_decay=0.01 \
    -S generator.n_samples_per_prompt=8 \
    -S generator.eval_n_samples_per_prompt=1 \
    -S trainer.project_name=dc-agent \
    -S trainer.logger="$LOGGER" \
    -S trainer.run_name="$RUN_NAME" \
    -S trainer.resume_mode=latest \
    -S generator.backend=vllm \
    -S generator.run_engines_locally=true \
    -S generator.weight_sync_backend=nccl \
    -S generator.async_engine=true \
    -S generator.batched=false \
    -S generator.enable_http_endpoint=true \
    -S generator.http_endpoint_host=127.0.0.1 \
    -S generator.http_endpoint_port=8000 \
    -S generator.append_eos_token_after_stop_str_in_multi_turn=true \
    -S +generator.engine_init_kwargs.custom_chat_template_chat_completion_path=$CHAT_TEMPLATE_PATH \
    -S +generator.engine_init_kwargs.served_model_name=nl2bash-bugsseq_Qwen3-8B \
    -S +generator.engine_init_kwargs.max_model_len=32768 \
    "$@"
