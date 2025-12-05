#!/bin/bash
# This script is intended to be sourced by specific run scripts.
# It expects the following variables to be set:
ERROR_MESSAGE="is not set. Please define it in the calling script."
: "${RUN_NAME:?RUN_NAME $ERROR_MESSAGE}"
: "${NUM_NODES:?NUM_NODES $ERROR_MESSAGE}"
: "${TRAIN_DATA_DIR:?TRAIN_DATA_DIR $ERROR_MESSAGE}"
: "${EVAL_DATA_DIR:?EVAL_DATA_DIR $ERROR_MESSAGE}"
: "${MODEL_PATH:?MODEL_PATH $ERROR_MESSAGE}"
: "${MAX_RESTARTS:?MAX_RESTARTS $ERROR_MESSAGE}"
: "${SANDBOXES_DIR:?SANDBOXES_DIR $ERROR_MESSAGE}"
: "${SKYRL_EXPORT_PATH:?SKYRL_EXPORT_PATH $ERROR_MESSAGE}"
: "${SKYRL_CKPT_PATH:?SKYRL_CKPT_PATH $ERROR_MESSAGE}"
: "${EVAL_BATCH_SIZE:?EVAL_BATCH_SIZE $ERROR_MESSAGE}"
: "${TRAIN_AND_MINI_BATCH_SIZE:?TRAIN_AND_MINI_BATCH_SIZE $ERROR_MESSAGE}"
: "${LOGGER:?LOGGER $ERROR_MESSAGE}"
: "${EPOCHS:?EPOCHS $ERROR_MESSAGE}"
: "${EVAL_INTERVAL:?EVAL_INTERVAL $ERROR_MESSAGE}"


python3 -m hpc.launch \
 --job_name "$RUN_NAME" \
 --final_model_name "$RUN_NAME" \
 --enable_hf_upload \
 --hf_repo_name "DCAgent2/${RUN_NAME}" \
 --time_limit 24:00:00 \
 --num_nodes "$NUM_NODES" \
 --train_data "$TRAIN_DATA_DIR" \
 --val_data "$EVAL_DATA_DIR" \
 --model_path "$MODEL_PATH" \
 --max_restarts "$MAX_RESTARTS" \
 --partition gh \
 --skyrl_entrypoint examples.terminal_bench.entrypoints.main_tbench \
 -S hydra.searchpath="['file://examples/terminal_bench']" \
 -S +terminal_bench_config=terminal_bench \
 -S +terminal_bench_config.agent_name=terminus \
 -S +terminal_bench_config.max_episodes=8 \
 -S +terminal_bench_config.trials_dir="$SANDBOXES_DIR" \
 -S +terminal_bench_config.override_memory_mb=1024 \
 -S +terminal_bench_config.override_storage_mb=1024 \
 -S +terminal_bench_config.override_cpus=1 \
 -S trainer.export_path="$SKYRL_EXPORT_PATH" \
 -S trainer.ckpt_path="$SKYRL_CKPT_PATH" \
 -S trainer.algorithm.advantage_estimator="grpo" \
 -S trainer.placement.colocate_all=true \
 -S trainer.strategy=fsdp2 \
 -S trainer.placement.policy_num_nodes="$NUM_NODES" \
 -S trainer.placement.ref_num_nodes="$NUM_NODES" \
 -S trainer.placement.policy_num_gpus_per_node=1 \
 -S trainer.placement.ref_num_gpus_per_node=1 \
 -S generator.num_inference_engines="$NUM_NODES" \
 -S generator.inference_engine_tensor_parallel_size=1 \
 -S trainer.epochs="$EPOCHS" \
 -S trainer.eval_batch_size="$EVAL_BATCH_SIZE" \
 -S trainer.eval_before_train=false \
 -S trainer.eval_interval="$EVAL_INTERVAL" \
 -S trainer.update_epochs_per_batch=1 \
 -S trainer.train_batch_size="$TRAIN_AND_MINI_BATCH_SIZE" \
 -S trainer.policy_mini_batch_size="$TRAIN_AND_MINI_BATCH_SIZE" \
 -S trainer.micro_forward_batch_size_per_gpu=1 \
 -S trainer.micro_train_batch_size_per_gpu=1 \
 -S trainer.ckpt_interval=5 \
 -S trainer.hf_save_interval=5 \
 -S trainer.max_prompt_length=2048 \
 -S generator.sampling_params.max_generate_length=30720 \
 -S trainer.policy.optimizer_config.lr=1.0e-6 \
 -S trainer.algorithm.use_kl_loss=true \
 -S generator.n_samples_per_prompt=8 \
 -S generator.eval_n_samples_per_prompt=8 \
 -S generator.gpu_memory_utilization=0.8 \
 -S trainer.logger="$LOGGER" \
 -S trainer.project_name=dc-agent \
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
 "$@"
