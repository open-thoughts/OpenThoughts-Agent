export TERMINAL_BENCH_CACHE_DIR="/p/project/laionize/marianna/terminal_bench/terminal-bench-main/.cache"
CONDA_ENV="/p/project1/ccstdl/envs/marianna/py3.12"
MINICONDA_PATH="/p/project1/ccstdl/nezhurina1/miniconda/miniconda"
source ${MINICONDA_PATH}/bin/activate ${CONDA_ENV}
export PYTHONPATH=$PYTHONPATH:/p/project/laionize/marianna/terminal_bench/terminal-bench-main

echo "IP Head: $ip_head"
echo "Testing Ray connection..."
for i in {1..6}; do
  if ray status --address $ip_head >/dev/null 2>&1; then
    echo "✓ Ray cluster connected"
    break
  else
    echo "Waiting for Ray... (attempt $i/6)"
    sleep 10
  fi
  
  if [ $i -eq 6 ]; then
    echo "✗ Ray cluster not accessible"
    exit 1
  fi
done

ray status --address $ip_head
export PYTHONPATH=/p/project/laionize/marianna/terminal_bench/terminal-bench-main:$PYTHONPATH

SKYRL_HOME=/p/project/laionize/marianna/terminal_bench/SkyRL/skyrl-train
cd $SKYRL_HOME
SKYRL_OUTPUT_DIR="/p/project/laionize/marianna/terminal_bench"
DATA_DIR="${SKYRL_OUTPUT_DIR}/data_rl/terminal-bench"
CHECKPOINT_PATH="${SKYRL_OUTPUT_DIR}/checkpoints/terminal-bench"
mkdir -p $CHECKPOINT_PATH

MODEL_PATH="/p/data1/mmlaion/marianna/models/Qwen/Qwen2.5-0.5B-Instruct"

# number of GPUs is NUM_GPUS_PER_NODE * SLURM_JOB_NUM_NODES
NUM_GPUS=$(($NUM_GPUS_PER_NODE * $SLURM_JOB_NUM_NODES))
echo "Number of GPUs: $NUM_GPUS"
LOGGER="console"  # change to "console" to print to stdout

export HYDRA_FULL_ERROR=1 # to see full error messages
CMD="python -m skyrl_train.entrypoints.main_base \
  data.train_data=\"['$DATA_DIR/train.parquet']\" \
  data.val_data=\"['$DATA_DIR/validation.parquet']\" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.policy.model.path=$MODEL_PATH \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes=$SLURM_JOB_NUM_NODES \
  trainer.placement.ref_num_nodes=$SLURM_JOB_NUM_NODES \
  trainer.placement.reward_num_nodes=$SLURM_JOB_NUM_NODES \
  trainer.placement.critic_num_nodes=$SLURM_JOB_NUM_NODES \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.reward_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1  \
  trainer.eval_batch_size=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1 \
  trainer.policy_mini_batch_size=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=16000 \
  generator.sampling_params.max_generate_length=16000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  +generator.trial_runs_dir="/p/project/laionize/marianna/terminal_bench/sandboxes/runs" \
  +generator.agent_name="terminus" \
  generator.use_http_server_inference_engine_client=true \
  generator.http_server_inference_engine_client_host="127.0.0.1" \
  generator.http_server_inference_engine_client_port=8000 \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  +generator.sandboxes_dir="/p/project/laionize/marianna/terminal_bench/sandboxes" \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger=$LOGGER \
  trainer.project_name="gsm8k" \
  trainer.run_name="gsm8k_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path=$CHECKPOINT_PATH \
  trainer.export_path=/p/project/laionize/marianna/terminal_bench/SkyRL/skyrl-train/exports "

echo "Running command: $CMD"
bash -c "$CMD" || exit 1