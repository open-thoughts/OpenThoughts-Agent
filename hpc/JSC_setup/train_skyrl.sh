# ------------- Environment Setup ------------
set -e
export PORT_TO_USE=25001
export USER_NAME="$(whoami)"
export SSH_KEY="${HOME}/.ssh/docker" # Path to your SSH private key
export PATH=/p/home/jusers/nezhurina1/jureca/.local/bin:$PATH
export LOGIN_NODE=$(hostname -f | cut -d. -f1)
export LOGIN_NODE="${LOGIN_NODE}i"
mkdir -p ~/.proxychains

# SLURM resources

HOST=$(hostname)
if [[ $HOST == *jureca ]]; then
  SBATCH_ACCOUNT="westai0066"
  SBATCH_PARTITION="dc-hwai"
  # SBATCH_ACCOUNT="synthlaion"
  # SBATCH_PARTITION="dc-gpu-devel"
elif [[ $HOST == *juwels ]]; then
  SBATCH_ACCOUNT="laionize"
  SBATCH_PARTITION="booster"
else
  echo "This script must be run on a SLURM login node (jureca or juwels). Current host: $HOST"
  exit 1
fi

# account: synthlaion if machine is jureca and laionize if machine is juwels

SBATCH_JOB_NAME="tbench-test"
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-"synthlaion"}
SBATCH_PARTITION=${SBATCH_PARTITION:-"dc-gpu"}
# SBATCH_ACCOUNT="westai0066"
# SBATCH_PARTITION="dc-hwai"
SBATCH_NODES=2
SBATCH_NUM_GPUS=4
NUM_GPUS=8
SBATCH_NTASKS_PER_NODE=1
SBATCH_CPUS_PER_TASK=48
SBATCH_TIME="08:00:00"
OUT_DIR="/p/project/laionize/marianna/terminal_bench/slurm-output"
mkdir -p "$OUT_DIR"
export NUM_GPUS_PER_NODE=$SBATCH_NUM_GPUS
# ------------ RUNTIME STAGING ------------
ts="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_DIR}/skyrl_tbench_${ts}"
mkdir -p "$RUN_DIR"
export RUN_DIR
export NODEFILE="${RUN_DIR}/node.txt"
export READYFILE="${RUN_DIR}/tunnel.ready"
export ALLOC_LOG="${RUN_DIR}/alloc.log"
export SRUN_LOG="${RUN_DIR}/srun.log"
export TUNNEL_LOG="${RUN_DIR}/tunnel.log"

JOBID=""
TUNNEL_PID=""

cleanup() {
  set +e
  if [[ -n "${TUNNEL_PID}" ]] && ps -p "${TUNNEL_PID}" >/dev/null 2>&1; then
    echo "[cleanup] Killing tunnel pid ${TUNNEL_PID}" | tee -a "$TUNNEL_LOG"
    kill "${TUNNEL_PID}" || true
  fi
  if [[ -n "${JOBID}" ]]; then
    if squeue -j "${JOBID}" -h >/dev/null 2>&1; then
      echo "[cleanup] Cancelling allocation ${JOBID}" | tee -a "$ALLOC_LOG"
      scancel "${JOBID}" || true
    fi
  fi
}
trap cleanup EXIT



ALLOC_CMD=$(cat <<'EOS'
    export TERMINAL_BENCH_CACHE_DIR="/p/project/laionize/marianna/terminal_bench/terminal-bench-main/.cache"
    export VLLM_CACHE_ROOT=/p/project/laionize/marianna/terminal_bench/vllm
    export VLLM_CONFIG_ROOT=/p/project/laionize/marianna/terminal_bench/vllm_config
    export TRITON_DUMP_DIR=/p/project/laionize/marianna/terminal_bench/triton_dump_dir
    export TRITON_OVERRIDE_DIR=/p/project/laionize/marianna/terminal_bench/triton_override_dir
    export TRITON_CACHE_DIR=/p/project/laionize/marianna/terminal_bench/triton_cache_dir
    export MASTER_PORT=12345

    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=ib0
    export GLOO_SOCKET_IFNAME=ib0
    export NCCL_IB_TIMEOUT=120
    export NCCL_ALGO=Ring

    export SKYRL_HOME=/p/project/laionize/marianna/terminal_bench/SkyRL/skyrl-train
    cd $SKYRL_HOME
    export SKYRL_OUTPUT_DIR="/p/project/laionize/marianna/terminal_bench"
    export CHECKPOINT_PATH="/p/project/laionize/etashg/checkpoints/terminal_bench"
    mkdir -p $CHECKPOINT_PATH
    export TRIALS_DIR="/p/project/laionize/marianna/terminal_bench/sandboxes_marianna/runs"
    mkdir -p $TRIALS_DIR

    export MODEL_PATH="/p/data1/mmlaion/marianna/models/Qwen/Qwen2.5-0.5B-Instruct"
    export LOGGER="console"
    export HYDRA_FULL_ERROR=1

    UV_CACHE_DIR=/p/scratch/synthlaion/marianna/uv_cache
    mkdir -p "$UV_CACHE_DIR"
    export UV_CACHE_DIR

    UV_TOOL_DIR=/p/scratch/synthlaion/marianna/uv_tool
    mkdir -p "$UV_TOOL_DIR"
    export UV_TOOL_DIR

    # Conda / env
    export MINICONDA_PATH="/p/project1/ccstdl/nezhurina1/miniconda/miniconda"
    export CONDA_ENV="/p/project1/ccstdl/envs/marianna/py3.12"
    export TOKENIZERS_PARALLELISM=false

    if [ -n "${DC_AGENT_SECRET_ENV:-}" ] && [ -f "${DC_AGENT_SECRET_ENV}" ]; then
        # shellcheck disable=SC1090
        source "${DC_AGENT_SECRET_ENV}"
    else
        echo "[train_skyrl] WARNING: DC_AGENT_SECRET_ENV not set or file missing; secrets must be provided via environment." >&2
    fi

    port=20156
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)
    head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export head_node_i="${head_node}i"
    head_node_ip="$(nslookup "$head_node_i" | grep -oP '(?<=Address: ).*')"
    echo "Head node: $head_node_ip"
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    echo "strict_chain" >~/.proxychains/proxychains.conf
    echo "proxy_dns" >>~/.proxychains/proxychains.conf
    echo "tcp_read_time_out  30000" >>~/.proxychains/proxychains.conf
    echo "tcp_connect_time_out 15000" >>~/.proxychains/proxychains.conf
    echo "localnet 127.0.0.0/255.0.0.0" >>~/.proxychains/proxychains.conf
    echo "localnet 127.0.0.1/255.255.255.255" >>~/.proxychains/proxychains.conf
    echo "localnet 10.0.0.0/255.0.0.0" >>~/.proxychains/proxychains.conf
    echo "localnet 172.16.0.0/255.240.0.0" >>~/.proxychains/proxychains.conf
    echo "localnet 192.168.0.0/255.255.0.0" >>~/.proxychains/proxychains.conf
    echo "[ProxyList]" >>~/.proxychains/proxychains.conf
    echo "socks5  ${head_node_ip} ${PORT_TO_USE}" >>~/.proxychains/proxychains.conf
    echo "Proxychains config:"
    cat ~/.proxychains/proxychains.conf
    # export PROXYCHAINS_CONF=/p/home/jusers/nezhurina1/jureca/.proxychains/proxychains.conf
    # export LD_PRELOAD=/p/home/jusers/nezhurina1/jureca/.local/lib/libproxychains4.so

    srun --nodes=1 --output="${SRUN_LOG}_head.out" --ntasks=1 -w "$head_node" \
        bash -lc "source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate && \
                conda activate /p/project1/ccstdl/envs/marianna/py3.12 && \
                echo 'Setting up SSH tunnel on HEAD node...' && \
                ssh -g -f -N -D ${head_node_ip}:$PORT_TO_USE \
                    -o StrictHostKeyChecking=no \
                    -o ConnectTimeout=1000 \
                    -o ServerAliveInterval=15 \
                    -o ServerAliveCountMax=15 \
                    -o TCPKeepAlive=no \
                    -o ExitOnForwardFailure=yes \
                    -o BatchMode=yes \
                    -i $SSH_KEY \
                    ${USER_NAME}@$LOGIN_NODE && \
                    LD_PRELOAD=/p/home/jusers/nezhurina1/jureca/.local/lib/libproxychains4.so \
                    PROXYCHAINS_CONF=/p/home/jusers/nezhurina1/jureca/.proxychains/proxychains.conf \
                    ray start --head --node-ip-address=$head_node_ip \
                        --port=$port --num-cpus ${SLURM_CPUS_PER_TASK} \
                        --num-gpus=$NUM_GPUS_PER_NODE --block" &
    sleep 20
    worker_num=$((SLURM_JOB_NUM_NODES - 1))
    if [[ $worker_num -gt 0 ]]; then
        for ((i = 1; i <= worker_num; i++)); do
            node=${nodes_array[$i]}
            node_i="${node}i"
            echo "Starting WORKER $i at $node"
            this_node_ip="$(nslookup "$node_i" | grep -oP '(?<=Address: ).*')"
            srun --nodes=1 --output="${SRUN_LOG}_$i.out" --ntasks=1 -w "$node" \
                bash -lc "
                    source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate && \
                        conda activate /p/project1/ccstdl/envs/marianna/py3.12 && \
                        echo 'Starting WORKER '$i'.' && \
                        LD_PRELOAD=/p/home/jusers/nezhurina1/jureca/.local/lib/libproxychains4.so \
                        PROXYCHAINS_CONF=/p/home/jusers/nezhurina1/jureca/.proxychains/proxychains.conf \
                        ray start --address $ip_head --node-ip-address=$this_node_ip \
                                    --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus=$NUM_GPUS_PER_NODE --block" &
            sleep 5
        done
    else
        echo "Single node setup - no workers to start"
    fi

    srun --nodes=1 --ntasks=1 -w "$head_node" --overlap --output="$SRUN_LOG" \
        bash -lc "
            source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate && \
            conda activate /p/project1/ccstdl/envs/marianna/py3.12 && \
            proxychains4 python -m skyrl_train.entrypoints.main_base \
            data.train_data=['/p/scratch/synthlaion/dc-agent-shared/hf_hub/datasets--mlfoundations-dev--sandboxes-tasks/snapshots/6fdf67053a80836ab1d5007104baded9f3513733'] \
            data.val_data=['/p/scratch/synthlaion/dc-agent-shared/hf_hub/datasets--mlfoundations-dev--sandboxes-tasks/snapshots/6fdf67053a80836ab1d5007104baded9f3513733'] \
            +data.cache_dir=/p/scratch/laionize/marianna/sky-rl/data/gsm8k \
            trainer.algorithm.advantage_estimator=grpo \
            trainer.policy.model.path=/p/data1/mmlaion/marianna/models/Qwen/Qwen2.5-0.5B-Instruct \
            trainer.placement.colocate_all=false \
            trainer.placement.policy_num_nodes=1 \
            trainer.placement.ref_num_nodes=1 \
            trainer.placement.reward_num_nodes=1 \
            trainer.placement.critic_num_nodes=1 \
            trainer.strategy=fsdp2 \
            trainer.placement.policy_num_gpus_per_node=4 \
            trainer.placement.ref_num_gpus_per_node=4 \
            trainer.placement.reward_num_gpus_per_node=4 \
            trainer.placement.critic_num_gpus_per_node=4 \
            generator.num_inference_engines=4 \
            generator.http_server_inference_engine_client_host=${head_node_ip} \
            generator.http_server_inference_engine_client_port=8000 \
            generator.use_http_server_inference_engine_client=true \
            generator.inference_engine_tensor_parallel_size=1 \
            trainer.epochs=10 \
            trainer.eval_batch_size=1 \
            trainer.eval_before_train=false \
            trainer.eval_interval=5 \
            trainer.update_epochs_per_batch=1 \
            trainer.train_batch_size=4 \
            trainer.policy_mini_batch_size=4 \
            trainer.micro_forward_batch_size_per_gpu=1 \
            trainer.micro_train_batch_size_per_gpu=1 \
            trainer.ckpt_interval=10 \
            trainer.max_prompt_length=16000 \
            generator.sampling_params.max_generate_length=16000 \
            trainer.policy.optimizer_config.lr=1.0e-6 \
            trainer.algorithm.use_kl_loss=true \
            generator.backend=vllm \
            generator.run_engines_locally=true \
            generator.weight_sync_backend=nccl \
            generator.async_engine=true \
            generator.batched=true \
            environment.env_class=gsm8k \
            generator.n_samples_per_prompt=4 \
            generator.gpu_memory_utilization=0.8 \
            +generator.agent_name=terminus \
            +generator.sandboxes_dir=/p/project/laionize/marianna/terminal_bench/sandboxes \
            +generator.trial_runs_dir=/p/project/laionize/marianna/terminal_bench/sandboxes/runs \
            trainer.logger=$LOGGER \
            trainer.project_name=gsm8k \
            trainer.run_name=gsm8k_test \
            trainer.resume_mode=null \
            trainer.ckpt_path=$CHECKPOINT_PATH \
            trainer.export_path=/p/project/laionize/marianna/terminal_bench/SkyRL/skyrl-train/exports"
EOS
)

{
  salloc \
    --job-name="${SBATCH_JOB_NAME}" \
    --account="${SBATCH_ACCOUNT}" \
    --partition="${SBATCH_PARTITION}" \
    --nodes="${SBATCH_NODES}" \
    --ntasks-per-node="${SBATCH_NTASKS_PER_NODE}" \
    --cpus-per-task="${SBATCH_CPUS_PER_TASK}" \
    --time="${SBATCH_TIME}" \
    --gres=gpu:"${SBATCH_NUM_GPUS}" \
    bash -lc "${ALLOC_CMD}"
} >"${ALLOC_LOG}" 2>&1 &

echo "Allocated Log: $ALLOC_LOG"
