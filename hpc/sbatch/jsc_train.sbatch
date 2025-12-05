#!/bin/bash -x
#SBATCH --nodes={num_nodes}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_node}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --threads-per-core=1
#SBATCH --time={time_limit}
#SBATCH --output={experiments_dir}/logs/%x_%j.out
#SBATCH --job-name={job_name}
#SBATCH --qos={qos}

set -e

ml purge

ulimit -c 0 # disable core dumps

# APPTAINER
if [ -z "$IMAGE" ]; then
    APPTAINER_ARGS=""
else
    APPTAINER_ARGS="
    apptainer exec \
    --nv \
    $IMAGE 
    "
fi

if [ -z "$SSH_KEY" ]; then
    SSH_TUNNEL_CMD=""
else
    USER_NAME="$(whoami)"
    LOGIN_NODE={login_node}
    PORT_TO_USE=$((20000 + RANDOM % 10000))
    head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
     # Add i to the hostname  if it contains jureca or juwels
    # if [[ $head_node_ip == *"jrc"* ]]; then
    #     head_node_ip="${head_node_ip}i"
    # elif [[ $head_node_ip == *"jwb"* ]]; then
    #     head_node_ip="${head_node_ip}i"
    # fi
    head_node_ip="$(nslookup "$head_node_ip" | grep -oP '(?<=Address: ).*')"
    # head_node_ip=0.0.0.0
    SSH_TUNNEL_CMD="ssh -g -f -N -D 0.0.0.0:$PORT_TO_USE \
                    -o StrictHostKeyChecking=no \
                    -o ConnectTimeout=1000 \
                    -o ServerAliveInterval=15 \
                    -o ServerAliveCountMax=15 \
                    -o TCPKeepAlive=no \
                    -o ExitOnForwardFailure=yes \
                    -o BatchMode=yes \
                    -i $SSH_KEY \
                    ${USER_NAME}@$LOGIN_NODE"

    mkdir -p ~/.proxychains

    cat > ~/.proxychains/proxychains.conf <<-EOT
        strict_chain
        proxy_dns
        tcp_read_time_out 30000
        tcp_connect_time_out 15000
        localnet 127.0.0.0/255.0.0.0
        localnet 127.0.0.1/255.255.255.255
        localnet 10.0.0.0/255.0.0.0
        localnet 172.16.0.0/255.240.0.0
        [ProxyList]
        socks5 ${head_node_ip} ${PORT_TO_USE}
EOT

    cat ~/.proxychains/proxychains.conf


fi


# OFFLINE MODE
# export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
# export HF_HUB_OFFLINE=0

SECRET_FILE="${DC_AGENT_SECRET_ENV:-${KEYS:-}}"
if [[ -n "${SECRET_FILE}" ]]; then
    if [[ -f "${SECRET_FILE}" ]]; then
        echo "Sourcing secrets from ${SECRET_FILE}"
        set -a
        # shellcheck disable=SC1090
        source "${SECRET_FILE}"
        set +a
    else
        echo "Warning: secrets file ${SECRET_FILE} not found; database registration may fail." >&2
    fi
else
    echo "Warning: DC_AGENT_SECRET_ENV is not set; database registration may fail." >&2
fi

for _supabase_var in SUPABASE_URL SUPABASE_ANON_KEY SUPABASE_SERVICE_ROLE_KEY; do
    if [[ -n "${!_supabase_var:-}" ]]; then
        export "${_supabase_var}=${!_supabase_var}"
    else
        echo "Warning: ${_supabase_var} is not set; Supabase registration may fail." >&2
    fi
done

# NETWORKING
MASTER_ADDR_HOST=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1` # Allow communication over InfiniBand cells.

# Add i to the hostname  if it contains jureca or juwels
if [[ $MASTER_ADDR_HOST == *"jrc"* ]]; then
    MASTER_ADDR="${MASTER_ADDR_HOST}i"
elif [[ $MASTER_ADDR_HOST == *"jwb"* ]]; then
    MASTER_ADDR="${MASTER_ADDR_HOST}i"
else
    MASTER_ADDR="${MASTER_ADDR_HOST}"
fi

# get the IP address of the master node
MASTER_ADDR=$(nslookup "$MASTER_ADDR_HOST" | grep -oP '(?<=Address: ).*')
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=20043
export NCCL_NET_GDR_LEVEL=0 # for h100
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_TIMEOUT=60
# export NCCL_DEBUG=INFO

# RESOURCES
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export NUM_NODES=$SLURM_JOB_NUM_NODES
export NUM_GPUS_PER_NODE={gpus_per_node}
export NUM_GPUS=$((NUM_GPUS_PER_NODE*SLURM_NNODES))
DEEPSPEED_CONFIG_FILE={deepspeed}

# PATHS
CONFIG={train_config_path_out}
OUTPUT_DIR={experiments_dir}
TMP_DIR=$OUTPUT_DIR/tmp
mkdir -p $OUTPUT_DIR
mkdir -p $TMP_DIR

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}/${SLURM_JOB_ID}"
mkdir -p $TRITON_CACHE_DIR

# ACCELERATE
{accelerate_config_block}

# MAIN TRAIN COMMAND
LAUNCHER="python -u -m accelerate.commands.launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s): --tee 3 \
    "
CMD="$LAUNCHER sft/llamafactory/src/train.py $CONFIG"
SRUN_ARGS="
    --nodes=$NUM_NODES \
    --gres=gpu:$NUM_GPUS_PER_NODE \
    --cpus-per-task=$SLURM_CPUS_PER_TASK \
   "

# RUN
cd $DCFT_PRIVATE

export PATH=$PATH:~/.local/bin/

if [ -n "$SSH_KEY" ]; then
  eval "$SSH_TUNNEL_CMD" 
fi

sleep 1
srun $SRUN_ARGS bash -lc "proxychains4 $APPTAINER_ARGS $CMD"

rm -rf $ACCELERATE_CONFIG_FILE
