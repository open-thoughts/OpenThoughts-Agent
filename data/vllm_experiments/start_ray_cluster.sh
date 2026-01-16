#!/bin/bash
# Start Ray cluster across SLURM-allocated nodes with proper InfiniBand interface
# Usage: source start_ray_cluster.sh
# This script sets up RAY_ADDRESS and other environment variables for run_vllm_dp.sh

set -e

# Load modules
module load gcc/15.1.0 cuda/12.8

# Environment setup for GCC compatibility
export LD_PRELOAD=/home1/apps/gcc/15.1.0/lib64/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:$LD_LIBRARY_PATH
export PATH=/home1/apps/gcc/15.1.0/bin:$PATH
export TRITON_CC=$(which gcc)

# Load HF token
source /scratch/10000/eguha3/old-dc-agent/secret.env

# Ray/vLLM paths
RAY_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/ray"
PYTHON_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3"
export RAY_DEDUP_LOGS=0

# Get node info from SLURM
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
NUM_NODES=${#nodes_array[@]}

# Cleanup function
cleanup_ray_cluster() {
    echo "Cleaning up Ray cluster..."
    for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
        srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null &
        srun --nodes=1 --ntasks=1 --overlap -w "$node" pkill -9 -f raylet 2>/dev/null &
        srun --nodes=1 --ntasks=1 --overlap -w "$node" pkill -9 -f gcs_server 2>/dev/null &
    done
    wait
    sleep 3
}

# Kill any existing Ray processes on all nodes
echo "Stopping any existing Ray processes..."
cleanup_ray_cluster

# Clean up session dirs on all nodes
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" rm -rf /tmp/ray_${USER}/session_* 2>/dev/null &
done
wait
sleep 2

# Use InfiniBand interface for Ray head IP (critical for multi-node)
head_iface="ib0"
echo "Probing for head node InfiniBand interface..."
if srun --nodes=1 --ntasks=1 --overlap -w "$head_node" ip -o -4 addr show "$head_iface" >/dev/null 2>&1; then
    head_node_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" ip -o -4 addr show "$head_iface" | awk '{print $4}' | cut -d/ -f1)
    export RAY_INTERFACE="$head_iface"
    echo "Using ${head_iface} address for Ray: $head_node_ip"
else
    echo "Warning: ${head_iface} not available, falling back to hostname --ip-address"
    head_node_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" hostname --ip-address)
    head_node_ip=${head_node_ip%% *}
    export RAY_INTERFACE="eth0"
    echo "Using fallback interface address: $head_node_ip"
fi

RAY_PORT=6379
RAY_TEMP_DIR="/tmp/ray_${USER}"
export RAY_ADDRESS="${head_node_ip}:${RAY_PORT}"

echo "============================================"
echo "Ray Cluster Setup"
echo "============================================"
echo "Head node: $head_node ($head_node_ip)"
echo "Total nodes: $NUM_NODES"
echo "Nodes: ${nodes_array[@]}"
echo "RAY_ADDRESS: $RAY_ADDRESS"
echo "============================================"

# Environment to forward to workers
export SRUN_EXPORT_ENV="ALL,TRITON_CC=$TRITON_CC,LD_LIBRARY_PATH=$LD_LIBRARY_PATH,PATH=$PATH,LD_PRELOAD=$LD_PRELOAD,HF_TOKEN=$HF_TOKEN"
RAY_ENV_VARS="TRITON_CC=$TRITON_CC LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH LD_PRELOAD=$LD_PRELOAD HF_TOKEN=$HF_TOKEN"

# Start Ray head node (daemon mode)
echo "Starting Ray head on $head_node ($head_node_ip)..."
srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" bash -c "env $RAY_ENV_VARS $RAY_BIN start --head --node-ip-address=${head_node_ip} --port=${RAY_PORT} --num-gpus=1 --num-cpus=64 --temp-dir=${RAY_TEMP_DIR}"

sleep 10

# Start workers (daemon mode)
echo "Starting $((NUM_NODES - 1)) workers..."
for ((i = 1; i < NUM_NODES; i++)); do
    node_i=${nodes_array[$i]}
    echo "  Starting worker on ${node_i}..."
    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$node_i" bash -c "env $RAY_ENV_VARS $RAY_BIN start --address ${head_node_ip}:${RAY_PORT} --num-gpus=1 --num-cpus=64 --temp-dir=${RAY_TEMP_DIR}" &
    sleep 3
done

echo "Waiting for workers to connect..."
sleep 20

# Verify cluster
echo "============================================"
echo "Ray Cluster Status:"
echo "============================================"
$RAY_BIN status

echo ""
echo "Ray cluster started successfully!"
echo "RAY_ADDRESS=$RAY_ADDRESS"
echo "HEAD_NODE_IP=$head_node_ip"

# Export variables for run_vllm_dp.sh
export HEAD_NODE_IP="$head_node_ip"
export RAY_NODES_ARRAY="${nodes_array[*]}"
export RAY_NUM_NODES="$NUM_NODES"
