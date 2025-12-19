#!/bin/bash

CMD_PREFIX=${1:-""}

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_i="${head_node}i"
head_node_ip=$(nslookup "$head_node_i" | grep 'Address' | tail -n1 | awk '{print $2}')

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 --gres=gpu:4 --overlap -w "$head_node" \
    $CMD_PREFIX ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 48 --num-gpus 4 --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    node_i_ip="${node_i}i"
    node_i_ip_addr=$(nslookup "$node_i_ip" | grep 'Address' | tail -n1 | awk '{print $2}')
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 --gres=gpu:4 -w "$node_i" \
        $CMD_PREFIX ray start --address "$ip_head" --node-ip-address="$node_i_ip_addr" \
        --num-cpus 48 --num-gpus 4 --block &
    sleep 5
done

export RAY_ADDRESS=$ip_head
echo "RAY ADDRESS: $RAY_ADDRESS"
sleep 30



