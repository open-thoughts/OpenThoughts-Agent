CMD_PREFIX="${1:-""}"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# filter head node from nodes array
nodes_array=(${nodes_array[@]/$head_node/})


echo Head node: $head_node
echo Worker nodes: ${nodes_array[@]}


port=26379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --time=${TIME_LIMIT} --cpus-per-task=32 --partition=${PARTITION} --account=${ACCOUNT} --nodes=1 --ntasks=1 --gres=gpu:4 --overlap -w "$head_node" \
    $CMD_PREFIX ray start --head --node-ip-address="$head_node_ip"  --port=$port \
    --num-cpus 32 --num-gpus 4 --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10


# wait until the head node is up before starting workers, nc may not be available on all systems, so we can also just try to connect in a loop until it succeeds
while ! ray status --address "$ip_head" &> /dev/null; do
    echo "Waiting for Ray head to start at $ip_head..."
    sleep 5
done

echo "Ray head is up at $ip_head, starting workers..."

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 0; i < worker_num; i++)); do
    # Every 4 nodes, add a vllm_instance resource
    node=${nodes_array[$i]}
    node_i="${node}-interconnect-1.leonardo.local"
    # node_i=$node
    node_i_ip="${node_i}"
    node_i_ip_addr=$node_i_ip
    echo "Starting WORKER $i at $node_i"
    srun --time=${TIME_LIMIT} --cpus-per-task=32 --partition=${PARTITION} --account=${ACCOUNT} \
         --nodes=1 --ntasks=1 --gres=gpu:4 -w "$node" \
        $CMD_PREFIX ray start --address "$ip_head"  --node-ip-address="$node_i_ip_addr" \
        --num-cpus 32 --num-gpus 4 --block &
    sleep 5
done

export RAY_ADDRESS=$ip_head
echo "RAY ADDRESS: $RAY_ADDRESS"
sleep 10
