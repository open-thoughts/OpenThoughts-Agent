# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

# Get head node IP more reliably using hostname -I (lists all IPs) and take the first IPv4
# This avoids issues with hostname --ip-address returning wrong/multiple IPs
head_node_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" bash -c 'hostname -I | awk "{print \$1}"')

# Validate we got a proper IP
if [[ -z "$head_node_ip" ]] || [[ ! "$head_node_ip" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: Failed to get valid head node IP. Got: '$head_node_ip'"
    exit 1
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# Don't use --temp-dir on shared filesystem - sockets don't work there
# Ray will use /tmp by default for sockets
echo "Starting HEAD at $head_node ($head_node_ip)"
srun --nodes=1 --ntasks=1 --overlap -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 72 --num-gpus 1 --resources='{"vllm_instance": 1}' --block &

# Wait for head to be ready
sleep 15

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 --overlap -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus 72 --num-gpus 1 --resources='{"vllm_instance": 0}' --block &
    sleep 5
done

# Wait a bit for all workers to connect
sleep 10
echo "Ray cluster startup complete. Workers should be connecting to $ip_head"

# Write the head address to a file so the parent script can set RAY_ADDRESS
echo "$ip_head" > /tmp/ray_head_address_${SLURM_JOB_ID}
echo "RAY_ADDRESS written to /tmp/ray_head_address_${SLURM_JOB_ID}"
