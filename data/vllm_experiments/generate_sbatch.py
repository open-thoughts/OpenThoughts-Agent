#!/usr/bin/env python3
"""
Generate SLURM batch scripts for running vLLM experiments in parallel.

Creates sbatch files for multi-node vLLM deployment and benchmarking.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import argparse


class SbatchGenerator:
    """Generate SLURM batch scripts for vLLM experiments."""

    # Template for head node (node 0) - runs vLLM server and benchmark
    HEAD_NODE_TEMPLATE = """#!/bin/bash
#SBATCH -p gh
#SBATCH --time=04:00:00
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --account CCR24067
#SBATCH --output=logs/vllm_{exp_id}_%j.out
#SBATCH --job-name=vllm_{exp_name}

# Set up environment
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /work/10000/eguha3/vista/miniconda3/envs/dataagent
source /scratch/10000/eguha3/old-ot-agent/secret.env

# Get head node IP
export HEAD_NODE_IP=$(hostname -i)
echo "Head node IP: $HEAD_NODE_IP"

# Set up Ray cluster
echo "Starting Ray cluster..."
ray start --head --port=6379 --num-cpus=64 --disable-usage-stats

# Wait for all nodes to be ready
sleep 30

# NCCL configuration for multi-node
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_NET_GDR_READ=1
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_TIMEOUT=20
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_PCI_RELAXED_ORDERING=2
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_MIN_NCHANNELS=16
export NCCL_BUFFSIZE=8388608
export NCCL_DEBUG=INFO

# vLLM environment
export VLLM_HOST_IP=$HEAD_NODE_IP
export VLLM_RAY_DP_PACK_STRATEGY=span
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Start vLLM server
echo "Starting vLLM server on head node..."
cd /scratch/10000/eguha3/ot-agent/data/vllm_experiments

{vllm_command} &
VLLM_PID=$!

echo "vLLM server PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
for i in {{1..60}}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 10
done

# Run benchmark
echo "Running benchmark..."
python benchmark_runner.py \\
    --server-url http://localhost:8000 \\
    --prometheus-url http://localhost:8001 \\
    --dataset datasets/{dataset} \\
    --config-hash {config_hash} \\
    --request-rate {request_rate} \\
    --output-dir results/{exp_id}

# Collect results
echo "Benchmark complete!"

# Cleanup
echo "Shutting down..."
kill $VLLM_PID
ray stop
sleep 10

echo "Job complete: {exp_id}"
"""

    # Template for worker nodes (headless vLLM workers)
    WORKER_NODE_TEMPLATE = """#!/bin/bash
#SBATCH -p gh
#SBATCH --time=04:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --account CCR24067
#SBATCH --output=logs/vllm_worker_{exp_id}_node{node_id}_%j.out
#SBATCH --job-name=vllm_worker_{exp_name}_n{node_id}
#SBATCH --dependency=singleton

# Set up environment
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /work/10000/eguha3/vista/miniconda3/envs/dataagent
source /scratch/10000/eguha3/old-ot-agent/secret.env

# Get head node IP from environment or file
export HEAD_NODE_IP={head_node_ip}
echo "Connecting to head node: $HEAD_NODE_IP"

# Join Ray cluster
echo "Joining Ray cluster..."
ray start --address=$HEAD_NODE_IP:6379 --num-cpus=64 --disable-usage-stats

# NCCL configuration
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=PXB
export NCCL_NET_GDR_READ=1
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_TIMEOUT=20
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_PCI_RELAXED_ORDERING=2
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_MIN_NCHANNELS=16
export NCCL_BUFFSIZE=8388608
export NCCL_DEBUG=INFO

# vLLM environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Start headless vLLM worker
echo "Starting vLLM worker (headless) on node {node_id}..."
cd /scratch/10000/eguha3/ot-agent/data/vllm_experiments

{vllm_command}

# This will keep running until job is cancelled or head node shuts down
echo "Worker node {node_id} shutting down..."
ray stop
"""

    def __init__(self, output_dir: str = ".", logs_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def generate_exp_id(self, config: Dict[str, Any]) -> str:
        """Generate experiment ID from config."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def generate_vllm_command(self, config: Dict[str, Any], node_id: int = 0) -> str:
        """Generate vLLM serve command from configuration."""
        cmd_parts = ["vllm serve", config.get("model_path", "THUDM/glm-4-9b-chat")]

        # Distributed configuration
        if node_id == 0:
            # Head node
            cmd_parts.extend([
                f"--tensor-parallel-size {config.get('tensor_parallel_size', 1)}",
                f"--data-parallel-size {config.get('data_parallel_size', 64)}",
                f"--data-parallel-size-local {config.get('data_parallel_size_local', 8)}",
            ])

            if config.get("enable_expert_parallel", True):
                cmd_parts.append("--enable-expert-parallel")

            cmd_parts.extend([
                f"--distributed-executor-backend {config.get('distributed_executor_backend', 'ray')}",
                f"--api-server-count {config.get('api_server_count', 8)}",
                "--port 8000",
                "--host 0.0.0.0",
            ])
        else:
            # Worker nodes (headless)
            cmd_parts.extend([
                "--headless",
                f"--tensor-parallel-size {config.get('tensor_parallel_size', 1)}",
                f"--data-parallel-size {config.get('data_parallel_size', 64)}",
                f"--data-parallel-size-local {config.get('data_parallel_size_local', 8)}",
                f"--data-parallel-start-rank {node_id * config.get('data_parallel_size_local', 8)}",
            ])

            if config.get("enable_expert_parallel", True):
                cmd_parts.append("--enable-expert-parallel")

            cmd_parts.extend([
                "--data-parallel-address $HEAD_NODE_IP",
                "--data-parallel-rpc-port 13345",
                f"--distributed-executor-backend {config.get('distributed_executor_backend', 'ray')}",
            ])

        # Memory optimization
        cmd_parts.extend([
            f"--gpu-memory-utilization {config.get('gpu_memory_utilization', 0.92)}",
            f"--max-model-len {config.get('max_model_len', 32768)}",
            f"--kv-cache-dtype {config.get('kv_cache_dtype', 'auto')}",
            f"--swap-space-gb {config.get('swap_space_gb', 20)}",
            f"--block-size {config.get('block_size', 16)}",
        ])

        # Batching & scheduling
        cmd_parts.extend([
            f"--max-num-batched-tokens {config.get('max_num_batched_tokens', 16384)}",
            f"--max-num-seqs {config.get('max_num_seqs', 256)}",
            f"--num-scheduler-steps {config.get('num_scheduler_steps', 2)}",
        ])

        # Chunked prefill
        if config.get("enable_chunked_prefill", True):
            cmd_parts.extend([
                "--enable-chunked-prefill",
                f"--max-num-partial-prefills {config.get('max_num_partial_prefills', 1)}",
            ])

        # Expert parallelism
        if config.get("all2all_backend"):
            cmd_parts.append(f"--all2all-backend {config['all2all_backend']}")

        if config.get("enable_eplb", False):
            eplb_config = {
                "window_size": config.get("eplb_window_size", 1000),
                "step_interval": config.get("eplb_step_interval", 3000),
                "num_redundant_experts": config.get("eplb_num_redundant_experts", 32),
                "log_balancedness": True,
            }
            eplb_json = json.dumps(eplb_config).replace('"', '\\"')
            cmd_parts.append(f'--enable-eplb --eplb-config "{eplb_json}"')

        # Prefix caching
        if config.get("enable_prefix_caching", False):
            cmd_parts.append("--enable-prefix-caching")

        # Performance
        cmd_parts.extend([
            f"--max-seq-len-to-capture {config.get('max_seq_len_to_capture', 8192)}",
            f"--dtype {config.get('dtype', 'bfloat16')}",
            "--trust-remote-code",
        ])

        # Monitoring (head node only)
        if node_id == 0:
            cmd_parts.extend([
                "--disable-log-stats false",
                "--prometheus-port 8001",
            ])

        return " \\\n  ".join(cmd_parts)

    def generate_sbatch_script(
        self,
        config: Dict[str, Any],
        dataset: str = "medium_balanced.jsonl",
        request_rate: float = 10.0,
        exp_name: str = None,
    ) -> Dict[str, Path]:
        """Generate sbatch scripts for an experiment."""
        exp_id = self.generate_exp_id(config)

        if exp_name is None:
            exp_name = f"exp_{exp_id}"

        num_nodes = config.get("num_nodes", 8)
        vllm_head_cmd = self.generate_vllm_command(config, node_id=0)

        # Generate head node script
        head_script = self.HEAD_NODE_TEMPLATE.format(
            num_nodes=num_nodes,
            exp_id=exp_id,
            exp_name=exp_name,
            vllm_command=vllm_head_cmd,
            dataset=dataset,
            config_hash=exp_id,
            request_rate=request_rate,
        )

        head_script_path = self.output_dir / f"run_{exp_id}_head.sbatch"
        with open(head_script_path, 'w') as f:
            f.write(head_script)

        scripts = {"head": head_script_path}

        # Note: For SLURM multi-node jobs, all nodes are allocated together
        # The worker node scripts are not needed when using multi-node allocation
        # The head node script will handle all nodes

        return scripts

    def generate_batch_scripts(
        self,
        configs: List[Dict[str, Any]],
        dataset: str = "medium_balanced.jsonl",
        request_rate: float = 10.0,
    ) -> List[Dict[str, Path]]:
        """Generate sbatch scripts for multiple experiments."""
        all_scripts = []

        for i, config in enumerate(configs):
            exp_name = f"exp_{i:03d}"
            scripts = self.generate_sbatch_script(
                config=config,
                dataset=dataset,
                request_rate=request_rate,
                exp_name=exp_name,
            )
            all_scripts.append(scripts)

        return all_scripts

    def generate_launch_script(self, script_paths: List[Dict[str, Path]]) -> Path:
        """Generate a master script to launch all experiments."""
        launch_script = "#!/bin/bash\n\n"
        launch_script += "# Auto-generated launch script for vLLM experiments\n\n"

        for i, scripts in enumerate(script_paths):
            head_script = scripts["head"]
            launch_script += f"# Experiment {i}\n"
            launch_script += f"echo 'Submitting experiment {i}...'\n"
            launch_script += f"sbatch {head_script}\n"
            launch_script += f"sleep 2\n\n"

        launch_path = self.output_dir / "launch_all_experiments.sh"
        with open(launch_path, 'w') as f:
            f.write(launch_script)

        # Make executable
        launch_path.chmod(0o755)

        return launch_path


def main():
    """Generate sbatch scripts from command line."""
    parser = argparse.ArgumentParser(description="Generate SLURM batch scripts for vLLM experiments")
    parser.add_argument("--config-dir", default="configs", help="Directory containing config JSON files")
    parser.add_argument("--output-dir", default=".", help="Output directory for sbatch scripts")
    parser.add_argument("--phase", help="Generate scripts for specific phase")
    parser.add_argument("--dataset", default="medium_balanced.jsonl", help="Dataset file name")
    parser.add_argument("--request-rate", type=float, default=10.0, help="Requests per second")
    parser.add_argument("--single-config", help="Generate script for single config file")

    args = parser.parse_args()

    generator = SbatchGenerator(output_dir=args.output_dir)

    if args.single_config:
        # Single config
        with open(args.single_config, 'r') as f:
            config = json.load(f)

        scripts = generator.generate_sbatch_script(
            config=config,
            dataset=args.dataset,
            request_rate=args.request_rate,
        )

        print(f"Generated sbatch script:")
        print(f"  Head: {scripts['head']}")

    else:
        # Multiple configs from phase or all configs
        config_dir = Path(args.config_dir)

        if args.phase:
            # Load configs from specific phase
            phase_dir = config_dir / args.phase
            config_files = sorted(phase_dir.glob("*_config_*.json"))
        else:
            # Load all config files
            config_files = sorted(config_dir.rglob("*_config_*.json"))

        if not config_files:
            print(f"No config files found in {config_dir}")
            return

        print(f"Found {len(config_files)} config files")

        configs = []
        for config_file in config_files:
            with open(config_file, 'r') as f:
                configs.append(json.load(f))

        # Generate scripts
        all_scripts = generator.generate_batch_scripts(
            configs=configs,
            dataset=args.dataset,
            request_rate=args.request_rate,
        )

        print(f"Generated {len(all_scripts)} experiment scripts")

        # Generate launch script
        launch_script = generator.generate_launch_script(all_scripts)
        print(f"Generated launch script: {launch_script}")
        print(f"\nTo launch all experiments:")
        print(f"  ./{launch_script}")


if __name__ == "__main__":
    main()
