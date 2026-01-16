#!/usr/bin/env python3
"""
Generate SLURM batch scripts for running vLLM experiments with Ray Serve data parallelism.

Creates sbatch files that:
1. Source start_ray_cluster.sh to set up the Ray cluster
2. Run run_vllm_dp.sh to launch vLLM via Ray Serve with replicas for data parallelism
3. Run benchmarks against the vLLM endpoint
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List
import argparse


class RayServeSbatchGenerator:
    """Generate SLURM batch scripts for vLLM experiments using Ray Serve."""

    SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -p gh
#SBATCH --time=02:00:00
#SBATCH --nodes {num_nodes}
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --account CCR24067
#SBATCH --output={experiments_dir}/logs/vllm_{config_hash}_%j.out
#SBATCH --job-name=vllm_{config_name}

set -eo pipefail

EXPERIMENTS_DIR="{experiments_dir}"
CONFIG_FILE="{config_file}"
CONFIG_HASH="{config_hash}"
CONFIG_NAME="{config_name}"

echo "============================================"
echo "vLLM Experiment: $CONFIG_NAME ($CONFIG_HASH)"
echo "============================================"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG_FILE"
echo "============================================"

# Create results directory
mkdir -p "$EXPERIMENTS_DIR/results/$CONFIG_HASH"

# Step 1: Start Ray cluster
echo ""
echo "=== Step 1: Starting Ray Cluster ==="
source "$EXPERIMENTS_DIR/start_ray_cluster.sh"

# Verify Ray cluster
if [ -z "$RAY_ADDRESS" ]; then
    echo "ERROR: Ray cluster failed to start"
    exit 1
fi

echo "Ray cluster ready: $RAY_ADDRESS"

# Step 2: Set vLLM configuration from JSON
echo ""
echo "=== Step 2: Configuring vLLM ==="

# Parse config file and set ALL environment variables for dp_debug.py
export VLLM_MODEL_PATH=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('model_path', 'zai-org/GLM-4.6'))")
export VLLM_GPU_MEMORY_UTILIZATION=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('gpu_memory_utilization', 0.95))")
export VLLM_MAX_NUM_SEQS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_num_seqs', 256))")
export VLLM_MAX_MODEL_LEN=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_model_len', 32768))")
export VLLM_MAX_NUM_BATCHED_TOKENS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('max_num_batched_tokens', 16384))")
export VLLM_SWAP_SPACE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('swap_space_gb', 20))")
export VLLM_BLOCK_SIZE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('block_size', 16))")
export VLLM_TENSOR_PARALLEL_SIZE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('tensor_parallel_size', 8))")
export VLLM_PIPELINE_PARALLEL_SIZE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('pipeline_parallel_size', 1))")

# Boolean settings
export VLLM_ENABLE_EXPERT_PARALLEL=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_expert_parallel', True) else 'false')")
export VLLM_ENABLE_CHUNKED_PREFILL=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_chunked_prefill', True) else 'false')")
export VLLM_ENABLE_PREFIX_CACHING=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_prefix_caching', False) else 'false')")
export VLLM_ENABLE_EPLB=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print('true' if c.get('enable_eplb', False) else 'false')")

# KV cache dtype
export VLLM_KV_CACHE_DTYPE=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('kv_cache_dtype', 'auto'))")

# Expert parallel settings
export VLLM_ALL2ALL_BACKEND=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('all2all_backend', 'pplx'))")
export VLLM_EPLB_NUM_REDUNDANT_EXPERTS=$(python3 -c "import json; c=json.load(open('$CONFIG_FILE')); print(c.get('eplb_num_redundant_experts', 32))")

echo "Configuration:"
echo "  Model: $VLLM_MODEL_PATH"
echo "  Tensor Parallel Size: $VLLM_TENSOR_PARALLEL_SIZE"
echo "  Pipeline Parallel Size: $VLLM_PIPELINE_PARALLEL_SIZE"
echo "  GPU Memory Util: $VLLM_GPU_MEMORY_UTILIZATION"
echo "  Max Model Len: $VLLM_MAX_MODEL_LEN"
echo "  Max Num Seqs: $VLLM_MAX_NUM_SEQS"
echo "  Max Batched Tokens: $VLLM_MAX_NUM_BATCHED_TOKENS"
echo "  Enable Expert Parallel: $VLLM_ENABLE_EXPERT_PARALLEL"
echo "  KV Cache Dtype: $VLLM_KV_CACHE_DTYPE"
echo "  Enable Prefix Caching: $VLLM_ENABLE_PREFIX_CACHING"
echo "  Enable Chunked Prefill: $VLLM_ENABLE_CHUNKED_PREFILL"
echo "  All2All Backend: $VLLM_ALL2ALL_BACKEND"
echo "  Enable EPLB: $VLLM_ENABLE_EPLB"

# Step 3: Start vLLM via Ray Serve
echo ""
echo "=== Step 3: Starting vLLM Ray Serve ==="
source "$EXPERIMENTS_DIR/run_vllm_dp.sh"

# Verify vLLM is running
if [ -z "$VLLM_ENDPOINT_URL" ]; then
    echo "ERROR: vLLM failed to start"
    exit 1
fi

echo "vLLM ready at: $VLLM_ENDPOINT_URL"

# Step 4: Run benchmark from head node (server binds to localhost)
echo ""
echo "=== Step 4: Running Benchmark ==="

# Get head node
HEAD_NODE=$(echo "$RAY_NODES_ARRAY" | awk '{{print $1}}')

# Run benchmark from head node so it can access http://127.0.0.1:8000
srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$HEAD_NODE" \\
    env TRITON_CC="$TRITON_CC" \\
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \\
        PATH="$PATH" \\
        LD_PRELOAD="$LD_PRELOAD" \\
    /scratch/10000/eguha3/vllm_sandboxes_backup/bin/python3 "$EXPERIMENTS_DIR/benchmark_runner.py" \\
        --server-url "http://127.0.0.1:8000" \\
        --dataset "$EXPERIMENTS_DIR/datasets/{dataset}" \\
        --config-hash "$CONFIG_HASH" \\
        --request-rate {request_rate} \\
        --output-dir "$EXPERIMENTS_DIR/results/$CONFIG_HASH"

BENCHMARK_EXIT_CODE=$?

# Step 5: Cleanup
echo ""
echo "=== Step 5: Cleanup ==="

# Stop vLLM server
if [ -n "$VLLM_PID" ]; then
    echo "Stopping vLLM server (PID: $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
fi

# Stop Ray on all nodes
RAY_BIN="/scratch/10000/eguha3/vllm_sandboxes_backup/bin/ray"
for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    srun --nodes=1 --ntasks=1 --overlap -w "$node" $RAY_BIN stop --force 2>/dev/null &
done
wait

echo ""
echo "============================================"
if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo "Experiment COMPLETED: $CONFIG_NAME ($CONFIG_HASH)"
else
    echo "Experiment FAILED: $CONFIG_NAME ($CONFIG_HASH)"
fi
echo "Results: $EXPERIMENTS_DIR/results/$CONFIG_HASH"
echo "============================================"

exit $BENCHMARK_EXIT_CODE
"""

    def __init__(self, experiments_dir: str = None):
        if experiments_dir is None:
            experiments_dir = os.path.expandvars("${DCAGENT_DIR}/data/vllm_experiments")
        self.experiments_dir = Path(experiments_dir)
        self.logs_dir = self.experiments_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate a unique hash for the config."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def generate_sbatch_script(
        self,
        config: Dict[str, Any],
        config_file: str,
        dataset: str = "large_throughput.jsonl",
        request_rate: float = 10.0,
    ) -> Path:
        """Generate sbatch script for an experiment."""
        config_hash = self.generate_config_hash(config)
        config_name = config.get("name", f"exp_{config_hash}")
        num_nodes = config.get("num_nodes", 8)

        # For Ray Serve, we use 8 nodes with 1 GPU each
        # The "gpus_per_node" in config is for reference, actual is 1 GPU per node

        script_content = self.SBATCH_TEMPLATE.format(
            num_nodes=num_nodes,
            experiments_dir=str(self.experiments_dir),
            config_file=config_file,
            config_hash=config_hash,
            config_name=config_name,
            dataset=dataset,
            request_rate=request_rate,
        )

        script_path = self.experiments_dir / f"run_{config_hash}_{config_name}.sbatch"
        with open(script_path, 'w') as f:
            f.write(script_content)

        script_path.chmod(0o755)
        return script_path

    def generate_all_scripts(
        self,
        config_dir: str = None,
        dataset: str = "large_throughput.jsonl",
        request_rate: float = 10.0,
    ) -> List[Path]:
        """Generate sbatch scripts for all configs in a directory."""
        if config_dir is None:
            config_dir = self.experiments_dir / "configs" / "focused"

        config_dir = Path(config_dir)
        config_files = sorted(config_dir.glob("config_*.json"))

        if not config_files:
            print(f"No config files found in {config_dir}")
            return []

        scripts = []
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config = json.load(f)

            script_path = self.generate_sbatch_script(
                config=config,
                config_file=str(config_file),
                dataset=dataset,
                request_rate=request_rate,
            )
            scripts.append(script_path)
            print(f"Generated: {script_path.name}")

        return scripts

    def generate_launch_script(self, script_paths: List[Path]) -> Path:
        """Generate a master script to launch all experiments."""
        launch_content = """#!/bin/bash
# Launch all vLLM experiments with Ray Serve data parallelism
# Generated automatically - do not edit

set -e

EXPERIMENTS_DIR="{experiments_dir}"
cd "$EXPERIMENTS_DIR"

echo "Launching {num_experiments} vLLM experiments..."
echo ""

""".format(experiments_dir=str(self.experiments_dir), num_experiments=len(script_paths))

        for i, script_path in enumerate(script_paths):
            launch_content += f"""
# Experiment {i+1}/{len(script_paths)}
echo "Submitting: {script_path.name}"
JOB_ID=$(sbatch --parsable "{script_path}")
echo "  Job ID: $JOB_ID"
sleep 2
"""

        launch_content += """
echo ""
echo "All experiments submitted!"
echo "Check status with: squeue -u $USER"
echo "View results in: $EXPERIMENTS_DIR/results/"
"""

        launch_path = self.experiments_dir / "launch_all.sh"
        with open(launch_path, 'w') as f:
            f.write(launch_content)

        launch_path.chmod(0o755)
        return launch_path


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM batch scripts for Ray Serve vLLM experiments")
    parser.add_argument("--config-dir", help="Directory containing config JSON files")
    parser.add_argument("--experiments-dir", default=None,
                        help="Base experiments directory (default: $DCAGENT_DIR/data/vllm_experiments)")
    parser.add_argument("--dataset", default="large_throughput.jsonl", help="Dataset file name")
    parser.add_argument("--request-rate", type=float, default=10.0, help="Requests per second")
    parser.add_argument("--single-config", help="Generate script for single config file")

    args = parser.parse_args()

    generator = RayServeSbatchGenerator(experiments_dir=args.experiments_dir)

    if args.single_config:
        with open(args.single_config, 'r') as f:
            config = json.load(f)

        script_path = generator.generate_sbatch_script(
            config=config,
            config_file=args.single_config,
            dataset=args.dataset,
            request_rate=args.request_rate,
        )
        print(f"Generated: {script_path}")
    else:
        config_dir = args.config_dir or str(generator.experiments_dir / "configs" / "focused")
        scripts = generator.generate_all_scripts(
            config_dir=config_dir,
            dataset=args.dataset,
            request_rate=args.request_rate,
        )

        if scripts:
            launch_script = generator.generate_launch_script(scripts)
            print(f"\nGenerated {len(scripts)} sbatch scripts")
            print(f"Launch script: {launch_script}")
            print(f"\nTo launch all experiments:")
            print(f"  ./{launch_script.name}")


if __name__ == "__main__":
    main()
