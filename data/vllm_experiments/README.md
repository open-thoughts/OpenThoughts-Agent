# vLLM Multi-Node Experiment Framework

A comprehensive framework for running parameter sweep experiments on GLM-4.6 (or other MoE models) across multi-node H200 GPU clusters using vLLM.

## Overview

This framework automates:
- **Configuration generation**: Parameter sweeps for critical vLLM settings
- **Dataset creation**: Synthetic benchmark prompts with controlled distributions
- **Multi-node deployment**: SLURM scripts for 8-node (64 GPU) vLLM clusters
- **Benchmarking**: Automated inference with metrics collection
- **Analysis**: Aggregation and comparison of results

## Directory Structure

```
vllm_experiments/
├── config_generator.py      # Generate experiment configurations
├── dataset_loader.py         # Create/load benchmark datasets
├── benchmark_runner.py       # Run benchmarks and collect metrics
├── generate_sbatch.py        # Generate SLURM batch scripts
├── aggregate_results.py      # Analyze and compare results
├── configs/                  # Generated configurations
│   ├── quick_test.json
│   ├── memory_batch/
│   ├── quantization_caching/
│   ├── expert_parallel/
│   ├── chunked_prefill/
│   └── scheduler/
├── datasets/                 # Benchmark datasets
│   ├── quick_test.jsonl
│   ├── medium_balanced.jsonl
│   ├── large_throughput.jsonl
│   └── ...
├── results/                  # Benchmark results
│   └── exp_*/
└── logs/                     # SLURM job logs
```

## Quick Start

### 1. Generate Datasets

Create benchmark datasets with different characteristics:

```bash
# Create full benchmark suite (recommended)
python dataset_loader.py --create-suite --show-stats

# Or create a custom dataset
python dataset_loader.py --num-prompts 500 --max-output-tokens 512
```

This generates:
- `quick_test.jsonl` (100 prompts) - for validation
- `medium_balanced.jsonl` (500 prompts) - for standard benchmarks
- `large_throughput.jsonl` (1000 prompts) - for throughput testing
- `short_latency.jsonl` (500 short prompts) - for latency testing
- `long_context.jsonl` (300 long prompts) - for context testing

### 2. Generate Experiment Configurations

Generate parameter sweep configurations:

```bash
# Generate quick test config (single run)
python config_generator.py --quick-test

# Generate all phase configurations
python config_generator.py --phase all

# Generate specific phase
python config_generator.py --phase memory_batch
python config_generator.py --phase expert_parallel
```

Phases:
- **memory_batch**: GPU memory utilization, batch sizes, concurrent sequences
- **quantization_caching**: FP8 KV cache, prefix caching
- **expert_parallel**: All-to-all backends, EPLB configurations
- **chunked_prefill**: Prefill/decode mixing strategies
- **scheduler**: Scheduler configurations

### 3. Generate SLURM Scripts

Create sbatch scripts for running experiments:

```bash
# Generate scripts for all configs in a phase
python generate_sbatch.py \
  --config-dir configs/memory_batch \
  --dataset medium_balanced.jsonl \
  --request-rate 10.0

# Generate script for single config
python generate_sbatch.py \
  --single-config configs/quick_test.json \
  --dataset quick_test.jsonl \
  --request-rate 5.0

# Generate for all phases
python generate_sbatch.py \
  --config-dir configs \
  --dataset medium_balanced.jsonl \
  --request-rate 10.0
```

This creates:
- Individual `run_<exp_id>_head.sbatch` files for each experiment
- `launch_all_experiments.sh` master script

### 4. Launch Experiments

Run experiments in parallel using SLURM:

```bash
# Launch all experiments
./launch_all_experiments.sh

# Or launch individual experiments
sbatch run_<exp_id>_head.sbatch

# Check job status
squeue -u $USER

# Monitor logs
tail -f logs/vllm_<exp_id>_<job_id>.out
```

### 5. Aggregate and Analyze Results

After experiments complete, aggregate results:

```bash
# Basic aggregation
python aggregate_results.py \
  --results-dir results \
  --config-dir configs

# With config enrichment (adds parameter columns)
python aggregate_results.py \
  --results-dir results \
  --config-dir configs \
  --enrich-configs

# Compare specific parameter
python aggregate_results.py \
  --enrich-configs \
  --compare-param max_num_batched_tokens \
  --metric throughput_tokens_per_sec
```

This generates:
- `experiment_results.csv` - All results in tabular format
- `experiment_report.txt` - Human-readable summary
- `viz_data.json` - Data for visualization tools

## Configuration Parameters

### Critical Parameters (Highest Impact)

**Memory & Batch:**
- `gpu_memory_utilization`: [0.90, 0.92, 0.95] - KV cache allocation
- `max_num_batched_tokens`: [4096, 8192, 16384, 32768] - Batch size control
- `max_num_seqs`: [64, 128, 256, 512] - Concurrent requests
- `max_model_len`: [16384, 32768] - Context window

**Quantization:**
- `kv_cache_dtype`: ["auto", "fp8_e5m2"] - 2x memory capacity with FP8
- `enable_prefix_caching`: [False, True] - Reuse shared prefixes

**Expert Parallelism (MoE):**
- `all2all_backend`: ["pplx", "deepep_low_latency", "deepep_high_throughput"]
- `enable_eplb`: [False, True] - Expert load balancing
- `eplb_num_redundant_experts`: [16, 32] - Expert replication count

**Chunked Prefill:**
- `enable_chunked_prefill`: [False, True] - Mix prefill/decode
- `max_num_partial_prefills`: [1, 2, 4] - Chunk aggressiveness

### Metrics Collected

**Throughput:**
- `throughput_tokens_per_sec` - Tokens generated per second
- `throughput_requests_per_sec` - Requests completed per second

**Latency:**
- `ttft_p50/p95/p99` - Time to first token (ms)
- `e2e_latency_p50/p95/p99` - End-to-end latency (ms)
- `itl_p50/p95/p99` - Inter-token latency (ms)

**System:**
- `success_rate` - Percentage of successful requests
- `prom_preemptions` - Number of preempted requests
- `prom_cache_usage` - GPU cache utilization percentage

## Usage Examples

### Example 1: Quick Validation Test

Test the setup with a minimal configuration:

```bash
# 1. Generate quick test config and dataset
python config_generator.py --quick-test
python dataset_loader.py --create-suite

# 2. Generate sbatch script
python generate_sbatch.py \
  --single-config configs/quick_test.json \
  --dataset datasets/quick_test.jsonl \
  --request-rate 5.0

# 3. Launch
sbatch run_*_head.sbatch

# 4. Monitor
tail -f logs/vllm_*.out

# 5. Check results
python aggregate_results.py
```

### Example 2: Memory Configuration Sweep

Find optimal memory and batch settings:

```bash
# 1. Generate configs
python config_generator.py --phase memory_batch

# 2. Generate sbatch scripts
python generate_sbatch.py \
  --config-dir configs/memory_batch \
  --dataset datasets/medium_balanced.jsonl \
  --request-rate 10.0

# 3. Launch all experiments
./launch_all_experiments.sh

# 4. Analyze results
python aggregate_results.py \
  --enrich-configs \
  --compare-param max_num_batched_tokens
```

### Example 3: Expert Parallelism Comparison

Compare all-to-all backends and EPLB:

```bash
# 1. Generate configs
python config_generator.py --phase expert_parallel

# 2. Generate sbatch scripts
python generate_sbatch.py \
  --config-dir configs/expert_parallel \
  --dataset datasets/medium_balanced.jsonl \
  --request-rate 10.0

# 3. Launch
./launch_all_experiments.sh

# 4. Compare all2all backends
python aggregate_results.py \
  --enrich-configs \
  --compare-param all2all_backend

# 5. Compare EPLB impact
python aggregate_results.py \
  --enrich-configs \
  --compare-param enable_eplb
```

### Example 4: Full Ablation Study

Run complete parameter sweep (all phases):

```bash
# 1. Generate everything
python config_generator.py --phase all
python dataset_loader.py --create-suite

# 2. Generate all sbatch scripts
python generate_sbatch.py \
  --config-dir configs \
  --dataset datasets/medium_balanced.jsonl \
  --request-rate 10.0

# 3. Launch (this will submit many jobs!)
./launch_all_experiments.sh

# 4. Wait for completion, then analyze
python aggregate_results.py --enrich-configs

# 5. View report
cat results/experiment_report.txt

# 6. Export for analysis
# Results are in experiment_results.csv - use pandas, Excel, etc.
```

## Customization

### Custom Parameter Ranges

Edit `config_generator.py` to modify sweep ranges:

```python
PARAMETER_SWEEPS = {
    "memory_batch": {
        "gpu_memory_utilization": [0.90, 0.92, 0.95],  # Add more values
        "max_num_batched_tokens": [8192, 16384, 32768, 65536],  # Extend range
        # ...
    },
}
```

### Custom Benchmark Dataset

Create a custom dataset with specific characteristics:

```python
from dataset_loader import DatasetLoader

loader = DatasetLoader()
prompts = loader.generate_dataset(
    num_prompts=1000,
    length_distribution={"short": 0.2, "medium": 0.6, "long": 0.2},
    max_output_tokens=1024,
)
loader.save_dataset(prompts, "custom_benchmark.jsonl")
```

### Custom Metrics Analysis

Add custom analysis in `aggregate_results.py`:

```python
# Find configurations with best latency/throughput tradeoff
def find_pareto_optimal(df):
    # Lower latency + higher throughput = better
    df['score'] = df['throughput_tokens_per_sec'] / df['ttft_p95']
    return df.nlargest(10, 'score')
```

## Troubleshooting

### Job Fails Immediately

Check SLURM output logs:
```bash
cat logs/vllm_*_<job_id>.out
```

Common issues:
- Model path incorrect: Update `model_path` in configs
- Conda environment: Check path in sbatch scripts
- Ray startup: Increase sleep time after `ray start`

### OOM Errors

Reduce memory parameters:
- Lower `gpu_memory_utilization` (0.95 → 0.90)
- Decrease `max_model_len`
- Reduce `max_num_batched_tokens`

### Low Throughput

Increase batch parameters:
- Increase `max_num_batched_tokens` (16K → 32K)
- Increase `max_num_seqs` (256 → 512)
- Enable FP8: `kv_cache_dtype: "fp8_e5m2"`

### High Latency

Reduce batch sizes:
- Decrease `max_num_batched_tokens` (32K → 8K)
- Enable chunked prefill
- Increase `max_num_partial_prefills`

### Network Issues (Multi-Node)

Check NCCL configuration:
- Verify InfiniBand: `ibstat`
- Check IB interfaces: `NCCL_IB_HCA=mlx5_0,mlx5_1`
- Enable NCCL debug: `export NCCL_DEBUG=TRACE`

## Performance Expectations

### Target Metrics (8 nodes × 8 H200 = 64 GPUs)

**Throughput:**
- Expected: 1000-2000 tokens/sec
- Best case (optimized): 2500+ tokens/sec

**Latency (P95):**
- TTFT: 500-2000ms
- ITL: 50-100ms (real-time streaming)

**Success Rate:**
- Target: >99%

## Hardware Requirements

- **Nodes**: 8 nodes
- **GPUs per node**: 8× H200 (96GB each)
- **CPU RAM**: 100GB+ per node
- **Network**: InfiniBand with GPUDirect RDMA
- **Storage**: Shared filesystem for model weights

## Model Requirements

**GLM-4.6:**
- Model path: Update in `config_generator.py`
- Ensure model is accessible from all nodes
- Requires `trust_remote_code=True`

**Alternative Models:**
- Any MoE model supported by vLLM
- Adjust `tensor_parallel_size` for non-MoE models

## Cost Management

To minimize cost while getting meaningful results:

1. **Start small**: Use `quick_test` dataset and config
2. **Selective sweeps**: Run one phase at a time
3. **Reduce dataset size**: Use `--max-prompts 100` in benchmark
4. **Sequential runs**: Launch experiments sequentially, not in parallel
5. **Monitor jobs**: Cancel failing experiments early

Example cost-effective workflow:
```bash
# Phase 1: Quick validation (1 experiment)
python config_generator.py --quick-test
python generate_sbatch.py --single-config configs/quick_test.json
sbatch run_*_head.sbatch

# Phase 2: Memory sweep (12 experiments)
python config_generator.py --phase memory_batch
# Edit configs to reduce sweep size if needed
python generate_sbatch.py --config-dir configs/memory_batch

# Phase 3: Best parameters only
# Based on Phase 2 results, create focused sweep
```

## Files Generated

**Configs**: `configs/*.json` - Experiment configurations
**Datasets**: `datasets/*.jsonl` - Benchmark prompts
**Scripts**: `run_*.sbatch` - SLURM batch scripts
**Results**: `results/*/results_*.json` - Per-experiment results
**Logs**: `logs/vllm_*.out` - Job output logs
**Reports**: `results/experiment_report.txt` - Analysis summary
**Data**: `results/experiment_results.csv` - Tabular results

## Advanced Usage

### Ray Cluster Management

Check Ray cluster status:
```bash
ray status
ray list nodes
```

### Prometheus Metrics

Query vLLM metrics directly:
```bash
curl http://<head_node>:8001/metrics
```

### Custom NCCL Configuration

Edit `generate_sbatch.py` to modify NCCL environment variables for your network topology.

### Distributed Tracing

Enable detailed tracing:
```bash
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
```

## Citation

Based on vLLM optimization guide for multi-node H200 MoE deployment.

## Support

For issues:
1. Check logs in `logs/` directory
2. Verify SLURM job status: `squeue -u $USER`
3. Test single-node first before multi-node
4. Enable debug logging in vLLM

## License

Part of the ot-agent project.
