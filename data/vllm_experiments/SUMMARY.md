# vLLM Experiment Framework - Summary

## Overview

A complete framework for running GLM-4.6 inference experiments across 8 nodes (64 H200 GPUs) with systematic parameter sweeps and performance analysis.

## Files Created

### Core Scripts (Python)
1. **config_generator.py** - Generates experiment configurations
   - Creates parameter sweep configs for all critical vLLM settings
   - Supports phases: memory_batch, quantization_caching, expert_parallel, chunked_prefill, scheduler
   - Output: JSON config files

2. **dataset_loader.py** - Creates benchmark datasets
   - Generates synthetic prompts with controlled distributions
   - Creates multiple dataset sizes (100-1000 prompts)
   - Output: JSONL files with prompts and metadata

3. **benchmark_runner.py** - Runs inference benchmarks
   - Sends requests to vLLM server
   - Collects detailed metrics (throughput, latency, success rate)
   - Output: JSON results files

4. **generate_sbatch.py** - Generates SLURM scripts
   - Creates sbatch files for multi-node vLLM deployment
   - Handles Ray cluster setup, NCCL configuration
   - Output: .sbatch files and launch script

5. **aggregate_results.py** - Analyzes results
   - Aggregates all experiment results
   - Compares parameter impacts
   - Output: CSV, text report, visualization data

### Helper Scripts
6. **setup_experiments.sh** - One-command setup
   - Runs entire setup pipeline
   - Creates datasets → configs → sbatch scripts
   - Usage: `./setup_experiments.sh [phase] [dataset] [request_rate]`

### Documentation
7. **README.md** - Complete documentation
8. **QUICKSTART.md** - 5-minute quick start guide
9. **requirements.txt** - Python dependencies

## Workflow

```
┌─────────────────────┐
│  1. Generate        │
│     Datasets        │
│  (dataset_loader)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Generate        │
│     Configs         │
│  (config_generator) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Generate        │
│     SLURM Scripts   │
│  (generate_sbatch)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Launch          │
│     Experiments     │
│  (sbatch)           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. Run Benchmark   │
│     (automatic in   │
│      sbatch job)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  6. Aggregate       │
│     Results         │
│  (aggregate_results)│
└─────────────────────┘
```

## Parameter Sweeps

### Phase 1: Memory & Batch (12 configs)
- `gpu_memory_utilization`: [0.90, 0.92, 0.95]
- `max_num_batched_tokens`: [4096, 8192, 16384, 32768]
- `max_num_seqs`: [64, 128, 256, 512]

### Phase 2: Quantization (4 configs)
- `kv_cache_dtype`: ["auto", "fp8_e5m2"]
- `enable_prefix_caching`: [False, True]

### Phase 3: Expert Parallelism (18 configs)
- `all2all_backend`: ["pplx", "deepep_low_latency", "deepep_high_throughput"]
- `enable_eplb`: [False, True]
- `eplb_num_redundant_experts`: [16, 32]
- `eplb_window_size`: [500, 1000, 2000]

### Phase 4: Chunked Prefill (6 configs)
- `enable_chunked_prefill`: [False, True]
- `max_num_partial_prefills`: [1, 2, 4]

### Phase 5: Scheduler (3 configs)
- `num_scheduler_steps`: [1, 2, 4]

**Total**: ~50+ unique configurations per complete sweep

## Datasets

- **quick_test.jsonl** (100 prompts) - Fast validation
- **medium_balanced.jsonl** (500 prompts) - Standard benchmarks
- **large_throughput.jsonl** (1000 prompts) - Throughput testing
- **short_latency.jsonl** (500 prompts) - Latency testing
- **long_context.jsonl** (300 prompts) - Context testing

## Metrics Collected

### Primary Metrics
- **Throughput**: tokens/sec, requests/sec
- **Latency**: TTFT (P50/P95/P99), ITL (P50/P95/P99), E2E (P50/P95/P99)
- **Reliability**: Success rate, failed requests

### System Metrics (from Prometheus)
- Preemptions count
- Cache usage percentage
- Requests running/waiting

## Quick Start Commands

### Fastest Path to Results
```bash
cd /scratch/10000/eguha3/ot-agent/data/vllm_experiments

# Setup and launch quick test
./setup_experiments.sh quick
./launch_all_experiments.sh

# Wait ~30 minutes, then analyze
python aggregate_results.py
cat results/experiment_report.txt
```

### Production Workflow
```bash
# 1. Memory sweep to find optimal batch settings
./setup_experiments.sh memory_batch
./launch_all_experiments.sh

# 2. Wait for completion, analyze
python aggregate_results.py --enrich-configs --compare-param max_num_batched_tokens

# 3. Expert parallelism with best memory settings
# (manually update BASE_CONFIG in config_generator.py with best values)
./setup_experiments.sh expert_parallel
./launch_all_experiments.sh

# 4. Final analysis
python aggregate_results.py --enrich-configs
```

## Expected Performance

### Target Metrics (64 H200 GPUs)
- **Throughput**: 1000-2500 tokens/sec
- **TTFT P95**: 500-2000ms
- **ITL P95**: 50-100ms
- **Success Rate**: >99%

### Optimal Settings (typical)
- `gpu_memory_utilization`: 0.92-0.95
- `max_num_batched_tokens`: 16384-32768
- `max_num_seqs`: 256-512
- `kv_cache_dtype`: "fp8_e5m2" (2x capacity)
- `all2all_backend`: "pplx" (single-node) or "deepep_low_latency" (multi-node)
- `enable_eplb`: True
- `enable_chunked_prefill`: True

## Cost Estimation

### Time per Experiment
- Quick test: ~30 min
- Medium benchmark: ~60 min
- Large benchmark: ~90 min

### Full Sweep Cost
- Quick validation: 1 job × 0.5 hr = 0.5 node-hours
- Memory sweep: 12 jobs × 1 hr = 12 node-hours
- Expert parallel: 18 jobs × 1 hr = 18 node-hours
- All phases: ~50 jobs × 1 hr = ~50 node-hours

**Recommendation**: Start with quick test, then memory_batch (highest impact).

## Customization

### Change Model Path
Edit `config_generator.py`:
```python
BASE_CONFIG = {
    "model_path": "/path/to/your/GLM-4.6",  # Update this
    # ...
}
```

### Reduce Sweep Size
Edit parameter ranges in `config_generator.py`:
```python
PARAMETER_SWEEPS = {
    "memory_batch": {
        "max_num_batched_tokens": [8192, 16384],  # Reduced from [4096, 8192, 16384, 32768]
        # ...
    }
}
```

### Custom Dataset
```bash
python dataset_loader.py --num-prompts 200 --max-output-tokens 256
python generate_sbatch.py --dataset datasets/custom_200.jsonl
```

## Output Files

### Generated During Setup
- `configs/*.json` - Experiment configurations
- `datasets/*.jsonl` - Benchmark prompts
- `run_*.sbatch` - SLURM job scripts
- `launch_all_experiments.sh` - Master launch script

### Generated During Experiments
- `logs/vllm_*.out` - Job output logs
- `results/*/results_*.json` - Individual experiment results

### Generated During Analysis
- `results/experiment_results.csv` - All results in tabular format
- `results/experiment_report.txt` - Summary report
- `results/viz_data.json` - Visualization data

## Troubleshooting

### Before Launching
1. Update model path in `config_generator.py`
2. Verify conda environment in sbatch scripts
3. Check SLURM partition (-p gh for multi-node)
4. Ensure model weights accessible from all nodes

### During Experiments
- Monitor: `squeue -u $USER`
- Check logs: `tail -f logs/vllm_*.out`
- Look for "vLLM server is ready!" in logs

### After Experiments
- Failed jobs: Check logs for errors
- OOM: Reduce `gpu_memory_utilization`
- Low throughput: Increase `max_num_batched_tokens`
- High latency: Decrease `max_num_batched_tokens`

## Key Features

✅ **Complete automation**: One command to setup and launch
✅ **Parallel execution**: Run multiple experiments via SLURM
✅ **Comprehensive metrics**: Throughput, latency, system health
✅ **Multi-node support**: 8 nodes × 8 GPUs = 64 GPUs
✅ **Cost-effective**: Configurable sweep sizes
✅ **Analysis tools**: CSV export, reports, comparisons
✅ **Production-ready**: Based on vLLM best practices

## Next Steps

1. **Validate**: Run quick test to verify setup
2. **Optimize**: Run memory_batch to find best batch settings
3. **Refine**: Run expert_parallel with optimal memory config
4. **Deploy**: Use best config for production serving

## Support

- Full docs: [README.md](README.md)
- Quick start: [QUICKSTART.md](QUICKSTART.md)
- Example configs: `configs/quick_test.json`
- Example output: Check `logs/` after first run

---

**Created**: 2025-11-22
**Location**: `/scratch/10000/eguha3/ot-agent/data/vllm_experiments/`
**Target**: 8-node H200 cluster (64 GPUs)
**Model**: GLM-4.6 (or any MoE model)
