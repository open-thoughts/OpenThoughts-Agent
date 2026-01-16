# Quick Start Guide

Get your vLLM experiments running in 5 minutes.

## Prerequisites

- Access to 8-node H200 cluster
- SLURM access
- Model weights for GLM-4.6 (update path in configs)

## Step-by-Step

### 1. One-Command Setup (Recommended)

```bash
cd ${DCAGENT_DIR}/data/vllm_experiments

# Quick test (single experiment)
./setup_experiments.sh quick

# Full memory sweep (12 experiments)
./setup_experiments.sh memory_batch

# All experiments (~100+ experiments - expensive!)
./setup_experiments.sh all
```

### 2. Launch Experiments

```bash
# Launch all prepared experiments
./launch_all_experiments.sh

# Or launch single experiment
sbatch run_<exp_id>_head.sbatch
```

### 3. Monitor Progress

```bash
# Check job queue
squeue -u $USER

# Watch logs in real-time
tail -f logs/vllm_*.out

# Check if vLLM server is ready
# (look for "vLLM server is ready!" in logs)
```

### 4. View Results

```bash
# After experiments complete
python aggregate_results.py --enrich-configs

# View report
cat results/experiment_report.txt

# Or analyze CSV
python -c "import pandas as pd; df = pd.read_csv('results/experiment_results.csv'); print(df.nlargest(5, 'throughput_tokens_per_sec'))"
```

## Manual Setup (Alternative)

If you prefer step-by-step control:

```bash
# 1. Generate datasets
python dataset_loader.py --create-suite

# 2. Generate configs
python config_generator.py --phase memory_batch

# 3. Generate sbatch scripts
python generate_sbatch.py \
  --config-dir configs/memory_batch \
  --dataset datasets/medium_balanced.jsonl \
  --request-rate 10.0

# 4. Launch
./launch_all_experiments.sh

# 5. Analyze
python aggregate_results.py --enrich-configs
```

## Cost Management

**Recommended workflow for cost-effectiveness:**

```bash
# Phase 1: Validate setup (1 job, ~30 min)
./setup_experiments.sh quick
./launch_all_experiments.sh
# Wait and verify results

# Phase 2: Memory sweep (12 jobs, ~6 hours)
./setup_experiments.sh memory_batch
./launch_all_experiments.sh
# Analyze results, identify best configs

# Phase 3: Expert parallelism (based on best from Phase 2)
# Manually edit configs to use best memory settings
./setup_experiments.sh expert_parallel
./launch_all_experiments.sh
```

## Configuration

### Model Path

Update in `config_generator.py`:
```python
BASE_CONFIG = {
    "model_path": "/path/to/GLM-4.6",  # Change this
    # ...
}
```

### Dataset Size

Edit dataset generation:
```bash
# Smaller dataset (faster, cheaper)
python dataset_loader.py --num-prompts 100 --max-output-tokens 256

# Then use it:
python generate_sbatch.py --dataset datasets/custom_100.jsonl
```

### Request Rate

Lower request rate for more stable latency measurements:
```bash
./setup_experiments.sh memory_batch medium_balanced.jsonl 5.0
```

## Troubleshooting

### "Model not found"
Update `model_path` in `config_generator.py`

### "Ray failed to start"
Increase sleep time in sbatch scripts (line ~40)

### "OOM errors"
Reduce `gpu_memory_utilization` in configs (0.95 → 0.90)

### "Jobs pending forever"
Check partition: `squeue` - may need to change `-p gh` in sbatch scripts

## What Each File Does

- `config_generator.py` - Creates experiment configs (JSON)
- `dataset_loader.py` - Creates benchmark prompts (JSONL)
- `generate_sbatch.py` - Creates SLURM job scripts (sbatch)
- `benchmark_runner.py` - Runs inference and measures perf
- `aggregate_results.py` - Analyzes results
- `setup_experiments.sh` - One-command setup

## Expected Results

After running memory_batch phase:
- 12 experiments × 30-60 min each
- Results show optimal `max_num_batched_tokens` and `max_num_seqs`
- Best config typically: 16K-32K batched tokens, 256-512 seqs
- Expected throughput: 1000-2000 tok/s on 64 GPUs

## Next Steps

1. Run quick test to validate setup
2. Run memory_batch to find optimal batch configs
3. Run expert_parallel with best batch settings
4. Analyze all results to find best overall config
5. Use best config for production deployment

## Getting Help

Check full docs: [README.md](README.md)

View example outputs: `cat logs/vllm_*.out`

Debug mode: Edit sbatch to add `export NCCL_DEBUG=TRACE`
