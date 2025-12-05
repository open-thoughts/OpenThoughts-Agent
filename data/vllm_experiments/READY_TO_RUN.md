# ✅ Ready to Run - All Scripts Generated!

## What's Ready

✅ **16 experiment configurations** testing different batch sizes, FP8, expert parallelism
✅ **16 SLURM batch scripts** for 8-node (64 GPU) deployment
✅ **5 benchmark datasets** with controlled prompt distributions
✅ **Automated metrics collection** for throughput, latency, and system health

## Metrics That Will Be Collected

Each experiment automatically measures:

### 🚀 Speed Metrics (Primary)
- **throughput_tokens_per_sec** - Tokens generated per second (target: 1000-2500)
- **throughput_requests_per_sec** - Requests completed per second
- **total_tokens_generated** - Total output tokens

### ⏱️ Latency Metrics
- **ttft_p50/p95/p99** - Time to First Token in ms (target P95: <2000ms)
- **itl_p50/p95/p99** - Inter-Token Latency in ms (target P95: <100ms)
- **e2e_latency_p50/p95/p99** - End-to-End latency in ms

### 📊 Reliability Metrics
- **success_rate** - % successful requests (target: >99%)
- **failed_requests** - Count of failures
- **prom_preemptions** - Preempted requests (target: 0)

### 💾 System Metrics
- **prom_cache_usage** - GPU cache utilization %
- **prom_requests_running** - Active concurrent requests
- **prom_requests_waiting** - Queued requests

## Quick Start - Run One Experiment

```bash
cd /scratch/10000/eguha3/ot-agent/data/vllm_experiments

# ⚠️ IMPORTANT: Update model path first!
# Edit configs/focused/config_00_baseline_conservative_9b6f4b39.json
# Change "model_path": "THUDM/glm-4-9b-chat" to your GLM-4.6 path

# Launch baseline experiment
sbatch run_9b6f4b39_head.sbatch

# Monitor job
squeue -u $USER
tail -f logs/vllm_9b6f4b39_*.out

# Look for this in logs:
# "vLLM server is ready!"
# "Running benchmark..."
# "Benchmark complete!"

# Check results (after ~60 min)
ls results/9b6f4b39/
python -c "import json; print(json.dumps(json.load(open('results/9b6f4b39/results_9b6f4b39_*.json')), indent=2))" | head -50
```

## Launch All Experiments

```bash
# Launch all 16 experiments
./launch_all.sh

# They'll run in parallel if you have quota for 16 × 8 nodes = 128 nodes
# Otherwise SLURM will queue them and run sequentially

# Monitor
watch -n 10 'squeue -u $USER | head -20'
```

## After Completion

```bash
# Aggregate all results
python aggregate_results.py --enrich-configs

# View comparison report
cat results/experiment_report.txt

# Find best config
python -c "
import pandas as pd
df = pd.read_csv('results/experiment_results.csv')
best = df.nlargest(5, 'throughput_tokens_per_sec')
print('\n=== Top 5 Fastest Configs ===\n')
print(best[['config_hash', 'throughput_tokens_per_sec', 'ttft_p95', 'success_rate']].to_string())
"

# Compare batch sizes
python aggregate_results.py --enrich-configs --compare-param max_num_batched_tokens --metric throughput_tokens_per_sec
```

## Experiment Configurations

| # | Config | Batch Tokens | Max Seqs | KV Cache | What It Tests |
|---|--------|--------------|----------|----------|---------------|
| 1 | 9b6f4b39 | 8,192 | 128 | BF16 | Baseline (conservative) |
| 2-10 | 6cd2bdd3-27aa356a | 8K-32K | 128-512 | BF16 | **Batch size sweep** |
| 11-12 | e07256a0-2594871d | 16K-32K | 512 | **FP8** | **2x memory capacity** |
| 13-14 | 8051b7d5-6244069d | 16K | 256 | BF16 | **EP backends** |
| 15 | 1c91ae1c | 16K | 256 | BF16 | **EPLB balancing** |
| 16 | d8eabc0d | 32K | 512 | FP8 | **Max throughput** |

**Most likely winner**: Config 5ee123b8 (16K batch, 256 seqs) or d8eabc0d (aggressive)

## Files Generated

```
vllm_experiments/
├── Configs (16)
│   └── configs/focused/config_*.json
├── Datasets (5)
│   └── datasets/*.jsonl (100-1000 prompts each)
├── Batch Scripts (16)
│   └── run_*_head.sbatch
├── Launch Script
│   └── launch_all.sh
└── Documentation
    ├── README.md (complete guide)
    ├── QUICKSTART.md (5-min guide)
    ├── EXPERIMENT_MANIFEST.md (config details)
    └── READY_TO_RUN.md (this file)
```

## Expected Runtime

- **Single experiment**: ~60 minutes on 8 nodes
- **All 16 experiments**:
  - Parallel: ~60 minutes (if 128 nodes available)
  - Sequential: ~16 hours (if only 8 nodes available)

## Expected Results

### Throughput Comparison
```
Config               Batch    Expected tok/s
------------------   ------   --------------
baseline             8K       800-1000
batch_16K_256        16K      1500-2000 ⭐
batch_32K_512        32K      1800-2200
fp8_32K              32K      2000-2500
aggressive           32K      2200-2800 (if no OOM)
```

### Latency Tradeoffs
- Lower batch = lower latency, lower throughput
- Higher batch = higher latency, higher throughput
- FP8 = similar latency, higher capacity
- EPLB = better throughput if experts imbalanced

## Output Format

Each experiment produces a JSON file with all metrics:

```json
{
  "config_hash": "9b6f4b39",
  "num_requests": 500,
  "successful_requests": 498,
  "success_rate": 0.996,

  "throughput_tokens_per_sec": 1234.56,
  "throughput_requests_per_sec": 2.41,

  "ttft_mean": 0.856,
  "ttft_p50": 0.812,
  "ttft_p95": 1.234,
  "ttft_p99": 1.567,

  "itl_mean": 0.045,
  "itl_p50": 0.042,
  "itl_p95": 0.089,
  "itl_p99": 0.123,

  "prometheus_metrics": {
    "num_preemptions_total": 0,
    "gpu_cache_usage_perc": 78.5
  }
}
```

## Troubleshooting

### Before Running
- [ ] Update model path in all configs
- [ ] Check conda environment path in sbatch scripts
- [ ] Verify you have access to 8 nodes on partition 'gh'
- [ ] Check model weights accessible from all nodes

### During Run
- **Job fails immediately**: Check `logs/vllm_*.out` for errors
- **OOM errors**: Config too aggressive, reduce `gpu_memory_utilization`
- **Ray connection failed**: Network issue, check InfiniBand
- **Benchmark not starting**: vLLM failed to initialize, check model path

### After Run
- **No results files**: Job failed, check logs
- **Low throughput**: May need larger batch size or more seqs
- **High preemptions**: Reduce `gpu_memory_utilization`
- **Request failures**: Model or network issues

## Next Steps

1. **Test one experiment first**: `sbatch run_9b6f4b39_head.sbatch`
2. **Verify it completes successfully**: Check logs and results
3. **Launch all experiments**: `./launch_all.sh`
4. **Analyze results**: `python aggregate_results.py --enrich-configs`
5. **Identify best config for your workload**
6. **Deploy winner to production**

## Cost Estimate

- **Test run (1 config)**: 8 nodes × 1 hour = 8 node-hours
- **Full sweep (16 configs)**: 8 nodes × 16 hours = 128 node-hours (sequential)
- **Parallel (if quota allows)**: 128 nodes × 1 hour = 128 node-hours

---

**Status**: ✅ Ready to launch
**Location**: `/scratch/10000/eguha3/ot-agent/data/vllm_experiments/`
**Command**: `sbatch run_9b6f4b39_head.sbatch` (single) or `./launch_all.sh` (all)
