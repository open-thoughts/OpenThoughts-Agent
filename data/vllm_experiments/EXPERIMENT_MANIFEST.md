# Experiment Manifest

## Overview
**16 experiments** ready to run for GLM-4.6 speed comparison across 8 nodes (64 H200 GPUs).

## Configurations

### 1. Baseline (Conservative)
- **Config**: `9b6f4b39` - baseline_conservative
- **Batch tokens**: 8,192
- **Max seqs**: 128
- **KV cache**: auto (BF16)
- **Purpose**: Conservative baseline for comparison

### 2-10. Batch Size Sweep (9 configs)
Tests impact of batch size and concurrent sequences on throughput:

| Config | Name | Batch Tokens | Max Seqs | Expected Impact |
|--------|------|--------------|----------|----------------|
| 6cd2bdd3 | batch_8192_seqs_128 | 8,192 | 128 | Low latency baseline |
| e2d27c22 | batch_8192_seqs_256 | 8,192 | 256 | Medium concurrency |
| ab1ac75a | batch_8192_seqs_512 | 8,192 | 512 | High concurrency |
| 7ad2141c | batch_16384_seqs_128 | 16,384 | 128 | Balanced batch |
| **5ee123b8** | **batch_16384_seqs_256** | **16,384** | **256** | **⭐ Likely optimal** |
| 8ec7ab0f | batch_16384_seqs_512 | 16,384 | 512 | High throughput |
| 6ac86abf | batch_32768_seqs_128 | 32,768 | 128 | Large batch, low concurrency |
| 485a5396 | batch_32768_seqs_256 | 32,768 | 256 | Large batch |
| 27aa356a | batch_32768_seqs_512 | 32,768 | 512 | Max batch |

### 11-12. FP8 Quantization (2 configs)
Tests 2x memory capacity with FP8 KV cache:

| Config | Name | Batch Tokens | Max Seqs | KV Cache | GPU Mem |
|--------|------|--------------|----------|----------|---------|
| e07256a0 | fp8_batch_16384 | 16,384 | 512 | **fp8_e5m2** | 0.95 |
| 2594871d | fp8_batch_32768 | 32,768 | 512 | **fp8_e5m2** | 0.95 |

**Expected**: ~2x increase in concurrent requests, minimal accuracy loss

### 13-14. Expert Parallelism Backends (2 configs)
Tests all-to-all communication strategies:

| Config | Name | Backend | Purpose |
|--------|------|---------|---------|
| 8051b7d5 | ep_pplx | pplx | Single-node optimized |
| 6244069d | ep_deepep_low_latency | deepep_low_latency | Multi-node decode |

**Expected**: PPLX often wins for balanced workloads, DeepEP better for pure decode

### 15. EPLB Load Balancing (1 config)
Tests expert load balancing:

| Config | Name | EPLB | Redundant Experts |
|--------|------|------|-------------------|
| 1c91ae1c | eplb_enabled | ✓ | 32 |

**Expected**: 10-30% improvement if expert usage is imbalanced

### 16. Aggressive Max Throughput (1 config)
Tests all optimizations combined:

| Config | Name | Batch | Seqs | KV | EPLB | Prefix Cache |
|--------|------|-------|------|----|----|--------------|
| d8eabc0d | aggressive_max_throughput | 32,768 | 512 | fp8 | ✓ | ✓ |

**Expected**: Highest throughput if no OOM

## Metrics Collected

Each experiment will measure:

### Throughput Metrics
✅ **throughput_tokens_per_sec** - Primary speed metric (target: 1000-2500 tok/s)
✅ **throughput_requests_per_sec** - Requests completed per second
✅ **total_tokens_generated** - Total tokens across all requests

### Latency Metrics (milliseconds)
✅ **ttft_p50/p95/p99** - Time to First Token (target P95: <2000ms)
✅ **itl_p50/p95/p99** - Inter-Token Latency (target P95: <100ms)
✅ **e2e_latency_p50/p95/p99** - End-to-End request latency

### Reliability Metrics
✅ **success_rate** - Percentage of successful requests (target: >99%)
✅ **failed_requests** - Count of failed requests
✅ **total_time** - Total benchmark duration

### System Metrics (from Prometheus)
✅ **prom_preemptions** - Request preemptions (target: 0)
✅ **prom_cache_usage** - GPU cache utilization %
✅ **prom_requests_running** - Concurrent requests
✅ **prom_requests_waiting** - Queued requests

## Dataset

**File**: `datasets/medium_balanced.jsonl`
- **Size**: 500 prompts
- **Distribution**: 30% short, 50% medium, 20% long
- **Max output**: 512 tokens per request
- **Request rate**: 10 requests/sec

## How to Run

### Run Single Experiment (Recommended First)
```bash
# Test with baseline config
sbatch run_9b6f4b39_head.sbatch

# Monitor
squeue -u $USER
tail -f logs/vllm_9b6f4b39_*.out

# After ~60 min, check results
ls results/9b6f4b39/
cat results/9b6f4b39/results_*.json | python -m json.tool | head -50
```

### Run All Experiments
```bash
# Launch all 16 experiments
./launch_all.sh

# Monitor progress
squeue -u $USER

# After completion (~16 hours if sequential)
python aggregate_results.py --enrich-configs

# View comparison report
cat results/experiment_report.txt

# View CSV for detailed analysis
python -c "import pandas as pd; df = pd.read_csv('results/experiment_results.csv'); print(df.sort_values('throughput_tokens_per_sec', ascending=False).head(10))"
```

## Expected Results

### Key Comparisons

1. **Batch Size Impact**
   - Expect throughput to increase with batch size up to a point
   - Latency will increase with larger batches
   - Optimal likely around 16K-32K batched tokens

2. **FP8 vs BF16**
   - FP8 should enable ~2x more concurrent requests
   - Throughput may be similar or slightly higher
   - Minimal accuracy degradation

3. **PPLX vs DeepEP**
   - PPLX often faster for mixed workloads
   - DeepEP may be better for pure decode
   - Network topology dependent

4. **EPLB Impact**
   - 10-30% improvement if experts are imbalanced
   - Negligible overhead if balanced

5. **Aggressive Config**
   - Should achieve highest throughput
   - May hit OOM or preemptions if pushed too far

### Performance Targets (64 H200 GPUs)

| Metric | Conservative | Optimal | Aggressive |
|--------|--------------|---------|------------|
| Throughput (tok/s) | 800-1200 | 1500-2000 | 2000-2500+ |
| TTFT P95 (ms) | <1500 | <2000 | <2500 |
| ITL P95 (ms) | <80 | <100 | <120 |
| Success Rate | >99% | >99% | >98% |

## Output Files

After each experiment completes:

```
results/<config_hash>/
└── results_<config_hash>_<timestamp>.json
```

Example result structure:
```json
{
  "config_hash": "9b6f4b39",
  "throughput_tokens_per_sec": 1234.56,
  "ttft_p95": 1.234,
  "itl_p95": 0.089,
  "success_rate": 0.998,
  "prometheus_metrics": {...}
}
```

## Troubleshooting

### Before Running
⚠️ **IMPORTANT**: Update model path in configs!
```bash
# Check current path
grep "model_path" configs/focused/config_00*.json

# Update if needed
# Edit generate_focused_configs.py line 13 and regenerate
```

### During Experiments
- **OOM errors**: Reduce gpu_memory_utilization (0.95 → 0.90)
- **Job fails immediately**: Check logs/vllm_*.out for errors
- **Low throughput**: Increase max_num_batched_tokens
- **High latency**: Decrease max_num_batched_tokens

### After Experiments
```bash
# Check completion status
ls results/*/results_*.json | wc -l  # Should be 16

# Check for failures
grep -l "Failed" results/*/results_*.json

# Re-run failed experiments
sbatch run_<failed_hash>_head.sbatch
```

## Cost Estimate

- **Single experiment**: ~60 minutes on 8 nodes = 8 node-hours
- **All 16 experiments**:
  - Sequential: ~16 hours × 8 nodes = 128 node-hours
  - Parallel (if resources available): ~1-2 hours × 8 nodes × 16 = 128 node-hours

## Next Steps After Results

1. **Identify best config**:
   - Highest throughput_tokens_per_sec
   - Acceptable latency (TTFT P95 < 2000ms)
   - High success rate (>99%)

2. **Validate best config**:
   - Run longer benchmark (1000+ prompts)
   - Test with real workload patterns
   - Check GPU utilization and memory

3. **Production deployment**:
   - Use best config for serving
   - Monitor metrics continuously
   - Fine-tune based on actual usage

---

**Created**: 2025-11-22
**Status**: Ready to run
**Estimated time**: 60 min per experiment
