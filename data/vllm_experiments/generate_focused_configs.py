#!/usr/bin/env python3
"""
Generate a focused set of configs for speed comparison with Ray Serve data parallelism.

Creates ~16 configs covering the most impactful parameters for vLLM performance.
These configs are designed for 8 nodes with 1 GPU each (GH200), using Ray Serve
replicas for data parallelism.

Key parameters to vary:
- gpu_memory_utilization (0.90 - 0.95)
- max_num_batched_tokens (8192, 16384, 32768)
- max_num_seqs (128, 256, 512)
- kv_cache_dtype (auto, fp8_e5m2)
- enable_expert_parallel (true/false)
- all2all_backend (pplx, deepep_low_latency)
- enable_eplb (true/false)
- enable_prefix_caching (true/false)

Fixed parameters:
- tensor_parallel_size: 1 (using Ray Serve replicas for data parallelism)
- num_replicas: 8 (one per node/GPU)
- model_path: zai-org/GLM-4.6
"""

import json
from pathlib import Path
import hashlib

# Base configuration for 8 nodes with 1 GPU each (Ray Serve data parallelism)
BASE_CONFIG = {
    "model_path": "zai-org/GLM-4.6",
    "num_nodes": 8,
    "gpus_per_node": 1,  # 1 GPU per node for GH200
    "tensor_parallel_size": 1,  # Fixed - don't vary!
    "num_replicas": 8,  # Data parallelism via Ray Serve replicas
    "enable_expert_parallel": True,
    "dtype": "bfloat16",
    "trust_remote_code": True,
    "block_size": 16,
    "swap_space_gb": 20,
    "enable_chunked_prefill": True,
    "max_num_partial_prefills": 1,
    "max_model_len": 32768,
}

# Focused parameter sweep for speed comparison
configs = []

# 1. Baseline (conservative)
configs.append({
    **BASE_CONFIG,
    "name": "baseline_conservative",
    "gpu_memory_utilization": 0.90,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 128,
    "kv_cache_dtype": "auto",
    "enable_prefix_caching": False,
    "all2all_backend": "pplx",
    "enable_eplb": False,
})

# 2. Batch size variations (CRITICAL for throughput)
for batch_tokens in [8192, 16384, 32768]:
    for num_seqs in [128, 256, 512]:
        # Skip baseline (already covered above)
        if batch_tokens == 8192 and num_seqs == 128:
            continue
        configs.append({
            **BASE_CONFIG,
            "name": f"batch_{batch_tokens}_seqs_{num_seqs}",
            "gpu_memory_utilization": 0.92,
            "max_num_batched_tokens": batch_tokens,
            "max_num_seqs": num_seqs,
            "kv_cache_dtype": "auto",
            "enable_prefix_caching": False,
            "all2all_backend": "pplx",
            "enable_eplb": False,
        })

# 3. FP8 variants (memory efficiency)
for batch_tokens in [16384, 32768]:
    configs.append({
        **BASE_CONFIG,
        "name": f"fp8_batch_{batch_tokens}",
        "gpu_memory_utilization": 0.95,
        "max_num_batched_tokens": batch_tokens,
        "max_num_seqs": 512,
        "kv_cache_dtype": "fp8_e5m2",
        "enable_prefix_caching": False,
        "all2all_backend": "pplx",
        "enable_eplb": False,
    })

# 4. Expert parallelism backends (MoE specific)
for backend in ["pplx", "deepep_low_latency"]:
    configs.append({
        **BASE_CONFIG,
        "name": f"ep_{backend}",
        "gpu_memory_utilization": 0.92,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 256,
        "kv_cache_dtype": "auto",
        "enable_prefix_caching": False,
        "all2all_backend": backend,
        "enable_eplb": False,
    })

# 5. EPLB enabled (load balancing for MoE)
configs.append({
    **BASE_CONFIG,
    "name": "eplb_enabled",
    "gpu_memory_utilization": 0.92,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 256,
    "kv_cache_dtype": "auto",
    "enable_prefix_caching": False,
    "all2all_backend": "pplx",
    "enable_eplb": True,
    "eplb_num_redundant_experts": 32,
    "eplb_window_size": 1000,
    "eplb_step_interval": 3000,
})

# 6. Prefix caching enabled
configs.append({
    **BASE_CONFIG,
    "name": "prefix_caching",
    "gpu_memory_utilization": 0.92,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 256,
    "kv_cache_dtype": "auto",
    "enable_prefix_caching": True,
    "all2all_backend": "pplx",
    "enable_eplb": False,
})

# 7. Aggressive configuration (max throughput)
configs.append({
    **BASE_CONFIG,
    "name": "aggressive_max_throughput",
    "gpu_memory_utilization": 0.95,
    "max_num_batched_tokens": 32768,
    "max_num_seqs": 512,
    "kv_cache_dtype": "fp8_e5m2",
    "enable_prefix_caching": True,
    "all2all_backend": "pplx",
    "enable_eplb": True,
    "eplb_num_redundant_experts": 32,
    "eplb_window_size": 1000,
    "eplb_step_interval": 3000,
})

# Save configs
output_dir = Path("configs/focused")
output_dir.mkdir(parents=True, exist_ok=True)

saved_configs = []
for i, config in enumerate(configs):
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

    filename = f"config_{i:02d}_{config.get('name', 'unnamed')}_{config_hash}.json"
    filepath = output_dir / filename

    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

    saved_configs.append(filepath)

    # Print summary
    kv_dtype = config['kv_cache_dtype']
    backend = config.get('all2all_backend', 'N/A')
    eplb = "Yes" if config.get('enable_eplb', False) else "No"
    prefix = "Yes" if config.get('enable_prefix_caching', False) else "No"

    print(f"{i+1:2d}. {config.get('name', 'unnamed'):30s} "
          f"batch={config['max_num_batched_tokens']:5d} "
          f"seqs={config['max_num_seqs']:3d} "
          f"kv={kv_dtype:8s} "
          f"backend={backend:20s} "
          f"eplb={eplb:3s} "
          f"prefix={prefix:3s}")

print(f"\nTotal configs created: {len(configs)}")
print(f"Output directory: {output_dir}")
print(f"\nThese configs test (with tensor_parallel_size=1, using Ray Serve replicas):")
print("  - Batch size impact on throughput (8 configs)")
print("  - FP8 KV cache quantization (2 configs)")
print("  - Expert parallelism backends (2 configs)")
print("  - EPLB load balancing (1 config)")
print("  - Prefix caching (1 config)")
print("  - Aggressive max throughput (1 config)")
print("  - Baseline for comparison (1 config)")
print(f"\nTo generate sbatch scripts:")
print(f"  python3 generate_sbatch_ray_serve.py")
