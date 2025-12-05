#!/usr/bin/env python3
"""
Configuration generator for vLLM parameter sweep experiments on GLM-4.6.

This script generates experimental configurations for ablating critical vLLM parameters
across 8 nodes (64 H200 GPUs total).
"""

import json
import itertools
from pathlib import Path
from typing import Dict, List, Any
import hashlib


class ExperimentConfigGenerator:
    """Generate experiment configurations for vLLM parameter sweeps."""

    # Base configuration for 8-node setup (8 nodes × 8 GPUs = 64 GPUs)
    BASE_CONFIG = {
        "model_path": "zai-org/GLM-4.6",
        "num_nodes": 8,
        "gpus_per_node": 8,
        "tensor_parallel_size": 1,  # Recommended for MoE
        "data_parallel_size": 64,    # Total GPUs
        "data_parallel_size_local": 8,  # GPUs per node
        "enable_expert_parallel": True,
        "distributed_executor_backend": "ray",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_seq_len_to_capture": 8192,
        "block_size": 16,
        "swap_space_gb": 20,
    }

    # Parameter sweep configurations
    PARAMETER_SWEEPS = {
        # Phase 1: Memory & Batch Configuration (CRITICAL)
        "memory_batch": {
            "gpu_memory_utilization": [0.90, 0.92, 0.95],
            "max_num_batched_tokens": [4096, 8192, 16384, 32768],
            "max_num_seqs": [64, 128, 256, 512],
            "max_model_len": [16384, 32768],
        },

        # Phase 2: Quantization & Caching (HIGH IMPACT)
        "quantization_caching": {
            "kv_cache_dtype": ["auto", "fp8_e5m2"],
            "enable_prefix_caching": [False, True],
        },

        # Phase 3: Expert Parallelism (MoE CRITICAL)
        "expert_parallel": {
            "all2all_backend": ["pplx", "deepep_low_latency", "deepep_high_throughput"],
            "enable_eplb": [False, True],
            "eplb_num_redundant_experts": [16, 32],
            "eplb_window_size": [500, 1000, 2000],
            "eplb_step_interval": [1500, 3000, 5000],
        },

        # Phase 4: Chunked Prefill (LATENCY OPTIMIZATION)
        "chunked_prefill": {
            "enable_chunked_prefill": [False, True],
            "max_num_partial_prefills": [1, 2, 4],
        },

        # Phase 5: Scheduler Configuration
        "scheduler": {
            "num_scheduler_steps": [1, 2, 4],
        },
    }

    # Quick test configuration (single run for validation)
    QUICK_TEST = {
        "gpu_memory_utilization": 0.90,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 128,
        "max_model_len": 16384,
        "kv_cache_dtype": "auto",
        "enable_prefix_caching": False,
        "all2all_backend": "pplx",
        "enable_eplb": False,
        "enable_chunked_prefill": True,
        "max_num_partial_prefills": 1,
        "num_scheduler_steps": 2,
    }

    def __init__(self, output_dir: str = "configs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate a unique hash for a configuration."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def build_eplb_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build EPLB configuration JSON."""
        if not config.get("enable_eplb", False):
            return {}

        return {
            "window_size": config.get("eplb_window_size", 1000),
            "step_interval": config.get("eplb_step_interval", 3000),
            "num_redundant_experts": config.get("eplb_num_redundant_experts", 32),
            "log_balancedness": True,
        }

    def create_vllm_command(self, config: Dict[str, Any], node_id: int = 0) -> str:
        """Generate vLLM serve command from configuration."""
        cmd_parts = ["vllm serve", config["model_path"]]

        # Distributed configuration
        if node_id == 0:
            # Head node
            cmd_parts.extend([
                f"--tensor-parallel-size {config['tensor_parallel_size']}",
                f"--data-parallel-size {config['data_parallel_size']}",
                f"--data-parallel-size-local {config['data_parallel_size_local']}",
                "--enable-expert-parallel" if config.get("enable_expert_parallel") else "",
                f"--distributed-executor-backend {config['distributed_executor_backend']}",
                "--api-server-count 8",
                "--port 8000",
                "--host 0.0.0.0",
            ])
        else:
            # Worker nodes (headless)
            cmd_parts.extend([
                "--headless",
                f"--tensor-parallel-size {config['tensor_parallel_size']}",
                f"--data-parallel-size {config['data_parallel_size']}",
                f"--data-parallel-size-local {config['data_parallel_size_local']}",
                f"--data-parallel-start-rank {node_id * config['data_parallel_size_local']}",
                "--enable-expert-parallel" if config.get("enable_expert_parallel") else "",
                f"--data-parallel-address $HEAD_NODE_IP",
                "--data-parallel-rpc-port 13345",
                f"--distributed-executor-backend {config['distributed_executor_backend']}",
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
            eplb_config = self.build_eplb_config(config)
            eplb_json = json.dumps(eplb_config).replace('"', '\\"')
            cmd_parts.append(f'--enable-eplb --eplb-config "{eplb_json}"')

        # Prefix caching
        if config.get("enable_prefix_caching", False):
            cmd_parts.append("--enable-prefix-caching")

        # Performance
        cmd_parts.extend([
            f"--max-seq-len-to-capture {config.get('max_seq_len_to_capture', 8192)}",
            f"--dtype {config.get('dtype', 'bfloat16')}",
            "--trust-remote-code" if config.get("trust_remote_code") else "",
        ])

        # Monitoring
        if node_id == 0:
            cmd_parts.extend([
                "--disable-log-stats false",
                "--prometheus-port 8001",
            ])

        # Filter empty strings and join
        cmd_parts = [p for p in cmd_parts if p]
        return " \\\n  ".join(cmd_parts)

    def generate_phase_configs(self, phase: str, baseline: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate configurations for a specific phase."""
        if baseline is None:
            baseline = {**self.BASE_CONFIG, **self.QUICK_TEST}

        if phase not in self.PARAMETER_SWEEPS:
            raise ValueError(f"Unknown phase: {phase}. Available: {list(self.PARAMETER_SWEEPS.keys())}")

        phase_params = self.PARAMETER_SWEEPS[phase]

        # Generate all combinations for this phase
        param_names = list(phase_params.keys())
        param_values = [phase_params[name] for name in param_names]

        configs = []
        for values in itertools.product(*param_values):
            config = baseline.copy()
            for name, value in zip(param_names, values):
                config[name] = value
            configs.append(config)

        return configs

    def generate_focused_sweep(self, focus_params: Dict[str, List[Any]], baseline: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate configurations for a focused parameter sweep."""
        if baseline is None:
            baseline = {**self.BASE_CONFIG, **self.QUICK_TEST}

        param_names = list(focus_params.keys())
        param_values = [focus_params[name] for name in param_names]

        configs = []
        for values in itertools.product(*param_values):
            config = baseline.copy()
            for name, value in zip(param_names, values):
                config[name] = value
            configs.append(config)

        return configs

    def generate_all_phases(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate configurations for all phases."""
        all_configs = {}
        baseline = {**self.BASE_CONFIG, **self.QUICK_TEST}

        for phase_name in self.PARAMETER_SWEEPS.keys():
            all_configs[phase_name] = self.generate_phase_configs(phase_name, baseline)

        return all_configs

    def save_config(self, config: Dict[str, Any], filename: str = None):
        """Save a single configuration to JSON file."""
        if filename is None:
            config_hash = self.generate_config_hash(config)
            filename = f"config_{config_hash}.json"

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        return output_path

    def save_phase_configs(self, phase: str, configs: List[Dict[str, Any]]):
        """Save all configurations for a phase."""
        phase_dir = self.output_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, config in enumerate(configs):
            config_hash = self.generate_config_hash(config)
            filename = f"{phase}_config_{i:03d}_{config_hash}.json"
            output_path = phase_dir / filename

            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)

            saved_paths.append(output_path)

        # Create a manifest
        manifest = {
            "phase": phase,
            "num_configs": len(configs),
            "configs": [str(p.relative_to(self.output_dir)) for p in saved_paths],
        }

        manifest_path = phase_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return saved_paths

    def generate_quick_test_config(self):
        """Generate a quick test configuration."""
        config = {**self.BASE_CONFIG, **self.QUICK_TEST}
        return self.save_config(config, "quick_test.json")


def main():
    """Generate experiment configurations."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate vLLM experiment configurations")
    parser.add_argument("--output-dir", default="configs", help="Output directory for configs")
    parser.add_argument("--phase", choices=["all", "memory_batch", "quantization_caching",
                                           "expert_parallel", "chunked_prefill", "scheduler"],
                       default="all", help="Which phase to generate configs for")
    parser.add_argument("--quick-test", action="store_true", help="Generate quick test config only")

    args = parser.parse_args()

    generator = ExperimentConfigGenerator(args.output_dir)

    if args.quick_test:
        config_path = generator.generate_quick_test_config()
        print(f"Generated quick test config: {config_path}")
        return

    if args.phase == "all":
        all_configs = generator.generate_all_phases()
        for phase_name, configs in all_configs.items():
            print(f"Generating {len(configs)} configs for phase: {phase_name}")
            saved_paths = generator.save_phase_configs(phase_name, configs)
            print(f"  Saved {len(saved_paths)} configs to {generator.output_dir / phase_name}")
    else:
        configs = generator.generate_phase_configs(args.phase)
        print(f"Generating {len(configs)} configs for phase: {args.phase}")
        saved_paths = generator.save_phase_configs(args.phase, configs)
        print(f"  Saved {len(saved_paths)} configs to {generator.output_dir / args.phase}")

    print(f"\nTotal configs generated: {sum(len(list((generator.output_dir / p).glob('*.json'))) for p in generator.PARAMETER_SWEEPS.keys())}")


if __name__ == "__main__":
    main()
