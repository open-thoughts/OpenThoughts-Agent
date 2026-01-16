#!/usr/bin/env python3
"""
Local SFT training runner.

Launches SFT training jobs on local GPUs without SLURM. Designed for machines
where we have exclusive access to GPUs and don't need job scheduling.

Usage:
    python -m sft.local.run_sft \
        --train_config path/to/train_config.yaml \
        --job_name my_training_run \
        --gpus 8

    # Or with a specific launcher
    python -m sft.local.run_sft \
        --train_config path/to/config.yaml \
        --launcher accelerate \
        --gpus 4
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from hpc.launch_utils import PROJECT_ROOT


@dataclass
class LocalSFTConfig:
    """Configuration for local SFT training."""

    train_config_path: str
    job_name: str
    experiments_dir: str = "experiments"
    gpus: int = 1
    cpus: int = 0  # 0 = auto-detect
    launcher: str = "torchrun"  # "torchrun" or "accelerate"
    deepspeed_config: Optional[str] = None
    master_port: int = 29500
    dry_run: bool = False


class LocalSFTRunner:
    """Runs SFT training on local GPUs without SLURM.

    This is a lightweight wrapper that directly invokes torchrun or accelerate
    for distributed training on the local machine.
    """

    def __init__(self, config: LocalSFTConfig):
        self.config = config
        self._processes: List[subprocess.Popen] = []

    def setup(self) -> None:
        """Validate configuration and set up directories."""
        # Validate train config exists
        train_config = Path(self.config.train_config_path).expanduser().resolve()
        if not train_config.exists():
            raise FileNotFoundError(f"Training config not found: {train_config}")
        self.config.train_config_path = str(train_config)

        # Set up experiments directory
        experiments_dir = Path(self.config.experiments_dir).expanduser().resolve()
        experiments_dir.mkdir(parents=True, exist_ok=True)
        self.config.experiments_dir = str(experiments_dir)

        # Auto-detect CPUs if not specified
        if self.config.cpus <= 0:
            self.config.cpus = os.cpu_count() or 16

        # Validate launcher
        if self.config.launcher not in ("torchrun", "accelerate"):
            raise ValueError(f"Unknown launcher: {self.config.launcher}")

        # Set up signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(signum, _frame):
            print(f"\nSignal {signum} received; shutting down...", file=sys.stderr)
            self.cleanup()
            sys.exit(1)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def cleanup(self) -> None:
        """Clean up any running processes."""
        for proc in self._processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def print_banner(self) -> None:
        """Print startup banner."""
        print("=== Local SFT Training Runner ===")
        print(f"  Job Name: {self.config.job_name}")
        print(f"  Config: {self.config.train_config_path}")
        print(f"  GPUs: {self.config.gpus}")
        print(f"  Launcher: {self.config.launcher}")
        if self.config.deepspeed_config:
            print(f"  DeepSpeed: {self.config.deepspeed_config}")
        print("=================================")

    def run(self) -> int:
        """Execute the SFT training job.

        Returns:
            Exit code (0 for success)
        """
        self.print_banner()

        if self.config.dry_run:
            print("\n[DRY RUN] Would execute training with above configuration.")
            return 0

        # Set environment for local distributed training
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(self.config.master_port))

        if self.config.launcher == "torchrun":
            return self._run_torchrun()
        else:
            return self._run_accelerate()

    def _run_torchrun(self) -> int:
        """Run training using torchrun."""
        # Build torchrun command
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.config.gpus}",
            f"--master_port={self.config.master_port}",
        ]

        # Add DeepSpeed config if specified
        train_script = str(PROJECT_ROOT / "sft" / "llamafactory" / "src" / "train.py")
        cmd.append(train_script)
        cmd.append(self.config.train_config_path)

        print(f"\nRunning: {' '.join(cmd)}\n")
        sys.stdout.flush()

        proc = subprocess.Popen(cmd)
        self._processes.append(proc)
        return proc.wait()

    def _run_accelerate(self) -> int:
        """Run training using accelerate."""
        # Generate accelerate config if needed
        accelerate_config = self._generate_accelerate_config()

        cmd = [
            "python", "-u", "-m", "accelerate.commands.launch",
            f"--config_file={accelerate_config}",
            f"--main_process_port={self.config.master_port}",
            "--tee=3",
            str(PROJECT_ROOT / "sft" / "llamafactory" / "src" / "train.py"),
            self.config.train_config_path,
        ]

        print(f"\nRunning: {' '.join(cmd)}\n")
        sys.stdout.flush()

        proc = subprocess.Popen(cmd)
        self._processes.append(proc)
        return proc.wait()

    def _generate_accelerate_config(self) -> str:
        """Generate an accelerate config file for local training."""
        import yaml

        config_dir = Path(self.config.experiments_dir) / "accelerate_configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / f"{self.config.job_name}_accelerate.yaml"

        # Build accelerate config
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "MULTI_GPU",
            "downcast_bf16": "no",
            "machine_rank": 0,
            "main_training_function": "main",
            "mixed_precision": "bf16",
            "num_machines": 1,
            "num_processes": self.config.gpus,
            "rdzv_backend": "static",
            "same_network": True,
            "tpu_env": [],
            "tpu_use_cluster": False,
            "tpu_use_sudo": False,
            "use_cpu": False,
        }

        if self.config.deepspeed_config:
            config["distributed_type"] = "DEEPSPEED"
            config["deepspeed_config"] = {
                "deepspeed_config_file": self.config.deepspeed_config,
                "zero3_init_flag": True,
            }
        else:
            # Default to FSDP for multi-GPU
            if self.config.gpus > 1:
                config["distributed_type"] = "FSDP"
                config["fsdp_config"] = {
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_backward_prefetch": "BACKWARD_PRE",
                    "fsdp_cpu_ram_efficient_loading": True,
                    "fsdp_forward_prefetch": False,
                    "fsdp_offload_params": False,
                    "fsdp_sharding_strategy": "FULL_SHARD",
                    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                    "fsdp_sync_module_states": True,
                    "fsdp_use_orig_params": True,
                }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Generated accelerate config: {config_path}")
        return str(config_path)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for local SFT runner."""
    parser = argparse.ArgumentParser(
        description="Run SFT training on local GPUs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--train_config",
        required=True,
        help="Path to LLaMA-Factory training config YAML.",
    )
    parser.add_argument("--train-config", dest="train_config", help=argparse.SUPPRESS)

    parser.add_argument(
        "--job_name",
        required=True,
        help="Name for this training job.",
    )
    parser.add_argument("--job-name", dest="job_name", help=argparse.SUPPRESS)

    # Optional arguments
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use.",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=0,
        help="Number of CPUs (0 = auto-detect).",
    )

    parser.add_argument(
        "--launcher",
        choices=["torchrun", "accelerate"],
        default="torchrun",
        help="Distributed training launcher to use.",
    )

    parser.add_argument(
        "--deepspeed_config",
        help="Path to DeepSpeed config JSON (enables DeepSpeed).",
    )
    parser.add_argument("--deepspeed-config", dest="deepspeed_config", help=argparse.SUPPRESS)

    parser.add_argument(
        "--master_port",
        type=int,
        default=29500,
        help="Master port for distributed training.",
    )
    parser.add_argument("--master-port", dest="master_port", help=argparse.SUPPRESS)

    parser.add_argument(
        "--experiments_dir",
        default=str(PROJECT_ROOT / "experiments"),
        help="Directory for experiment outputs.",
    )
    parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configuration without running.",
    )
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help=argparse.SUPPRESS)

    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    config = LocalSFTConfig(
        train_config_path=args.train_config,
        job_name=args.job_name,
        experiments_dir=args.experiments_dir,
        gpus=args.gpus,
        cpus=args.cpus,
        launcher=args.launcher,
        deepspeed_config=args.deepspeed_config,
        master_port=args.master_port,
        dry_run=args.dry_run,
    )

    runner = LocalSFTRunner(config)
    runner.setup()
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
