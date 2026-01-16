#!/usr/bin/env python3
"""
Local RL training runner.

Launches SkyRL reinforcement learning jobs on local GPUs without SLURM.
Designed for machines where we have exclusive access to GPUs and don't need
job scheduling.

Usage:
    python -m rl.local.run_rl \
        --rl_config terminal_bench.yaml \
        --train_data '["penfever/my-dataset"]' \
        --model_path Qwen/Qwen3-8B \
        --job_name my_rl_run \
        --gpus 4

    # Dry run to preview command
    python -m rl.local.run_rl \
        --rl_config terminal_bench.yaml \
        --train_data '["dataset"]' \
        --model_path Qwen/Qwen3-8B \
        --dry_run

Requirements:
    - RL environment must be set up (./hpc/setup_rl_env.sh)
    - SkyRL must be cloned ($SKYRL_HOME or ./SkyRL)
    - Ray must be available for distributed training
"""

from __future__ import annotations

import argparse
import ast
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from hpc.launch_utils import PROJECT_ROOT as HPC_PROJECT_ROOT
from hpc.rl_config_utils import (
    parse_rl_config,
    build_skyrl_hydra_args,
    get_skyrl_command_preview,
)
from hpc.rl_launch_utils import (
    check_rl_environment,
    resolve_rl_train_data,
    compute_num_inference_engines,
    derive_skyrl_export_path,
)


@dataclass
class LocalRLConfig:
    """Configuration for local RL training."""

    rl_config_path: str
    job_name: str
    model_path: str
    train_data: List[str] = field(default_factory=list)
    val_data: List[str] = field(default_factory=list)
    experiments_dir: str = "experiments"
    gpus: int = 4
    cpus: int = 0  # 0 = auto-detect
    ray_port: int = 6379
    master_port: int = 12345
    skyrl_overrides: List[str] = field(default_factory=list)
    dry_run: bool = False
    # Auto-derived
    tensor_parallel_size: int = 1


class LocalRLRunner:
    """Runs RL training on local GPUs without SLURM.

    This is a lightweight wrapper that:
    1. Starts a local Ray cluster
    2. Configures the environment for SkyRL
    3. Executes SkyRL training
    """

    def __init__(self, config: LocalRLConfig):
        self.config = config
        self._processes: List[subprocess.Popen] = []
        self._ray_started = False

    def setup(self) -> None:
        """Validate configuration and set up directories."""
        # Check RL environment
        rl_env = check_rl_environment()
        if rl_env:
            print(f"RL environment found: {rl_env}")
            self.rl_env_path = rl_env
        else:
            print("\n" + "=" * 60)
            print("WARNING: RL environment not found!")
            print("The RL environment is required for SkyRL training.")
            print("Create it with: ./hpc/setup_rl_env.sh")
            print("Or set DCFT_RL_ENV to point to an existing environment.")
            print("=" * 60 + "\n")
            self.rl_env_path = None

        # Check SKYRL_HOME
        skyrl_home = os.environ.get("SKYRL_HOME")
        if not skyrl_home:
            # Try common locations
            candidates = [
                PROJECT_ROOT / "SkyRL",
                Path.home() / "SkyRL",
            ]
            for candidate in candidates:
                if candidate.exists():
                    skyrl_home = str(candidate)
                    os.environ["SKYRL_HOME"] = skyrl_home
                    break

        if skyrl_home and Path(skyrl_home).exists():
            print(f"SkyRL home: {skyrl_home}")
            # Add to PYTHONPATH
            skyrl_train = os.path.join(skyrl_home, "skyrl-train")
            if skyrl_train not in sys.path:
                sys.path.insert(0, skyrl_train)
            pythonpath = os.environ.get("PYTHONPATH", "")
            if skyrl_train not in pythonpath:
                os.environ["PYTHONPATH"] = f"{skyrl_train}:{pythonpath}"
        else:
            print("\nWARNING: SKYRL_HOME not found!")
            print("Clone SkyRL or set SKYRL_HOME environment variable.")

        # Set up experiments directory
        experiments_dir = Path(self.config.experiments_dir).expanduser().resolve()
        experiments_dir.mkdir(parents=True, exist_ok=True)
        self.config.experiments_dir = str(experiments_dir)

        # Auto-detect CPUs if not specified
        if self.config.cpus <= 0:
            self.config.cpus = os.cpu_count() or 16

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
        """Clean up Ray and any running processes."""
        for proc in self._processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()

        if self._ray_started:
            try:
                import ray
                ray.shutdown()
                print("Ray cluster shut down.")
            except Exception:
                pass

    def print_banner(self) -> None:
        """Print startup banner."""
        print("=== Local RL Training Runner ===")
        print(f"  Job Name: {self.config.job_name}")
        print(f"  RL Config: {self.config.rl_config_path}")
        print(f"  Model: {self.config.model_path}")
        print(f"  GPUs: {self.config.gpus}")
        print(f"  Train Data: {self.config.train_data}")
        print(f"  Experiments Dir: {self.config.experiments_dir}")
        print("================================")

    def run(self) -> int:
        """Execute the RL training job.

        Returns:
            Exit code (0 for success)
        """
        self.print_banner()

        # Parse RL config
        parsed = parse_rl_config(
            self.config.rl_config_path,
            model_override=self.config.model_path,
        )
        print(f"Loaded RL config: {parsed.config_path}")
        self.config.tensor_parallel_size = parsed.tensor_parallel_size

        # Resolve train data (extract HF datasets to local directories)
        if self.config.train_data:
            print(f"\nResolving train_data: {self.config.train_data}")
            resolved_train = resolve_rl_train_data(self.config.train_data)
            self.config.train_data = resolved_train
            print(f"Resolved train_data: {resolved_train}")

        # Build experiment args dict (mimics exp_args from HPC launcher)
        exp_args = self._build_exp_args()

        # Build Hydra args using shared utility
        # Create a minimal HPC-like object for the builder
        hpc_stub = _LocalHPCStub(
            gpus_per_node=self.config.gpus,
            cpus_per_node=self.config.cpus,
        )
        hydra_args = build_skyrl_hydra_args(parsed, exp_args, hpc_stub)

        # Add CLI overrides
        if self.config.skyrl_overrides:
            hydra_args.extend(self.config.skyrl_overrides)

        if self.config.dry_run:
            print("\n[DRY RUN] Would execute SkyRL with:")
            print(get_skyrl_command_preview(parsed.entrypoint, hydra_args))
            return 0

        # Set up environment
        self._setup_environment(exp_args)

        # Start Ray and run SkyRL
        return self._run_with_ray(parsed.entrypoint, hydra_args)

    def _build_exp_args(self) -> Dict[str, Any]:
        """Build exp_args dict mimicking HPC launcher."""
        return {
            "job_name": self.config.job_name,
            "experiments_dir": self.config.experiments_dir,
            "model_path": self.config.model_path,
            "train_data": self.config.train_data,
            "val_data": self.config.val_data,
            "num_nodes": 1,  # Local = single node
            "gpus_per_node": self.config.gpus,
            "cpus_per_node": self.config.cpus,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "ray_port": self.config.ray_port,
            "master_port": self.config.master_port,
        }

    def _setup_environment(self, exp_args: Dict[str, Any]) -> None:
        """Configure environment variables for RL training."""
        # Tensor parallelism and inference engines
        os.environ["TENSOR_PARALLEL_SIZE"] = str(self.config.tensor_parallel_size)
        os.environ["NUM_INFERENCE_ENGINES"] = str(
            compute_num_inference_engines(
                1,  # num_nodes = 1 for local
                self.config.gpus,
                self.config.tensor_parallel_size,
            )
        )
        os.environ["POLICY_NUM_NODES"] = "1"

        # Export path
        export_path = derive_skyrl_export_path(
            self.config.experiments_dir,
            self.config.job_name,
        )
        os.environ["SKYRL_EXPORT_PATH"] = export_path

        # vLLM settings
        os.environ["VLLM_USE_V1"] = "1"

        # WandB directory
        wandb_dir = os.path.join(self.config.experiments_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir

        # CUDA settings
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(self.config.gpus)))

        print("\nEnvironment configured:")
        print(f"  TENSOR_PARALLEL_SIZE={os.environ['TENSOR_PARALLEL_SIZE']}")
        print(f"  NUM_INFERENCE_ENGINES={os.environ['NUM_INFERENCE_ENGINES']}")
        print(f"  SKYRL_EXPORT_PATH={export_path}")
        print(f"  WANDB_DIR={wandb_dir}")

    def _run_with_ray(self, entrypoint: str, hydra_args: List[str]) -> int:
        """Start local Ray cluster and run SkyRL."""
        try:
            import ray
        except ImportError:
            print("ERROR: Ray not installed. Activate the RL environment first.")
            print("  source ./hpc/activate_rl_env.sh")
            return 1

        # Start local Ray cluster
        print(f"\nStarting local Ray cluster with {self.config.gpus} GPUs...")
        ray.init(
            num_cpus=self.config.cpus,
            num_gpus=self.config.gpus,
            include_dashboard=False,
        )
        self._ray_started = True

        os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        print(f"Ray cluster ready at {os.environ['RAY_ADDRESS']}")

        try:
            return self._run_skyrl(entrypoint, hydra_args)
        finally:
            ray.shutdown()
            self._ray_started = False

    def _run_skyrl(self, entrypoint: str, hydra_args: List[str]) -> int:
        """Execute SkyRL training."""
        # Determine Python executable
        if self.rl_env_path:
            python_exe = str(self.rl_env_path / "bin" / "python")
        else:
            python_exe = sys.executable

        cmd = [python_exe, "-m", entrypoint] + hydra_args

        print(f"\nRunning SkyRL:")
        print(f"  Entrypoint: {entrypoint}")
        print(f"  Args: {len(hydra_args)} Hydra arguments")

        # Change to SKYRL_HOME/skyrl-train if available
        skyrl_home = os.environ.get("SKYRL_HOME")
        cwd = None
        if skyrl_home:
            cwd = os.path.join(skyrl_home, "skyrl-train")
            if os.path.isdir(cwd):
                print(f"  Working dir: {cwd}")
            else:
                cwd = None

        print(f"\nCommand: {' '.join(cmd[:3])} [... {len(cmd)-3} more args]")
        sys.stdout.flush()

        proc = subprocess.Popen(cmd, cwd=cwd)
        self._processes.append(proc)
        return proc.wait()


@dataclass
class _LocalHPCStub:
    """Minimal HPC-like object for build_skyrl_hydra_args compatibility."""
    gpus_per_node: int = 4
    cpus_per_node: int = 48
    name: str = "local"

    def get_module_commands(self) -> str:
        return "# No module commands for local execution"

    @property
    def conda_activate(self) -> str:
        return "# No conda activation for local execution"

    @property
    def dotenv_filename(self) -> str:
        return "local.env"

    @property
    def needs_ssh_tunnel(self) -> bool:
        return False


def parse_list_arg(value: str) -> List[str]:
    """Parse a list argument from CLI (JSON or Python literal)."""
    if not value:
        return []
    try:
        # Try JSON/Python literal
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except (ValueError, SyntaxError):
        # Treat as single value
        return [value]


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for local RL runner."""
    parser = argparse.ArgumentParser(
        description="Run RL training on local GPUs using SkyRL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--rl_config",
        required=True,
        help="Path to SkyRL config YAML (e.g., terminal_bench.yaml).",
    )
    parser.add_argument("--rl-config", dest="rl_config", help=argparse.SUPPRESS)

    parser.add_argument(
        "--model_path",
        required=True,
        help="Model path or HuggingFace ID (e.g., Qwen/Qwen3-8B).",
    )
    parser.add_argument("--model-path", dest="model_path", help=argparse.SUPPRESS)

    parser.add_argument(
        "--job_name",
        required=True,
        help="Name for this training job.",
    )
    parser.add_argument("--job-name", dest="job_name", help=argparse.SUPPRESS)

    # Data arguments
    parser.add_argument(
        "--train_data",
        default="[]",
        help="Training data paths as JSON list (e.g., '[\"org/dataset\"]').",
    )
    parser.add_argument("--train-data", dest="train_data", help=argparse.SUPPRESS)

    parser.add_argument(
        "--val_data",
        default="[]",
        help="Validation data paths as JSON list.",
    )
    parser.add_argument("--val-data", dest="val_data", help=argparse.SUPPRESS)

    # Resource arguments
    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="Number of GPUs to use.",
    )

    parser.add_argument(
        "--cpus",
        type=int,
        default=0,
        help="Number of CPUs (0 = auto-detect).",
    )

    # Network arguments
    parser.add_argument(
        "--ray_port",
        type=int,
        default=6379,
        help="Port for Ray cluster.",
    )
    parser.add_argument("--ray-port", dest="ray_port", help=argparse.SUPPRESS)

    parser.add_argument(
        "--master_port",
        type=int,
        default=12345,
        help="Master port for distributed training.",
    )
    parser.add_argument("--master-port", dest="master_port", help=argparse.SUPPRESS)

    # Override arguments
    parser.add_argument(
        "--skyrl_override",
        action="append",
        default=[],
        help="SkyRL Hydra override (can be specified multiple times).",
    )
    parser.add_argument("--skyrl-override", dest="skyrl_override", action="append", help=argparse.SUPPRESS)

    # Path arguments
    parser.add_argument(
        "--experiments_dir",
        default=str(PROJECT_ROOT / "experiments"),
        help="Directory for experiment outputs.",
    )
    parser.add_argument("--experiments-dir", dest="experiments_dir", help=argparse.SUPPRESS)

    # Control arguments
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print configuration and command without running.",
    )
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help=argparse.SUPPRESS)

    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Parse list arguments
    train_data = parse_list_arg(args.train_data)
    val_data = parse_list_arg(args.val_data)

    # Collect overrides (handle both underscore and hyphen versions)
    skyrl_overrides = args.skyrl_override or []

    config = LocalRLConfig(
        rl_config_path=args.rl_config,
        job_name=args.job_name,
        model_path=args.model_path,
        train_data=train_data,
        val_data=val_data,
        experiments_dir=args.experiments_dir,
        gpus=args.gpus,
        cpus=args.cpus,
        ray_port=args.ray_port,
        master_port=args.master_port,
        skyrl_overrides=skyrl_overrides,
        dry_run=args.dry_run,
    )

    runner = LocalRLRunner(config)
    runner.setup()
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
