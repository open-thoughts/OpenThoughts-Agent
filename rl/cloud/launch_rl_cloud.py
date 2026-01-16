#!/usr/bin/env python3
"""
Launch OpenThoughts RL training on a cloud VM via SkyPilot.

This wrapper mirrors the key arguments from rl/local/run_rl.py, then wraps the
whole invocation inside a SkyPilot Task so we can bring up short-lived GPU nodes.

Usage:
    python -m rl.cloud.launch_rl_cloud \
        --rl_config terminal_bench.yaml \
        --model_path Qwen/Qwen3-8B \
        --train_data '["mlfoundations-dev/dataset"]' \
        --job_name my_rl_run \
        --cloud_provider gcp \
        --accelerator "H100:4"

    # Use gpu-rl Docker image (default for RL)
    python -m rl.cloud.launch_rl_cloud \
        --rl_config qwen3_1.7b_4x80GB.yaml \
        --model_path Qwen/Qwen3-1.7B \
        --train_data '["dataset"]' \
        --docker_image ghcr.io/open-thoughts/openthoughts-agent:gpu-rl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add repo root to sys.path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

# Handle --list-providers before importing anything heavy
if "--list-providers" in sys.argv or "--list_providers" in sys.argv:
    from hpc.cloud_providers import list_providers
    print(list_providers(verbose=True))
    sys.exit(0)

from hpc.launch_utils import PROJECT_ROOT
from hpc.cloud_launch_utils import CloudLauncher, repo_relative, parse_gpu_count
from hpc.cloud_sync_utils import sync_outputs

# Default Docker image for RL (has SkyRL environment)
DEFAULT_RL_DOCKER_IMAGE = "ghcr.io/open-thoughts/openthoughts-agent:gpu-rl"


class RLCloudLauncher(CloudLauncher):
    """Cloud launcher for rl/local/run_rl.py."""

    task_name = "ot-rl-cloud"
    job_name_prefix = "rl"  # For auto-derived job names
    default_output_subdir = "rl_runs"
    default_n_concurrent = 1  # RL is typically single-job

    def add_task_specific_args(self, parser: argparse.ArgumentParser) -> None:
        """Add RL-specific arguments."""
        # Required arguments
        parser.add_argument(
            "--rl_config",
            required=True,
            help="Path to SkyRL config YAML (e.g., terminal_bench.yaml, qwen3_1.7b_4x80GB.yaml).",
        )
        parser.add_argument("--rl-config", dest="rl_config", help=argparse.SUPPRESS)

        parser.add_argument(
            "--model_path",
            required=True,
            help="Model path or HuggingFace ID (e.g., Qwen/Qwen3-8B).",
        )
        parser.add_argument("--model-path", dest="model_path", help=argparse.SUPPRESS)

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

        # Job identification
        parser.add_argument(
            "--job_name",
            help="Name for this training job (auto-derived if not set).",
        )
        parser.add_argument("--job-name", dest="job_name", help=argparse.SUPPRESS)

        # Resource arguments
        parser.add_argument(
            "--gpus",
            type=int,
            default=None,
            help="Number of GPUs (inferred from --accelerator if not set).",
        )

        parser.add_argument(
            "--cpus",
            type=int,
            default=0,
            help="Number of CPUs (0 = auto-detect on remote).",
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
        parser.add_argument(
            "--skyrl-override",
            dest="skyrl_override",
            action="append",
            help=argparse.SUPPRESS,
        )

        # Dry run
        parser.add_argument(
            "--dry_run",
            action="store_true",
            help="Print configuration and command without running.",
        )
        parser.add_argument("--dry-run", dest="dry_run", action="store_true", help=argparse.SUPPRESS)

    def _add_cloud_args(self, parser: argparse.ArgumentParser) -> None:
        """Override to set RL-specific defaults."""
        super()._add_cloud_args(parser)

        # Override docker_image default for RL
        for action in parser._actions:
            if action.dest == "docker_image":
                action.default = DEFAULT_RL_DOCKER_IMAGE
                break

    def get_dataset_arg_name(self) -> Optional[str]:
        """Return the dataset argument name for HF handling."""
        # RL uses train_data which can be a list, handled separately
        return None

    def normalize_paths(self, args: argparse.Namespace) -> None:
        """Normalize repo-relative paths and infer defaults."""
        # Infer --gpus from --accelerator if not explicitly provided
        if args.gpus is None:
            args.gpus = parse_gpu_count(args.accelerator)

        # Normalize rl_config path (check hpc/skyrl_yaml/ if not found)
        rl_config_path = Path(args.rl_config).expanduser()
        if not rl_config_path.exists():
            # Try hpc/skyrl_yaml/
            yaml_dir = self.repo_root / "hpc" / "skyrl_yaml"
            candidate = yaml_dir / args.rl_config
            if candidate.exists():
                args.rl_config = str(candidate)
            else:
                # Try with .yaml extension
                candidate_yaml = yaml_dir / f"{args.rl_config}.yaml"
                if candidate_yaml.exists():
                    args.rl_config = str(candidate_yaml)
                else:
                    # Keep original, let run_rl.py handle the error
                    pass
        else:
            args.rl_config = str(rl_config_path.resolve())

    def build_task_command(self, args: argparse.Namespace, remote_output_dir: str) -> List[str]:
        """Build the run_rl.py command."""
        # Use the RL environment's Python
        cmd: List[str] = [
            "/opt/openthoughts/envs/rl/bin/python",
            "-m", "rl.local.run_rl",
            "--rl_config", args.rl_config,
            "--model_path", args.model_path,
            "--job_name", args.job_name,
            "--gpus", str(args.gpus),
            "--experiments_dir", remote_output_dir,
        ]

        # Data arguments
        if args.train_data and args.train_data != "[]":
            cmd.extend(["--train_data", args.train_data])
        if args.val_data and args.val_data != "[]":
            cmd.extend(["--val_data", args.val_data])

        # Optional arguments
        if args.cpus > 0:
            cmd.extend(["--cpus", str(args.cpus)])
        if args.ray_port != 6379:
            cmd.extend(["--ray_port", str(args.ray_port)])
        if args.master_port != 12345:
            cmd.extend(["--master_port", str(args.master_port)])

        # SkyRL overrides
        for override in (args.skyrl_override or []):
            cmd.extend(["--skyrl_override", override])

        # Dry run (for debugging remote)
        if args.dry_run:
            cmd.append("--dry_run")

        return cmd

    def get_pre_task_commands(self, args: argparse.Namespace) -> List[str]:
        """Return commands to run before the main task on remote.

        For RL, we skip harbor reinstall (not needed) but ensure SkyRL
        environment is properly set up.
        """
        return [
            'echo "[cloud-setup] Setting up RL environment..."',
            # Ensure SKYRL_HOME is set
            'export SKYRL_HOME=${SKYRL_HOME:-/opt/skyrl}',
            # Add skyrl-train to PYTHONPATH
            'export PYTHONPATH="$SKYRL_HOME/skyrl-train:${PYTHONPATH:-}"',
            # vLLM settings
            'export VLLM_USE_V1=1',
        ]

    def get_periodic_sync_paths(
        self,
        args: argparse.Namespace,
        remote_output_dir: str,
        remote_workdir: str,
    ) -> List[tuple]:
        """Return paths to sync periodically during job execution.

        Syncs logs and checkpoints directory to track training progress.
        """
        return [
            (f"{remote_output_dir}/logs", str(Path(args.local_sync_dir) / "logs")),
            (f"{remote_output_dir}/{args.job_name}/exports", str(Path(args.local_sync_dir) / "exports")),
        ]

    def sync_additional_outputs(
        self,
        cluster_name: str,
        args: argparse.Namespace,
        remote_workdir: str,
    ) -> None:
        """Sync WandB directory (final sync after job completes)."""
        # Sync wandb logs if they exist
        wandb_remote = f"{remote_workdir}/{args.remote_output_subdir}/wandb"
        wandb_local = str(Path(args.local_sync_dir) / "wandb")
        print(f"[cloud-sync] Also syncing WandB logs from {wandb_remote}...")
        try:
            sync_outputs(
                cluster_name=cluster_name,
                remote_path=wandb_remote,
                local_dir=wandb_local,
            )
        except Exception as e:
            # WandB directory may not exist if not configured
            print(f"[cloud-sync] WandB sync skipped: {e}")


def main() -> None:
    launcher = RLCloudLauncher(PROJECT_ROOT)
    parser = launcher.create_argument_parser(
        description="Launch rl/local/run_rl.py on a cloud GPU node via SkyPilot."
    )
    args = parser.parse_args()
    launcher.run(args)


if __name__ == "__main__":
    main()
