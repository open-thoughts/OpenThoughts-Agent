"""
RL Training Launch Utilities

Helpers for computing distributed RL training parameters and deriving paths
for SkyRL-based reinforcement learning jobs.

This module provides:
- RLJobConfig: Configuration dataclass for RL training jobs
- launch_rl_job(): Main entry point for submitting RL jobs
- RLJobRunner: Class for executing RL jobs from sbatch
- resolve_rl_train_data(): Extracts HF datasets to local task directories
- Helper functions for computing inference engines, tensor parallelism, etc.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from hpc.hf_utils import is_hf_dataset_path


def resolve_rl_train_data(
    train_data: List[str],
    scratch_dir: Optional[str] = None,
    on_exist: str = "skip",
    verbose: bool = True,
) -> List[str]:
    """Resolve train_data paths, extracting HF datasets to local task directories.

    SkyRL's TerminalBenchTaskDataset expects local directory paths where each
    subdirectory is a task containing an instruction.md file. This function:
    1. Detects HuggingFace dataset identifiers (e.g., "org/repo-name")
    2. Extracts them to $SCRATCH/tasks/<repo-name>/ using extract_tasks_from_parquet
    3. Fixes permissions (chmod) to ensure tasks are readable
    4. Returns local filesystem paths for all datasets

    Args:
        train_data: List of dataset paths (local paths or HF repo IDs).
        scratch_dir: Base directory for extracted tasks (default: $SCRATCH/tasks or /tmp/tasks).
        on_exist: How to handle existing task directories ("skip", "overwrite", "error").
        verbose: Whether to print status messages.

    Returns:
        List of resolved local filesystem paths.

    Example:
        >>> resolve_rl_train_data(["penfever/my-dataset", "/local/path/tasks"])
        ['/scratch/tasks/my-dataset', '/local/path/tasks']
    """
    if not train_data:
        return []

    # Determine scratch directory for extracted tasks
    if scratch_dir is None:
        scratch_dir = os.environ.get("SCRATCH", "/tmp")
    tasks_base = Path(scratch_dir) / "tasks"

    resolved_paths = []

    for data_path in train_data:
        if is_hf_dataset_path(data_path):
            # It's a HuggingFace dataset - extract to local directory
            # Extract repo name from "org/repo-name" -> "repo-name"
            repo_name = data_path.split("/")[-1]
            output_dir = tasks_base / repo_name

            if verbose:
                print(f"[rl_launch_utils] Extracting HF dataset: {data_path}")
                print(f"[rl_launch_utils] Output directory: {output_dir}")

            # Check if already extracted (when on_exist="skip")
            if on_exist == "skip" and output_dir.exists() and any(output_dir.iterdir()):
                if verbose:
                    print(f"[rl_launch_utils] Tasks already extracted, skipping: {output_dir}")
                resolved_paths.append(str(output_dir))
                continue

            # Run extract_tasks_from_parquet
            cmd = [
                sys.executable, "-m", "scripts.datagen.extract_tasks_from_parquet",
                "--parquet", data_path,
                "--output_dir", str(output_dir),
                "--on_exist", on_exist,
            ]

            if verbose:
                print(f"[rl_launch_utils] Running: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if verbose and result.stdout:
                    print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"[rl_launch_utils] ERROR extracting {data_path}:")
                print(f"  stdout: {e.stdout}")
                print(f"  stderr: {e.stderr}")
                raise RuntimeError(f"Failed to extract HF dataset: {data_path}") from e

            # Fix permissions on extracted tasks (chmod -R a+rX)
            _fix_task_permissions(output_dir, verbose=verbose)

            resolved_paths.append(str(output_dir))
        else:
            # It's a local path - fix permissions just in case
            local_path = Path(data_path)
            if local_path.exists():
                _fix_task_permissions(local_path, verbose=verbose)
            resolved_paths.append(data_path)

    return resolved_paths


def _fix_task_permissions(task_dir: Path, verbose: bool = True) -> None:
    """Fix permissions on task directory to ensure files are readable.

    Runs chmod -R a+rX on the directory to make all files readable
    and directories traversable.

    Args:
        task_dir: Path to task directory.
        verbose: Whether to print status messages.
    """
    if not task_dir.exists():
        return

    if verbose:
        print(f"[rl_launch_utils] Fixing permissions on: {task_dir}")

    try:
        subprocess.run(
            ["chmod", "-R", "a+rX", str(task_dir)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        # Don't fail the whole job for permission issues
        print(f"[rl_launch_utils] Warning: chmod failed on {task_dir}: {e.stderr}")


def compute_num_inference_engines(
    num_nodes: int,
    gpus_per_node: int,
    tensor_parallel_size: int = 1,
) -> int:
    """
    Compute the number of vLLM inference engines for distributed RL training.

    In SkyRL, inference engines are used for rollout generation. The total number
    is determined by dividing the total GPU count by the tensor parallel size.

    Args:
        num_nodes: Number of nodes in the job.
        gpus_per_node: Number of GPUs per node.
        tensor_parallel_size: Tensor parallel size for vLLM (default: 1).

    Returns:
        Number of inference engines.

    Example:
        >>> compute_num_inference_engines(num_nodes=2, gpus_per_node=4, tensor_parallel_size=1)
        8
        >>> compute_num_inference_engines(num_nodes=2, gpus_per_node=4, tensor_parallel_size=2)
        4
    """
    total_gpus = num_nodes * gpus_per_node
    return total_gpus // tensor_parallel_size


def get_tensor_parallel_size(
    gpus_per_node: int,
    model_size_hint: Optional[str] = None,
) -> int:
    """
    Determine appropriate tensor parallel size based on model and GPU configuration.

    For most models, TP=1 is sufficient. Larger models (70B+) may need TP=2 or TP=4.

    Args:
        gpus_per_node: Number of GPUs per node.
        model_size_hint: Optional hint about model size (e.g., "7B", "70B", "405B").

    Returns:
        Recommended tensor parallel size.
    """
    # Default to TP=1 for most models
    if model_size_hint is None:
        return 1

    # Parse model size from hint
    model_size_hint = model_size_hint.upper()
    if "405B" in model_size_hint or "400B" in model_size_hint:
        # Very large models need high TP
        return min(8, gpus_per_node)
    elif "70B" in model_size_hint or "72B" in model_size_hint:
        # Large models benefit from TP=2 or TP=4
        return min(4, gpus_per_node)
    elif "32B" in model_size_hint or "34B" in model_size_hint:
        # Medium-large models may benefit from TP=2
        return min(2, gpus_per_node)
    else:
        # Smaller models (7B, 8B, 14B, etc.) work fine with TP=1
        return 1


def derive_skyrl_export_path(
    experiments_dir: str,
    run_name: str,
    exports_subdir: str = "exports",
) -> str:
    """
    Derive the SkyRL export path from experiments directory and run name.

    The export path is where SkyRL saves model checkpoints during training.

    Args:
        experiments_dir: Base experiments directory.
        run_name: Name of the training run.
        exports_subdir: Subdirectory name for exports (default: "exports").

    Returns:
        Full path to the SkyRL export directory.

    Example:
        >>> derive_skyrl_export_path("/scratch/experiments", "qwen3_8b_nl2bash")
        '/scratch/experiments/qwen3_8b_nl2bash/exports'
    """
    return str(Path(experiments_dir) / run_name / exports_subdir)


def build_rl_env_vars(
    exp_args: Dict[str, Any],
    hpc: Optional[Any] = None,
) -> Dict[str, str]:
    """
    Build environment variables dictionary for RL training jobs.

    Args:
        exp_args: Experiment arguments dictionary.
        hpc: Optional HPC configuration object.

    Returns:
        Dictionary of environment variable name -> value.
    """
    env_vars = {}

    # Tensor parallel and inference engine settings
    num_nodes = int(exp_args.get("num_nodes", 1))
    gpus_per_node = int(exp_args.get("gpus_per_node", 4))
    tensor_parallel_size = int(exp_args.get("tensor_parallel_size", 1))

    env_vars["TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
    env_vars["NUM_INFERENCE_ENGINES"] = str(
        compute_num_inference_engines(num_nodes, gpus_per_node, tensor_parallel_size)
    )

    # Policy nodes (can be different from total nodes for asymmetric setups)
    policy_num_nodes = exp_args.get("policy_num_nodes")
    if policy_num_nodes is not None:
        env_vars["POLICY_NUM_NODES"] = str(policy_num_nodes)
    else:
        env_vars["POLICY_NUM_NODES"] = str(num_nodes)

    # SkyRL export path
    experiments_dir = exp_args.get("experiments_dir", "")
    run_name = exp_args.get("run_name") or exp_args.get("job_name", "")
    if experiments_dir and run_name:
        env_vars["SKYRL_EXPORT_PATH"] = derive_skyrl_export_path(experiments_dir, run_name)

    # WANDB mode (inherit from HPC if available)
    if hpc is not None and hasattr(hpc, "env_vars"):
        hpc_env = hpc.env_vars or {}
        if "WANDB_MODE" in hpc_env:
            env_vars["WANDB_MODE"] = hpc_env["WANDB_MODE"]

    return env_vars


def get_rl_env_exports(exp_args: Dict[str, Any], hpc: Optional[Any] = None) -> str:
    """
    Generate shell export statements for RL environment variables.

    Args:
        exp_args: Experiment arguments dictionary.
        hpc: Optional HPC configuration object.

    Returns:
        Multi-line string of export statements.
    """
    env_vars = build_rl_env_vars(exp_args, hpc)

    if not env_vars:
        return "# No RL-specific environment variables"

    lines = ["# RL training environment variables"]
    for key, value in env_vars.items():
        lines.append(f'export {key}="{value}"')

    return "\n".join(lines)


def get_rl_env_activation(exp_args: Dict[str, Any]) -> str:
    """
    Generate shell code for RL environment activation.

    Supports two modes:
    1. Conda environment (--rl_use_conda --rl_conda_env NAME)
    2. venv created by setup_rl_env.sh (default)

    Args:
        exp_args: Experiment arguments dictionary.

    Returns:
        Multi-line shell script for environment activation.
    """
    use_conda = exp_args.get("rl_use_conda", False)
    conda_env = exp_args.get("rl_conda_env", "dcagent-rl")

    if use_conda:
        return f'''# Using conda environment for RL: {conda_env}
echo "Activating conda environment: {conda_env}"
# Disable unbound variable check during conda operations (conda scripts reference unset vars)
set +u
# Initialize conda for non-interactive shell (required before conda activate)
if [[ -n "${{CONDA_EXE:-}}" ]]; then
  # Use CONDA_EXE to find conda.sh
  CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
  source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [[ -f "${{HOME}}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${{HOME}}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${{HOME}}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${{HOME}}/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: Could not find conda installation for initialization"
  set -u
  exit 1
fi
conda activate {conda_env}
# Re-enable unbound variable check
set -u'''
    else:
        return '''# Using venv for RL (created by ./hpc/setup_rl_env.sh)
RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"
if [[ -d "$RL_ENV_DIR" ]]; then
  echo "Activating RL environment: $RL_ENV_DIR"
  source "$RL_ENV_DIR/bin/activate"
elif [[ -n "${DCFT_RL_ENV:-}" ]] && [[ -d "$DCFT_RL_ENV" ]]; then
  echo "Activating RL environment from DCFT_RL_ENV: $DCFT_RL_ENV"
  source "$DCFT_RL_ENV/bin/activate"
else
  echo "Warning: RL environment not found at $RL_ENV_DIR"
  echo "Run ./hpc/setup_rl_env.sh to create it, or set DCFT_RL_ENV"
fi'''


# =============================================================================
# RLJobConfig and Job Submission
# =============================================================================


@dataclass
class RLJobConfig:
    """Configuration for an RL training job (serialized to JSON for sbatch).

    This dataclass contains all information needed to run an RL training job
    via the universal_rl.sbatch template and RLJobRunner.
    """

    job_name: str
    experiments_dir: str
    cluster_name: str

    # SkyRL settings
    skyrl_entrypoint: str
    skyrl_hydra_args: List[str] = field(default_factory=list)

    # Model and data
    model_path: str = ""
    train_data: List[str] = field(default_factory=list)
    val_data: List[str] = field(default_factory=list)

    # Resource allocation
    num_nodes: int = 1
    gpus_per_node: int = 4
    cpus_per_node: int = 48
    tensor_parallel_size: int = 1

    # Networking
    ray_port: int = 6379
    master_port: int = 12345

    # Paths
    checkpoints_dir: Optional[str] = None
    export_path: Optional[str] = None

    # Cluster-specific flags
    needs_ssh_tunnel: bool = False
    needs_cuda_detection: bool = False


def build_skyrl_command_string(config: RLJobConfig) -> str:
    """Build the full SkyRL command string for the sbatch template.

    Args:
        config: RLJobConfig with entrypoint and hydra args.

    Returns:
        Shell command string with proper line continuations.
    """
    parts = [f"python -m {config.skyrl_entrypoint}"]

    for arg in config.skyrl_hydra_args:
        parts.append(f"  {arg}")

    return " \\\n".join(parts)


def construct_rl_sbatch_script(exp_args: dict, hpc) -> str:
    """Construct RL sbatch script using the universal template system.

    This follows the same pattern as construct_sft_sbatch_script() but for RL jobs.

    Args:
        exp_args: Experiment arguments dictionary.
        hpc: HPC cluster configuration.

    Returns:
        Path to the generated sbatch script.
    """
    from hpc.launch_utils import (
        resolve_job_and_paths,
        substitute_template,
        build_sbatch_directives,
    )
    from hpc.rl_config_utils import parse_rl_config, build_skyrl_hydra_args

    print("\n=== RL MODE (Universal Launcher) ===")

    # Parse RL config YAML
    rl_config_path = exp_args.get("rl_config")
    if not rl_config_path:
        raise ValueError("--rl_config is required for RL jobs")

    parsed = parse_rl_config(rl_config_path, model_override=exp_args.get("model_path"))
    print(f"Loaded RL config from: {parsed.config_path}")

    # Resolve train_data: extract HF datasets to local task directories
    # This must happen BEFORE building Hydra args so the local paths are used
    train_data_raw = exp_args.get("train_data") or []
    if isinstance(train_data_raw, str):
        # Handle JSON string from CLI
        import ast
        try:
            train_data_raw = ast.literal_eval(train_data_raw)
        except (ValueError, SyntaxError):
            train_data_raw = [train_data_raw]

    if train_data_raw:
        print(f"Resolving train_data: {train_data_raw}")
        resolved_train_data = resolve_rl_train_data(train_data_raw)
        exp_args["train_data"] = resolved_train_data
        print(f"Resolved train_data: {resolved_train_data}")

    # Resolve val_data similarly (eval datasets may also be HF repos)
    # Check CLI first, then fall back to YAML config default
    val_data_raw = exp_args.get("val_data")
    if val_data_raw is None:
        # Get default from YAML config
        val_data_raw = parsed.data.get("val_data", [])
    if isinstance(val_data_raw, str):
        import ast
        try:
            val_data_raw = ast.literal_eval(val_data_raw)
        except (ValueError, SyntaxError):
            val_data_raw = [val_data_raw]

    if val_data_raw:
        print(f"Resolving val_data: {val_data_raw}")
        resolved_val_data = resolve_rl_train_data(val_data_raw)
        exp_args["val_data"] = resolved_val_data
        print(f"Resolved val_data: {resolved_val_data}")

    # Build Hydra args from YAML + CLI overrides
    hydra_args = build_skyrl_hydra_args(parsed, exp_args, hpc)

    # Apply CLI overrides (--skyrl_override key=value)
    skyrl_overrides = exp_args.get("skyrl_override") or []
    if skyrl_overrides:
        hydra_args.extend(skyrl_overrides)
        print(f"Applied {len(skyrl_overrides)} CLI overrides")

    # Resolve job_name and paths (job_name already set by get_job_name() in launch.py)
    job_setup = resolve_job_and_paths(
        exp_args,
        job_type_label="RL",
    )
    job_name = job_setup.job_name
    exp_paths = job_setup.paths
    experiments_subdir = str(exp_paths.root)

    # Extract config values
    num_nodes = int(exp_args.get("num_nodes") or 1)
    gpus_per_node = int(exp_args.get("gpus_per_node") or hpc.gpus_per_node)
    cpus_per_node = int(exp_args.get("cpus_per_node") or hpc.cpus_per_node)

    # Build RLJobConfig
    job_config = RLJobConfig(
        job_name=job_name,
        experiments_dir=experiments_subdir,
        cluster_name=hpc.name,
        skyrl_entrypoint=parsed.entrypoint,
        skyrl_hydra_args=hydra_args,
        model_path=exp_args.get("model_path", ""),
        train_data=exp_args.get("train_data", []),
        val_data=exp_args.get("val_data", []),
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_node=cpus_per_node,
        tensor_parallel_size=parsed.tensor_parallel_size,
        ray_port=int(exp_args.get("ray_port") or 6379),
        master_port=int(exp_args.get("master_port") or 12345),
        export_path=derive_skyrl_export_path(experiments_subdir, job_name),
        needs_ssh_tunnel=hpc.needs_ssh_tunnel,
        needs_cuda_detection=getattr(hpc, "needs_cuda_detection", False),
    )

    # Write config JSON
    config_dir = exp_paths.configs
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{job_name}_rl_config.json"
    config_path.write_text(json.dumps(asdict(job_config), indent=2))
    print(f"Wrote RL job config to {config_path}")

    # Load and populate universal template
    template_path = Path(__file__).parent / "sbatch_rl" / "universal_rl.sbatch"
    if not template_path.exists():
        raise FileNotFoundError(f"RL sbatch template not found: {template_path}")
    template_text = template_path.read_text()

    # Build cluster-specific SBATCH directives
    sbatch_directives = build_sbatch_directives(hpc, exp_args)

    # Generate RL environment exports
    rl_env_exports = get_rl_env_exports(exp_args, hpc)

    # Generate CUDA setup code
    cuda_setup = ""
    if getattr(hpc, "needs_cuda_detection", False):
        cuda_setup = """# CUDA path detection (Perlmutter and similar)
if [[ -d /opt/nvidia/hpc_sdk ]]; then
    export CUDA_HOME=$(ls -d /opt/nvidia/hpc_sdk/*/Linux_x86_64/cuda/* 2>/dev/null | head -1)
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi"""

    # Build SkyRL command
    skyrl_command = build_skyrl_command_string(job_config)

    # Generate RL environment activation code (conda or venv)
    rl_env_activation = get_rl_env_activation(exp_args)

    substitutions = {
        "time_limit": exp_args.get("time_limit") or "24:00:00",
        "num_nodes": str(num_nodes),
        "cpus_per_node": str(cpus_per_node),
        "experiments_dir": experiments_subdir,
        "job_name": job_name,
        "sbatch_extra_directives": "\n".join(sbatch_directives),
        "module_commands": hpc.get_module_commands(),
        "conda_activate": hpc.conda_activate or "# No conda activation configured",
        "cluster_env_file": hpc.dotenv_filename,
        "cuda_setup": cuda_setup,
        "nccl_exports": hpc.get_nccl_exports(),
        "rl_env_exports": rl_env_exports,
        "rl_env_activation": rl_env_activation,
        "ssh_tunnel_setup": hpc.get_ssh_tunnel_setup(),
        "ray_port": str(job_config.ray_port),
        "master_port": str(job_config.master_port),
        "gpus_per_node": str(gpus_per_node),
        "config_path": str(config_path),
        "skyrl_command": skyrl_command,
        "email_address": os.environ.get("EMAIL_ADDRESS", ""),
    }

    sbatch_text = substitute_template(template_text, substitutions)

    # Write sbatch script
    sbatch_dir = exp_paths.sbatch
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    sbatch_output = sbatch_dir / f"{job_name}_rl.sbatch"
    sbatch_output.write_text(sbatch_text)
    os.chmod(sbatch_output, 0o750)
    print(f"Wrote RL sbatch script to {sbatch_output}")

    return str(sbatch_output)


def check_rl_environment() -> Optional[Path]:
    """Check if the RL environment exists and return its path.

    The RL environment is separate from the main environment due to
    dependency conflicts between SkyRL (torch 2.8, vllm 0.11.0) and
    datagen (torch 2.9, vllm 0.11.2).

    Returns:
        Path to RL environment if found, None otherwise.
    """
    # Check common locations
    candidates = []

    # DCFT_RL_ENV explicit override
    if os.environ.get("DCFT_RL_ENV"):
        candidates.append(Path(os.environ["DCFT_RL_ENV"]))

    # Standard location relative to DCFT
    if os.environ.get("DCFT"):
        candidates.append(Path(os.environ["DCFT"]) / "envs" / "rl")

    # Standard location relative to this file
    candidates.append(Path(__file__).parent.parent / "envs" / "rl")

    for candidate in candidates:
        if candidate.exists() and (candidate / "bin" / "activate").exists():
            return candidate

    return None


def launch_rl_job(exp_args: dict, hpc) -> Optional[str]:
    """Launch RL training job using universal template system.

    This is the main entry point for RL job submission from hpc/launch.py.

    Args:
        exp_args: Experiment arguments dictionary from CLI.
        hpc: HPC cluster configuration.

    Returns:
        Job ID if submitted, None if dry_run.
    """
    from hpc.launch_utils import launch_sbatch
    from hpc.rl_config_utils import get_skyrl_command_preview, parse_rl_config, build_skyrl_hydra_args

    # Check for RL environment
    rl_env_path = check_rl_environment()
    if rl_env_path:
        print(f"RL environment found: {rl_env_path}")
    else:
        print("\n" + "=" * 60)
        print("WARNING: RL environment not found!")
        print("The RL environment is required for SkyRL training.")
        print("Create it with: ./hpc/setup_rl_env.sh")
        print("Or set DCFT_RL_ENV to point to an existing environment.")
        print("=" * 60 + "\n")

    # Construct the sbatch script
    sbatch_path = construct_rl_sbatch_script(exp_args, hpc)

    # Dry run handling
    if exp_args.get("dry_run"):
        print(f"\nDRY RUN: RL sbatch script written to {sbatch_path}")

        # Show command preview
        rl_config_path = exp_args.get("rl_config")
        if rl_config_path:
            parsed = parse_rl_config(rl_config_path)
            hydra_args = build_skyrl_hydra_args(parsed, exp_args, hpc)
            skyrl_overrides = exp_args.get("skyrl_override") or []
            hydra_args.extend(skyrl_overrides)
            print("\nSkyRL command preview:")
            print(get_skyrl_command_preview(parsed.entrypoint, hydra_args))

        return None

    # Submit the job
    job_id = launch_sbatch(sbatch_path)
    print(f"\nRL job submitted: {job_id}")

    return job_id


# =============================================================================
# RLJobRunner - Runs within sbatch
# =============================================================================


class RLJobRunner:
    """Runner for RL training jobs executed from sbatch.

    This class is instantiated within the sbatch script and handles:
    - Ray cluster setup (using shared RayCluster utility)
    - Environment configuration
    - SkyRL execution

    Usage (from sbatch):
        python -m hpc.rl_launch_utils --config /path/to/config.json
    """

    def __init__(self, config: RLJobConfig):
        self.config = config
        self._hpc = None

    def _get_hpc(self):
        """Lazy-load HPC configuration."""
        if self._hpc is None:
            from hpc.hpc import detect_hpc, clusters
            if self.config.cluster_name:
                for c in clusters:
                    if c.name.lower() == self.config.cluster_name.lower():
                        self._hpc = c
                        break
                if self._hpc is None:
                    raise ValueError(f"Unknown cluster: {self.config.cluster_name}")
            else:
                self._hpc = detect_hpc()
        return self._hpc

    def run(self) -> int:
        """Execute the RL training job.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        print(f"=== RLJobRunner: {self.config.job_name} ===", flush=True)

        try:
            self._setup_environment()
            return self._run_with_ray()
        except Exception as e:
            print(f"RL job failed: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()
            return 1

    def _setup_environment(self) -> None:
        """Configure environment variables for RL training."""
        # Set common environment variables
        os.environ["TENSOR_PARALLEL_SIZE"] = str(self.config.tensor_parallel_size)
        os.environ["NUM_INFERENCE_ENGINES"] = str(
            compute_num_inference_engines(
                self.config.num_nodes,
                self.config.gpus_per_node,
                self.config.tensor_parallel_size,
            )
        )
        os.environ["POLICY_NUM_NODES"] = str(self.config.num_nodes)

        if self.config.export_path:
            os.environ["SKYRL_EXPORT_PATH"] = self.config.export_path

        # vLLM settings
        os.environ["VLLM_USE_V1"] = "1"

        # Ensure WandB directory is writable
        from hpc.wandb_launch_utils import ensure_wandb_dir
        wandb_dir = ensure_wandb_dir(
            experiments_dir=self.config.experiments_dir,
            verbose=True,
        )
        os.environ["WANDB_DIR"] = wandb_dir

        print(f"Environment configured:", flush=True)
        print(f"  TENSOR_PARALLEL_SIZE={os.environ['TENSOR_PARALLEL_SIZE']}", flush=True)
        print(f"  NUM_INFERENCE_ENGINES={os.environ['NUM_INFERENCE_ENGINES']}", flush=True)
        print(f"  POLICY_NUM_NODES={os.environ['POLICY_NUM_NODES']}", flush=True)
        print(f"  WANDB_DIR={wandb_dir}", flush=True)

    def _run_with_ray(self) -> int:
        """Run SkyRL training with managed Ray cluster.

        Uses RayCluster.from_slurm() to properly start Ray across all SLURM nodes
        using srun, ensuring all nodes join the cluster before training begins.
        """
        from hpc.ray_utils import (
            RayCluster,
            RayClusterConfig,
            compute_ray_memory_from_slurm,
            DEFAULT_OBJECT_STORE_MEMORY_BYTES,
        )

        hpc = self._get_hpc()
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", self.config.num_nodes))

        # Use config values (from CLI overrides) instead of cluster defaults
        gpus_per_node = self.config.gpus_per_node or hpc.gpus_per_node
        cpus_per_node = self.config.cpus_per_node or hpc.cpus_per_node

        # Compute Ray memory limit from SLURM allocation (prevents OOM from over-detection)
        ray_memory = compute_ray_memory_from_slurm()
        if ray_memory:
            print(f"[RLJobRunner] Ray memory limit: {ray_memory / (1024**3):.1f} GB", flush=True)

        ray_cfg = RayClusterConfig(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            cpus_per_node=cpus_per_node,
            ray_port=self.config.ray_port,
            srun_export_env=hpc.get_srun_export_env(),
            ray_env_vars=hpc.get_ray_env_vars(),
            memory_per_node=ray_memory,
            object_store_memory=DEFAULT_OBJECT_STORE_MEMORY_BYTES,
        )

        print(f"Starting Ray cluster with {num_nodes} nodes, {gpus_per_node} GPUs/node", flush=True)

        with RayCluster.from_slurm(ray_cfg) as ray_cluster:
            # Set RAY_ADDRESS for SkyRL to connect
            os.environ["RAY_ADDRESS"] = ray_cluster.address
            print(f"Ray cluster ready at {ray_cluster.address}", flush=True)
            print(f"Total GPUs available: {ray_cluster.total_gpus}", flush=True)

            return self._run_skyrl()

    def _run_skyrl(self) -> int:
        """Execute SkyRL training.

        Returns:
            Exit code from SkyRL process.
        """
        # Build command - use sys.executable to ensure we use the same Python
        # as the current process (respects conda/venv activation)
        cmd = [sys.executable, "-m", self.config.skyrl_entrypoint]
        cmd.extend(self.config.skyrl_hydra_args)

        print(f"\nRunning SkyRL:", flush=True)
        print(f"  Python: {sys.executable}", flush=True)
        print(f"  Entrypoint: {self.config.skyrl_entrypoint}", flush=True)
        print(f"  Args: {len(self.config.skyrl_hydra_args)} Hydra arguments", flush=True)

        # Change to SKYRL_HOME if set
        skyrl_home = os.environ.get("SKYRL_HOME")
        cwd = None
        if skyrl_home:
            cwd = os.path.join(skyrl_home, "skyrl-train")
            if os.path.isdir(cwd):
                print(f"  Working dir: {cwd}", flush=True)
            else:
                cwd = None

        result = subprocess.run(cmd, cwd=cwd)
        return result.returncode


def run_rl_job_main():
    """Entry point for running RL jobs from sbatch.

    This is invoked by the sbatch script via:
        python -m hpc.rl_launch_utils --config /path/to/config.json
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run RL training job")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = json.load(f)

    config = RLJobConfig(**config_dict)
    runner = RLJobRunner(config)
    sys.exit(runner.run())


if __name__ == "__main__":
    run_rl_job_main()
