"""Utilities for launching consolidation jobs on HPC."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Callable

from hpc.launch_utils import (
    sanitize_repo_for_job,
    resolve_job_and_paths,
    parse_bool_with_default,
    scale_memory_for_partial_gpus,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _derive_consolidate_preamble(hpc) -> str:
    """
    Generate environment setup preamble for consolidate jobs using HPC config.
    """
    lines: list[str] = []

    # Add module commands
    module_commands = getattr(hpc, "get_module_commands", lambda: "")()
    if module_commands and module_commands.strip():
        lines.append(module_commands)

    # Add conda activation
    conda_activate = getattr(hpc, "conda_activate", "")
    if conda_activate and conda_activate.strip():
        lines.append(conda_activate)

    return "\n".join(lines).strip()


def launch_consolidate_job(
    exp_args: dict,
    hpc,
    *,
    update_exp_args_fn: Callable[[dict, dict], dict],
    launch_sbatch_fn: Callable[..., str],  # Accepts (path, dependency=None)
) -> Optional[str]:
    """Launch a consolidation job and return the submitted job id."""

    print("\n=== CONSOLIDATE MODE ===")

    input_value = exp_args.get("consolidate_input")
    workdir = exp_args.get("consolidate_workdir")
    if not input_value:
        raise ValueError("--consolidate_input is required for consolidate jobs")
    if not workdir:
        raise ValueError("--consolidate_workdir is required for consolidate jobs")

    base_repo = exp_args.get("consolidate_base_repo")
    output_repo = exp_args.get("consolidate_output_repo")
    commit_message = exp_args.get("consolidate_commit_message") or "Merge ZeRO shards into safetensors"
    push_to_hub_flag = parse_bool_with_default(exp_args.get("push_to_hub"), default=True)

    expanded_input = Path(str(input_value)).expanduser()
    input_is_local = False
    if str(input_value).startswith(("/", "./", "../", "~")) or expanded_input.exists():
        input_is_local = True
        if not expanded_input.exists():
            print(
                f"Warning: --consolidate_input={input_value} was treated as a local path but does not currently exist.",
                "The job may fail if the path is unavailable on compute nodes.",
            )
    if input_is_local:
        if not output_repo:
            raise ValueError("--consolidate_output_repo is required when --consolidate_input points to a local path")
    else:
        if not output_repo:
            raise ValueError(
                "--consolidate_output_repo is required when --consolidate_input references a Hugging Face repo so the merged weights upload to a new repo."
            )
        if output_repo.strip() == str(input_value).strip():
            raise ValueError(
                "When --consolidate_input and --consolidate_output_repo both reference Hugging Face repos, they must differ."
            )
    effective_output_repo = output_repo
    input_kind = "local" if input_is_local else "repo"

    # Define local derive function that captures input_value
    def _derive_consolidate_job_name(_: dict) -> str:
        identifier = sanitize_repo_for_job(str(input_value))
        name = f"{identifier}_consolidate"
        return name[:96] if len(name) > 96 else name

    # Resolve job_name and paths
    job_setup = resolve_job_and_paths(
        exp_args,
        job_type_label="Consolidate",
        derive_job_name_fn=_derive_consolidate_job_name,
        sbatch_subdir="sbatch_scripts",
    )
    job_name = job_setup.job_name
    exp_paths = job_setup.paths

    # Update exp_args with derived job_name if it was auto-derived
    if not exp_args.get("job_name"):
        exp_args = update_exp_args_fn(exp_args, {"job_name": job_name})
    experiments_dir = str(exp_paths.root)
    exp_args = update_exp_args_fn(exp_args, {"logs_dir": str(exp_paths.logs)})

    hpc_name = getattr(hpc, "name", "").lower()

    sbatch_path = exp_paths.sbatch / f"{job_name}.sbatch"

    partition = exp_args.get("partition") or getattr(hpc, "partition", "")
    account = exp_args.get("account") or getattr(hpc, "account", "")
    time_limit = exp_args.get("time_limit") or getattr(hpc, "default_time_limit", "24:00:00")

    cpus_per_task = int(exp_args.get("cpus_per_task") or getattr(hpc, "cpus_per_node", 1) or 1)
    if hpc_name == "capella":
        cpus_per_task = min(cpus_per_task, 14)

    # Scale memory proportionally when requesting fewer GPUs than available on node
    # (required by ZIH Capella and other schedulers for fair sharing)
    mem_per_node = getattr(hpc, "mem_per_node", "") or ""
    gpus_requested = 1  # consolidate jobs use 1 GPU
    total_gpus = getattr(hpc, "gpus_per_node", 1) or 1
    if mem_per_node and gpus_requested < total_gpus:
        scaled_mem = scale_memory_for_partial_gpus(mem_per_node, gpus_requested, total_gpus)
        mem_directive = f"#SBATCH --mem={scaled_mem}"
    elif mem_per_node:
        mem_directive = f"#SBATCH --mem={mem_per_node}"
    else:
        mem_directive = "#SBATCH --mem=0"

    output_path = exp_paths.logs / f"{job_name}_%j.out"
    gpu_directive = "#SBATCH --gpus-per-node=1"
    if hpc_name in {"vista", "lonestar"}:
        gpu_directive = ""

    template_dir = os.path.join(os.path.dirname(__file__), "sbatch_consolidate")
    cluster_template = os.path.join(template_dir, f"{hpc_name}_consolidate.sbatch")
    if os.path.exists(cluster_template):
        template_path = cluster_template
        environment_preamble = ""
    else:
        template_path = os.path.join(template_dir, "consolidate_template.sbatch")
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Consolidate sbatch template not found at {template_path}")
        environment_preamble = _derive_consolidate_preamble(hpc).strip()

    python_script_path = os.path.join(os.path.dirname(__file__), "sbatch_consolidate", "consolidate.py")
    if not os.path.exists(python_script_path):
        raise FileNotFoundError(f"Consolidate helper script not found at {python_script_path}")

    substitutions = {
        "partition_directive": f"#SBATCH -p {partition}" if partition else "",
        "account_directive": f"#SBATCH --account {account}" if account else "",
        "time_limit": time_limit,
        "cpus_per_task": cpus_per_task,
        "gpu_directive": gpu_directive,
        "mem_directive": mem_directive,
        "job_name": job_name,
        "output_path": output_path,
        "experiments_dir": experiments_dir,
        "environment_preamble": environment_preamble,
        "consolidate_input": input_value,
        "consolidate_input_kind": input_kind,
        "consolidate_base_repo": base_repo or "",
        "consolidate_workdir": workdir,
        "consolidate_commit_message": commit_message,
        "consolidate_output_repo": effective_output_repo,
        "consolidate_push_to_hub": "true" if push_to_hub_flag else "false",
        "project_root": PROJECT_ROOT,
        "python_script": python_script_path,
    }

    with open(template_path, "r") as fh:
        template = fh.read()

    script_content = template.format(**substitutions)

    with open(sbatch_path, "w") as f:
        f.write(script_content)
    print(f"Wrote consolidation sbatch to {sbatch_path}")

    # Get dependency if specified
    dependency = exp_args.get("dependency")

    if exp_args.get("dry_run"):
        print("DRY RUN: Would submit consolidation job")
        if dependency:
            print(f"  Would submit with dependency: {dependency}")
        return None

    job_id = launch_sbatch_fn(sbatch_path, dependency=dependency)
    print(f"✓ Consolidation job submitted: {job_id}")
    return job_id


__all__ = [
    "_derive_consolidate_preamble",
    "launch_consolidate_job",
    "PROJECT_ROOT",
]
