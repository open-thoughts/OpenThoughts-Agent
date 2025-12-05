"""Utilities for launching consolidation jobs on HPC."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Callable

from hpc.launch_utils import sanitize_repo_for_job

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _derive_consolidate_preamble(hpc) -> str:
    """
    Attempt to reuse the environment setup section from the HPC training template so
    consolidate jobs inherit module loads and activation commands.
    """
    template_path = getattr(hpc, "train_sbatch_path", None)
    if not template_path or not os.path.exists(template_path):
        return ""

    lines: list[str] = []
    started = False
    try:
        with open(template_path, "r") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")
                if line.startswith("#SBATCH") or line.startswith("#!/bin/bash"):
                    continue
                striped = line.strip()
                if not started:
                    if not striped:
                        continue
                    started = True
                if "sft/llamafactory" in striped or striped.startswith("CONFIG="):
                    break
                lines.append(line)
    except Exception:
        return ""

    return "\n".join(lines).strip()


def launch_consolidate_job(
    exp_args: dict,
    hpc,
    *,
    update_exp_args_fn: Callable[[dict, dict], dict],
    launch_sbatch_fn: Callable[[str], str],
) -> Optional[str]:
    """Launch a consolidation job and return the submitted job id."""

    print("\n=== CONSOLIDATE MODE ===")

    repo_id = exp_args.get("consolidate_repo_id")
    workdir = exp_args.get("consolidate_workdir")
    if not repo_id:
        raise ValueError("--consolidate_repo_id is required for consolidate jobs")
    if not workdir:
        raise ValueError("--consolidate_workdir is required for consolidate jobs")

    base_repo = exp_args.get("consolidate_base_repo")
    commit_message = exp_args.get("consolidate_commit_message") or "Merge ZeRO shards into safetensors"

    job_name = exp_args.get("job_name")
    if not job_name:
        job_name = f"{sanitize_repo_for_job(repo_id)}_consolidate"
        if len(job_name) > 96:
            job_name = job_name[:96]
        exp_args = update_exp_args_fn(exp_args, {"job_name": job_name})

    experiments_dir = exp_args.get("experiments_dir") or "experiments"
    logs_dir = os.path.join(experiments_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    exp_args = update_exp_args_fn(exp_args, {"logs_dir": logs_dir})

    sbatch_dir = os.path.join(experiments_dir, "sbatch_scripts")
    os.makedirs(sbatch_dir, exist_ok=True)
    sbatch_path = os.path.join(sbatch_dir, f"{job_name}.sbatch")

    partition = exp_args.get("partition") or getattr(hpc, "partition", "")
    account = exp_args.get("account") or getattr(hpc, "account", "")
    time_limit = exp_args.get("time_limit") or os.environ.get("DEFAULT_TIME_LIMIT", "24:00:00")

    cpus_per_task = int(exp_args.get("cpus_per_task") or getattr(hpc, "cpus_per_node", 1) or 1)
    mem_per_node = getattr(hpc, "mem_per_node", "") or ""
    mem_directive = f"#SBATCH --mem={mem_per_node}" if mem_per_node else "#SBATCH --mem=0"

    output_path = os.path.join(logs_dir, f"{job_name}_%j.out")
    hpc_name = getattr(hpc, "name", "").lower()
    gpu_directive = ""
    if hpc_name not in {"vista", "lonestar"}:
        gpu_directive = "#SBATCH --gpus-per-node=1"

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
        "environment_preamble": environment_preamble,
        "consolidate_repo_id": repo_id,
        "consolidate_base_repo": base_repo or "",
        "consolidate_workdir": workdir,
        "consolidate_commit_message": commit_message,
        "project_root": PROJECT_ROOT,
        "python_script": python_script_path,
    }

    with open(template_path, "r") as fh:
        template = fh.read()

    script_content = template.format(**substitutions)

    with open(sbatch_path, "w") as f:
        f.write(script_content)
    print(f"Wrote consolidation sbatch to {sbatch_path}")

    if exp_args.get("dry_run"):
        print("DRY RUN: Would submit consolidation job")
        return None

    job_id = launch_sbatch_fn(sbatch_path)
    print(f"✓ Consolidation job submitted: {job_id}")
    return job_id


__all__ = [
    "_derive_consolidate_preamble",
    "launch_consolidate_job",
    "PROJECT_ROOT",
]
