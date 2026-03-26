#!/usr/bin/env python3
"""
C1 Dataset Generation: Annotate sandboxes with different teacher models to generate traces.

Takes the best-ranked B1 model's sandbox datasets and generates new traces using
different teacher models (e.g., GPT-5, Claude, GLM-4.7). Each C1 dataset is 10K traces.

Workflow:
    1. Pick the best B1 model from eval results
    2. Select sandbox datasets to annotate (from the B1 model's constituent A1 datasets)
    3. Run harbor trace generation with each teacher model
    4. Export traces and upload to HF
    5. Train C1 SFT models on the new traces

Usage:
    # Generate traces for a single sandbox dataset with a teacher model:
    python c1_teacher_annotation.py single \
        --sandbox-dataset DCAgent/exp_rpt_stack-bash \
        --teacher-model glm-4.7 \
        --agent terminus-2 \
        --output-size 10000 \
        --push

    # Generate traces for multiple sandbox datasets from a config:
    python c1_teacher_annotation.py batch \
        --config c1_config.yaml \
        --push

    # Generate harbor eval config for SLURM submission:
    python c1_teacher_annotation.py gen-config \
        --config c1_config.yaml \
        --output-dir /e/scratch/jureap59/raoof1/c1_configs/

Config YAML format:
    best_b1_model: DCAgent/b1-top10_equal
    output_size: 10000
    hf_org: DCAgent
    teacher_models:
      - name: glm-4.7
        agent: terminus-2
        agent_kwargs: {}
      - name: gpt-5-mini
        agent: terminus-2
        agent_kwargs: {}
      - name: claude-sonnet-4
        agent: claude-code
        agent_kwargs: {}
    sandbox_datasets:
      - name: stack_bash
        hf_repo: DCAgent/exp_rpt_stack-bash
        n_tasks: 10000
      - name: crosscodeeval_python
        hf_repo: DCAgent/exp_rpt_crosscodeeval-python-v2
        n_tasks: 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_SIZE = 10_000
DEFAULT_HF_ORG = "DCAgent"
DEFAULT_JOBS_DIR = "/e/scratch/jureap59/raoof1/c1_jobs"
DEFAULT_N_CONCURRENT = 4
DEFAULT_N_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Harbor config generation
# ---------------------------------------------------------------------------

def generate_harbor_config(
    sandbox_dataset: str,
    teacher_model: str,
    agent: str = "terminus-2",
    agent_kwargs: dict | None = None,
    n_concurrent: int = DEFAULT_N_CONCURRENT,
    n_attempts: int = DEFAULT_N_ATTEMPTS,
    n_tasks: int | None = None,
    jobs_dir: str = DEFAULT_JOBS_DIR,
    export_repo: str | None = None,
) -> dict:
    """Generate a harbor job config for trace annotation."""
    config = {
        "jobs_dir": jobs_dir,
        "n_attempts": n_attempts,
        "timeout_multiplier": 1.0,
        "orchestrator": {
            "type": "local",
            "n_concurrent_trials": n_concurrent,
            "quiet": False,
            "retry": {
                "max_retries": 10,
                "exclude_exceptions": [
                    "AgentTimeoutError",
                    "EnvironmentStartTimeoutError",
                    "SandboxBuildFailedError",
                    "VerifierTimeoutError",
                    "SummarizationTimeout",
                    "SummarizationTimeoutError",
                    "ContextLengthExceededError",
                ],
                "wait_multiplier": 1.0,
                "min_wait_sec": 1.0,
                "max_wait_sec": 60.0,
            },
        },
        "environment": {
            "type": "daytona",
            "force_build": False,
            "delete": False,
            "kwargs": {"auto_snapshot": True},
        },
        "agents": [
            {
                "name": agent,
                "model": teacher_model,
                "trajectory_config": {
                    "raw_content": True,
                    "linear_history": True,
                },
            }
        ],
        "datasets": [
            {"path": sandbox_dataset},
        ],
    }

    if agent_kwargs:
        config["agents"][0]["kwargs"] = agent_kwargs

    if n_tasks:
        config["n_tasks"] = n_tasks

    if export_repo:
        config["export"] = {
            "push": True,
            "repo": export_repo,
        }

    return config


def generate_sbatch_script(
    config_path: str,
    job_name: str,
    account: str = "reformo",
    time_limit: str = "12:00:00",
    nodes: int = 1,
) -> str:
    """Generate an sbatch script for running harbor trace generation."""
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition=booster
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --time={time_limit}
#SBATCH --output=/e/scratch/jureap59/raoof1/c1_logs/{job_name}_%j.out

source /e/scratch/jureap59/feuer1/miniforge3/etc/profile.d/conda.sh
conda activate otagent
source ~/secrets.env
export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH:-}}"

# Proxy for Daytona access
source /e/scratch/jureap59/feuer1/harbor/src/hpc/dotenv/jupiter.env 2>/dev/null || true

export DAYTONA_API_KEY="${{DAYTONA_API_KEY}}"
export DAYTONA_MAX_RETRIES=5
export DAYTONA_RETRY_DELAY=30

echo "=== C1 Trace Generation: {job_name} ==="
echo "Config: {config_path}"
echo "Started: $(date)"

harbor run -c {config_path}

echo "Finished: $(date)"
"""


# ---------------------------------------------------------------------------
# Single annotation run
# ---------------------------------------------------------------------------

def run_single(args):
    """Run trace gen for one sandbox dataset with one teacher model."""
    teacher_safe = args.teacher_model.replace("/", "_").replace("-", "_")
    sandbox_safe = args.sandbox_dataset.split("/")[-1]
    c1_name = f"c1_{sandbox_safe}_{teacher_safe}_traces"
    export_repo = f"{args.hf_org}/{c1_name}" if args.push else None

    config = generate_harbor_config(
        sandbox_dataset=args.sandbox_dataset,
        teacher_model=args.teacher_model,
        agent=args.agent,
        n_tasks=args.output_size,
        jobs_dir=args.jobs_dir,
        export_repo=export_repo,
    )

    config_path = Path(args.jobs_dir) / f"{c1_name}_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    log.info(f"Config written to {config_path}")

    if args.submit:
        sbatch = generate_sbatch_script(str(config_path), c1_name)
        sbatch_path = Path(args.jobs_dir) / f"{c1_name}.sbatch"
        with open(sbatch_path, "w") as f:
            f.write(sbatch)

        result = subprocess.run(["sbatch", str(sbatch_path)], capture_output=True, text=True)
        log.info(f"Submitted: {result.stdout.strip()}")
    else:
        log.info(f"Config ready. Submit with: harbor run -c {config_path}")


# ---------------------------------------------------------------------------
# Batch annotation from config
# ---------------------------------------------------------------------------

def run_batch(args):
    """Generate configs + sbatch scripts for all teacher × sandbox combinations."""
    with open(args.config) as f:
        config = yaml.safe_load(f)

    hf_org = config.get("hf_org", DEFAULT_HF_ORG)
    output_size = config.get("output_size", DEFAULT_OUTPUT_SIZE)
    jobs_dir = config.get("jobs_dir", DEFAULT_JOBS_DIR)
    teachers = config["teacher_models"]
    sandboxes = config["sandbox_datasets"]

    os.makedirs(jobs_dir, exist_ok=True)
    os.makedirs(f"{jobs_dir}/../c1_logs", exist_ok=True)

    generated = []

    for teacher in teachers:
        teacher_name = teacher["name"]
        agent = teacher.get("agent", "terminus-2")
        agent_kwargs = teacher.get("agent_kwargs", {})
        teacher_safe = teacher_name.replace("/", "_").replace("-", "_").replace(".", "_")

        for sandbox in sandboxes:
            sandbox_name = sandbox["name"]
            sandbox_repo = sandbox["hf_repo"]
            n_tasks = sandbox.get("n_tasks", output_size)

            c1_name = f"c1_{sandbox_name}_{teacher_safe}_traces"
            export_repo = f"{hf_org}/{c1_name}" if args.push else None

            harbor_config = generate_harbor_config(
                sandbox_dataset=sandbox_repo,
                teacher_model=teacher_name,
                agent=agent,
                agent_kwargs=agent_kwargs,
                n_tasks=n_tasks,
                jobs_dir=jobs_dir,
                export_repo=export_repo,
            )

            config_path = Path(jobs_dir) / f"{c1_name}_config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(harbor_config, f, default_flow_style=False)

            sbatch_content = generate_sbatch_script(str(config_path), c1_name)
            sbatch_path = Path(jobs_dir) / f"{c1_name}.sbatch"
            with open(sbatch_path, "w") as f:
                f.write(sbatch_content)

            generated.append({
                "name": c1_name,
                "teacher": teacher_name,
                "sandbox": sandbox_name,
                "config": str(config_path),
                "sbatch": str(sbatch_path),
            })
            log.info(f"Generated: {c1_name}")

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"Generated {len(generated)} C1 configs ({len(teachers)} teachers × {len(sandboxes)} sandboxes)")
    log.info(f"Configs in: {jobs_dir}/")

    if args.submit:
        log.info("Submitting all jobs...")
        for g in generated:
            result = subprocess.run(["sbatch", g["sbatch"]], capture_output=True, text=True)
            job_id = result.stdout.strip()
            log.info(f"  {g['name']}: {job_id}")
    else:
        log.info("To submit all:")
        log.info(f"  for f in {jobs_dir}/c1_*.sbatch; do sbatch $f; done")

    # Write manifest
    manifest_path = Path(jobs_dir) / "c1_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(generated, f, indent=2)
    log.info(f"Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# Generate config template
# ---------------------------------------------------------------------------

def run_gen_config(args):
    """Generate a template C1 config YAML."""
    template = {
        "best_b1_model": "DCAgent/b1-top10_equal",
        "output_size": 10_000,
        "hf_org": "DCAgent",
        "jobs_dir": DEFAULT_JOBS_DIR,
        "teacher_models": [
            {"name": "glm-4.7", "agent": "terminus-2"},
            {"name": "gpt-5-mini", "agent": "terminus-2"},
            {"name": "claude-sonnet-4", "agent": "claude-code"},
        ],
        "sandbox_datasets": [
            {"name": "stack_bash", "hf_repo": "DCAgent/exp_rpt_stack-bash", "n_tasks": 10000},
            {"name": "crosscodeeval_python", "hf_repo": "DCAgent/exp_rpt_crosscodeeval-python-v2", "n_tasks": 10000},
        ],
    }

    output = Path(args.output)
    with open(output, "w") as f:
        yaml.dump(template, f, default_flow_style=False)
    log.info(f"Template config written to {output}")
    log.info("Edit the teacher_models and sandbox_datasets, then run:")
    log.info(f"  python c1_teacher_annotation.py batch --config {output} --push --submit")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="C1: Generate traces with different teacher models")
    sub = p.add_subparsers(dest="mode", required=True)

    # Single run
    sp = sub.add_parser("single", help="Annotate one sandbox with one teacher")
    sp.add_argument("--sandbox-dataset", required=True, help="HF sandbox dataset repo")
    sp.add_argument("--teacher-model", required=True, help="Teacher model name")
    sp.add_argument("--agent", default="terminus-2", help="Agent to use")
    sp.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE)
    sp.add_argument("--hf-org", default=DEFAULT_HF_ORG)
    sp.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)

    # Batch run
    bp = sub.add_parser("batch", help="Batch annotate from config")
    bp.add_argument("--config", required=True, help="C1 config YAML")

    # Generate template
    gp = sub.add_parser("gen-config", help="Generate template C1 config")
    gp.add_argument("--output", default="c1_config.yaml", help="Output path")

    for s in [sp, bp]:
        s.add_argument("--push", action="store_true", help="Push traces to HF")
        s.add_argument("--submit", action="store_true", help="Submit SLURM jobs")

    args = p.parse_args()
    {"single": run_single, "batch": run_batch, "gen-config": run_gen_config}[args.mode](args)


if __name__ == "__main__":
    main()
