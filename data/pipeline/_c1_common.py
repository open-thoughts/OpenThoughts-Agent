#!/usr/bin/env python3
"""
Shared runner for C1 trace generation scripts.
Each c1_*.py file defines a teacher model and calls run_c1_pipeline().
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

import yaml
from c1_teacher_annotation import generate_harbor_config, generate_sbatch_script

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_SIZE = 10_000
DEFAULT_JOBS_DIR = "/e/scratch/jureap59/raoof1/c1_jobs"
DEFAULT_HF_ORG = "DCAgent"


def run_c1_pipeline(
    teacher_model: str,
    agent: str,
    c1_prefix: str,
    agent_kwargs: dict | None = None,
):
    """Run C1 trace generation for a specific teacher model."""

    p = argparse.ArgumentParser(description=f"C1 trace generation with {teacher_model}")
    p.add_argument(
        "--sandbox-datasets", nargs="+", required=True,
        help="HF sandbox dataset repos to annotate",
    )
    p.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE)
    p.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    p.add_argument("--hf-org", default=DEFAULT_HF_ORG)
    p.add_argument("--push", action="store_true", help="Push traces to HF")
    p.add_argument("--submit", action="store_true", help="Submit SLURM jobs")
    p.add_argument("--n-concurrent", type=int, default=4)
    p.add_argument("--time-limit", default="12:00:00")
    args = p.parse_args()

    os.makedirs(args.jobs_dir, exist_ok=True)
    logs_dir = str(Path(args.jobs_dir).parent / "c1_logs")
    os.makedirs(logs_dir, exist_ok=True)

    generated = []

    for sandbox_repo in args.sandbox_datasets:
        sandbox_name = sandbox_repo.split("/")[-1]
        c1_name = f"{c1_prefix}_{sandbox_name}_traces"
        export_repo = f"{args.hf_org}/{c1_name}" if args.push else None

        config = generate_harbor_config(
            sandbox_dataset=sandbox_repo,
            teacher_model=teacher_model,
            agent=agent,
            agent_kwargs=agent_kwargs,
            n_concurrent=args.n_concurrent,
            n_tasks=args.output_size,
            jobs_dir=args.jobs_dir,
            export_repo=export_repo,
        )

        config_path = Path(args.jobs_dir) / f"{c1_name}_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        sbatch_content = generate_sbatch_script(
            str(config_path), c1_name, time_limit=args.time_limit,
        )
        sbatch_path = Path(args.jobs_dir) / f"{c1_name}.sbatch"
        with open(sbatch_path, "w") as f:
            f.write(sbatch_content)

        generated.append({"name": c1_name, "config": str(config_path), "sbatch": str(sbatch_path)})
        log.info(f"Generated: {c1_name}")

    log.info(f"\n{'='*60}")
    log.info(f"{c1_prefix}: {len(generated)} configs generated")

    if args.submit:
        for g in generated:
            result = subprocess.run(["sbatch", g["sbatch"]], capture_output=True, text=True)
            log.info(f"  {g['name']}: {result.stdout.strip()}")
    else:
        log.info(f"To submit: for f in {args.jobs_dir}/{c1_prefix}_*.sbatch; do sbatch $f; done")

    manifest = Path(args.jobs_dir) / f"{c1_prefix}_manifest.json"
    with open(manifest, "w") as f:
        json.dump(generated, f, indent=2)
