#!/usr/bin/env python3
"""
D1 Mixing: Combine tasks from different sandbox sources into one dataset.

Unlike B1 (which mixes traces), D1 mix operates on sandboxes — combining
task instructions from different domains before trace generation.

Usage:
    python d1_mix.py \
        --sandbox-datasets DCAgent/stackexchange-tezos-sandboxes DCAgent/exp_rpt_stack-bash \
        --output-repo DCAgent/d1-mix-tezos-bash \
        --output-size 10000 \
        --push
"""

from __future__ import annotations

import argparse
import io
import logging
import random
import tarfile
import tempfile
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from data.commons import upload_tasks_to_hf
from d1_harden import extract_instructions_from_sandbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(description="D1: Mix tasks from multiple sandbox sources")
    p.add_argument("--sandbox-datasets", nargs="+", required=True, help="HF sandbox repos to mix")
    p.add_argument("--output-repo", required=True)
    p.add_argument("--output-size", type=int, default=10000)
    p.add_argument("--strategy", default="equal", choices=["equal", "proportional"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--push", action="store_true")
    args = p.parse_args()

    rng = random.Random(args.seed)
    all_tasks = {}

    for repo in args.sandbox_datasets:
        log.info(f"Loading {repo}...")
        ds = load_dataset(repo, split="train")
        tasks = extract_instructions_from_sandbox(ds)
        all_tasks[repo] = tasks
        log.info(f"  {len(tasks)} tasks")

    # Sample from each source
    n_sources = len(all_tasks)
    out_dir = Path(tempfile.mkdtemp(prefix="d1_mix_"))
    idx = 0

    if args.strategy == "equal":
        per_source = args.output_size // n_sources
        remainder = args.output_size % n_sources

        for i, (repo, tasks) in enumerate(all_tasks.items()):
            take = per_source + (1 if i < remainder else 0)
            if len(tasks) >= take:
                sampled = rng.sample(tasks, take)
            else:
                sampled = tasks * (take // len(tasks))
                sampled += rng.sample(tasks, take - len(sampled))
            source_name = repo.split("/")[-1][:20]

            for task in sampled:
                task_dir = out_dir / f"mixed-{idx:05d}-{source_name}"
                task_dir.mkdir()
                for name, content in task["files"].items():
                    fpath = task_dir / name
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_bytes(content)
                idx += 1
            log.info(f"  {repo}: {take} tasks sampled")

    elif args.strategy == "proportional":
        total = sum(len(t) for t in all_tasks.values())
        for repo, tasks in all_tasks.items():
            take = int(args.output_size * len(tasks) / total)
            sampled = rng.sample(tasks, min(take, len(tasks)))
            source_name = repo.split("/")[-1][:20]
            for task in sampled:
                task_dir = out_dir / f"mixed-{idx:05d}-{source_name}"
                task_dir.mkdir()
                for name, content in task["files"].items():
                    fpath = task_dir / name
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    fpath.write_bytes(content)
                idx += 1
            log.info(f"  {repo}: {len(sampled)} tasks sampled")

    log.info(f"Mixed {idx} tasks in {out_dir}")

    if args.push:
        upload_tasks_to_hf(str(out_dir), args.output_repo)

    log.info("Done!")


if __name__ == "__main__":
    main()
