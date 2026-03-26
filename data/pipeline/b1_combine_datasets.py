#!/usr/bin/env python3
"""
B1 Dataset Generation: Combine top-ranked A1 trace datasets into mixed training sets.

Given a ranking of A1 datasets (from eval scores), creates B1 combined datasets
by mixing traces from the top-N A1 datasets. Each B1 dataset is 10K samples.

Workflow:
    1. Run A1 evals to get per-dataset scores
    2. Create rankings.yaml with scores
    3. Run this script to generate B1 combined datasets
    4. Train B1 models using the SFT pipeline

Usage:
    # From rankings (auto top-N):
    python b1_combine_datasets.py rankings --rankings rankings.yaml --top-n 5 10 20 --push

    # From explicit config:
    python b1_combine_datasets.py config --config b1_config.yaml --push

Rankings YAML format:
    datasets:
      - name: a1_stack_bash
        trace_dataset: DCAgent/exp_rpt_stack-bash_glm_4.7_traces_jupiter
        eval_score: 0.45
      - name: a1_crosscodeeval_python
        trace_dataset: DCAgent/exp_rpt_crosscodeeval-python-v2_10k_glm_4.7_traces_jupiter
        eval_score: 0.42

Config YAML format:
    output_size: 10000
    hf_org: DCAgent
    combinations:
      - name: b1_top5_equal
        datasets: [a1_stack_bash, a1_crosscodeeval_python, a1_bugsinpy, a1_nemotron_bash, a1_manybugs]
        strategy: equal
      - name: b1_top10_weighted
        datasets: [...]
        strategy: weighted
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_SIZE = 10_000
DEFAULT_HF_ORG = "DCAgent"
DEFAULT_CSV = Path.home() / "datagen" / "dataset_mappings.csv"


def load_a1_registry(csv_path: str | Path | None = None) -> dict[str, str]:
    """Load A1 name -> HF trace dataset mapping from the dataset mappings CSV."""
    csv_path = Path(csv_path) if csv_path else DEFAULT_CSV
    registry = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "").strip() == "ON HF":
                link = row["hf_sandbox_or_trace_link"].strip()
                if link and "{" not in link:
                    registry[row["a1_name"].strip()] = link
    log.info(f"Loaded {len(registry)} A1 datasets from {csv_path}")
    return registry


# ---------------------------------------------------------------------------
# Mixing strategies
# ---------------------------------------------------------------------------

def mix_equal(datasets: dict[str, Dataset], target_size: int, seed: int = 42) -> Dataset:
    """Equal samples from each dataset. Upsamples small datasets if needed."""
    n = len(datasets)
    per_ds = target_size // n
    remainder = target_size % n
    rng = random.Random(seed)

    parts = []
    for i, (name, ds) in enumerate(datasets.items()):
        take = per_ds + (1 if i < remainder else 0)
        if len(ds) >= take:
            indices = rng.sample(range(len(ds)), take)
        else:
            indices = list(range(len(ds))) * (take // len(ds))
            indices += rng.sample(range(len(ds)), take - len(indices))
        parts.append(ds.select(indices))
        log.info(f"  {name}: {take} samples (from {len(ds)})")

    return concatenate_datasets(parts).shuffle(seed=seed)


def mix_weighted(datasets: dict[str, Dataset], weights: dict[str, float],
                 target_size: int, seed: int = 42) -> Dataset:
    """Sample proportional to eval scores or explicit weights."""
    total_w = sum(weights.get(name, 1.0) for name in datasets)
    rng = random.Random(seed)
    parts, allocated = [], 0
    names = list(datasets.keys())

    for i, name in enumerate(names):
        w = weights.get(name, 1.0) / total_w
        take = target_size - allocated if i == len(names) - 1 else int(target_size * w)
        allocated += take

        ds = datasets[name]
        if len(ds) >= take:
            indices = rng.sample(range(len(ds)), take)
        else:
            indices = list(range(len(ds))) * (take // len(ds))
            indices += rng.sample(range(len(ds)), take - len(indices))
        parts.append(ds.select(indices))
        log.info(f"  {name}: {take} samples (weight={w:.3f})")

    return concatenate_datasets(parts).shuffle(seed=seed)


def mix_proportional(datasets: dict[str, Dataset], target_size: int, seed: int = 42) -> Dataset:
    """Proportional to original dataset sizes."""
    total = sum(len(ds) for ds in datasets.values())
    weights = {name: len(ds) / total for name, ds in datasets.items()}
    return mix_weighted(datasets, weights, target_size, seed)


STRATEGIES = {"equal": mix_equal, "weighted": mix_weighted, "proportional": mix_proportional}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def create_b1_dataset(a1_names: list[str], registry: dict[str, str],
                      strategy: str = "equal", target_size: int = DEFAULT_OUTPUT_SIZE,
                      weights: dict[str, float] | None = None, seed: int = 42) -> Dataset:
    """Load A1 traces, mix them, return a single B1 dataset."""
    log.info(f"B1: combining {len(a1_names)} datasets, strategy={strategy}, target={target_size}")

    loaded = {}
    for name in a1_names:
        repo = registry.get(name)
        if not repo:
            log.warning(f"  {name}: not in registry, skipping")
            continue
        ds = load_dataset(repo, split="train")
        loaded[name] = ds
        log.info(f"  {name}: {len(ds)} rows")

    if not loaded:
        raise ValueError("No datasets loaded — check names and registry")

    if strategy == "weighted" and weights:
        return mix_weighted(loaded, weights, target_size, seed)
    if strategy == "proportional":
        return mix_proportional(loaded, target_size, seed)
    return mix_equal(loaded, target_size, seed)


def upload_b1(dataset: Dataset, name: str, hf_org: str = DEFAULT_HF_ORG, push: bool = True) -> str:
    repo_id = f"{hf_org}/{name}"
    if push:
        log.info(f"Pushing {len(dataset)} rows to {repo_id}...")
        dataset.push_to_hub(repo_id, private=False)
        log.info(f"Uploaded: https://huggingface.co/datasets/{repo_id}")
    return repo_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_rankings(args):
    with open(args.rankings) as f:
        rankings = yaml.safe_load(f)

    registry = load_a1_registry(args.csv)
    ranked = sorted(rankings["datasets"], key=lambda x: x.get("eval_score", 0), reverse=True)

    for n in args.top_n:
        top = ranked[:n]
        a1_names = [d["name"] for d in top]
        weights = {d["name"]: d["eval_score"] for d in top} if args.strategy == "weighted" else None
        b1_name = f"b1_top{n}_{args.strategy}"

        log.info(f"\n{'='*60}\nCreating {b1_name} from top-{n}\n{'='*60}")
        ds = create_b1_dataset(a1_names, registry, args.strategy, args.output_size, weights, args.seed)
        upload_b1(ds, b1_name, args.hf_org, args.push)


def run_config(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    registry = load_a1_registry(args.csv)
    output_size = config.get("output_size", DEFAULT_OUTPUT_SIZE)
    hf_org = config.get("hf_org", DEFAULT_HF_ORG)

    for combo in config["combinations"]:
        name = combo["name"]
        a1_names = combo["datasets"]
        strategy = combo.get("strategy", "equal")
        weights = combo.get("weights")

        log.info(f"\n{'='*60}\nCreating {name}\n{'='*60}")
        ds = create_b1_dataset(a1_names, registry, strategy, output_size, weights, args.seed)
        upload_b1(ds, name, hf_org, args.push)


def main():
    p = argparse.ArgumentParser(description="B1: Combine top-ranked A1 trace datasets")
    sub = p.add_subparsers(dest="mode", required=True)

    rp = sub.add_parser("rankings", help="Auto top-N from ranked list")
    rp.add_argument("--rankings", required=True, help="Rankings YAML path")
    rp.add_argument("--top-n", type=int, nargs="+", default=[5, 10, 20])
    rp.add_argument("--strategy", default="equal", choices=STRATEGIES)
    rp.add_argument("--output-size", type=int, default=DEFAULT_OUTPUT_SIZE)
    rp.add_argument("--hf-org", default=DEFAULT_HF_ORG)

    cp = sub.add_parser("config", help="Explicit B1 combinations")
    cp.add_argument("--config", required=True, help="B1 config YAML path")

    for sp in [rp, cp]:
        sp.add_argument("--csv", default=None, help="dataset_mappings.csv path")
        sp.add_argument("--push", action="store_true", help="Push to HuggingFace")
        sp.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    {"rankings": run_rankings, "config": run_config}[args.mode](args)


if __name__ == "__main__":
    main()
