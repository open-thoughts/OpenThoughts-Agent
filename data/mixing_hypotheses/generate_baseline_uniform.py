#!/usr/bin/env python3
"""Baseline: Uniform random mix from all Python-base datasets.

Samples uniformly from all 20 Python-base datasets with binary reward.
Every hypothesis is compared against this baseline.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from exp_general_utils import mix_datasets
from mixing_hypotheses.dataset_registry import PYTHON_BASE, DATASETS

PREFIX = "mix-baseline-uniform"
HF_REPO = "DCAgent/mix_baseline_uniform"
TOTAL = 5000

if __name__ == "__main__":
    # Equal weight per dataset
    n = len(PYTHON_BASE)
    weight = 1.0 / n
    repos_weights = [(DATASETS[d].hf_repo, weight) for d in PYTHON_BASE]

    print(f"Baseline: uniform mix from {n} Python-base datasets, {TOTAL} total tasks")
    for name in sorted(PYTHON_BASE):
        print(f"  {name}: pass_rate={DATASETS[name].pass_rate:.0%}, weight={weight:.3f}")

    mix_datasets(
        source_repos_with_weights=repos_weights,
        total_limit=TOTAL,
        prefix=PREFIX,
        hf_repo=HF_REPO,
    )
