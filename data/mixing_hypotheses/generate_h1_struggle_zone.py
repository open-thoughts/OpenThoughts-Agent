#!/usr/bin/env python3
"""H1: Struggle Zone — overweight 20-50% pass-rate tasks for maximum RL signal.

Claim: Oversampling 20-50% pass-rate tasks produces better RL signal than uniform sampling.
Why: RL learns fastest when the agent succeeds ~30-50% of the time.

Mix: 70% medium tier (20-49%), 20% easy (>=50%), 10% hard (<20%)
All Python-base datasets, 1 Dockerfile.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from exp_general_utils import mix_datasets
from mixing_hypotheses.dataset_registry import (
    DATASETS, PYTHON_EASY, PYTHON_MEDIUM, PYTHON_HARD,
)

PREFIX = "mix-h1-struggle-zone"
HF_REPO = "DCAgent/mix_h1_struggle_zone"
TOTAL = 5000

# Tier weights
WEIGHT_MEDIUM = 0.70  # 20-49% pass rate — the "struggle zone"
WEIGHT_EASY = 0.20    # >=50% pass rate
WEIGHT_HARD = 0.10    # <20% pass rate

if __name__ == "__main__":
    repos_weights = []

    # Distribute weight equally within each tier
    for tier_name, tier_datasets, tier_weight in [
        ("easy", PYTHON_EASY, WEIGHT_EASY),
        ("medium", PYTHON_MEDIUM, WEIGHT_MEDIUM),
        ("hard", PYTHON_HARD, WEIGHT_HARD),
    ]:
        if not tier_datasets:
            print(f"WARNING: no datasets in {tier_name} tier")
            continue
        per_dataset = tier_weight / len(tier_datasets)
        for d in tier_datasets:
            repos_weights.append((DATASETS[d].hf_repo, per_dataset))
        print(f"{tier_name} tier ({tier_weight:.0%}): {len(tier_datasets)} datasets @ {per_dataset:.4f} each")
        for d in tier_datasets:
            print(f"  {d}: pass_rate={DATASETS[d].pass_rate:.0%}")

    print(f"\nTotal weight: {sum(w for _, w in repos_weights):.4f}")
    mix_datasets(
        source_repos_with_weights=repos_weights,
        total_limit=TOTAL,
        prefix=PREFIX,
        hf_repo=HF_REPO,
    )
