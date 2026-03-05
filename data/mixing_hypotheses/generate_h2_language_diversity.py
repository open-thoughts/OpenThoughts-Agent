#!/usr/bin/env python3
"""H2: Language Diversity Bonus — language-balanced sampling across 6 families.

Claim: Language-balanced sampling improves generalization vs Python-heavy sampling.
Why: 17/38 datasets are Python. Language-balanced forces language-agnostic reasoning.

Mix: Equal weight per language family (6 families, 1-2 datasets each).
  Python: stack-pytest + bigcodebench (1 Dockerfile)
  Java: stack-junit + methods2test (1 Dockerfile)
  Bash: stack-bash-withtests (1 Dockerfile)
  C#: crosscodeeval-csharp (1 Dockerfile)
  TS: crosscodeeval-typescript (1 Dockerfile)
  C++: stack-cpp (1 Dockerfile)
→ 6 Dockerfiles total.

Baseline: Same total tasks, sampled proportional to dataset size (Python-dominated).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from exp_general_utils import mix_datasets
from mixing_hypotheses.dataset_registry import DATASETS, LANGUAGE_FAMILIES

PREFIX_TEST = "mix-h2-lang-balanced"
PREFIX_BASELINE = "mix-h2-lang-proportional"
HF_REPO_TEST = "DCAgent/mix_h2_language_balanced"
HF_REPO_BASELINE = "DCAgent/mix_h2_language_proportional"
TOTAL = 5000

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["test", "baseline", "both"], default="both")
    args = parser.parse_args()

    # Flatten all datasets used across families
    all_datasets = []
    for lang, ds_list in LANGUAGE_FAMILIES.items():
        all_datasets.extend(ds_list)

    # ── Test arm: equal weight per language family ──
    if args.arm in ("test", "both"):
        n_families = len(LANGUAGE_FAMILIES)
        weight_per_family = 1.0 / n_families
        repos_weights = []
        for lang, ds_list in sorted(LANGUAGE_FAMILIES.items()):
            per_ds = weight_per_family / len(ds_list)
            for d in ds_list:
                repos_weights.append((DATASETS[d].hf_repo, per_ds))
            print(f"  {lang}: {len(ds_list)} datasets @ {per_ds:.4f} each")

        print(f"\n[TEST] Language-balanced mix ({n_families} families, {TOTAL} tasks)")
        mix_datasets(
            source_repos_with_weights=repos_weights,
            total_limit=TOTAL,
            prefix=PREFIX_TEST,
            hf_repo=HF_REPO_TEST,
        )

    # ── Baseline arm: proportional to available tasks (Python-dominated) ──
    if args.arm in ("baseline", "both"):
        # Weight proportional to number of datasets per language
        total_ds = len(all_datasets)
        repos_weights = []
        for lang, ds_list in sorted(LANGUAGE_FAMILIES.items()):
            per_ds = len(ds_list) / total_ds / len(ds_list)
            for d in ds_list:
                repos_weights.append((DATASETS[d].hf_repo, per_ds))

        print(f"\n[BASELINE] Proportional mix ({total_ds} datasets, {TOTAL} tasks)")
        mix_datasets(
            source_repos_with_weights=repos_weights,
            total_limit=TOTAL,
            prefix=PREFIX_BASELINE,
            hf_repo=HF_REPO_BASELINE,
        )
