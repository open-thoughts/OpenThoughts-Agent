#!/usr/bin/env python3
"""Dataset registry for RL data mixing hypotheses.

Centralizes dataset metadata: HF repos, pass rates, language, Dockerfile group.
All 38 good datasets from GOOD_DATASETS.md (2026-02-19).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatasetInfo:
    name: str
    hf_repo: str
    pass_rate: float  # 0-1, measured on gpt-5-mini unless noted
    language: str
    dockerfile_group: str  # datasets sharing a Dockerfile can be mixed cheaply
    n_tasks: Optional[int] = None  # None = original (unknown exact count)


# ── All 38 good datasets ──────────────────────────────────────────────────────

DATASETS: Dict[str, DatasetInfo] = {
    # Python base (16 datasets, 1 Dockerfile)
    "r2egym-easy": DatasetInfo("r2egym-easy", "DCAgent/exp-rdb-r2egym-easy", 0.65, "python", "python"),
    "r2egym-trivial": DatasetInfo("r2egym-trivial", "DCAgent/exp-rdb-r2egym-trivial", 0.60, "python", "python"),
    "r2egym-medium": DatasetInfo("r2egym-medium", "DCAgent/exp-rdb-r2egym-medium", 0.56, "python", "python"),
    "r2egym-hard": DatasetInfo("r2egym-hard", "DCAgent/exp-rdb-r2egym-hard", 0.56, "python", "python"),
    "r2egym-very_hard": DatasetInfo("r2egym-very_hard", "DCAgent/exp-rdb-r2egym-very_hard", 0.28, "python", "python"),
    "bigcodebench": DatasetInfo("bigcodebench", "DCAgent/exp_rpt_bigcodebench-v3", 0.54, "python", "python", 500),
    "crosscodeeval-python": DatasetInfo("crosscodeeval-python", "DCAgent/exp_rpt_crosscodeeval-python-v2", 0.48, "python", "python", 500),
    "e2egit": DatasetInfo("e2egit", "DCAgent/exp_rpt_e2egit-v2", 0.48, "python", "python", 500),
    "pymethods2test": DatasetInfo("pymethods2test", "DCAgent/exp_rpt_pymethods2test-v3", 0.40, "python", "python", 500),
    "stack-pytest": DatasetInfo("stack-pytest", "DCAgent/exp_rpt_stack-pytest-v2", 0.40, "python", "python", 500),
    "unitsyn-python": DatasetInfo("unitsyn-python", "DCAgent/exp_rpt_unitsyn-python-v3", 0.36, "python", "python", 500),
    "codereval-python": DatasetInfo("codereval-python", "DCAgent/exp_rpt_codereval-python-v2", 0.24, "python", "python", 230),
    "exercism-python": DatasetInfo("exercism-python", "DCAgent/exp_rpt_exercism-python", 0.24, "python", "python"),
    "codenet-python": DatasetInfo("codenet-python", "DCAgent/exp_rpt_codenet-python", 0.24, "python", "python"),
    "stack-selfdoc": DatasetInfo("stack-selfdoc", "DCAgent/exp_rpt_stack-selfdoc-v2", 0.16, "python", "python", 500),
    "softwareheritage": DatasetInfo("softwareheritage", "DCAgent/exp_rpt_softwareheritage-v2", 0.12, "python", "python", 500),
    "stack-pytest-gpt5mini": DatasetInfo("stack-pytest-gpt5mini", "DCAgent/exp_rpt_stack-pytest-gpt5mini", 0.08, "python", "python"),
    "bugsinpy": DatasetInfo("bugsinpy", "DCAgent/exp_rpt_bugsinpy-v4", 0.08, "python", "python", 500),
    "bugsinpy-mf": DatasetInfo("bugsinpy-mf", "DCAgent/exp_rpt_bugsinpy-mf", 0.08, "python", "python", 500),
    "nemotron-pytest": DatasetInfo("nemotron-pytest", "DCAgent/exp_rpt_nemotron-pytest-gpt5mini-v2", 0.30, "python", "python", 10),
    # Java base (4 datasets, 1 Dockerfile)
    "stack-junit": DatasetInfo("stack-junit", "DCAgent/exp_rpt_stack-junit", 0.84, "java", "java"),
    "crosscodeeval-java": DatasetInfo("crosscodeeval-java", "DCAgent/exp_rpt_crosscodeeval-java", 0.48, "java", "java"),
    "methods2test": DatasetInfo("methods2test", "DCAgent/exp_rpt_methods2test-v2", 0.28, "java", "java", 500),
    "defects4j": DatasetInfo("defects4j", "DCAgent/exp_rpt_defects4j-v3", 0.04, "java", "java", 464),
    # Bash base (4 datasets, 1 Dockerfile)
    "stack-bash": DatasetInfo("stack-bash", "DCAgent/exp_rpt_stack-bash", 0.20, "bash", "bash"),
    "stack-bash-withtests": DatasetInfo("stack-bash-withtests", "DCAgent/exp_rpt_stack-bash-withtests", 0.24, "bash", "bash"),
    "stack-bash-withtests-gpt5mini": DatasetInfo("stack-bash-withtests-gpt5mini", "DCAgent/exp_rpt_stack-bash-withtests-gpt5mini", 0.32, "bash", "bash"),
    "nemotron-bash": DatasetInfo("nemotron-bash", "DCAgent/exp_rpt_nemotron-bash-v2", 0.33, "bash", "bash", 10),
    # C# base (2 datasets, 1 Dockerfile)
    "crosscodeeval-csharp": DatasetInfo("crosscodeeval-csharp", "DCAgent/exp_rpt_crosscodeeval-csharp", 0.68, "csharp", "csharp"),
    "stack-csharp": DatasetInfo("stack-csharp", "DCAgent/exp_rpt_stack-csharp", 0.12, "csharp", "csharp"),
    # Single-dataset Dockerfile groups
    "stack-php": DatasetInfo("stack-php", "DCAgent/exp_rpt_stack-php-v2", 1.00, "php", "php", 500),
    "stack-cpp": DatasetInfo("stack-cpp", "DCAgent/exp_rpt_stack-cpp", 0.24, "cpp", "cpp"),
    "stack-jest": DatasetInfo("stack-jest", "DCAgent/exp_rpt_stack-jest-v2", 0.08, "javascript", "node", 500),
    "crosscodeeval-typescript": DatasetInfo("crosscodeeval-typescript", "DCAgent/exp_rpt_crosscodeeval-typescript", 0.48, "typescript", "node"),
    "stack-ruby": DatasetInfo("stack-ruby", "DCAgent/exp_rpt_stack-ruby", 0.12, "ruby", "ruby"),
    "stack-rust": DatasetInfo("stack-rust", "DCAgent/exp_rpt_stack-rust", 0.12, "rust", "rust"),
    "manybugs": DatasetInfo("manybugs", "DCAgent/exp_rpt_manybugs-v2", 0.07, "c", "c"),
    "stack-dockerfile": DatasetInfo("stack-dockerfile", "DCAgent/exp_rpt_stack-dockerfile-v2", 0.16, "dockerfile", "dockerfile", 497),
    "codeelo": DatasetInfo("codeelo", "DCAgent/exp_rpt_codeelo-v2", 0.52, "multi", "multi", 500),
    "taco": DatasetInfo("taco", "DCAgent/exp_rpt_taco", 0.24, "multi", "multi"),
}

# ── Convenience groupings ──────────────────────────────────────────────────────

PYTHON_BASE = [k for k, v in DATASETS.items() if v.dockerfile_group == "python"]
JAVA_BASE = [k for k, v in DATASETS.items() if v.dockerfile_group == "java"]
BASH_BASE = [k for k, v in DATASETS.items() if v.dockerfile_group == "bash"]
CSHARP_BASE = [k for k, v in DATASETS.items() if v.dockerfile_group == "csharp"]

# Difficulty tiers based on pass rate
EASY_DATASETS = [k for k, v in DATASETS.items() if v.pass_rate >= 0.50]
MEDIUM_DATASETS = [k for k, v in DATASETS.items() if 0.20 <= v.pass_rate < 0.50]
HARD_DATASETS = [k for k, v in DATASETS.items() if v.pass_rate < 0.20]

# Python-only difficulty tiers
PYTHON_EASY = [k for k in PYTHON_BASE if DATASETS[k].pass_rate >= 0.50]
PYTHON_MEDIUM = [k for k in PYTHON_BASE if 0.20 <= DATASETS[k].pass_rate < 0.50]
PYTHON_HARD = [k for k in PYTHON_BASE if DATASETS[k].pass_rate < 0.20]

# Language families (for H2)
LANGUAGE_FAMILIES = {
    "python": ["stack-pytest", "bigcodebench"],
    "java": ["stack-junit", "methods2test"],
    "bash": ["stack-bash-withtests"],
    "csharp": ["crosscodeeval-csharp"],
    "typescript": ["crosscodeeval-typescript"],
    "cpp": ["stack-cpp"],
}


def get_repos(dataset_names: List[str]) -> List[str]:
    """Get HF repo IDs for a list of dataset names."""
    return [DATASETS[n].hf_repo for n in dataset_names]


def get_repos_with_weights(dataset_names: List[str], weights: List[float]) -> list:
    """Get (repo, weight) tuples for mix_datasets()."""
    assert len(dataset_names) == len(weights)
    return [(DATASETS[n].hf_repo, w) for n, w in zip(dataset_names, weights)]
