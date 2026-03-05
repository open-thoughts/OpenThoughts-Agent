#!/usr/bin/env python3
"""H6: Test Quality Filter — keep only top-25% quality tests.

Claim: Keeping only top-25% quality tests produces better RL agents than training
on all tasks.

Why: Weak tests (`assert isinstance(x, list)`) reward trivially correct but wrong
solutions. High-quality tests force genuinely correct code.

Mix: Score all tasks via test quality pipeline, keep score >= 75.
Smaller but higher signal-to-noise. Python-base only, 1 Dockerfile.
"""
import json, random, sys, tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from exp_general_utils import DEFAULT_MODEL
from data.commons import load_harbor_tasks_as_dicts, create_harbor_task_directory_generic, upload_tasks_to_hf
from data.completions import run_completions
from datasets import Dataset
from tqdm import tqdm
from mixing_hypotheses.dataset_registry import DATASETS, PYTHON_BASE

PREFIX = "mix-h6-test-quality"
HF_REPO = "DCAgent/mix_h6_test_quality_top25"
LOAD_PER_DATASET = 1000
QUALITY_THRESHOLD = 75  # keep top 25%

QUALITY_PROMPT = """Rate the quality of these test cases on a scale of 0-100.

Consider:
1. Number and variety of assertions (not just file existence or type checks)
2. Edge case coverage (empty input, large input, boundary values, error cases)
3. Specificity (would a trivially wrong solution still pass these tests?)
4. Independence (do tests check distinct behaviors, not redundant checks?)
5. Correctness (are the expected values reasonable for the described task?)

A score of 100 = comprehensive, specific, covers edge cases, catches subtle bugs.
A score of 0 = trivial checks that any implementation would pass.

TASK:
{{instruction}}

TESTS:
{{test_files_json}}

Output only a single integer 0-100:"""


def main():
    # Load from all Python-base datasets
    all_tasks = []
    for name in PYTHON_BASE:
        info = DATASETS[name]
        tasks = load_harbor_tasks_as_dicts(info.hf_repo, limit=LOAD_PER_DATASET)
        all_tasks.extend(tasks)
        print(f"  Loaded {len(tasks)} from {name}")

    print(f"Total pool: {len(all_tasks)} tasks")

    # Score test quality via LLM
    dataset = Dataset.from_list(all_tasks)
    result = run_completions(
        dataset, model=DEFAULT_MODEL, map_type="chat",
        map_config={"user_message": QUALITY_PROMPT, "output_column": "test_quality_score"},
        require_all_responses=False,
    )

    # Filter to top 25%
    kept = []
    rejected = 0
    for row in result.dataset:
        try:
            score = int(row.get("test_quality_score", "0").strip())
        except (ValueError, AttributeError):
            rejected += 1
            continue
        if score >= QUALITY_THRESHOLD:
            kept.append((row, score))
        else:
            rejected += 1

    print(f"Kept {len(kept)} / {len(kept) + rejected} tasks (threshold >= {QUALITY_THRESHOLD})")

    # Sort by quality descending, then create
    kept.sort(key=lambda x: -x[1])
    random.shuffle(kept)  # shuffle for training

    out = Path(tempfile.mkdtemp(prefix=f"{PREFIX}_"))
    for i, (row, score) in enumerate(tqdm(kept, desc="Creating tasks")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i, instruction=row["instruction"],
            dockerfile=row["dockerfile"], test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix=PREFIX,
            metadata={"source": row.get("task_dir_name", ""), "test_quality_score": score},
        )

    print(f"\nUploading to {HF_REPO}...")
    upload_tasks_to_hf(str(out), HF_REPO)


if __name__ == "__main__":
    main()
