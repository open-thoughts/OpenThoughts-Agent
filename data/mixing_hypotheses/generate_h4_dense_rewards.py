#!/usr/bin/env python3
"""H4: Dense Rewards on Hard Tasks > Binary Rewards on Easy Tasks.

Claim: Proportional rewards (passed/total tests) on hard datasets produce better
learning than binary rewards on easy datasets.

Why: At 8% binary pass rate, 92% of rollouts yield zero signal. Proportional reward
means passing 3/10 tests → reward 0.3. Dense rewards turn hard tasks into medium-
difficulty learning.

Mix A (test):  Hard Python datasets with PROPORTIONAL reward. 1 Dockerfile.
Mix B (baseline): Easy Python datasets with BINARY reward. 1 Dockerfile.
Same total training steps.
"""
import json, sys, tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from exp_general_utils import mix_datasets, transform_tasks
from mixing_hypotheses.dataset_registry import DATASETS, PYTHON_EASY, PYTHON_HARD

PREFIX_TEST = "mix-h4-dense-hard"
PREFIX_BASELINE = "mix-h4-binary-easy"
HF_REPO_TEST = "DCAgent/mix_h4_dense_rewards_hard"
HF_REPO_BASELINE = "DCAgent/mix_h4_binary_easy"
TOTAL = 5000

# Hard Python datasets for the test arm
HARD_DATASETS = [
    "bugsinpy", "softwareheritage", "stack-selfdoc",
    "stack-pytest-gpt5mini", "bugsinpy-mf",
]

# Easy Python datasets for the baseline arm
EASY_DATASETS = [
    "r2egym-easy", "bigcodebench", "crosscodeeval-python",
    "r2egym-trivial", "r2egym-medium",
]

PROPORTIONAL_TEST_SH = '''#!/bin/bash
set -uo pipefail

mkdir -p /logs/verifier

# Run all test files, count pass/fail
TOTAL=0
PASSED=0

for test_file in /tests/test_*.py; do
    [ -f "$test_file" ] || continue
    TEST_COUNT=$(grep -c "def test_" "$test_file" 2>/dev/null || true)
    TEST_COUNT=${TEST_COUNT:-0}
    TOTAL=$((TOTAL + TEST_COUNT))

    RESULT=$(cd /app && python -m pytest "$test_file" -v --tb=no 2>&1 || true)
    FILE_PASSED=$(echo "$RESULT" | grep -c " PASSED" || true)
    FILE_PASSED=${FILE_PASSED:-0}
    PASSED=$((PASSED + FILE_PASSED))
done

for test_file in /tests/test_*.sh; do
    [ -f "$test_file" ] || continue
    TOTAL=$((TOTAL + 1))
    if bash "$test_file" 2>/dev/null; then
        PASSED=$((PASSED + 1))
    fi
done

if [ $TOTAL -eq 0 ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi

REWARD=$(python3 -c "print(round($PASSED / $TOTAL, 4))")
echo "$REWARD" > /logs/verifier/reward.txt
echo "Passed $PASSED / $TOTAL tests (reward=$REWARD)"

if [ "$PASSED" -eq "$TOTAL" ]; then
    exit 0
else
    exit 1
fi
'''


def make_proportional(task):
    """Replace test.sh with proportional reward version."""
    task = dict(task)
    task["test_sh"] = PROPORTIONAL_TEST_SH
    task["metadata"] = {"source": task.get("task_dir_name", ""), "reward_type": "proportional"}
    return task


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["test", "baseline", "both"], default="both")
    args = parser.parse_args()

    # ── Test arm: hard datasets with proportional reward ──
    if args.arm in ("test", "both"):
        print("[TEST] Hard datasets with proportional reward")
        n = len(HARD_DATASETS)
        weight = 1.0 / n
        repos_weights = [(DATASETS[d].hf_repo, weight) for d in HARD_DATASETS]
        for d in HARD_DATASETS:
            print(f"  {d}: pass_rate={DATASETS[d].pass_rate:.0%}")

        # First mix, then transform to proportional reward
        from data.commons import load_harbor_tasks_as_dicts, create_harbor_task_directory_generic, upload_tasks_to_hf
        from tqdm import tqdm
        import random

        all_tasks = []
        per_ds = TOTAL // n
        for d in HARD_DATASETS:
            tasks = load_harbor_tasks_as_dicts(DATASETS[d].hf_repo, limit=per_ds)
            random.shuffle(tasks)
            all_tasks.extend(tasks[:per_ds])
            print(f"  Loaded {min(per_ds, len(tasks))} from {d}")

        random.shuffle(all_tasks)
        all_tasks = all_tasks[:TOTAL]

        out = Path(tempfile.mkdtemp(prefix=f"{PREFIX_TEST}_"))
        for i, row in enumerate(tqdm(all_tasks, desc="Creating test arm")):
            row = make_proportional(row)
            create_harbor_task_directory_generic(
                output_dir=out, task_id=i, instruction=row["instruction"],
                dockerfile=row["dockerfile"], test_sh=row["test_sh"],
                test_files=json.loads(row["test_files_json"]),
                dataset_prefix=PREFIX_TEST,
                metadata=row["metadata"],
            )

        print(f"\nUploading test arm to {HF_REPO_TEST}...")
        upload_tasks_to_hf(str(out), HF_REPO_TEST)

    # ── Baseline arm: easy datasets with binary reward (original test.sh) ──
    if args.arm in ("baseline", "both"):
        print("\n[BASELINE] Easy datasets with binary reward")
        n = len(EASY_DATASETS)
        weight = 1.0 / n
        repos_weights = [(DATASETS[d].hf_repo, weight) for d in EASY_DATASETS]
        for d in EASY_DATASETS:
            print(f"  {d}: pass_rate={DATASETS[d].pass_rate:.0%}")

        mix_datasets(
            source_repos_with_weights=repos_weights,
            total_limit=TOTAL,
            prefix=PREFIX_BASELINE,
            hf_repo=HF_REPO_BASELINE,
        )
