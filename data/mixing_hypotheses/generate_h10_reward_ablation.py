#!/usr/bin/env python3
"""H10: Reward Structure Ablation — binary vs proportional vs staged rewards.

Claim: Staged rewards (basic → edge → robustness test ordering) outperform both
proportional and binary rewards.

Why: Proportional rewards treat all tests equally, but tests have natural ordering.
Staged rewards capture this: agent first passes basic tests, then edges, then
robustness. Intra-episode curriculum.

Mix: Same medium-tier Python dataset, three arms:
  Arm A: Binary (pass/fail all)
  Arm B: Proportional (passed/total)
  Arm C: Staged (max_stage/total from exp 7.5)
1 Dockerfile across all arms.
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
from mixing_hypotheses.dataset_registry import DATASETS, PYTHON_MEDIUM

PREFIX_A = "mix-h10-binary"
PREFIX_B = "mix-h10-proportional"
PREFIX_C = "mix-h10-staged"
HF_REPO_A = "DCAgent/mix_h10_reward_binary"
HF_REPO_B = "DCAgent/mix_h10_reward_proportional"
HF_REPO_C = "DCAgent/mix_h10_reward_staged"
TOTAL = 5000
LOAD_PER_DATASET = 1000

SOURCE_DATASETS = PYTHON_MEDIUM

# ── Arm B: Proportional reward test.sh ──
PROPORTIONAL_TEST_SH = '''#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier

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

if [ $TOTAL -eq 0 ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 1
fi

REWARD=$(python3 -c "print(round($PASSED / $TOTAL, 4))")
echo "$REWARD" > /logs/verifier/reward.txt
echo "Passed $PASSED / $TOTAL tests (reward=$REWARD)"
[ "$PASSED" -eq "$TOTAL" ] && exit 0 || exit 1
'''

# ── Arm C: Staged reward test.sh template ──
STAGED_TEST_SH = '''#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier

STAGES_JSON='STAGES_PLACEHOLDER'
TOTAL_STAGES=$(echo "$STAGES_JSON" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")
MAX_STAGE=0

if [ "$TOTAL_STAGES" -eq 0 ]; then
    echo "0" > /logs/verifier/reward.txt
    echo "No stages defined"
    exit 1
fi

for stage in $(seq 1 $TOTAL_STAGES); do
    TESTS=$(echo "$STAGES_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(' '.join(['-k '+t for t in d.get(str($stage), [])]))")
    if [ -z "$TESTS" ]; then continue; fi

    cd /app && python -m pytest /tests/ $TESTS --tb=short 2>&1 | tee -a /logs/verifier/output.txt
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        MAX_STAGE=$stage
    else
        break
    fi
done

REWARD=$(python3 -c "print(round($MAX_STAGE / $TOTAL_STAGES, 4))")
echo "$REWARD" > /logs/verifier/reward.txt
echo "Reached stage $MAX_STAGE / $TOTAL_STAGES (reward=$REWARD)"
[ "$MAX_STAGE" -eq "$TOTAL_STAGES" ] && exit 0 || exit 1
'''

DECOMPOSE_PROMPT = """Split this test suite into 3-5 ordered stages. Each stage builds on the previous:
Stage 1: Code exists and is syntactically valid
Stage 2: Basic/happy-path tests pass
Stage 3: Edge cases pass
Stage 4: Performance/robustness tests pass (if applicable)

TESTS:
{{test_files_json}}

TASK:
{{instruction}}

Output a JSON object mapping stage numbers to lists of test function names. Example:
{"1": ["test_file_exists"], "2": ["test_basic_output"], "3": ["test_edge_empty", "test_edge_large"]}

Output only the JSON:"""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "B", "C", "all"], default="all")
    args = parser.parse_args()

    # Load same tasks for all arms
    all_tasks = []
    for name in SOURCE_DATASETS:
        info = DATASETS[name]
        tasks = load_harbor_tasks_as_dicts(info.hf_repo, limit=LOAD_PER_DATASET)
        all_tasks.extend(tasks)
        print(f"  Loaded {len(tasks)} from {name}")

    random.shuffle(all_tasks)
    all_tasks = all_tasks[:TOTAL]
    print(f"Pool: {len(all_tasks)} tasks")

    # ── Arm A: Binary reward (original test.sh) ──
    if args.arm in ("A", "all"):
        print("\n[ARM A] Binary reward (original test.sh)")
        out = Path(tempfile.mkdtemp(prefix=f"{PREFIX_A}_"))
        for i, row in enumerate(tqdm(all_tasks, desc="Arm A")):
            create_harbor_task_directory_generic(
                output_dir=out, task_id=i, instruction=row["instruction"],
                dockerfile=row["dockerfile"], test_sh=row["test_sh"],
                test_files=json.loads(row["test_files_json"]),
                dataset_prefix=PREFIX_A,
                metadata={"source": row.get("task_dir_name", ""), "reward_type": "binary"},
            )
        upload_tasks_to_hf(str(out), HF_REPO_A)

    # ── Arm B: Proportional reward ──
    if args.arm in ("B", "all"):
        print("\n[ARM B] Proportional reward")
        out = Path(tempfile.mkdtemp(prefix=f"{PREFIX_B}_"))
        for i, row in enumerate(tqdm(all_tasks, desc="Arm B")):
            create_harbor_task_directory_generic(
                output_dir=out, task_id=i, instruction=row["instruction"],
                dockerfile=row["dockerfile"], test_sh=PROPORTIONAL_TEST_SH,
                test_files=json.loads(row["test_files_json"]),
                dataset_prefix=PREFIX_B,
                metadata={"source": row.get("task_dir_name", ""), "reward_type": "proportional"},
            )
        upload_tasks_to_hf(str(out), HF_REPO_B)

    # ── Arm C: Staged reward (requires LLM decomposition) ──
    if args.arm in ("C", "all"):
        print("\n[ARM C] Staged reward (LLM decomposition)")
        dataset = Dataset.from_list(all_tasks)
        result = run_completions(
            dataset, model=DEFAULT_MODEL, map_type="chat",
            map_config={"user_message": DECOMPOSE_PROMPT, "output_column": "stages_json"},
            require_all_responses=False,
        )

        out = Path(tempfile.mkdtemp(prefix=f"{PREFIX_C}_"))
        created = 0
        for i, row in enumerate(tqdm(result.dataset, desc="Arm C")):
            stages = row.get("stages_json", "{}").strip()
            try:
                parsed = json.loads(stages)
                if not parsed or not isinstance(parsed, dict) or len(parsed) == 0:
                    continue
            except (json.JSONDecodeError, TypeError):
                continue

            test_sh = STAGED_TEST_SH.replace("STAGES_PLACEHOLDER", stages.replace("'", "'\\''"))
            create_harbor_task_directory_generic(
                output_dir=out, task_id=created, instruction=row["instruction"],
                dockerfile=row["dockerfile"], test_sh=test_sh,
                test_files=json.loads(row["test_files_json"]),
                dataset_prefix=PREFIX_C,
                metadata={"source": row.get("task_dir_name", ""), "reward_type": "staged", "stages": stages},
            )
            created += 1

        print(f"Created {created} staged tasks")
        upload_tasks_to_hf(str(out), HF_REPO_C)


if __name__ == "__main__":
    main()
