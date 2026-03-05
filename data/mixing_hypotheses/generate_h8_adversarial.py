#!/usr/bin/env python3
"""H8: Adversarial Test Augmentation.

Claim: Augmenting tasks with adversarial tests (targeting off-by-one, empty input,
unicode, type errors) makes agents more robust.

Why: Standard tests miss edge cases. Adversarial tests lower initial pass rates but
force the model to handle real-world failure modes.

Mix: Medium-tier Python datasets + 5 adversarial tests per task. 1 Dockerfile.
Baseline: Same datasets with original tests only.
"""
import json, random, sys, tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from exp_general_utils import DEFAULT_MODEL, mix_datasets
from data.commons import load_harbor_tasks_as_dicts, create_harbor_task_directory_generic, upload_tasks_to_hf
from data.completions import run_completions
from datasets import Dataset
from tqdm import tqdm
from mixing_hypotheses.dataset_registry import DATASETS, PYTHON_MEDIUM

PREFIX_TEST = "mix-h8-adversarial"
PREFIX_BASELINE = "mix-h8-original"
HF_REPO_TEST = "DCAgent/mix_h8_adversarial_tests"
HF_REPO_BASELINE = "DCAgent/mix_h8_original_tests"
TOTAL = 5000
LOAD_PER_DATASET = 1000

# Use medium-tier Python datasets (20-49% pass rate)
SOURCE_DATASETS = PYTHON_MEDIUM

ADVERSARIAL_PROMPT = """Generate 5 adversarial pytest test functions targeting common failure modes for this task.

TASK:
{{instruction}}

EXISTING TESTS (for reference, do NOT duplicate):
{{test_files_json}}

Generate tests checking:
1. Off-by-one / boundary conditions
2. Empty input / missing data
3. Unicode / special characters in input
4. Type errors (wrong types, None values)
5. Large input / performance edge cases

Write ONLY the test functions (no imports needed, they'll be appended to existing test file).
Each function must start with `def test_adversarial_`.
Output only the Python code:"""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["test", "baseline", "both"], default="both")
    args = parser.parse_args()

    # Load medium-tier tasks
    all_tasks = []
    for name in SOURCE_DATASETS:
        info = DATASETS[name]
        tasks = load_harbor_tasks_as_dicts(info.hf_repo, limit=LOAD_PER_DATASET)
        all_tasks.extend(tasks)
        print(f"  Loaded {len(tasks)} from {name}")

    random.shuffle(all_tasks)
    all_tasks = all_tasks[:TOTAL]
    print(f"Pool: {len(all_tasks)} tasks")

    # ── Baseline arm: original tests ──
    if args.arm in ("baseline", "both"):
        print("\n[BASELINE] Original tests")
        out = Path(tempfile.mkdtemp(prefix=f"{PREFIX_BASELINE}_"))
        for i, row in enumerate(tqdm(all_tasks, desc="Creating baseline")):
            create_harbor_task_directory_generic(
                output_dir=out, task_id=i, instruction=row["instruction"],
                dockerfile=row["dockerfile"], test_sh=row["test_sh"],
                test_files=json.loads(row["test_files_json"]),
                dataset_prefix=PREFIX_BASELINE,
                metadata={"source": row.get("task_dir_name", "")},
            )
        upload_tasks_to_hf(str(out), HF_REPO_BASELINE)

    # ── Test arm: augmented with adversarial tests ──
    if args.arm in ("test", "both"):
        print("\n[TEST] Generating adversarial tests via LLM...")
        dataset = Dataset.from_list(all_tasks)
        result = run_completions(
            dataset, model=DEFAULT_MODEL, map_type="chat",
            map_config={"user_message": ADVERSARIAL_PROMPT, "output_column": "adversarial_tests"},
            require_all_responses=False,
        )

        out = Path(tempfile.mkdtemp(prefix=f"{PREFIX_TEST}_"))
        created = 0
        for i, row in enumerate(tqdm(result.dataset, desc="Creating test arm")):
            adv_code = row.get("adversarial_tests", "").strip()
            if not adv_code or "def test_adversarial_" not in adv_code:
                # Skip if LLM didn't produce valid tests; use original
                adv_code = ""

            # Merge adversarial tests into existing test files
            test_files = json.loads(row["test_files_json"])
            if adv_code:
                # Find existing test file to append to, or create new one
                appended = False
                for fname in list(test_files.keys()):
                    if fname.startswith("test_") and fname.endswith(".py"):
                        test_files[fname] = test_files[fname] + "\n\n# --- Adversarial tests ---\n" + adv_code
                        appended = True
                        break
                if not appended:
                    test_files["test_adversarial.py"] = adv_code

            create_harbor_task_directory_generic(
                output_dir=out, task_id=created, instruction=row["instruction"],
                dockerfile=row["dockerfile"], test_sh=row["test_sh"],
                test_files=test_files,
                dataset_prefix=PREFIX_TEST,
                metadata={"source": row.get("task_dir_name", ""), "has_adversarial": bool(adv_code)},
            )
            created += 1

        print(f"Created {created} tasks with adversarial augmentation")
        print(f"\nUploading to {HF_REPO_TEST}...")
        upload_tasks_to_hf(str(out), HF_REPO_TEST)


if __name__ == "__main__":
    main()
