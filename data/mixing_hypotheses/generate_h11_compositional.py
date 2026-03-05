#!/usr/bin/env python3
"""H11: Compositional Complexity Gradient.

Claim: Mixing single-skill + multi-skill tasks produces better generalization
than single-skill only.

Why: Real tasks require composing capabilities. 1-skill tasks teach individual skills;
2/3-skill tasks teach integration. Prevents narrow specialization.

Mix: 50% single-skill, 30% 2-skill, 20% 3-skill. 1 Dockerfile (all Python).
Baseline: 100% single-skill of equivalent total difficulty.
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

PREFIX_TEST = "mix-h11-compositional"
PREFIX_BASELINE = "mix-h11-single-skill"
HF_REPO_TEST = "DCAgent/mix_h11_compositional_gradient"
HF_REPO_BASELINE = "DCAgent/mix_h11_single_skill_only"
TOTAL = 5000
LOAD_PER_DATASET = 1000

SOURCE_DATASETS = PYTHON_MEDIUM

# ── LLM prompts for multi-skill rewriting ──

REWRITE_2SKILL_PROMPT = """Rewrite this task to require EXACTLY 2 skills: the original skill PLUS file I/O.

The rewritten task should:
1. Keep the original algorithmic requirement
2. Add a requirement to read input from a file and write output to a file
3. Be a single cohesive task (not two separate tasks)

ORIGINAL TASK:
{{instruction}}

Output only the rewritten 2-skill task:"""

REWRITE_3SKILL_PROMPT = """Rewrite this task to require EXACTLY 3 skills: the original skill PLUS file I/O PLUS JSON parsing.

The rewritten task should:
1. Keep the original algorithmic requirement
2. Read input from a JSON file
3. Write structured JSON output to a file
4. Be a single cohesive task (not three separate tasks)

ORIGINAL TASK:
{{instruction}}

Output only the rewritten 3-skill task:"""

TEST_2SKILL_PROMPT = """Adapt these tests for a version of the task that also requires file I/O (reading input from a file, writing output to a file). Add tests for the file I/O aspect while keeping the core logic tests.

ORIGINAL TESTS:
{{test_files_json}}

Write adapted pytest tests. Output only the test code:"""

TEST_3SKILL_PROMPT = """Adapt these tests for a version of the task that requires file I/O AND JSON parsing (reading JSON input, writing JSON output). Add tests for file I/O and JSON aspects while keeping core logic tests.

ORIGINAL TESTS:
{{test_files_json}}

Write adapted pytest tests. Output only the test code:"""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["test", "baseline", "both"], default="both")
    args = parser.parse_args()

    # Load source tasks
    all_tasks = []
    for name in SOURCE_DATASETS:
        info = DATASETS[name]
        tasks = load_harbor_tasks_as_dicts(info.hf_repo, limit=LOAD_PER_DATASET)
        all_tasks.extend(tasks)
        print(f"  Loaded {len(tasks)} from {name}")

    random.shuffle(all_tasks)
    all_tasks = all_tasks[:TOTAL]
    print(f"Pool: {len(all_tasks)} tasks")

    # Split pool: 50% single, 30% for 2-skill rewrite, 20% for 3-skill rewrite
    n_single = int(TOTAL * 0.50)
    n_2skill = int(TOTAL * 0.30)
    n_3skill = TOTAL - n_single - n_2skill

    single_tasks = all_tasks[:n_single]
    tasks_for_2skill = all_tasks[n_single:n_single + n_2skill]
    tasks_for_3skill = all_tasks[n_single + n_2skill:]

    # ── Baseline arm: 100% single-skill ──
    if args.arm in ("baseline", "both"):
        print(f"\n[BASELINE] {TOTAL} single-skill tasks")
        out = Path(tempfile.mkdtemp(prefix=f"{PREFIX_BASELINE}_"))
        for i, row in enumerate(tqdm(all_tasks, desc="Baseline")):
            create_harbor_task_directory_generic(
                output_dir=out, task_id=i, instruction=row["instruction"],
                dockerfile=row["dockerfile"], test_sh=row["test_sh"],
                test_files=json.loads(row["test_files_json"]),
                dataset_prefix=PREFIX_BASELINE,
                metadata={"source": row.get("task_dir_name", ""), "n_skills": 1},
            )
        upload_tasks_to_hf(str(out), HF_REPO_BASELINE)

    # ── Test arm: 50/30/20 mix ──
    if args.arm in ("test", "both"):
        print(f"\n[TEST] Compositional: {n_single} single + {n_2skill} 2-skill + {n_3skill} 3-skill")

        # Rewrite 2-skill tasks
        print("Rewriting 2-skill tasks...")
        ds_2 = Dataset.from_list(tasks_for_2skill)
        result_2_instr = run_completions(
            ds_2, model=DEFAULT_MODEL, map_type="chat",
            map_config={"user_message": REWRITE_2SKILL_PROMPT, "output_column": "rewritten_instruction"},
            require_all_responses=False,
        )
        result_2_tests = run_completions(
            result_2_instr.dataset, model=DEFAULT_MODEL, map_type="chat",
            map_config={"user_message": TEST_2SKILL_PROMPT, "output_column": "rewritten_tests"},
            require_all_responses=False,
        )

        # Rewrite 3-skill tasks
        print("Rewriting 3-skill tasks...")
        ds_3 = Dataset.from_list(tasks_for_3skill)
        result_3_instr = run_completions(
            ds_3, model=DEFAULT_MODEL, map_type="chat",
            map_config={"user_message": REWRITE_3SKILL_PROMPT, "output_column": "rewritten_instruction"},
            require_all_responses=False,
        )
        result_3_tests = run_completions(
            result_3_instr.dataset, model=DEFAULT_MODEL, map_type="chat",
            map_config={"user_message": TEST_3SKILL_PROMPT, "output_column": "rewritten_tests"},
            require_all_responses=False,
        )

        # Assemble final dataset
        out = Path(tempfile.mkdtemp(prefix=f"{PREFIX_TEST}_"))
        task_id = 0

        # Single-skill tasks (unchanged)
        for row in tqdm(single_tasks, desc="Single-skill"):
            create_harbor_task_directory_generic(
                output_dir=out, task_id=task_id, instruction=row["instruction"],
                dockerfile=row["dockerfile"], test_sh=row["test_sh"],
                test_files=json.loads(row["test_files_json"]),
                dataset_prefix=PREFIX_TEST,
                metadata={"source": row.get("task_dir_name", ""), "n_skills": 1},
            )
            task_id += 1

        # 2-skill tasks
        for row in tqdm(result_2_tests.dataset, desc="2-skill"):
            instr = row.get("rewritten_instruction", "").strip()
            tests = row.get("rewritten_tests", "").strip()
            if not instr or not tests or "def test_" not in tests:
                # Fallback to original if rewrite failed
                instr = row["instruction"]
                tests = None

            test_files = json.loads(row["test_files_json"]) if tests is None else {"test_2skill.py": tests}
            create_harbor_task_directory_generic(
                output_dir=out, task_id=task_id,
                instruction=instr,
                dockerfile=row["dockerfile"],
                test_sh=row["test_sh"],
                test_files=test_files,
                dataset_prefix=PREFIX_TEST,
                metadata={"source": row.get("task_dir_name", ""), "n_skills": 2},
            )
            task_id += 1

        # 3-skill tasks
        for row in tqdm(result_3_tests.dataset, desc="3-skill"):
            instr = row.get("rewritten_instruction", "").strip()
            tests = row.get("rewritten_tests", "").strip()
            if not instr or not tests or "def test_" not in tests:
                instr = row["instruction"]
                tests = None

            test_files = json.loads(row["test_files_json"]) if tests is None else {"test_3skill.py": tests}
            create_harbor_task_directory_generic(
                output_dir=out, task_id=task_id,
                instruction=instr,
                dockerfile=row["dockerfile"],
                test_sh=row["test_sh"],
                test_files=test_files,
                dataset_prefix=PREFIX_TEST,
                metadata={"source": row.get("task_dir_name", ""), "n_skills": 3},
            )
            task_id += 1

        print(f"Created {task_id} compositional tasks")
        upload_tasks_to_hf(str(out), HF_REPO_TEST)


if __name__ == "__main__":
    main()
