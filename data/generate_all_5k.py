#!/usr/bin/env python3
"""Generate all 18 datasets at 5000 tasks each and upload to HuggingFace."""

import json
import sys
import os
import tempfile
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from data.exp_8_utils import (
    generate_instruction_transform,
    generate_instruction_and_tests_transform,
    generate_error_state,
)
from data.commons import load_harbor_tasks_as_dicts, create_harbor_task_directory_generic, upload_tasks_to_hf
from data.completions import run_completions
from datasets import Dataset

LIMIT = 5000
HF_ORG = "DCAgent"
SOURCE = "DCAgent/exp_rpt_stack-pytest"

# ─── Prompts ─────────────────────────────────────────────────

STYLE_PROMPT = """Rewrite the following coding task as a {{variant}}.

Styles:
- "github_issue": GitHub issue with title, description, steps to reproduce
- "slack_message": Casual Slack message from a coworker
- "stackoverflow_question": SO question with context and code
- "code_review_comment": Code review comment on what needs to change
- "error_report": Bug report with observed vs expected behavior

ORIGINAL TASK:
{{instruction}}

Rewrite as a {{variant}}. Preserve ALL technical requirements. Output only the rewritten instruction:"""

MINIMAL_PROMPT = """Rewrite the following coding task using ONLY the bare minimum information. Strip away all examples, context, hints, and details. Keep only the core requirement in 1-2 sentences.

ORIGINAL TASK:
{{instruction}}

Output only the minimal instruction:"""

GRANULARITY_PROMPT = """Rewrite the following coding task at "{{variant}}" granularity level.

- "vague": Single vague sentence. No details, no file hints.
- "bullets": 3-5 requirement bullets. Key requirements but no implementation hints.
- "detailed": Step-by-step with file structure hints, function signatures, edge cases.

ORIGINAL TASK:
{{instruction}}

Rewrite at "{{variant}}" granularity. Output only the rewritten instruction:"""

PADDING_PROMPT = """Add irrelevant context/noise to this task at "{{variant}}" level. Real requirements must stay intact but be buried in noise.

- "mild": 1-2 irrelevant sentences of backstory
- "moderate": Paragraph of project context + red herring requirements
- "heavy": Verbose logs, unrelated code snippets, misleading hints. Real task buried deep.

ORIGINAL TASK:
{{instruction}}

Add "{{variant}}" padding. Output only the padded instruction:"""

KNOWLEDGE_PROMPT = """Rewrite this task at "{{variant}}" prerequisite knowledge level.

- "expert": Terse, assumes complete domain knowledge. No algorithm explanation.
- "intermediate": Explains algorithm steps but assumes programming competence.
- "novice": Fully explained with pseudocode or step-by-step algorithm.

ORIGINAL TASK:
{{instruction}}

Rewrite at "{{variant}}" level. Output only the rewritten instruction:"""

AMBIGUITY_PROMPT = """Rewrite this task to be deliberately underspecified at "{{variant}}" level.

- "partial": Remove some details (exact return types, edge cases) but keep function names
- "minimal": Only the high-level goal. Example: "Build something that processes the data correctly."

ORIGINAL TASK:
{{instruction}}

Output only the underspecified instruction:"""

PROPERTY_TEST_PROMPT = """Convert these exact-output tests into PROPERTY-BASED tests that accept any correct implementation. Replace equality checks with property assertions.

ORIGINAL TESTS:
{{test_files_json}}

Write property-based pytest tests. Output only the new test code:"""

ADVERSARIAL_INSTR_PROMPT = """Return the following task instruction exactly as-is, unchanged.

{{instruction}}

Output only the instruction:"""

ADVERSARIAL_TEST_PROMPT = """Generate ADVERSARIAL pytest tests targeting common agent failure modes for this task.

TASK:
{{instruction}}

EXISTING TESTS (for reference, do NOT copy):
{{test_files_json}}

Generate tests that check for these common mistakes:
1. Off-by-one errors (boundary conditions)
2. Empty input handling
3. Large input handling
4. Wrong file paths / missing files
5. Type errors (string vs int, None handling)
6. Unicode / special characters
7. Concurrent access issues (if applicable)

Write 5-10 adversarial pytest test functions. Output only the test code:"""

BROKEN_CODE_PROMPT = """Generate a PARTIALLY BROKEN implementation for this task at "{{variant}}" severity.

- "subtle": 90% correct, 1-2 subtle bugs (off-by-one, wrong import, missing edge case)
- "structural": Structure correct but core logic broken (wrong algorithm, missing steps)

TASK:
{{instruction}}

Output only the broken Python code:"""

ERROR_INSTR_PROMPT = """Rewrite this task to say the workspace contains a partially broken implementation at /app/starter_code.py. The agent must fix the bugs and complete the implementation.

ORIGINAL TASK:
{{instruction}}

Output only the rewritten instruction:"""

MODALITY_PROMPT = """Rewrite the following coding task using ONLY the "{{variant}}" specification modality.

- "nl_prose": Clear natural language paragraph
- "pseudocode": Pseudocode only, no natural language
- "io_examples": Only input/output examples (3+ examples, no prose)
- "type_signatures": Only Python type signatures and docstrings
- "failing_test_description": Describe as "write code that makes these tests pass" with tests described in words

ORIGINAL TASK:
{{instruction}}

Rewrite using ONLY "{{variant}}" modality. Output only the rewritten instruction:"""

CURATE_PROMPT = """Rate this coding task on a scale of 1-5 for training quality:
- Clarity of requirements (1-5)
- Testability (1-5)
- Appropriate difficulty (1-5)

TASK:
{{instruction}}

TESTS:
{{test_files_json}}

Output ONLY a single number 1-5 (the average):"""

# ─── Generator functions ─────────────────────────────────────

# Group 1: No LLM calls (fast)
def gen_baseline():
    print("=== Generating baseline (5000 tasks) ===")
    tasks = load_harbor_tasks_as_dicts(SOURCE, limit=LIMIT)
    out = Path(tempfile.mkdtemp(prefix="5k-baseline_"))
    for i, row in enumerate(tqdm(tasks, desc="baseline")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"], dockerfile=row["dockerfile"],
            test_sh=row["test_sh"], test_files=json.loads(row["test_files_json"]),
            dataset_prefix="5k-baseline", metadata={"source": row["task_dir_name"]},
        )
    upload_tasks_to_hf(str(out), f"{HF_ORG}/exp_flat25_baseline", commit_message="Upload 5000-task dataset")
    return str(out)

def gen_speed_bonus():
    print("=== Generating speed_bonus (5000 tasks) ===")
    tasks = load_harbor_tasks_as_dicts(SOURCE, limit=LIMIT)
    out = Path(tempfile.mkdtemp(prefix="5k-speed_"))
    for i, row in enumerate(tqdm(tasks, desc="speed_bonus")):
        instruction = row["instruction"] + "\n\nBONUS: Complete this task as quickly as possible. Faster solutions are preferred."
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=instruction, dockerfile=row["dockerfile"],
            test_sh=row["test_sh"], test_files=json.loads(row["test_files_json"]),
            dataset_prefix="5k-speed", metadata={"source": row["task_dir_name"]},
        )
    upload_tasks_to_hf(str(out), f"{HF_ORG}/exp_flat25_speed_bonus", commit_message="Upload 5000-task dataset")
    return str(out)

def gen_proportional():
    print("=== Generating proportional (5000 tasks) ===")
    tasks = load_harbor_tasks_as_dicts(SOURCE, limit=LIMIT)
    out = Path(tempfile.mkdtemp(prefix="5k-proportional_"))

    proportional_test_sh = r'''#!/bin/bash
set -uo pipefail
mkdir -p /logs/verifier
if [ ! -d /app/.venv ]; then
    python3 -m venv /app/.venv
fi
source /app/.venv/bin/activate
if ! command -v pytest &> /dev/null; then
    pip install --quiet pytest
fi
WHITELIST="requests numpy pandas scipy scikit-learn sklearn torch tensorflow keras httpx aiohttp ddtrace django flask fastapi matplotlib seaborn pillow pydantic pytest-mock requests-mock faker pyyaml pytz cryptography bcrypt hypothesis"
for pkg in $WHITELIST; do
    python3 -c "import ${pkg//-/_}" 2>/dev/null || pip install --quiet "$pkg" 2>/dev/null || true
done
cd /app
export PYTHONPATH="/app:${PYTHONPATH:-}"
TOTAL=0
PASSED=0
for test_file in /tests/test_*.py; do
    [ -f "$test_file" ] || continue
    TEST_COUNT=$(grep -c "def test_" "$test_file" 2>/dev/null || true)
    TEST_COUNT=${TEST_COUNT:-0}
    TOTAL=$((TOTAL + TEST_COUNT))
    RESULT=$(python -m pytest "$test_file" -v --tb=no 2>&1 || true)
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
if [ "$PASSED" -eq "$TOTAL" ]; then exit 0; else exit 1; fi
'''

    for i, row in enumerate(tqdm(tasks, desc="proportional")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"], dockerfile=row["dockerfile"],
            test_sh=proportional_test_sh,
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix="5k-proportional",
            metadata={"source": row["task_dir_name"], "reward_type": "proportional"},
        )
    upload_tasks_to_hf(str(out), f"{HF_ORG}/exp_rle_proportional", commit_message="Upload 5000-task dataset")
    return str(out)

# Group 2: Category A — 1 LLM call per task (instruction rewrite)
def gen_error_report():
    print("=== Generating error_report (5000 tasks) ===")
    path = generate_instruction_transform(
        prompt=STYLE_PROMPT, prefix="5k-error-report",
        hf_repo=f"{HF_ORG}/exp_rle_error_report",
        variant_col="variant", variant_value="error_report", limit=LIMIT,
    )
    return path

def gen_github_issue():
    print("=== Generating github_issue (5000 tasks) ===")
    return generate_instruction_transform(
        prompt=STYLE_PROMPT, prefix="5k-github-issue",
        hf_repo=f"{HF_ORG}/exp_rle_github_issue",
        variant_col="variant", variant_value="github_issue", limit=LIMIT,
    )

def gen_minimal():
    print("=== Generating minimal (5000 tasks) ===")
    return generate_instruction_transform(
        prompt=MINIMAL_PROMPT, prefix="5k-minimal",
        hf_repo=f"{HF_ORG}/exp_rle_minimal_instructions",
        variant_col="variant", variant_value="minimal", limit=LIMIT,
    )

def gen_detailed():
    print("=== Generating detailed (5000 tasks) ===")
    return generate_instruction_transform(
        prompt=GRANULARITY_PROMPT, prefix="5k-detailed",
        hf_repo=f"{HF_ORG}/exp_rle_detailed",
        variant_col="variant", variant_value="detailed", limit=LIMIT,
    )

def gen_expert():
    print("=== Generating expert (5000 tasks) ===")
    return generate_instruction_transform(
        prompt=KNOWLEDGE_PROMPT, prefix="5k-expert",
        hf_repo=f"{HF_ORG}/exp_rle_expert",
        variant_col="variant", variant_value="expert", limit=LIMIT,
    )

def gen_heavy():
    print("=== Generating heavy_padding (5000 tasks) ===")
    return generate_instruction_transform(
        prompt=PADDING_PROMPT, prefix="5k-heavy",
        hf_repo=f"{HF_ORG}/exp_rle_heavy_padding",
        variant_col="variant", variant_value="heavy", limit=LIMIT,
    )

def gen_moderate():
    print("=== Generating moderate_padding (5000 tasks) ===")
    return generate_instruction_transform(
        prompt=PADDING_PROMPT, prefix="5k-moderate",
        hf_repo=f"{HF_ORG}/exp_rle_moderate_padding",
        variant_col="variant", variant_value="moderate", limit=LIMIT,
    )

def gen_pseudocode():
    print("=== Generating pseudocode (5000 tasks) ===")
    return generate_instruction_transform(
        prompt=MODALITY_PROMPT, prefix="5k-pseudocode",
        hf_repo=f"{HF_ORG}/exp_flat25_pseudocode",
        variant_col="variant", variant_value="pseudocode", limit=LIMIT,
    )

def gen_stackoverflow():
    print("=== Generating stackoverflow (5000 tasks) ===")
    return generate_instruction_transform(
        prompt=STYLE_PROMPT, prefix="5k-stackoverflow",
        hf_repo=f"{HF_ORG}/exp_flat25_stackoverflow",
        variant_col="variant", variant_value="stackoverflow_question", limit=LIMIT,
    )

# Group 3: Category B — 2 LLM calls per task (instruction + tests)
def gen_2skill():
    print("=== Generating 2skill (5000 tasks) ===")
    instr_prompt = """Rewrite this task to require EXACTLY 2 skills: the original skill PLUS file I/O.

The rewritten task should:
1. Keep the original algorithmic requirement
2. Add a requirement to read input from a file and write output to a file
3. Be a single cohesive task (not two separate tasks)

ORIGINAL TASK:
{{instruction}}

Output only the rewritten 2-skill task:"""

    test_prompt = """Adapt these tests for a version of the task that also requires file I/O (reading input from a file, writing output to a file). Add tests for the file I/O aspect while keeping the core logic tests.

ORIGINAL TESTS:
{{test_files_json}}

Write adapted tests. Output only the test code:"""

    return generate_instruction_and_tests_transform(
        instruction_prompt=instr_prompt, test_prompt=test_prompt,
        test_filename="test_2skill.py", prefix="5k-2skill",
        hf_repo=f"{HF_ORG}/exp_rle_2skill",
        variant_col="n_skills", variant_value="2", limit=LIMIT,
    )

def gen_partial():
    print("=== Generating partial_ambiguity (5000 tasks) ===")
    return generate_instruction_and_tests_transform(
        instruction_prompt=AMBIGUITY_PROMPT, test_prompt=PROPERTY_TEST_PROMPT,
        test_filename="test_properties.py", prefix="5k-partial",
        hf_repo=f"{HF_ORG}/exp_rle_partial_ambiguity",
        variant_col="variant", variant_value="partial", limit=LIMIT,
    )

def gen_adversarial():
    print("=== Generating adversarial (5000 tasks) ===")
    return generate_instruction_and_tests_transform(
        instruction_prompt=ADVERSARIAL_INSTR_PROMPT, test_prompt=ADVERSARIAL_TEST_PROMPT,
        test_filename="test_adversarial.py", prefix="5k-adversarial",
        hf_repo=f"{HF_ORG}/exp_rle_adversarial",
        variant_col="variant", variant_value="adversarial", limit=LIMIT,
    )

# Group 4: Category C — 2 LLM calls per task (broken code + instruction rewrite)
def gen_structural():
    print("=== Generating structural_debug (5000 tasks) ===")
    return generate_error_state(
        broken_code_prompt=BROKEN_CODE_PROMPT, instruction_prompt=ERROR_INSTR_PROMPT,
        prefix="5k-structural", hf_repo=f"{HF_ORG}/exp_rle_structural_debug",
        variant_col="variant", variant_value="structural", limit=LIMIT,
    )

def gen_subtle():
    print("=== Generating subtle_debug (5000 tasks) ===")
    return generate_error_state(
        broken_code_prompt=BROKEN_CODE_PROMPT, instruction_prompt=ERROR_INSTR_PROMPT,
        prefix="5k-subtle", hf_repo=f"{HF_ORG}/exp_flat25_subtle_debug",
        variant_col="variant", variant_value="subtle", limit=LIMIT,
    )

# Group 5: Curated (special — scores 10000 then picks top 5000)
def gen_curated():
    print("=== Generating curated (5000 tasks) ===")
    tasks = load_harbor_tasks_as_dicts(SOURCE, limit=9985)
    print(f"  Loaded {len(tasks)} tasks for scoring")

    dataset = Dataset.from_list(tasks)
    result = run_completions(
        dataset, model="gpt-5-nano-2025-08-07", map_type="chat",
        map_config={"user_message": CURATE_PROMPT, "output_column": "quality_score"},
        require_all_responses=False,
    )
    scored = result.dataset
    print(f"  Scored {len(scored)} tasks")

    scored_list = []
    for row in scored:
        try:
            score = float(row["quality_score"].strip()[:1])
            scored_list.append((score, row))
        except (ValueError, IndexError):
            continue

    scored_list.sort(key=lambda x: -x[0])
    top_tasks = [row for _, row in scored_list[:LIMIT]]
    print(f"  Selected top {len(top_tasks)} tasks")

    out = Path(tempfile.mkdtemp(prefix="5k-curated_"))
    for i, row in enumerate(tqdm(top_tasks, desc="Creating tasks")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"], dockerfile=row["dockerfile"],
            test_sh=row["test_sh"], test_files=json.loads(row["test_files_json"]),
            dataset_prefix="5k-curated",
            metadata={"source": row.get("task_dir_name", ""), "quality_score": row["quality_score"]},
        )

    upload_tasks_to_hf(str(out), f"{HF_ORG}/exp_rle_curated", commit_message="Upload 5000-task dataset")
    return str(out)


# ─── Main ────────────────────────────────────────────────────

ALL_GENERATORS = {
    # No LLM (fast)
    "baseline": gen_baseline,
    "speed_bonus": gen_speed_bonus,
    "proportional": gen_proportional,
    # 1 LLM call
    "error_report": gen_error_report,
    "github_issue": gen_github_issue,
    "minimal": gen_minimal,
    "detailed": gen_detailed,
    "expert": gen_expert,
    "heavy": gen_heavy,
    "moderate": gen_moderate,
    "pseudocode": gen_pseudocode,
    "stackoverflow": gen_stackoverflow,
    # 2 LLM calls
    "2skill": gen_2skill,
    "partial": gen_partial,
    "adversarial": gen_adversarial,
    "structural": gen_structural,
    "subtle": gen_subtle,
    # Special
    "curated": gen_curated,
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", help="Subset to generate")
    parser.add_argument("--workers", type=int, default=6, help="Max parallel workers")
    args = parser.parse_args()

    gens = ALL_GENERATORS
    if args.experiments:
        gens = {k: v for k, v in gens.items() if any(k.startswith(p) or p in k for p in args.experiments)}

    print(f"Generating {len(gens)} datasets, {LIMIT} tasks each")
    print(f"Workers: {args.workers}")
    print(f"Datasets: {', '.join(gens.keys())}\n")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fn): name for name, fn in gens.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                task_dir = future.result()
                n_tasks = len(list(Path(task_dir).glob("*/instruction.md"))) if task_dir else 0
                results[name] = {"task_dir": task_dir, "n_tasks": n_tasks, "status": "ok"}
                print(f"\n  DONE {name}: {n_tasks} tasks at {task_dir}")
            except Exception as e:
                results[name] = {"error": str(e), "status": "failed"}
                print(f"\n  FAILED {name}: {e}")
                import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    for name in sorted(results):
        r = results[name]
        if r["status"] == "ok":
            print(f"  {name:<25} {r['n_tasks']:>5} tasks  {r['task_dir']}")
        else:
            print(f"  {name:<25} FAILED: {r['error']}")

    ok = sum(1 for r in results.values() if r["status"] == "ok")
    total_tasks = sum(r.get("n_tasks", 0) for r in results.values() if r["status"] == "ok")
    print(f"\n{ok}/{len(gens)} succeeded, {total_tasks} total tasks generated")
