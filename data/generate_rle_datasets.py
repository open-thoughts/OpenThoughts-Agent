#!/usr/bin/env python3
"""Generate 500-task RL experiment datasets for experiments with best scaling progression.

Selected experiments (monotone increasing pass rate across nano→mini→gpt5):
  8.1_error_report:  20% → 60% → 60%  (+40pp)
  5.4_2skill:        20% → 40% → 40%  (+20pp)
  8.7_partial:       20% → 40% → 40%  (+20pp)
  5.2_adversarial:   20% → 20% → 40%  (+20pp)
  8.6_structural:    40% → 60% → 60%  (+20pp)

Usage:
    python data/generate_rle_datasets.py
"""

import sys
import os
import concurrent.futures
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from data.exp_8_utils import (
    generate_instruction_transform,
    generate_instruction_and_tests_transform,
    generate_error_state,
)

LIMIT = 500
HF_ORG = "DCAgent"

# ─── Prompts (identical to test_all_exp8.py) ─────────────────

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

COMPOUND_PROMPT = """Combine these 2 separate tasks into ONE compound task with a connecting narrative.

TASK 1:
{{instruction_1}}

TASK 2:
{{instruction_2}}

Write a compound instruction. Output only the compound instruction:"""

# ─── Generator functions ─────────────────────────────────────

def gen_error_report():
    """8.1_error_report: Bug report style instructions (Category A)."""
    print("=== Generating 8.1_error_report (500 tasks) ===")
    return generate_instruction_transform(
        prompt=STYLE_PROMPT,
        prefix="exp-rle-error-report",
        hf_repo=f"{HF_ORG}/exp_rle_error_report",
        variant_col="variant",
        variant_value="error_report",
        limit=LIMIT,
    )

def gen_2skill():
    """5.4_2skill: 2-skill compositional tasks (Category B)."""
    print("=== Generating 5.4_2skill (500 tasks) ===")
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
        instruction_prompt=instr_prompt,
        test_prompt=test_prompt,
        test_filename="test_2skill.py",
        prefix="exp-rle-2skill",
        hf_repo=f"{HF_ORG}/exp_rle_2skill",
        variant_col="n_skills",
        variant_value="2",
        limit=LIMIT,
    )

def gen_partial():
    """8.7_partial: Partial ambiguity with property tests (Category B)."""
    print("=== Generating 8.7_partial (500 tasks) ===")
    return generate_instruction_and_tests_transform(
        instruction_prompt=AMBIGUITY_PROMPT,
        test_prompt=PROPERTY_TEST_PROMPT,
        test_filename="test_properties.py",
        prefix="exp-rle-partial",
        hf_repo=f"{HF_ORG}/exp_rle_partial_ambiguity",
        variant_col="variant",
        variant_value="partial",
        limit=LIMIT,
    )

def gen_adversarial():
    """5.2_adversarial: Adversarial test generation (Category B)."""
    print("=== Generating 5.2_adversarial (500 tasks) ===")
    return generate_instruction_and_tests_transform(
        instruction_prompt=ADVERSARIAL_INSTR_PROMPT,
        test_prompt=ADVERSARIAL_TEST_PROMPT,
        test_filename="test_adversarial.py",
        prefix="exp-rle-adversarial",
        hf_repo=f"{HF_ORG}/exp_rle_adversarial",
        variant_col="variant",
        variant_value="adversarial",
        limit=LIMIT,
    )

def gen_structural():
    """8.6_structural: Broken starter code, structural bugs (Category C)."""
    print("=== Generating 8.6_structural (500 tasks) ===")
    return generate_error_state(
        broken_code_prompt=BROKEN_CODE_PROMPT,
        instruction_prompt=ERROR_INSTR_PROMPT,
        prefix="exp-rle-structural",
        hf_repo=f"{HF_ORG}/exp_rle_structural_debug",
        variant_col="variant",
        variant_value="structural",
        limit=LIMIT,
    )


# ─── Main ────────────────────────────────────────────────────

GENERATORS = {
    "8.1_error_report": gen_error_report,
    "5.4_2skill": gen_2skill,
    "8.7_partial": gen_partial,
    "5.2_adversarial": gen_adversarial,
    "8.6_structural": gen_structural,
}

if __name__ == "__main__":
    import json

    print(f"Generating {len(GENERATORS)} datasets, {LIMIT} tasks each")
    print(f"Running in parallel...\n")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GENERATORS)) as pool:
        futures = {pool.submit(fn): name for name, fn in GENERATORS.items()}
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

    # Summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    for name in sorted(results):
        r = results[name]
        if r["status"] == "ok":
            print(f"  {name:<25} {r['n_tasks']:>4} tasks  {r['task_dir']}")
        else:
            print(f"  {name:<25} FAILED: {r['error']}")

    # Save manifest
    manifest = {k: v["task_dir"] for k, v in results.items() if v["status"] == "ok"}
    manifest_path = PROJECT_ROOT / "data" / "rle_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
