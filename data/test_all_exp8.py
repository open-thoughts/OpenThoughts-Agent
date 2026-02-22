#!/usr/bin/env python3
"""
Generate 25 tasks per Theme 8 experiment variant, then run harbor pass rates with gpt-5-nano.

Usage:
    python data/test_all_exp8.py
    python data/test_all_exp8.py --generate-only
    python data/test_all_exp8.py --test-only
    python data/test_all_exp8.py --experiments 8.1 8.5
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "harbor"))

LIMIT = 5
AGENT_MODEL = "openai/gpt-5-nano-2025-08-07"
CONFIG = str(PROJECT_ROOT / "eval/tacc/dcagent_eval_config.yaml")
HF_SUFFIX = "_test5"

from data.exp_8_utils import (
    generate_instruction_transform,
    generate_instruction_and_tests_transform,
    generate_error_state,
    generate_task_stacking,
)

# ─── Prompts ────────────────────────────────────────────────────

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

MODALITY_PROMPT = """Rewrite the following coding task using ONLY the "{{variant}}" specification modality.

- "nl_prose": Clear natural language paragraph
- "pseudocode": Pseudocode only, no natural language
- "io_examples": Only input/output examples (3+ examples, no prose)
- "type_signatures": Only Python type signatures and docstrings
- "failing_test_description": Describe as "write code that makes these tests pass" with tests described in words

ORIGINAL TASK:
{{instruction}}

Rewrite using ONLY "{{variant}}" modality. Output only the rewritten instruction:"""

GRANULARITY_PROMPT = """Rewrite the following coding task at "{{variant}}" granularity level.

- "vague": Single vague sentence. No details, no file hints.
- "bullets": 3-5 requirement bullets. Key requirements but no implementation hints.
- "detailed": Step-by-step with file structure hints, function signatures, edge cases.

ORIGINAL TASK:
{{instruction}}

Rewrite at "{{variant}}" granularity. Output only the rewritten instruction:"""

CONSTRAINT_INSTR_PROMPT = """Add {{n_constraints}} escalating constraints to this task from this list:
1. No external libraries (stdlib only)
2. Single file solution
3. Memory limit (max 50MB)
4. Time limit (complete in < 2 seconds)

ORIGINAL TASK:
{{instruction}}

Rewrite with the first {{n_constraints}} constraints. Output only the rewritten instruction:"""

CONSTRAINT_TEST_PROMPT = """Given these constraints and existing test code, write ADDITIONAL pytest assertions verifying constraint compliance.

CONSTRAINTS: First {{n_constraints}} of: no external libs, single file, memory < 50MB, time < 2s

EXISTING TESTS:
{{test_files_json}}

Output only the additional test code to append:"""

PADDING_PROMPT = """Add irrelevant context/noise to this task at "{{variant}}" level. Real requirements must stay intact but be buried in noise.

- "mild": 1-2 irrelevant sentences of backstory
- "moderate": Paragraph of project context + red herring requirements
- "heavy": Verbose logs, unrelated code snippets, misleading hints. Real task buried deep.

ORIGINAL TASK:
{{instruction}}

Add "{{variant}}" padding. Output only the padded instruction:"""

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

DOMAIN_INSTR_PROMPT = """Transplant this algorithmic task into "{{variant}}" domain with domain-specific vocabulary.

- "medical": Patient records, HL7, clinical data
- "financial": Transactions, SWIFT, portfolio data
- "devops": Logs, metrics, YAML configs
- "ecommerce": Product catalogs, carts, inventory

ORIGINAL TASK:
{{instruction}}

Rewrite for "{{variant}}" domain. Same core algorithm. Output only the rewritten instruction:"""

DOMAIN_TEST_PROMPT = """Adapt these tests for "{{variant}}" domain. Replace generic test data with domain-realistic data while testing the same core logic.

ORIGINAL TESTS:
{{test_files_json}}

Write adapted tests. Output only the new test code:"""

COMPOUND_PROMPT = """Combine these 2 separate tasks into ONE compound task with a connecting narrative.

TASK 1:
{{instruction_1}}

TASK 2:
{{instruction_2}}

Write a compound instruction. Output only the compound instruction:"""

COMBINED_TEST_PROMPT = """Combine these separate test suites into one file. All tests must pass independently.

TEST SUITE 1:
{{test_files_1}}

TEST SUITE 2:
{{test_files_2}}

Write a combined test file. Output only the combined test code:"""

KNOWLEDGE_PROMPT = """Rewrite this task at "{{variant}}" prerequisite knowledge level.

- "expert": Terse, assumes complete domain knowledge. No algorithm explanation.
- "intermediate": Explains algorithm steps but assumes programming competence.
- "novice": Fully explained with pseudocode or step-by-step algorithm.

ORIGINAL TASK:
{{instruction}}

Rewrite at "{{variant}}" level. Output only the rewritten instruction:"""


# ─── Experiment definitions ─────────────────────────────────────

def _cat_a(prompt, prefix, hf_repo, variant_col, variant_value):
    """Category A: instruction-only transform."""
    return generate_instruction_transform(
        prompt=prompt, prefix=prefix, hf_repo=hf_repo + HF_SUFFIX,
        variant_col=variant_col, variant_value=variant_value, limit=LIMIT,
    )

def _cat_b(instr_prompt, test_prompt, test_filename, prefix, hf_repo, variant_col, variant_value):
    """Category B: instruction + test transform."""
    return generate_instruction_and_tests_transform(
        instruction_prompt=instr_prompt, test_prompt=test_prompt,
        test_filename=test_filename, prefix=prefix, hf_repo=hf_repo + HF_SUFFIX,
        variant_col=variant_col, variant_value=variant_value, limit=LIMIT,
    )

EXPERIMENTS = {
    # 8.1 Style Transfer
    "8.1_github_issue": lambda: _cat_a(STYLE_PROMPT, "exp-8-1-gh", "DCAgent/exp_8_1_style_transfer_github_issue", "variant", "github_issue"),
    "8.1_slack_message": lambda: _cat_a(STYLE_PROMPT, "exp-8-1-sl", "DCAgent/exp_8_1_style_transfer_slack_message", "variant", "slack_message"),
    "8.1_stackoverflow": lambda: _cat_a(STYLE_PROMPT, "exp-8-1-so", "DCAgent/exp_8_1_style_transfer_stackoverflow_question", "variant", "stackoverflow_question"),
    "8.1_code_review": lambda: _cat_a(STYLE_PROMPT, "exp-8-1-cr", "DCAgent/exp_8_1_style_transfer_code_review_comment", "variant", "code_review_comment"),
    "8.1_error_report": lambda: _cat_a(STYLE_PROMPT, "exp-8-1-er", "DCAgent/exp_8_1_style_transfer_error_report", "variant", "error_report"),
    # 8.2 Spec Modality
    "8.2_nl_prose": lambda: _cat_a(MODALITY_PROMPT, "exp-8-2-nl", "DCAgent/exp_8_2_spec_modality_nl_prose", "variant", "nl_prose"),
    "8.2_pseudocode": lambda: _cat_a(MODALITY_PROMPT, "exp-8-2-ps", "DCAgent/exp_8_2_spec_modality_pseudocode", "variant", "pseudocode"),
    "8.2_io_examples": lambda: _cat_a(MODALITY_PROMPT, "exp-8-2-io", "DCAgent/exp_8_2_spec_modality_io_examples", "variant", "io_examples"),
    "8.2_type_signatures": lambda: _cat_a(MODALITY_PROMPT, "exp-8-2-ts", "DCAgent/exp_8_2_spec_modality_type_signatures", "variant", "type_signatures"),
    "8.2_failing_test": lambda: _cat_a(MODALITY_PROMPT, "exp-8-2-ft", "DCAgent/exp_8_2_spec_modality_failing_test_description", "variant", "failing_test_description"),
    # 8.3 Granularity
    "8.3_vague": lambda: _cat_a(GRANULARITY_PROMPT, "exp-8-3-vg", "DCAgent/exp_8_3_granularity_vague", "variant", "vague"),
    "8.3_bullets": lambda: _cat_a(GRANULARITY_PROMPT, "exp-8-3-bl", "DCAgent/exp_8_3_granularity_bullets", "variant", "bullets"),
    "8.3_detailed": lambda: _cat_a(GRANULARITY_PROMPT, "exp-8-3-dt", "DCAgent/exp_8_3_granularity_detailed", "variant", "detailed"),
    # 8.4 Constraint Ladder
    "8.4_1constraint": lambda: _cat_b(CONSTRAINT_INSTR_PROMPT, CONSTRAINT_TEST_PROMPT, "test_constraints.py", "exp-8-4-c1", "DCAgent/exp_8_4_constraint_ladder_1", "n_constraints", "1"),
    "8.4_2constraints": lambda: _cat_b(CONSTRAINT_INSTR_PROMPT, CONSTRAINT_TEST_PROMPT, "test_constraints.py", "exp-8-4-c2", "DCAgent/exp_8_4_constraint_ladder_2", "n_constraints", "2"),
    "8.4_3constraints": lambda: _cat_b(CONSTRAINT_INSTR_PROMPT, CONSTRAINT_TEST_PROMPT, "test_constraints.py", "exp-8-4-c3", "DCAgent/exp_8_4_constraint_ladder_3", "n_constraints", "3"),
    "8.4_4constraints": lambda: _cat_b(CONSTRAINT_INSTR_PROMPT, CONSTRAINT_TEST_PROMPT, "test_constraints.py", "exp-8-4-c4", "DCAgent/exp_8_4_constraint_ladder_4", "n_constraints", "4"),
    # 8.5 Context Padding
    "8.5_mild": lambda: _cat_a(PADDING_PROMPT, "exp-8-5-mi", "DCAgent/exp_8_5_context_padding_mild", "variant", "mild"),
    "8.5_moderate": lambda: _cat_a(PADDING_PROMPT, "exp-8-5-mo", "DCAgent/exp_8_5_context_padding_moderate", "variant", "moderate"),
    "8.5_heavy": lambda: _cat_a(PADDING_PROMPT, "exp-8-5-hv", "DCAgent/exp_8_5_context_padding_heavy", "variant", "heavy"),
    # 8.6 Error State
    "8.6_subtle": lambda: generate_error_state(
        broken_code_prompt=BROKEN_CODE_PROMPT, instruction_prompt=ERROR_INSTR_PROMPT,
        prefix="exp-8-6-sub", hf_repo="DCAgent/exp_8_6_error_state_subtle" + HF_SUFFIX,
        variant_col="variant", variant_value="subtle", limit=LIMIT,
    ),
    "8.6_structural": lambda: generate_error_state(
        broken_code_prompt=BROKEN_CODE_PROMPT, instruction_prompt=ERROR_INSTR_PROMPT,
        prefix="exp-8-6-str", hf_repo="DCAgent/exp_8_6_error_state_structural" + HF_SUFFIX,
        variant_col="variant", variant_value="structural", limit=LIMIT,
    ),
    # 8.7 Ambiguity
    "8.7_partial": lambda: _cat_b(AMBIGUITY_PROMPT, PROPERTY_TEST_PROMPT, "test_properties.py", "exp-8-7-pa", "DCAgent/exp_8_7_ambiguity_partial", "variant", "partial"),
    "8.7_minimal": lambda: _cat_b(AMBIGUITY_PROMPT, PROPERTY_TEST_PROMPT, "test_properties.py", "exp-8-7-mi", "DCAgent/exp_8_7_ambiguity_minimal", "variant", "minimal"),
    # 8.8 Domain Shift
    "8.8_medical": lambda: _cat_b(DOMAIN_INSTR_PROMPT, DOMAIN_TEST_PROMPT, "test_domain.py", "exp-8-8-med", "DCAgent/exp_8_8_domain_shift_medical", "variant", "medical"),
    "8.8_financial": lambda: _cat_b(DOMAIN_INSTR_PROMPT, DOMAIN_TEST_PROMPT, "test_domain.py", "exp-8-8-fin", "DCAgent/exp_8_8_domain_shift_financial", "variant", "financial"),
    "8.8_devops": lambda: _cat_b(DOMAIN_INSTR_PROMPT, DOMAIN_TEST_PROMPT, "test_domain.py", "exp-8-8-dev", "DCAgent/exp_8_8_domain_shift_devops", "variant", "devops"),
    "8.8_ecommerce": lambda: _cat_b(DOMAIN_INSTR_PROMPT, DOMAIN_TEST_PROMPT, "test_domain.py", "exp-8-8-eco", "DCAgent/exp_8_8_domain_shift_ecommerce", "variant", "ecommerce"),
    # 8.9 Task Stacking
    "8.9_pairs": lambda: generate_task_stacking(
        compound_prompt=COMPOUND_PROMPT, test_prompt=COMBINED_TEST_PROMPT,
        prefix="exp-8-9-pr", hf_repo="DCAgent/exp_8_9_task_stacking_pairs" + HF_SUFFIX,
        limit=LIMIT,
    ),
    # 8.10 Knowledge Gradient
    "8.10_expert": lambda: _cat_a(KNOWLEDGE_PROMPT, "exp-8-10-ex", "DCAgent/exp_8_10_knowledge_gradient_expert", "variant", "expert"),
    "8.10_intermediate": lambda: _cat_a(KNOWLEDGE_PROMPT, "exp-8-10-in", "DCAgent/exp_8_10_knowledge_gradient_intermediate", "variant", "intermediate"),
    "8.10_novice": lambda: _cat_a(KNOWLEDGE_PROMPT, "exp-8-10-no", "DCAgent/exp_8_10_knowledge_gradient_novice", "variant", "novice"),
}


# ─── Harbor testing ─────────────────────────────────────────────

def run_harbor_passrate(task_dir: str, name: str) -> dict:
    """Run harbor pass rate test on a local task directory."""
    print(f"\n--- Harbor test: {name} ---")
    job_name = f"exp8_{name}_{os.getpid()}"

    cmd = [
        "harbor", "jobs", "start",
        "-p", task_dir,
        "--n-concurrent", "8",
        "--agent", "terminus-2",
        "--model", AGENT_MODEL,
        "--env", "daytona",
        "--n-attempts", "1",
        "--max-retries", "0",
        "--job-name", job_name,
        "--config", CONFIG,
    ]

    print(f"  Running harbor ({AGENT_MODEL})...")
    os.chdir(str(PROJECT_ROOT))
    # Source secret.env for Daytona API key
    env = os.environ.copy()
    secret_env = Path("/scratch/10000/eguha3/old-dc-agent/secret.env")
    if secret_env.exists():
        for line in secret_env.read_text().splitlines():
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                key, val = line[7:].split("=", 1)
                env[key] = os.path.expandvars(val)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    if result.returncode != 0 and result.stderr:
        print(f"  stderr: {result.stderr[:300]}")

    # Read results from individual trial directories
    job_dir = PROJECT_ROOT / "jobs" / job_name
    if not job_dir.exists():
        print(f"  No job directory found")
        return {"name": name, "error": "no job dir"}

    passed = 0
    total = 0
    errors = 0
    for trial_dir in sorted(job_dir.iterdir()):
        result_file = trial_dir / "result.json"
        if not result_file.exists():
            continue
        with open(result_file) as f:
            data = json.load(f)
        vr = data.get("verifier_result")
        if vr and "rewards" in vr:
            reward = vr["rewards"].get("reward", 0)
            total += 1
            if reward > 0:
                passed += 1
        elif data.get("exception_info"):
            errors += 1
            total += 1

    if total == 0:
        print(f"  No results found in trial dirs")
        return {"name": name, "error": "no results"}

    rate = (passed / total * 100) if total > 0 else 0
    print(f"  Result: {passed}/{total} ({rate:.1f}%) [{errors} errors]")
    return {"name": name, "passed": passed, "total": total, "pass_rate": rate, "errors": errors}


# ─── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--experiments", nargs="+",
                        help="Filter by prefix (e.g. '8.1' '8.5_mild')")
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.experiments:
        exps = {k: v for k, v in exps.items()
                if any(k.startswith(p) for p in args.experiments)}
    print(f"Running {len(exps)} experiments with limit={LIMIT}")

    manifest_path = PROJECT_ROOT / "data" / "exp8_test_manifest.json"
    task_dirs = {}

    # Phase 1: Generate
    if not args.test_only:
        print(f"\n{'#'*60}")
        print(f"PHASE 1: GENERATING ({len(exps)} experiments, {LIMIT} tasks each)")
        print(f"{'#'*60}")

        for name, gen_fn in exps.items():
            print(f"\n{'='*60}")
            print(f"GENERATING: {name}")
            print(f"{'='*60}")
            try:
                task_dir = gen_fn()
                if task_dir and Path(task_dir).exists():
                    n = len(list(Path(task_dir).glob("*/instruction.md")))
                    task_dirs[name] = task_dir
                    print(f"  -> OK: {n} tasks at {task_dir}")
                else:
                    print(f"  -> FAILED (no output)")
            except Exception as e:
                print(f"  -> ERROR: {e}")
                import traceback; traceback.print_exc()

        print(f"\nGeneration: {len(task_dirs)}/{len(exps)} succeeded")
        # Merge with existing manifest (don't overwrite entries from other runs)
        existing = {}
        if manifest_path.exists():
            with open(manifest_path) as f:
                existing = json.load(f)
        existing.update(task_dirs)
        with open(manifest_path, "w") as f:
            json.dump(existing, f, indent=2)

    if args.generate_only:
        print("--generate-only: done")
        return

    # Phase 2: Harbor tests
    if args.test_only:
        if manifest_path.exists():
            with open(manifest_path) as f:
                task_dirs = json.load(f)
            if args.experiments:
                task_dirs = {k: v for k, v in task_dirs.items()
                             if any(k.startswith(p) for p in args.experiments)}
            print(f"Loaded {len(task_dirs)} from manifest")
        else:
            print("No manifest. Run without --test-only first.")
            return

    print(f"\n{'#'*60}")
    print(f"PHASE 2: HARBOR PASS RATE ({len(task_dirs)} experiments)")
    print(f"Model: {AGENT_MODEL}")
    print(f"{'#'*60}")

    all_results = []
    for name, task_dir in task_dirs.items():
        if not Path(task_dir).exists():
            all_results.append({"name": name, "error": "dir missing"})
            continue
        try:
            r = run_harbor_passrate(task_dir, name)
            all_results.append(r)
        except Exception as e:
            all_results.append({"name": name, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"{'Experiment':<30} {'Pass Rate':>12} {'Detail':>12}")
    print("-" * 56)
    for r in sorted(all_results, key=lambda x: x["name"]):
        if "error" in r:
            print(f"{r['name']:<30} {'ERROR':>12} {r['error']}")
        else:
            print(f"{r['name']:<30} {r['pass_rate']:>11.1f}% {r['passed']}/{r['total']}")

    with open(PROJECT_ROOT / "data" / "exp8_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to data/exp8_test_results.json")


if __name__ == "__main__":
    main()
