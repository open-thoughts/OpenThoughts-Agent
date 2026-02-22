#!/usr/bin/env python3
"""
Generate 25 tasks from a test-generating experiment and run through harbor.
Usage: python generate_25_and_test.py <experiment_name>

Example: python generate_25_and_test.py 8.4_constraint_1
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "harbor"))
sys.path.insert(0, str(Path(__file__).parent))

from data.commons import load_harbor_tasks_as_dicts, create_harbor_task_directory_generic, create_standard_dockerfile, upload_tasks_to_hf
from data.completions import run_completions
from datasets import Dataset, load_dataset
from tqdm import tqdm
import itertools

MODEL = "gpt-5-nano-2025-08-07"
HARBOR_MODEL = "openai/gpt-5-nano-2025-08-07"
SOURCE_REPO = "DCAgent/exp_rpt_stack-pytest"
LIMIT = 25
N_CONCURRENT = 8


def _strip_llm_code(text: str) -> str:
    """Strip markdown fences and LLM preamble from generated code."""
    import re
    # Extract code from markdown fences if present
    match = re.search(r'```(?:python|rust|go|typescript|bash)?\s*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fences, strip any preamble lines before first code line
    lines = text.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(('import ', 'from ', '#!', 'def ', 'class ', 'use ', 'package ', 'const ', 'fn ', '@')):
            return '\n'.join(lines[i:]).strip()
    return text.strip()


def _make_test_sh(test_file="/tests/test_solution.py", test_cmd=None):
    """Create a proper test.sh that checks exit codes correctly."""
    if test_cmd is None:
        test_cmd = f"python -m pytest {test_file} -v --tb=short"
    return f"""#!/bin/bash
set -e
mkdir -p /logs/verifier

cleanup() {{
    if [ $? -eq 0 ]; then
        echo "1" > /logs/verifier/reward.txt
    else
        echo "0" > /logs/verifier/reward.txt
    fi
}}
trap cleanup EXIT

cd /app
{test_cmd} 2>&1 | tee /logs/verifier/output.txt

TEST_EXIT=${{PIPESTATUS[0]}}
if [ $TEST_EXIT -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed."
    exit 1
fi
"""


# ============================================================================
# Experiment definitions: each generates 25 tasks with new LLM-written tests
# ============================================================================

def gen_8_4_constraint_1():
    """8.4 Constraint Ladder — 1 constraint."""
    tasks = load_harbor_tasks_as_dicts(SOURCE_REPO, limit=LIMIT)
    for t in tasks:
        t["n_constraints"] = "1"
    ds = Dataset.from_list(tasks)

    r1 = run_completions(ds, model=MODEL, map_type="chat", map_config={
        "user_message": "Add {{n_constraints}} escalating constraints to this task from: 1. No external libs (stdlib only) 2. Single file 3. Memory < 50MB 4. Time < 2s\n\nORIGINAL:\n{{instruction}}\n\nRewrite with first {{n_constraints}} constraints. Output only the rewritten instruction:",
        "output_column": "new_instruction"}, require_all_responses=False)

    r2 = run_completions(r1.dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Write ADDITIONAL pytest assertions for constraint: no external libs.\n\nEXISTING TESTS:\n{{test_files_json}}\n\nOutput only the test code:",
        "output_column": "new_tests"}, require_all_responses=False)

    return r2.dataset, "exp-8-4", lambda row: {
        "instruction": row["new_instruction"], "dockerfile": row["dockerfile"],
        "test_sh": row["test_sh"],
        "test_files": {**json.loads(row["test_files_json"]), "test_constraints.py": _strip_llm_code(row["new_tests"])},
        "metadata": {"source": row["task_dir_name"], "n_constraints": "1"},
    }


def gen_8_7_ambiguity():
    """8.7 Ambiguity — partial underspecification + property tests."""
    tasks = load_harbor_tasks_as_dicts(SOURCE_REPO, limit=LIMIT)
    for t in tasks:
        t["variant"] = "partial"
    ds = Dataset.from_list(tasks)

    r1 = run_completions(ds, model=MODEL, map_type="chat", map_config={
        "user_message": "Rewrite this task to be deliberately underspecified at \"partial\" level. Remove some details but keep function names.\n\nORIGINAL:\n{{instruction}}\n\nOutput only the underspecified instruction:",
        "output_column": "new_instruction"}, require_all_responses=False)

    r2 = run_completions(r1.dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Convert these tests into PROPERTY-BASED tests. Replace equality checks with property assertions.\n\nORIGINAL TESTS:\n{{test_files_json}}\n\nWrite property-based pytest tests:",
        "output_column": "new_tests"}, require_all_responses=False)

    return r2.dataset, "exp-8-7", lambda row: {
        "instruction": row["new_instruction"], "dockerfile": row["dockerfile"],
        "test_sh": row["test_sh"],
        "test_files": {"test_properties.py": _strip_llm_code(row["new_tests"])},
        "metadata": {"source": row["task_dir_name"], "variant": "partial"},
    }


def gen_8_8_domain_medical():
    """8.8 Domain Shift — medical domain."""
    tasks = load_harbor_tasks_as_dicts(SOURCE_REPO, limit=LIMIT)
    for t in tasks:
        t["variant"] = "medical"
    ds = Dataset.from_list(tasks)

    r1 = run_completions(ds, model=MODEL, map_type="chat", map_config={
        "user_message": "Transplant this task into medical domain with domain-specific vocabulary (patient records, HL7, clinical data).\n\nORIGINAL:\n{{instruction}}\n\nRewrite for medical domain. Output only the instruction:",
        "output_column": "new_instruction"}, require_all_responses=False)

    r2 = run_completions(r1.dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Adapt these tests for medical domain. Replace generic data with domain-realistic data.\n\nORIGINAL TESTS:\n{{test_files_json}}\n\nWrite adapted tests:",
        "output_column": "new_tests"}, require_all_responses=False)

    return r2.dataset, "exp-8-8", lambda row: {
        "instruction": row["new_instruction"], "dockerfile": row["dockerfile"],
        "test_sh": row["test_sh"],
        "test_files": {"test_domain.py": _strip_llm_code(row["new_tests"])},
        "metadata": {"source": row["task_dir_name"], "variant": "medical"},
    }


def gen_8_9_stacking():
    """8.9 Task Stacking — compound task pairs."""
    tasks = load_harbor_tasks_as_dicts(SOURCE_REPO, limit=LIMIT * 2)
    grouped = []
    for i in range(0, len(tasks) - 1, 2):
        grouped.append({
            "instruction_1": tasks[i]["instruction"], "instruction_2": tasks[i+1]["instruction"],
            "test_files_1": tasks[i]["test_files_json"], "test_files_2": tasks[i+1]["test_files_json"],
            "dockerfile": tasks[i]["dockerfile"], "test_sh": tasks[i]["test_sh"],
            "source_1": tasks[i]["task_dir_name"], "source_2": tasks[i+1]["task_dir_name"],
        })
    ds = Dataset.from_list(grouped[:LIMIT])

    r1 = run_completions(ds, model=MODEL, map_type="chat", map_config={
        "user_message": "Combine these 2 tasks into ONE compound task.\n\nTASK 1:\n{{instruction_1}}\n\nTASK 2:\n{{instruction_2}}\n\nWrite compound instruction:",
        "output_column": "new_instruction"}, require_all_responses=False)

    r2 = run_completions(r1.dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Combine these test suites into one file.\n\nSUITE 1:\n{{test_files_1}}\n\nSUITE 2:\n{{test_files_2}}\n\nWrite combined test code:",
        "output_column": "new_tests"}, require_all_responses=False)

    return r2.dataset, "exp-8-9", lambda row: {
        "instruction": row["new_instruction"], "dockerfile": row["dockerfile"],
        "test_sh": row["test_sh"],
        "test_files": {"test_combined.py": _strip_llm_code(row["new_tests"])},
        "metadata": {"source_1": row["source_1"], "source_2": row["source_2"]},
    }


def gen_5_2_adversarial():
    """5.2 Adversarial Tests."""
    tasks = load_harbor_tasks_as_dicts(SOURCE_REPO, limit=LIMIT)
    for t in tasks:
        t["variant"] = "adversarial"  # discriminator to avoid curator cache collision
    ds = Dataset.from_list(tasks)

    r1 = run_completions(ds, model=MODEL, map_type="chat", map_config={
        "user_message": "Generate ADVERSARIAL pytest tests for common failure modes (off-by-one, empty input, type errors).\n\nTASK:\n{{instruction}}\n\nEXISTING TESTS:\n{{test_files_json}}\n\nWrite 5-10 adversarial test functions:",
        "output_column": "new_tests"}, require_all_responses=False)

    return r1.dataset, "exp-5-2", lambda row: {
        "instruction": row["instruction"], "dockerfile": row["dockerfile"],
        "test_sh": row["test_sh"],
        "test_files": {**json.loads(row["test_files_json"]), "test_adversarial.py": _strip_llm_code(row["new_tests"])},
        "metadata": {"source": row["task_dir_name"]},
    }


def gen_5_4_2skill():
    """5.4 Compositional — 2-skill."""
    tasks = load_harbor_tasks_as_dicts(SOURCE_REPO, limit=LIMIT)
    for t in tasks:
        t["n_skills"] = "2"
    ds = Dataset.from_list(tasks)

    r1 = run_completions(ds, model=MODEL, map_type="chat", map_config={
        "user_message": "Rewrite to require 2 skills: original + file I/O (read from file, write to file).\n\nORIGINAL:\n{{instruction}}\n\nOutput 2-skill task:",
        "output_column": "new_instruction"}, require_all_responses=False)

    r2 = run_completions(r1.dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Adapt tests for version with file I/O.\n\nORIGINAL TESTS:\n{{test_files_json}}\n\nWrite adapted tests:",
        "output_column": "new_tests"}, require_all_responses=False)

    return r2.dataset, "exp-5-4", lambda row: {
        "instruction": row["new_instruction"], "dockerfile": row["dockerfile"],
        "test_sh": row["test_sh"],
        "test_files": {"test_2skill.py": _strip_llm_code(row["new_tests"])},
        "metadata": {"source": row["task_dir_name"], "n_skills": "2"},
    }


def gen_5_3_test_synthesis():
    """5.3 Test Synthesis — mine code from The Stack, generate tests."""
    print("Mining Python files from The Stack (streaming, max 50k scan)...")
    ds = load_dataset('bigcode/the-stack', data_dir='data/python', split='train', streaming=True)

    samples = []
    for sample in itertools.islice(ds, 50_000):
        content = sample.get('content', '')
        path = sample.get('path', '')
        if 'test' in path.lower() or len(content) < 200 or len(content) > 5000:
            continue
        if 'from .' in content or 'def ' not in content:
            continue
        samples.append({'code': content, 'path': path, 'repo': sample.get('repository_name', '')})
        if len(samples) >= LIMIT:
            break

    print(f"Found {len(samples)} candidate files")
    dataset = Dataset.from_list(samples)

    r1 = run_completions(dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Given this Python code, write a clear task description that asks an AI agent to implement this functionality from scratch.\n\nCODE:\n{{code}}\n\nOutput only the task description:",
        "output_column": "instruction"}, require_all_responses=False)

    r2 = run_completions(r1.dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Given this Python code, generate a comprehensive pytest test suite (5-10 tests covering happy path, edge cases, errors).\n\nCODE:\n{{code}}\n\nOutput only the pytest test code:",
        "output_column": "generated_tests"}, require_all_responses=False)

    dockerfile = create_standard_dockerfile()
    test_sh = _make_test_sh("/tests/test_synthesized.py")

    return r2.dataset, "exp-5-3", lambda row: {
        "instruction": row["instruction"], "dockerfile": dockerfile,
        "test_sh": test_sh,
        "test_files": {"test_synthesized.py": _strip_llm_code(row["generated_tests"])},
        "metadata": {"source_path": row.get("path", ""), "source_repo": row.get("repo", "")},
    }


def gen_7_3_cross_language_rust():
    """7.3 Cross-Language Transplant — Python to Rust."""
    tasks = load_harbor_tasks_as_dicts(SOURCE_REPO, limit=LIMIT)
    for t in tasks:
        t["target_language"] = "rust"  # discriminator to avoid curator cache collision
    dataset = Dataset.from_list(tasks)

    r1 = run_completions(dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Transplant this Python coding task to Rust. Rewrite the instruction so it asks for a Rust implementation. Adapt data types and idioms.\n\nPYTHON TASK:\n{{instruction}}\n\nOutput only the Rust task instruction:",
        "output_column": "new_instruction"}, require_all_responses=False)

    r2 = run_completions(r1.dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Convert these Python tests into Rust test code (using #[test] and assert! macros).\n\nPYTHON TESTS:\n{{test_files_json}}\n\nOutput only the Rust test code:",
        "output_column": "rust_tests"}, require_all_responses=False)

    # Standard Rust Dockerfile — don't use LLM-generated ones (they try to build solutions)
    rust_dockerfile = """FROM rust:1.75-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential pkg-config && rm -rf /var/lib/apt/lists/*
"""

    test_sh = _make_test_sh(test_cmd="cd /app && cargo test 2>&1")

    return r2.dataset, "exp-7-3", lambda row: {
        "instruction": row["new_instruction"], "dockerfile": rust_dockerfile,
        "test_sh": test_sh,
        "test_files": {"test_main.rs": _strip_llm_code(row["rust_tests"])},
        "metadata": {"source": row["task_dir_name"], "target_language": "rust"},
    }


def gen_7_4_test_first():
    """7.4 Test-First — mine pytest files from The Stack, generate instructions."""
    print("Mining pytest files from The Stack (streaming, max 50k scan)...")
    ds = load_dataset('bigcode/the-stack', data_dir='data/python', split='train', streaming=True)

    samples = []
    for sample in itertools.islice(ds, 50_000):
        content = sample.get('content', '')
        if len(content) < 100 or len(content) > 10000:
            continue
        if ('import pytest' in content or 'from pytest' in content) and 'def test_' in content and 'assert ' in content and 'from .' not in content:
            samples.append({'test_code': content, 'path': sample.get('path', ''), 'repo': sample.get('repository_name', '')})
            if len(samples) >= LIMIT:
                break

    print(f"Found {len(samples)} pytest files")
    dataset = Dataset.from_list(samples)

    r1 = run_completions(dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Given this pytest test file, write a clear task instruction describing what code needs to be written to make ALL tests pass. Do NOT reference the test file directly.\n\nPYTEST CODE:\n{{test_code}}\n\nOutput only the task instruction:",
        "output_column": "instruction"}, require_all_responses=False)

    r2 = run_completions(r1.dataset, model=MODEL, map_type="chat", map_config={
        "user_message": "Is this (instruction, test) pair consistent? Can a developer write code that satisfies the instruction AND passes the tests?\n\nINSTRUCTION:\n{{instruction}}\n\nTESTS:\n{{test_code}}\n\nOutput only: yes or no",
        "output_column": "consistent"}, require_all_responses=False)

    # Filter to only consistent pairs
    consistent_rows = [row for row in r2.dataset if row.get("consistent", "").strip().lower() == "yes"]
    print(f"Consistent pairs: {len(consistent_rows)} / {len(r2.dataset)}")
    filtered = Dataset.from_list(consistent_rows) if consistent_rows else r2.dataset

    dockerfile = create_standard_dockerfile()
    test_sh = _make_test_sh("/tests/test_solution.py")

    return filtered, "exp-7-4", lambda row: {
        "instruction": row["instruction"], "dockerfile": dockerfile,
        "test_sh": test_sh,
        "test_files": {"test_solution.py": row["test_code"]},
        "metadata": {"source_path": row.get("path", ""), "source_repo": row.get("repo", "")},
    }


# Registry
EXPERIMENTS = {
    "8.4_constraint_1": gen_8_4_constraint_1,
    "8.7_ambiguity": gen_8_7_ambiguity,
    "8.8_domain_medical": gen_8_8_domain_medical,
    "8.9_stacking": gen_8_9_stacking,
    "5.2_adversarial": gen_5_2_adversarial,
    "5.3_test_synthesis": gen_5_3_test_synthesis,
    "5.4_2skill": gen_5_4_2skill,
    "7.3_cross_language_rust": gen_7_3_cross_language_rust,
    "7.4_test_first": gen_7_4_test_first,
}


def create_tasks(dataset, prefix, build_fn):
    """Create harbor task directories from dataset."""
    out = Path(tempfile.mkdtemp(prefix=f"{prefix}_validate_"))
    for i, row in enumerate(tqdm(dataset, desc="Creating tasks")):
        task = build_fn(row)
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=task["instruction"],
            dockerfile=task["dockerfile"],
            test_sh=task["test_sh"],
            test_files=task["test_files"],
            dataset_prefix=prefix,
            metadata=task.get("metadata", {}),
        )
    return str(out)


def run_harbor(task_dir, name):
    """Run harbor evaluation and return results."""
    os.chdir(Path(__file__).parent.parent)
    job_name = f"validate_{name}_{os.getpid()}"
    config_path = Path(__file__).parent.parent / "eval/tacc/dcagent_eval_config.yaml"

    cmd = [
        "harbor", "jobs", "start",
        "-p", task_dir,
        "--n-concurrent", str(N_CONCURRENT),
        "--agent", "terminus-2",
        "--model", HARBOR_MODEL,
        "--env", "daytona",
        "--n-attempts", "1",
        "--max-retries", "0",
        "--job-name", job_name,
        "--config", str(config_path),
    ]

    print(f"\n  Harbor cmd: {' '.join(cmd)}")
    subprocess.run(cmd, capture_output=False)

    result_file = Path("jobs") / job_name / "result.json"
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        results = data.get("results", [])
        passed = sum(1 for r in results if r.get("reward", 0) > 0)
        total = len(results)
        started = sum(1 for r in results if "reward" in r)
        errors = data.get("exception_stats", {})
        n_errors = sum(len(v) for v in errors.values()) if isinstance(errors, dict) else 0
        return {
            "experiment": name, "started": started, "passed": passed,
            "total": total, "pass_rate": (passed / total * 100) if total > 0 else 0,
            "errors": n_errors,
        }
    return {"experiment": name, "error": "No result.json"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments", nargs="*", default=list(EXPERIMENTS.keys()),
                        help=f"Experiments to run. Available: {', '.join(EXPERIMENTS.keys())}")
    parser.add_argument("--generate-only", action="store_true", help="Generate tasks but don't run harbor")
    args = parser.parse_args()

    all_results = []

    for name in args.experiments:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}. Available: {', '.join(EXPERIMENTS.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"Experiment: {name}")
        print(f"{'='*60}")

        # Step 1: Generate 25 tasks
        print("\n--- Generating tasks ---")
        try:
            dataset, prefix, build_fn = EXPERIMENTS[name]()
            task_dir = create_tasks(dataset, prefix, build_fn)
            n_tasks = len(list(Path(task_dir).glob("*/instruction.md")))
            print(f"Created {n_tasks} tasks at {task_dir}")
        except Exception as e:
            print(f"ERROR generating: {e}")
            all_results.append({"experiment": name, "error": str(e)})
            continue

        if args.generate_only:
            print(f"  --generate-only: skipping harbor. Tasks at {task_dir}")
            all_results.append({"experiment": name, "task_dir": task_dir, "n_tasks": n_tasks})
            continue

        # Step 2: Run through harbor
        print("\n--- Running harbor ---")
        result = run_harbor(task_dir, name)
        all_results.append(result)
        print(f"\n  Result: {json.dumps(result, indent=2)}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Experiment':<30} {'Started':>8} {'Passed':>8} {'Rate':>8} {'Errors':>8}")
    print("-" * 70)
    for r in all_results:
        if "error" in r:
            print(f"{r['experiment']:<30} {'ERROR':>8} {r['error']}")
        elif "task_dir" in r:
            print(f"{r['experiment']:<30} {r['n_tasks']:>8} tasks generated (harbor skipped)")
        else:
            print(f"{r['experiment']:<30} {r['started']:>8} {r['passed']:>8} {r['pass_rate']:>7.1f}% {r['errors']:>8}")

    # Save results
    out_file = Path(__file__).parent / "test_gen_passrate_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
