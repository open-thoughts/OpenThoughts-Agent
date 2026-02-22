#!/usr/bin/env python3
"""
Validate all Theme experiments.
- Non-test experiments: generate 25 samples locally to verify they work, confirm 10k config
- Test-generating experiments: generate 25 samples + run through harbor to check pass rate
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

# ============================================================================
# Categorization
# ============================================================================

# Experiments that DO generate new LLM-written tests (need harbor validation)
TEST_GENERATING = {
    # (folder, script, description)
    ("exp_8_4_constraint_ladder", "generate_1_constraint.py", "8.4 - 1 constraint"),
    ("exp_8_7_ambiguity", "generate_partial.py", "8.7 - partial ambiguity"),
    ("exp_8_8_domain_shift", "generate_medical.py", "8.8 - medical domain"),
    ("exp_8_9_task_stacking", "generate_pairs.py", "8.9 - task stacking"),
    ("exp_5_2_adversarial_tests", "generate_adversarial.py", "5.2 - adversarial tests"),
    ("exp_5_3_test_synthesis", "generate_synthesized_tests.py", "5.3 - test synthesis"),
    ("exp_5_4_compositional_tasks", "generate_2skill.py", "5.4 - 2-skill composite"),
    ("exp_7_3_cross_language", "generate_rust.py", "7.3 - cross-lang rust"),
    ("exp_7_4_test_first", "generate_pytest_reverse.py", "7.4 - test-first reverse"),
}

# Experiments that DON'T generate tests (just need local validation + 10k confirmed)
NON_TEST = {
    ("exp_8_1_style_transfer", "generate_github_issue.py", "8.1 - github issue style"),
    ("exp_8_2_spec_modality", "generate_pseudocode.py", "8.2 - pseudocode spec"),
    ("exp_8_3_granularity", "generate_vague.py", "8.3 - vague granularity"),
    ("exp_8_5_context_padding", "generate_mild.py", "8.5 - mild padding"),
    ("exp_8_6_error_state", "generate_subtle.py", "8.6 - subtle error state"),
    ("exp_8_10_knowledge_gradient", "generate_expert.py", "8.10 - expert knowledge"),
    ("exp_1_1_difficulty_mixing", "generate_goldilocks.py", "1.1 - goldilocks mix"),
    ("exp_1_2_instruction_specificity", "generate_minimal.py", "1.2 - minimal spec"),
    ("exp_1_3_test_quality", "generate_top50.py", "1.3 - top 50% quality"),
    ("exp_1_4_deduplication", "generate_deduplicated.py", "1.4 - deduplicated"),
    ("exp_2_1_dense_rewards", "generate_proportional.py", "2.1 - proportional reward"),
    ("exp_4_3_skill_distillation", "generate_curated_1k.py", "4.3 - skill distillation"),
    ("exp_5_1_consistency_verification", "generate_filtered_solvable.py", "5.1 - consistency filter"),
    ("exp_7_1_dockerfile_curriculum", "generate_d5_bare.py", "7.1 - bare dockerfile"),
    ("exp_7_2_info_dropout", "generate_random_30.py", "7.2 - 30% info dropout"),
    ("exp_7_5_staged_rewards", "generate_staged.py", "7.5 - staged rewards"),
    ("exp_7_6_temporal_reward", "generate_speed_bonus.py", "7.6 - speed bonus"),
}


def run_harbor_passrate(task_dir: str, job_name: str, model: str = "openai/gpt-5-nano-2025-08-07"):
    """Run harbor evaluation on a task directory and return pass rate."""
    config_path = Path(__file__).parent.parent / "eval/tacc/dcagent_eval_config.yaml"

    cmd = [
        "harbor", "jobs", "start",
        "-p", task_dir,
        "--n-concurrent", "8",
        "--agent", "terminus-2",
        "--model", model,
        "--env", "daytona",
        "--n-attempts", "1",
        "--max-retries", "0",
        "--job-name", job_name,
        "--config", str(config_path),
    ]

    print(f"  Running: {' '.join(cmd)}")
    os.chdir(Path(__file__).parent.parent)
    subprocess.run(cmd, capture_output=False)

    result_file = Path("jobs") / job_name / "result.json"
    if result_file.exists():
        with open(result_file) as f:
            data = json.load(f)
        results = data.get("results", [])
        passed = sum(1 for r in results if r.get("reward", 0) > 0)
        total = len(results)
        started = sum(1 for r in results if r.get("reward", 0) is not None)
        return {"passed": passed, "total": total, "started": started,
                "pass_rate": (passed / total * 100) if total > 0 else 0}
    return {"error": "No result.json found"}


def validate_task_dir(task_dir: str) -> dict:
    """Validate a generated task directory has correct structure."""
    task_path = Path(task_dir)
    tasks = sorted([d for d in task_path.iterdir() if d.is_dir()])

    stats = {"n_tasks": len(tasks), "valid": 0, "issues": []}
    for td in tasks:
        has_instruction = (td / "instruction.md").exists()
        has_dockerfile = (td / "environment" / "Dockerfile").exists()
        has_test_sh = (td / "tests" / "test.sh").exists()
        has_task_toml = (td / "task.toml").exists()

        if all([has_instruction, has_dockerfile, has_test_sh, has_task_toml]):
            stats["valid"] += 1
        else:
            missing = []
            if not has_instruction: missing.append("instruction.md")
            if not has_dockerfile: missing.append("Dockerfile")
            if not has_test_sh: missing.append("test.sh")
            if not has_task_toml: missing.append("task.toml")
            stats["issues"].append(f"{td.name}: missing {', '.join(missing)}")

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test-gen", "non-test", "all"], default="all")
    parser.add_argument("--harbor", action="store_true", help="Run harbor evaluation for test-generating experiments")
    parser.add_argument("--model", default="openai/gpt-5-nano-2025-08-07")
    args = parser.parse_args()

    data_dir = Path(__file__).parent

    if args.mode in ("all", "non-test"):
        print("\n" + "=" * 70)
        print("NON-TEST-GENERATING EXPERIMENTS (keep original tests)")
        print("These are configured for 10k tasks in production.")
        print("=" * 70)
        for folder, script, desc in sorted(NON_TEST, key=lambda x: x[2]):
            script_path = data_dir / folder / script
            print(f"\n  {desc}: {script_path.name}")
            if script_path.exists():
                print(f"    ✓ Script exists")
            else:
                print(f"    ✗ MISSING: {script_path}")

    if args.mode in ("all", "test-gen"):
        print("\n" + "=" * 70)
        print("TEST-GENERATING EXPERIMENTS (LLM-written tests)")
        if args.harbor:
            print("Will generate 25 tasks + run through harbor for pass rate.")
        else:
            print("Use --harbor to generate 25 tasks and run pass rate test.")
        print("=" * 70)
        for folder, script, desc in sorted(TEST_GENERATING, key=lambda x: x[2]):
            script_path = data_dir / folder / script
            print(f"\n  {desc}: {script_path.name}")
            if script_path.exists():
                print(f"    ✓ Script exists")
            else:
                print(f"    ✗ MISSING: {script_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Non-test-generating experiments: {len(NON_TEST)} (one representative each)")
    print(f"Test-generating experiments: {len(TEST_GENERATING)} (one representative each)")
    print(f"Total experiment folders: {len(list(data_dir.glob('exp_*/')))} ")
    print(f"Total generate scripts: {len(list(data_dir.glob('exp_*/generate_*.py')))}")


if __name__ == "__main__":
    main()
