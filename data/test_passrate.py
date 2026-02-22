#!/usr/bin/env python3
"""
Test pass rate for a Harbor dataset with gpt-5-nano on 25 tasks.
Usage: python test_passrate.py <dataset_repo_id>
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add scripts/harbor to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "harbor"))

def main():
    parser = argparse.ArgumentParser(description="Test pass rate for a dataset")
    parser.add_argument("dataset", help="HuggingFace dataset repo ID (e.g., DCAgent/exp_rpt_stack-bash)")
    parser.add_argument("--n-tasks", type=int, default=25, help="Number of tasks to test")
    parser.add_argument("--model", default="openai/gpt-4.1-nano", help="Model to use")
    args = parser.parse_args()

    dataset_name = args.dataset.split("/")[-1]
    print(f"Testing {args.dataset} with {args.n_tasks} tasks using {args.model}")

    # Download dataset
    print("\n=== Downloading dataset ===")
    from tasks_parquet_converter import from_hf_dataset
    dataset_path = from_hf_dataset(args.dataset)
    print(f"Dataset path: {dataset_path}")

    # Count available tasks
    task_dirs = list(Path(dataset_path).glob("*/instruction.md"))
    total_tasks = len(task_dirs)
    print(f"Total tasks available: {total_tasks}")

    if total_tasks == 0:
        print("ERROR: No tasks found!")
        return

    # Limit to n_tasks
    n_tasks = min(args.n_tasks, total_tasks)

    # Create subset directory with only n_tasks
    subset_dir = tempfile.mkdtemp(prefix=f"{dataset_name}_subset_")
    task_dirs_sorted = sorted(task_dirs)[:n_tasks]

    for i, task_md in enumerate(task_dirs_sorted):
        task_dir = task_md.parent
        target = Path(subset_dir) / task_dir.name
        # Symlink the task directory
        target.symlink_to(task_dir)

    print(f"Created subset with {n_tasks} tasks at: {subset_dir}")

    # Run Harbor
    print("\n=== Running Harbor ===")
    job_name = f"passrate_{dataset_name}"

    harbor_cmd = [
        "harbor", "jobs", "start",
        "-p", subset_dir,
        "--n-concurrent", "8",
        "--agent", "terminus-2",
        "--model", args.model,
        "--env", "daytona",
        "--n-attempts", "1",
        "--max-retries", "0",
        "--job-name", job_name,
        "--config", str(Path(__file__).parent.parent / "eval/tacc/dcagent_eval_config.yaml"),
    ]

    print(f"Command: {' '.join(harbor_cmd)}")

    result = subprocess.run(harbor_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Parse results
    print("\n=== Results ===")
    jobs_dir = Path("jobs") / job_name
    if jobs_dir.exists():
        result_file = jobs_dir / "result.json"
        if result_file.exists():
            with open(result_file) as f:
                results = json.load(f)

            passed = sum(1 for r in results.get("results", []) if r.get("reward", 0) > 0)
            total = len(results.get("results", []))
            pass_rate = (passed / total * 100) if total > 0 else 0

            print(f"Dataset: {args.dataset}")
            print(f"Tasks tested: {total}")
            print(f"Passed: {passed}")
            print(f"Pass rate: {pass_rate:.1f}%")

            return {"dataset": args.dataset, "passed": passed, "total": total, "pass_rate": pass_rate}

    print("Could not find results")
    return None


if __name__ == "__main__":
    main()
