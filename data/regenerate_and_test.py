#!/usr/bin/env python3
"""
Regenerate discrepant datasets (25 tasks each), test with harbor, report results.

Usage:
    python data/regenerate_and_test.py [--datasets DS1,DS2,...] [--limit 25] [--skip-harbor]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Datasets that need regeneration, with their generator scripts and args
DATASETS = {
    # No LLM needed
    "bigcodebench": {
        "generator": "data/dclm-mine/generate_bigcodebench_tasks.py",
        "args": ["--no-upload"],
        "old_rate": 60,
        "hf_repo": "DCAgent/exp_rpt_bigcodebench",
    },
    "bugsinpy-mf": {
        "generator": "data/dclm-mine/generate_bugsinpy_mf_tasks.py",
        "args": ["--no-upload"],
        "old_rate": 52,
        "hf_repo": "DCAgent/exp_rpt_bugsinpy-mf",
    },
    "bugswarm": {
        "generator": "data/dclm-mine/generate_bugswarm_tasks.py",
        "args": ["--no-upload"],
        "old_rate": 72,
        "hf_repo": "DCAgent/exp_rpt_bugswarm",
    },
    "codeelo": {
        "generator": "data/dclm-mine/generate_codeelo_tasks.py",
        "args": ["--no-upload"],
        "old_rate": 84,
        "hf_repo": "DCAgent/exp_rpt_codeelo",
    },
    "crosscodeeval-python": {
        "generator": "data/dclm-mine/generate_crosscodeeval_tasks.py",
        "args": ["--no-upload", "--language", "python"],
        "old_rate": 60,
        "hf_repo": "DCAgent/exp_rpt_crosscodeeval-python",
    },
    "exercism-python": {
        "generator": "data/dclm-mine/generate_exercism_tasks.py",
        "args": ["--language", "python"],
        "old_rate": 80,
        "hf_repo": "DCAgent/exp_rpt_exercism-python",
        "no_upload_flag": False,  # doesn't have --no-upload
    },
    "manybugs": {
        "generator": "data/dclm-mine/generate_manybugs_tasks.py",
        "args": ["--no-upload"],
        "old_rate": 60,
        "hf_repo": "DCAgent/exp_rpt_manybugs",
    },
    "swebench": {
        "generator": "data/dclm-mine/generate_swebench_tasks.py",
        "args": ["--no-upload"],
        "old_rate": 64,
        "hf_repo": "DCAgent/exp_rpt_swebench",
    },
    # LLM needed (gpt-4o-mini)
    "bugsinpy": {
        "generator": "data/dclm-mine/generate_bugsinpy_tasks.py",
        "args": [],
        "old_rate": 40,
        "hf_repo": "DCAgent/exp_rpt_bugsinpy",
        "no_upload_flag": False,
        "needs_llm": True,
    },
    "codenet-python": {
        "generator": "data/dclm-mine/generate_codenet_tasks.py",
        "args": ["--no-upload", "--language", "python"],
        "old_rate": 92,
        "hf_repo": "DCAgent/exp_rpt_codenet-python",
        "needs_llm": True,
    },
    "codereval-python": {
        "generator": "data/dclm-mine/generate_codereval_tasks.py",
        "args": ["--language", "python"],
        "old_rate": 52,
        "hf_repo": "DCAgent/exp_rpt_codereval-python",
        "no_upload_flag": False,
        "needs_llm": True,
    },
    "defects4j": {
        "generator": "data/dclm-mine/generate_defects4j_tasks.py",
        "args": [],
        "old_rate": 48,
        "hf_repo": "DCAgent/exp_rpt_defects4j",
        "no_upload_flag": False,
        "needs_llm": True,
    },
    "e2egit": {
        "generator": "data/dclm-mine/generate_e2egit_tasks.py",
        "args": ["--no-upload"],
        "old_rate": 56,
        "hf_repo": "DCAgent/exp_rpt_e2egit",
        "needs_llm": True,
    },
    "ghactions": {
        "generator": "data/dclm-mine/generate_ghactions_tasks.py",
        "args": ["--skip-clone"],
        "old_rate": 84,
        "hf_repo": "DCAgent/exp_rpt_ghactions",
        "no_upload_flag": False,
        "needs_llm": True,
    },
    "methods2test": {
        "generator": "data/dclm-mine/generate_methods2test_tasks.py",
        "args": [],
        "old_rate": 92,
        "hf_repo": "DCAgent/exp_rpt_methods2test",
        "no_upload_flag": False,
        "needs_llm": True,
    },
    "pymethods2test": {
        "generator": "data/dclm-mine/generate_pymethods2test_tasks.py",
        "args": ["--no-upload"],
        "old_rate": 76,
        "hf_repo": "DCAgent/exp_rpt_pymethods2test",
        "needs_llm": True,
    },
    "softwareheritage": {
        "generator": "data/dclm-mine/generate_softwareheritage_tasks.py",
        "args": [],
        "old_rate": 72,
        "hf_repo": "DCAgent/exp_rpt_softwareheritage",
        "no_upload_flag": False,
        "needs_llm": True,
    },
    "taco": {
        "generator": "data/dclm-mine/generate_taco_tasks.py",
        "args": [],
        "old_rate": 72,
        "hf_repo": "DCAgent/exp_rpt_taco",
        "no_upload_flag": False,
        "needs_llm": True,
    },
    "unitsyn-python": {
        "generator": "data/dclm-mine/generate_unitsyn_tasks.py",
        "args": ["--no-upload", "--language", "python"],
        "old_rate": 80,
        "hf_repo": "DCAgent/exp_rpt_unitsyn-python",
        "needs_llm": True,
    },
}


def run_generator(name: str, config: dict, limit: int = 25) -> str | None:
    """Run a generator and return the output directory path."""
    generator = config["generator"]
    extra_args = config.get("args", [])
    has_no_upload = config.get("no_upload_flag", True)  # default True

    cmd = [sys.executable, generator, "--limit", str(limit)]
    cmd.extend(extra_args)

    # For generators without --no-upload, set env var to skip
    env = os.environ.copy()
    if not has_no_upload:
        env["SKIP_HF_UPLOAD"] = "1"

    print(f"\n{'='*60}")
    print(f"Regenerating: {name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
            env=env,
        )
        output = result.stdout + result.stderr
        print(output[-2000:] if len(output) > 2000 else output)

        # Extract temp directory from output
        # Look for patterns like "Generating harbor tasks in: /tmp/xxx"
        # or "Task directory: /tmp/xxx" or "Output directory: /tmp/xxx"
        for pattern in [
            r"(?:Task directory|Output directory|Generating harbor tasks in): (.+)",
            r"Generated \d+ harbor tasks.*\n.*?(/tmp/\S+)",
            r"(/tmp/[a-zA-Z0-9_]+tasks_[a-zA-Z0-9]+)",
        ]:
            match = re.search(pattern, output)
            if match:
                task_dir = match.group(1).strip()
                if os.path.isdir(task_dir):
                    return task_dir

        # Fallback: find most recent temp dir matching pattern
        print(f"WARNING: Could not extract task dir from output for {name}")
        return None

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Generator for {name} timed out after 30 min")
        return None
    except Exception as e:
        print(f"ERROR: Generator for {name} failed: {e}")
        return None


def run_harbor(name: str, task_dir: str, job_name: str) -> dict | None:
    """Run harbor on a task directory and return results."""
    cmd = [
        "harbor", "jobs", "start",
        "-p", task_dir,
        "--n-concurrent", "5",
        "--agent", "terminus-2",
        "--model", "openai/gpt-5-nano-2025-08-07",
        "--env", "daytona",
        "--n-attempts", "1",
        "--max-retries", "0",
        "--job-name", job_name,
        "--config", "eval/tacc/dcagent_eval_config.yaml",
    ]

    print(f"\nRunning harbor for {name}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        print(result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
        if result.returncode != 0:
            print(f"Harbor stderr: {result.stderr[-500:]}")

        # Parse result.json
        result_path = f"jobs/{job_name}/result.json"
        if os.path.exists(result_path):
            with open(result_path) as f:
                data = json.load(f)
            stats = data.get("stats", {})
            evals = stats.get("evals", {})
            for k, v in evals.items():
                metrics = v.get("metrics", [])
                if metrics:
                    return {
                        "pass_rate": metrics[0].get("mean", 0),
                        "n_trials": stats.get("n_trials", 0),
                        "n_errors": stats.get("n_errors", 0),
                    }
        return None

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Harbor for {name} timed out after 1 hour")
        return None
    except Exception as e:
        print(f"ERROR: Harbor for {name} failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Regenerate and test discrepant datasets")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated list of datasets to process (default: all)")
    parser.add_argument("--limit", type=int, default=25,
                        help="Number of tasks to generate per dataset")
    parser.add_argument("--skip-harbor", action="store_true",
                        help="Only regenerate, skip harbor testing")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip datasets that need LLM calls")
    args = parser.parse_args()

    # Select datasets
    if args.datasets:
        selected = args.datasets.split(",")
    else:
        selected = list(DATASETS.keys())

    if args.skip_llm:
        selected = [d for d in selected if not DATASETS[d].get("needs_llm", False)]

    print(f"Will process {len(selected)} datasets: {', '.join(selected)}")
    print(f"Limit: {args.limit} tasks per dataset")

    results = {}
    for name in selected:
        config = DATASETS[name]

        # Step 1: Regenerate
        task_dir = run_generator(name, config, limit=args.limit)
        if not task_dir:
            results[name] = {"status": "GENERATE_FAILED", "task_dir": None}
            continue

        # Count tasks (directories may have different naming conventions)
        task_count = len([d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))])
        print(f"  Generated {task_count} tasks in {task_dir}")

        if args.skip_harbor:
            results[name] = {
                "status": "GENERATED",
                "task_dir": task_dir,
                "task_count": task_count,
            }
            continue

        # Step 2: Run harbor
        import hashlib
        ts = str(int(time.time()))[-6:]
        job_name = f"regen_{name.replace('-', '_')}_{ts}"
        harbor_result = run_harbor(name, task_dir, job_name)

        if harbor_result:
            new_rate = harbor_result["pass_rate"] * 100
            old_rate = config["old_rate"]
            diff = new_rate - old_rate
            status = "GOOD" if abs(diff) < 25 else "STILL_BAD"
            results[name] = {
                "status": status,
                "task_dir": task_dir,
                "task_count": task_count,
                "new_rate": new_rate,
                "old_rate": old_rate,
                "diff": diff,
                "n_trials": harbor_result["n_trials"],
                "n_errors": harbor_result["n_errors"],
            }
            print(f"  {name}: new={new_rate:.0f}% old={old_rate}% diff={diff:+.0f}% -> {status}")
        else:
            results[name] = {
                "status": "HARBOR_FAILED",
                "task_dir": task_dir,
                "task_count": task_count,
            }

    # Save results
    output_path = "data/regeneration_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("REGENERATION SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        status = r["status"]
        if "new_rate" in r:
            print(f"  {name}: {status} (new={r['new_rate']:.0f}% old={r['old_rate']}% diff={r['diff']:+.0f}%)")
        else:
            print(f"  {name}: {status}")

    good = sum(1 for r in results.values() if r["status"] == "GOOD")
    bad = sum(1 for r in results.values() if r["status"] == "STILL_BAD")
    failed = sum(1 for r in results.values() if r["status"] in ("GENERATE_FAILED", "HARBOR_FAILED"))
    print(f"\nGOOD: {good}, STILL_BAD: {bad}, FAILED: {failed}")


if __name__ == "__main__":
    main()
