#!/usr/bin/env python3
"""
Test all dataset generators by creating 25 samples from each and optionally
running harbor tests with gpt-5-nano.

Usage:
    python test_all_datasets.py --generate  # Generate tasks only
    python test_all_datasets.py --test      # Generate and test with harbor
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Working datasets and their generation commands
DATASETS = {
    "codenet": {
        "script": "generate_codenet_tasks.py",
        "args": ["--limit", "25", "--no-upload"],
        "status": "working",
    },
    "crosscodeeval": {
        "script": "generate_crosscodeeval_tasks.py",
        "args": ["--limit", "25", "--no-upload",
                 "--local-data", "/scratch/10000/eguha3/dc-agent/data/dclm-mine/.cache/github/amazon-science_cceval/data"],
        "status": "working",
    },
    "pymethods2test": {
        "script": "generate_pymethods2test_tasks.py",
        "args": ["--limit", "25", "--no-upload"],
        "status": "working",
    },
    "unitsyn": {
        "script": "generate_unitsyn_tasks.py",
        "args": ["--limit", "25", "--no-upload"],
        "status": "working",  # Uses same HF dataset as pymethods2test
    },
    "bugswarm": {
        "script": "generate_bugswarm_tasks.py",
        "args": ["--limit", "25", "--no-upload"],
        "status": "sample_only",  # Only 2 sample tasks available
    },
    "manybugs": {
        "script": "generate_manybugs_tasks.py",
        "args": ["--limit", "25", "--no-upload"],
        "status": "sample_only",  # Complex BugZoo structure
    },
    "e2egit": {
        "script": "generate_e2egit_tasks.py",
        "args": ["--limit", "25", "--no-upload"],
        "status": "sample_only",  # Database unavailable
    },
    "ghactions": {
        "script": "generate_ghactions_tasks.py",
        "args": ["--limit", "25", "--skip-clone", "--no-upload",
                 "--local-data", "/scratch/10000/eguha3/dc-agent/data/dclm-mine/.cache/zenodo_10566003/workflows"],
        "status": "working",  # Downloads from Zenodo
    },
}


def run_generator(dataset_name: str, config: dict) -> dict:
    """Run a dataset generator and return results."""
    script_path = Path(__file__).parent / config["script"]

    print(f"\n{'='*60}")
    print(f"Generating: {dataset_name}")
    print(f"{'='*60}")

    cmd = ["python3", str(script_path)] + config["args"]
    print(f"Command: {' '.join(cmd)}")

    result = {
        "dataset": dataset_name,
        "status": "unknown",
        "task_dir": None,
        "num_tasks": 0,
        "error": None,
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=Path(__file__).parent.parent.parent,  # dc-agent root
        )

        output = proc.stdout + proc.stderr

        # Parse output for task directory
        for line in output.split("\n"):
            if "Output directory:" in line or "Task directory:" in line:
                task_dir = line.split(":")[-1].strip()
                result["task_dir"] = task_dir
            if "Successfully generated" in line:
                # Extract number of tasks
                import re
                match = re.search(r"(\d+)", line)
                if match:
                    result["num_tasks"] = int(match.group(1))

        if result["task_dir"] and result["num_tasks"] > 0:
            result["status"] = "success"
        elif proc.returncode != 0:
            result["status"] = "error"
            result["error"] = output[-500:]  # Last 500 chars
        else:
            result["status"] = "no_output"
            result["error"] = "No task directory found in output"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Generation timed out after 10 minutes"
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)

    print(f"Result: {result['status']}, Tasks: {result['num_tasks']}, Dir: {result['task_dir']}")
    return result


def run_harbor_test(task_dir: str, model: str = "openai/gpt-5-nano") -> dict:
    """Run harbor test on a task directory."""
    print(f"\nRunning harbor test with {model} on {task_dir}")

    harbor_config = "/scratch/10000/eguha3/dc-agent/eval/tacc/dcagent_eval_config.yaml"

    cmd = [
        "harbor", "jobs", "start",
        "-p", task_dir,
        "--n-concurrent", "8",
        "--agent", "terminus-2",
        "--model", model,
        "--env", "daytona",
        "--n-attempts", "1",
        "--max-retries", "1",
        "--config", harbor_config,
    ]

    result = {
        "task_dir": task_dir,
        "model": model,
        "status": "unknown",
        "passed": 0,
        "total": 0,
        "pass_rate": 0.0,
        "error": None,
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )

        output = proc.stdout + proc.stderr

        # Parse for results
        for line in output.split("\n"):
            if "Results written to" in line:
                # Find results JSON
                import re
                match = re.search(r"jobs/([^/]+)", line)
                if match:
                    job_name = match.group(1)
                    results_file = f"jobs/{job_name}/result.json"
                    if Path(results_file).exists():
                        with open(results_file) as f:
                            results = json.load(f)
                        result["total"] = len(results)
                        result["passed"] = sum(1 for r in results if r.get("reward") == 1)
                        if result["total"] > 0:
                            result["pass_rate"] = result["passed"] / result["total"] * 100

        result["status"] = "success" if proc.returncode == 0 else "error"

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Harbor test timed out"
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)

    print(f"Harbor result: {result['passed']}/{result['total']} passed ({result['pass_rate']:.1f}%)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Test dataset generators and harbor")
    parser.add_argument("--generate", action="store_true", help="Generate tasks only")
    parser.add_argument("--test", action="store_true", help="Generate and test with harbor")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to test")
    parser.add_argument("--model", default="openai/gpt-5-nano", help="Model for harbor tests")
    parser.add_argument("--working-only", action="store_true", help="Only test working datasets")

    args = parser.parse_args()

    if not args.generate and not args.test:
        args.generate = True  # Default to generate only

    # Select datasets
    if args.datasets:
        datasets = {k: v for k, v in DATASETS.items() if k in args.datasets}
    elif args.working_only:
        datasets = {k: v for k, v in DATASETS.items() if v["status"] == "working"}
    else:
        datasets = DATASETS

    print(f"Testing {len(datasets)} datasets: {list(datasets.keys())}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "generation": {},
        "harbor": {},
    }

    # Generate tasks
    for name, config in datasets.items():
        gen_result = run_generator(name, config)
        results["generation"][name] = gen_result

        # Run harbor test if requested and generation succeeded
        if args.test and gen_result["status"] == "success" and gen_result["task_dir"]:
            harbor_result = run_harbor_test(gen_result["task_dir"], args.model)
            results["harbor"][name] = harbor_result

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(__file__).parent.parent / f"generation_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    print("\nGeneration Results:")
    for name, result in results["generation"].items():
        status = "✓" if result["status"] == "success" else "✗"
        print(f"  {status} {name}: {result['num_tasks']} tasks")

    if results["harbor"]:
        print("\nHarbor Test Results:")
        for name, result in results["harbor"].items():
            print(f"  {name}: {result['passed']}/{result['total']} ({result['pass_rate']:.1f}%)")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
