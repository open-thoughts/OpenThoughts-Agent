#!/usr/bin/env python3
"""
Generate 500 tasks for each improved dataset and upload to HF as v2.
Then create a manifest of all good datasets.
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from data.commons import upload_tasks_to_hf

DATA_CACHE = "/scratch/10000/eguha3/data_cache"

# The 12 improved datasets and their generator commands
DATASETS = [
    {
        "name": "bigcodebench",
        "repo": "DCAgent/exp_rpt_bigcodebench-v2",
        "cmd": ["python3", "data/dclm-mine/generate_bigcodebench_tasks.py", "--limit", "500", "--no-upload"],
        "dir_pattern": "bigcodebench_tasks_*",
    },
    {
        "name": "codeelo",
        "repo": "DCAgent/exp_rpt_codeelo-v2",
        "cmd": ["python3", "data/dclm-mine/generate_codeelo_tasks.py", "--limit", "500", "--no-upload"],
        "dir_pattern": "codeelo_tasks_*",
    },
    {
        "name": "crosscodeeval-python",
        "repo": "DCAgent/exp_rpt_crosscodeeval-python-v2",
        "cmd": ["python3", "data/dclm-mine/generate_crosscodeeval_tasks.py", "--language", "python", "--limit", "500", "--no-upload"],
        "dir_pattern": "crosscodeeval-python_tasks_*",
    },
    {
        "name": "e2egit",
        "repo": "DCAgent/exp_rpt_e2egit-v2",
        "cmd": ["python3", "data/dclm-mine/generate_e2egit_tasks.py", "--limit", "500", "--no-upload"],
        "dir_pattern": "e2egit_tasks_*",
    },
    {
        "name": "methods2test",
        "repo": "DCAgent/exp_rpt_methods2test-v2",
        "cmd": ["python3", "data/dclm-mine/generate_methods2test_tasks.py", "--limit", "500"],
        "dir_pattern": "methods2test_tasks_*",
    },
    {
        "name": "codereval-python",
        "repo": "DCAgent/exp_rpt_codereval-python-v2",
        "cmd": ["python3", "data/dclm-mine/generate_codereval_tasks.py", "--limit", "500", "--no-upload"],
        "dir_pattern": "codereval-python_tasks_*",
    },
    {
        "name": "softwareheritage",
        "repo": "DCAgent/exp_rpt_softwareheritage-v2",
        "cmd": ["python3", "data/dclm-mine/generate_softwareheritage_tasks.py", "--limit", "500"],
        "dir_pattern": "swh_tasks_*",
    },
    {
        "name": "stack-php",
        "repo": "DCAgent/exp_rpt_stack-php-v2",
        "cmd": ["python3", "data/dclm-mine/generate_phpunit_tasks.py", "--limit", "500"],
        "dir_pattern": "stack-php_tasks_*",
    },
    {
        "name": "stack-pytest",
        "repo": "DCAgent/exp_rpt_stack-pytest-v2",
        "cmd": ["python3", "data/dclm-mine/generate_pytest_tasks.py", "--limit", "500"],
        "dir_pattern": "stack-pytest_tasks_*",
    },
    {
        "name": "stack-dockerfile",
        "repo": "DCAgent/exp_rpt_stack-dockerfile-v2",
        "cmd": ["python3", "data/dclm-mine/generate_dockerfile_tasks.py", "--limit", "500"],
        "dir_pattern": "stack-dockerfile_tasks_*",
    },
    {
        "name": "stack-selfdoc",
        "repo": "DCAgent/exp_rpt_stack-selfdoc-v2",
        "cmd": ["python3", "data/dclm-mine/generate_self_documented_tasks.py", "--limit", "500"],
        "dir_pattern": "stack-selfdoc_tasks_*",
    },
    {
        "name": "stack-jest",
        "repo": "DCAgent/exp_rpt_stack-jest-v2",
        "cmd": ["python3", "data/dclm-mine/generate_jest_tasks.py", "--limit", "500"],
        "dir_pattern": "stack-jest_tasks_*",
    },
]

def find_newest_dir(pattern):
    """Find the most recently created directory matching pattern."""
    dirs = sorted(glob.glob(f"{DATA_CACHE}/{pattern}"), key=os.path.getmtime, reverse=True)
    return dirs[0] if dirs else None


def main():
    os.environ["SKIP_HF_UPLOAD"] = "1"  # Skip built-in uploads
    os.chdir("/scratch/10000/eguha3/dc-agent")

    results = []

    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"[{ds['name']}] Generating 500 tasks...")
        print(f"{'='*60}")

        # Run generator
        try:
            proc = subprocess.run(
                ds["cmd"],
                capture_output=False,
                timeout=1800,  # 30 min max per generator
            )
            if proc.returncode != 0:
                print(f"  WARNING: Generator exited with code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: Generator timed out after 30 min")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": ds["name"], "status": "GENERATION_FAILED", "error": str(e)})
            continue

        # Find the generated directory
        task_dir = find_newest_dir(ds["dir_pattern"])
        if not task_dir:
            print(f"  ERROR: No task directory found matching {ds['dir_pattern']}")
            results.append({"name": ds["name"], "status": "NO_TASK_DIR"})
            continue

        task_count = len([d for d in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, d))])
        print(f"  Task dir: {task_dir} ({task_count} tasks)")

        # Upload to HF with v2 name
        print(f"  Uploading to {ds['repo']}...")
        del os.environ["SKIP_HF_UPLOAD"]  # Enable upload
        try:
            url = upload_tasks_to_hf(task_dir, ds["repo"])
            print(f"  Uploaded: {url}")
            results.append({
                "name": ds["name"],
                "repo": ds["repo"],
                "task_dir": task_dir,
                "task_count": task_count,
                "status": "UPLOADED",
                "url": url,
            })
        except Exception as e:
            print(f"  UPLOAD ERROR: {e}")
            results.append({"name": ds["name"], "status": "UPLOAD_FAILED", "error": str(e)})
        finally:
            os.environ["SKIP_HF_UPLOAD"] = "1"  # Re-enable skip for next gen

    # Print summary
    print(f"\n{'='*60}")
    print("UPLOAD SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = r["status"]
        name = r["name"]
        if status == "UPLOADED":
            print(f"  OK: {name} -> {r['repo']} ({r['task_count']} tasks)")
        else:
            print(f"  FAIL: {name} -> {status}: {r.get('error', '')}")

    # Save results
    with open("data/regen_results/upload_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Now create the "all good datasets" manifest
    print(f"\n{'='*60}")
    print("CREATING ALL-GOOD-DATASETS MANIFEST")
    print(f"{'='*60}")

    all_good = []

    # Already good on HF (no changes needed)
    already_good = [
        ("DCAgent/exp_rpt_stack-junit", "stack-junit", "84%"),
        ("DCAgent/exp_rpt_crosscodeeval-csharp", "crosscodeeval-csharp", "68%"),
        ("DCAgent/exp-rdb-r2egym-easy", "r2egym-easy", "65%"),
        ("DCAgent/exp-rdb-r2egym-trivial", "r2egym-trivial", "60%"),
        ("DCAgent/exp-rdb-r2egym-medium", "r2egym-medium", "56%"),
        ("DCAgent/exp-rdb-r2egym-hard", "r2egym-hard", "56%"),
        ("DCAgent/exp_rpt_crosscodeeval-java", "crosscodeeval-java", "48%"),
        ("DCAgent/exp_rpt_crosscodeeval-typescript", "crosscodeeval-typescript", "48%"),
        ("DCAgent/exp_rpt_stack-bash-withtests-gpt5mini", "stack-bash-withtests-gpt5mini", "32%"),
        ("DCAgent/exp-rdb-r2egym-very_hard", "r2egym-very_hard", "28%"),
        ("DCAgent/exp_rpt_stack-cpp", "stack-cpp", "24%"),
        ("DCAgent/exp_rpt_stack-bash-withtests", "stack-bash-withtests", "24%"),
        ("DCAgent/exp_rpt_stack-bash", "stack-bash", "20%"),
        ("DCAgent/exp_rpt_stack-csharp", "stack-csharp", "12%"),
        ("DCAgent/exp_rpt_stack-ruby", "stack-ruby", "12%"),
        ("DCAgent/exp_rpt_stack-rust", "stack-rust", "12%"),
        ("DCAgent/exp_rpt_stack-pytest-gpt5mini", "stack-pytest-gpt5mini", "8%"),
    ]

    for repo, name, rate in already_good:
        all_good.append({
            "repo_id": repo,
            "name": name,
            "pass_rate_nano": rate,
            "version": "original",
            "status": "good",
        })

    # Fixed v2 datasets
    fixed_rates = {
        "stack-php": "100%", "bigcodebench": "54%", "codeelo": "52%",
        "e2egit": "48%", "crosscodeeval-python": "48%", "stack-pytest": "40%",
        "methods2test": "28%", "codereval-python": "24%",
        "stack-dockerfile": "16%", "stack-selfdoc": "16%",
        "softwareheritage": "12%", "stack-jest": "8%",
    }

    for r in results:
        if r["status"] == "UPLOADED":
            all_good.append({
                "repo_id": r["repo"],
                "name": r["name"],
                "pass_rate_nano": fixed_rates.get(r["name"], "unknown"),
                "version": "v2-fixed",
                "status": "fixed",
            })

    # Save manifest
    manifest = {
        "description": "All datasets with 8%+ pass rate on gpt-5-nano",
        "total_datasets": len(all_good),
        "datasets": all_good,
    }
    with open("data/all_good_datasets.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved to data/all_good_datasets.json ({len(all_good)} datasets)")
    for d in all_good:
        print(f"  {d['repo_id']}: {d['pass_rate_nano']} ({d['version']})")


if __name__ == "__main__":
    main()
