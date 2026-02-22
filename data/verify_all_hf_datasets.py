#!/usr/bin/env python3
"""
Verify and validate all HuggingFace datasets under DCAgent/ and DCAgent2/.

For each dataset:
1. Download from HF (load_dataset, sample N rows, extract only those)
2. Run harbor with gpt-5-nano on the sample
3. Compare pass rate against old results
4. Report discrepancies

Usage:
    python data/verify_all_hf_datasets.py [--dataset DATASET] [--n-tasks 25] [--dry-run] [--skip-verified]
"""

import argparse
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path, PurePosixPath

DCAGENT_DIR = Path("/scratch/10000/eguha3/dc-agent")
RESULTS_JSON = DCAGENT_DIR / "data" / "hf_dataset_verification.json"
RESULTS_MD = DCAGENT_DIR / "data" / "hf_dataset_verification.md"
MODEL = "openai/gpt-5-nano-2025-08-07"
CONFIG = "eval/tacc/dcagent_eval_config.yaml"

# All datasets with their OLD known pass rates (from passrate_logs + jobs/dataset_*_full)
# Format: (repo_id, old_pass_rate_pct_or_None)
DATASETS = [
    # Stack-based (19) - old pass rates from passrate_logs
    ("DCAgent/exp_rpt_stack-bash", 4.0),
    ("DCAgent/exp_rpt_stack-bash-withtests", 4.0),
    ("DCAgent/exp_rpt_stack-bash-withtests-gpt5mini", None),  # no old data
    ("DCAgent/exp_rpt_stack-cpp", 36.0),
    ("DCAgent/exp_rpt_stack-csharp", 12.0),
    ("DCAgent/exp_rpt_stack-dockerfile", 0.0),
    ("DCAgent/exp_rpt_stack-dockerfile-gpt5mini", None),
    ("DCAgent/exp_rpt_stack-go", 0.0),
    ("DCAgent/exp_rpt_stack-jest", 0.0),
    ("DCAgent/exp_rpt_stack-junit", None),  # incomplete log
    ("DCAgent/exp_rpt_stack-php", 0.0),
    ("DCAgent/exp_rpt_stack-pytest", 0.0),
    ("DCAgent/exp_rpt_stack-pytest-gpt5mini", None),
    ("DCAgent/exp_rpt_stack-pytest-synthetic-gpt5nano", 0.0),
    ("DCAgent/exp_rpt_stack-pytest-withtests", 0.0),
    ("DCAgent/exp_rpt_stack-ruby", 0.0),
    ("DCAgent/exp_rpt_stack-rust", 0.0),
    ("DCAgent/exp_rpt_stack-selfdoc", 0.0),
    ("DCAgent/exp_rpt_stack-selfdoc-gpt5mini", None),
    # DCLM-mine (22) - old pass rates from jobs/dataset_*_full
    ("DCAgent/exp_rpt_bigcodebench", 60.0),
    ("DCAgent/exp_rpt_bugsinpy", 40.0),
    ("DCAgent/exp_rpt_bugsinpy-mf", 52.0),
    ("DCAgent/exp_rpt_bugswarm", 72.0),
    ("DCAgent/exp_rpt_codeelo", 84.0),
    ("DCAgent/exp_rpt_codenet-python", 92.0),
    ("DCAgent/exp_rpt_codereval-python", 52.0),
    ("DCAgent/exp_rpt_crosscodeeval-python", 60.0),  # crosscodeeval_full combined
    ("DCAgent/exp_rpt_crosscodeeval-java", None),
    ("DCAgent/exp_rpt_crosscodeeval-typescript", None),
    ("DCAgent/exp_rpt_crosscodeeval-csharp", None),
    ("DCAgent/exp_rpt_defects4j", 48.0),
    ("DCAgent/exp_rpt_e2egit", 56.0),
    ("DCAgent/exp_rpt_exercism-python", 80.0),
    ("DCAgent/exp_rpt_ghactions", 84.0),
    ("DCAgent/exp_rpt_manybugs", 60.0),
    ("DCAgent/exp_rpt_methods2test", 92.0),
    ("DCAgent/exp_rpt_pymethods2test", 76.0),
    ("DCAgent/exp_rpt_quixbugs-python-10k", 84.0),
    ("DCAgent/exp_rpt_softwareheritage", 72.0),
    ("DCAgent/exp_rpt_swebench", 64.0),
    ("DCAgent/exp_rpt_taco", 72.0),
    ("DCAgent/exp_rpt_unitsyn-python", 80.0),
    # 10K variants (5 unique)
    ("DCAgent/exp_rpt_bigcodebench-10k", None),
    ("DCAgent/exp_rpt_bugsinpy-10k", None),
    ("DCAgent/exp_rpt_codeelo-10k", None),
    ("DCAgent/exp_rpt_exercism-python-10k", None),
    ("DCAgent/exp_rpt_swebench-10k", None),
    # R2E-Gym (6)
    ("DCAgent/exp-rdb-r2egym-impossible", None),
    ("DCAgent/exp-rdb-r2egym-very_hard", None),
    ("DCAgent/exp-rdb-r2egym-hard", None),
    ("DCAgent/exp-rdb-r2egym-medium", None),
    ("DCAgent/exp-rdb-r2egym-easy", None),
    ("DCAgent/exp-rdb-r2egym-trivial", None),
    # Other (1)
    ("DCAgent2/logsmith-easy-500", None),
]


def safe_name(repo_id: str) -> str:
    """Convert repo_id to a safe job/dir name."""
    return repo_id.replace("/", "_").replace("-", "_")


def extract_tar_to_dir(archive_bytes: bytes, dest_dir: Path) -> None:
    """Extract a tar archive to a directory (from tasks_parquet_converter logic)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        for member in tf.getmembers():
            name = member.name
            # Strip leading ./ or /
            while name.startswith("./") or name.startswith("/"):
                name = name[1:] if name.startswith("/") else name[2:]
            if not name or ".." in name.split("/"):
                continue
            target = dest_dir / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                f = tf.extractfile(member)
                if f is not None:
                    target.write_bytes(f.read())


def download_and_sample(repo_id: str, n_tasks: int, seed: int = 42) -> tuple[str, Path | None, int]:
    """Download HF dataset, sample N tasks, extract only those to disk.

    Returns (status, sample_dir_or_None, total_task_count).
    status: "OK", "MISSING", "ERROR: ..."
    """
    try:
        from datasets import load_dataset

        print(f"  Loading HF dataset: {repo_id}")
        dataset = load_dataset(repo_id)
        train = dataset["train"]
        total = len(train)
        print(f"  Total rows: {total}")

        if total == 0:
            return "EMPTY", None, 0

        # Sample indices
        random.seed(seed)
        n = min(n_tasks, total)
        indices = random.sample(range(total), n)

        # Extract only sampled rows
        sample_dir = Path(tempfile.mkdtemp(prefix="verify_hf_"))
        for idx in indices:
            row = train[idx]
            rel_path = row["path"]
            archive_bytes = row["task_binary"]

            # Use the last component of path as task dir name
            task_name = PurePosixPath(rel_path).name if "/" in rel_path else rel_path
            task_dir = sample_dir / task_name
            extract_tar_to_dir(bytes(archive_bytes), task_dir)

        return "OK", sample_dir, total

    except Exception as e:
        error_msg = str(e)
        if any(k in error_msg.lower() for k in ["doesn't exist", "404", "not found", "couldn't find"]):
            return "MISSING", None, 0
        return f"ERROR: {error_msg[:300]}", None, 0


def run_harbor(task_dir: Path, job_name: str, n_concurrent: int = 8) -> dict | None:
    """Run harbor on the task directory. Returns parsed result.json or None."""
    # Clean up any old job with same name
    old_job_dir = DCAGENT_DIR / "jobs" / job_name
    if old_job_dir.exists():
        shutil.rmtree(old_job_dir, ignore_errors=True)

    cmd = [
        "harbor", "jobs", "start",
        "-p", str(task_dir),
        "--n-concurrent", str(n_concurrent),
        "--agent", "terminus-2",
        "--model", MODEL,
        "--env", "daytona",
        "--n-attempts", "1",
        "--max-retries", "0",
        "--job-name", job_name,
        "--config", CONFIG,
    ]

    print(f"  Running: {' '.join(cmd)}")
    sys.stdout.flush()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(DCAGENT_DIR),
            timeout=1800,  # 30 min timeout per dataset
        )
        print(f"  Harbor exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 30 minutes")
        return None
    except Exception as e:
        print(f"  ERROR running harbor: {e}")
        return None

    # Find result.json
    result_file = None
    jobs_dir = DCAGENT_DIR / "jobs"
    for d in sorted(jobs_dir.glob(f"{job_name}*"), reverse=True):
        rf = d / "result.json"
        if rf.exists():
            result_file = rf
            break

    if result_file is None:
        print(f"  No result.json found for {job_name}")
        return None

    try:
        with open(result_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error reading result.json: {e}")
        return None


def parse_pass_rate(result_data: dict) -> tuple[float | None, int, int]:
    """Parse pass rate from harbor result.json. Returns (pct, passed, total)."""
    if "stats" in result_data:
        stats = result_data["stats"]
        evals = stats.get("evals", {})
        if evals:
            eval_key = list(evals.keys())[0]
            metrics = evals[eval_key].get("metrics", [])
            if metrics:
                mean = metrics[0].get("mean", 0)
                n = metrics[0].get("n", 0)
                return mean * 100, int(round(mean * n)), n

    if "results" in result_data:
        results = result_data["results"]
        total = len(results)
        passed = sum(1 for r in results if r.get("reward", 0) > 0)
        return (passed / total * 100) if total > 0 else 0, passed, total

    return None, 0, 0


def verify_dataset(repo_id: str, old_rate: float | None, n_tasks: int = 25,
                   n_concurrent: int = 8, dry_run: bool = False) -> dict:
    """Verify a single dataset. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"Verifying: {repo_id} (old pass rate: {old_rate}%)" if old_rate is not None
          else f"Verifying: {repo_id} (no old data)")
    print(f"{'='*60}")

    result = {
        "repo_id": repo_id,
        "status": None,
        "total_tasks": 0,
        "sampled_tasks": 0,
        "pass_rate": None,
        "old_pass_rate": old_rate,
        "passed": 0,
        "tested": 0,
        "discrepancy": False,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    # Phase A: Download & sample
    status, sample_dir, total_count = download_and_sample(repo_id, n_tasks)
    result["total_tasks"] = total_count

    if status == "MISSING":
        result["status"] = "MISSING"
        result["error"] = "Dataset not found on HuggingFace"
        print(f"  MISSING - dataset does not exist on HF")
        return result

    if status == "EMPTY":
        result["status"] = "EMPTY"
        result["error"] = "Dataset has 0 tasks"
        print(f"  EMPTY - no tasks found")
        return result

    if status.startswith("ERROR"):
        result["status"] = "ERROR"
        result["error"] = status
        print(f"  {status}")
        return result

    actual_sampled = len(list(sample_dir.iterdir())) if sample_dir else 0
    result["sampled_tasks"] = actual_sampled
    print(f"  Downloaded & sampled {actual_sampled} tasks (of {total_count} total)")

    if dry_run:
        result["status"] = "DRY_RUN"
        if sample_dir:
            shutil.rmtree(sample_dir, ignore_errors=True)
        print(f"  DRY_RUN - skipping harbor")
        return result

    # Phase B: Run harbor
    sname = safe_name(repo_id)
    job_name = f"verify_{sname}"
    harbor_result = run_harbor(sample_dir, job_name, n_concurrent)

    # Cleanup sample dir
    if sample_dir:
        shutil.rmtree(sample_dir, ignore_errors=True)

    if harbor_result is None:
        result["status"] = "HARBOR_FAILED"
        result["error"] = "Harbor did not produce results"
        print(f"  HARBOR_FAILED")
        return result

    # Parse pass rate
    pass_rate, passed, total = parse_pass_rate(harbor_result)
    result["pass_rate"] = pass_rate
    result["passed"] = passed
    result["tested"] = total
    result["status"] = "VERIFIED"

    # Check discrepancy (>20% difference from old rate)
    if old_rate is not None and pass_rate is not None:
        diff = abs(pass_rate - old_rate)
        if diff > 20:
            result["discrepancy"] = True
            print(f"  *** DISCREPANCY: new={pass_rate:.1f}% vs old={old_rate:.1f}% (diff={diff:.1f}%)")
        else:
            print(f"  OK: {pass_rate:.1f}% (old={old_rate:.1f}%, diff={diff:.1f}%)")
    else:
        print(f"  VERIFIED: {pass_rate:.1f}% ({passed}/{total})")

    return result


def save_results(results: list[dict]):
    """Save results to JSON and markdown."""
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    lines = [
        "# HuggingFace Dataset Verification Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {MODEL}",
        f"Total datasets: {len(results)}",
        "",
        "## Summary",
        "",
        f"- Verified: {sum(1 for r in results if r['status'] == 'VERIFIED')}",
        f"- Missing: {sum(1 for r in results if r['status'] == 'MISSING')}",
        f"- Empty: {sum(1 for r in results if r['status'] == 'EMPTY')}",
        f"- Error: {sum(1 for r in results if r['status'] in ('ERROR', 'HARBOR_FAILED'))}",
        f"- Dry run: {sum(1 for r in results if r['status'] == 'DRY_RUN')}",
        f"- Discrepancies: {sum(1 for r in results if r.get('discrepancy'))}",
        "",
        "## Results Table",
        "",
        "| Dataset | Status | Total | New Rate | Old Rate | Diff | Flag |",
        "|---------|--------|-------|----------|----------|------|------|",
    ]

    for r in results:
        repo = r["repo_id"]
        status = r["status"]
        total = r["total_tasks"]
        new_rate = f"{r['pass_rate']:.1f}%" if r["pass_rate"] is not None else "N/A"
        old_rate = f"{r['old_pass_rate']:.1f}%" if r.get("old_pass_rate") is not None else "N/A"
        if r["pass_rate"] is not None and r.get("old_pass_rate") is not None:
            diff = f"{r['pass_rate'] - r['old_pass_rate']:+.1f}%"
        else:
            diff = "-"
        flag = "**DISC**" if r.get("discrepancy") else ""
        if status == "MISSING":
            flag = "**MISSING**"
        elif status in ("ERROR", "HARBOR_FAILED"):
            flag = f"**{status}**"
        lines.append(f"| {repo} | {status} | {total} | {new_rate} | {old_rate} | {diff} | {flag} |")

    # Problem summary
    problems = [r for r in results if r["status"] != "VERIFIED" or r.get("discrepancy")]
    if problems:
        lines.extend(["", "## Action Required", ""])
        for r in problems:
            if r["status"] == "MISSING":
                lines.append(f"- **{r['repo_id']}**: MISSING - needs regeneration and upload")
            elif r.get("discrepancy"):
                lines.append(f"- **{r['repo_id']}**: Pass rate changed from {r['old_pass_rate']:.1f}% to {r['pass_rate']:.1f}% - needs regeneration")
            else:
                lines.append(f"- **{r['repo_id']}**: {r['status']} - {r.get('error', '')}")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to {RESULTS_JSON} and {RESULTS_MD}")


def load_existing_results() -> dict[str, dict]:
    """Load existing results so we can skip already-verified datasets."""
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON) as f:
                return {r["repo_id"]: r for r in json.load(f)}
        except Exception:
            pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Verify all HF datasets")
    parser.add_argument("--dataset", type=str, help="Verify a single dataset (repo_id)")
    parser.add_argument("--n-tasks", type=int, default=25, help="Tasks to sample per dataset")
    parser.add_argument("--n-concurrent", type=int, default=8, help="Harbor concurrency")
    parser.add_argument("--dry-run", action="store_true", help="Only download check, skip harbor")
    parser.add_argument("--skip-verified", action="store_true", help="Skip already-verified datasets")
    parser.add_argument("--category", choices=["stack", "dclm", "10k", "r2e", "other"])
    args = parser.parse_args()

    # Filter datasets
    all_datasets = DATASETS
    if args.dataset:
        all_datasets = [(r, p) for r, p in DATASETS if r == args.dataset]
        if not all_datasets:
            # Allow arbitrary dataset not in the list
            all_datasets = [(args.dataset, None)]
    elif args.category:
        ranges = {"stack": (0, 19), "dclm": (19, 42), "10k": (42, 47), "r2e": (47, 53), "other": (53, 54)}
        s, e = ranges[args.category]
        all_datasets = DATASETS[s:e]

    existing = load_existing_results() if args.skip_verified else {}

    results = []
    # Preserve already-verified results
    for repo_id, _ in DATASETS:
        if repo_id in existing and existing[repo_id].get("status") == "VERIFIED":
            if not any(repo_id == r for r, _ in all_datasets):
                results.append(existing[repo_id])

    print(f"Verifying {len(all_datasets)} datasets ({args.n_tasks} tasks each)")
    print(f"Dry run: {args.dry_run}")
    if args.skip_verified:
        skip_count = sum(1 for r, _ in all_datasets if r in existing and existing[r].get("status") == "VERIFIED")
        print(f"Will skip {skip_count} already-verified")
    print()
    sys.stdout.flush()

    for i, (repo_id, old_rate) in enumerate(all_datasets):
        if args.skip_verified and repo_id in existing and existing[repo_id].get("status") == "VERIFIED":
            print(f"[{i+1}/{len(all_datasets)}] SKIP: {repo_id}")
            results.append(existing[repo_id])
            continue

        print(f"\n[{i+1}/{len(all_datasets)}] {repo_id}")
        sys.stdout.flush()
        result = verify_dataset(repo_id, old_rate, args.n_tasks, args.n_concurrent, args.dry_run)
        results.append(result)
        save_results(results)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    verified = [r for r in results if r["status"] == "VERIFIED"]
    missing = [r for r in results if r["status"] == "MISSING"]
    disc = [r for r in results if r.get("discrepancy")]

    if verified:
        rates = [r["pass_rate"] for r in verified if r["pass_rate"] is not None]
        print(f"Verified: {len(verified)}/{len(results)}")
        if rates:
            print(f"Pass rates: min={min(rates):.1f}%, max={max(rates):.1f}%, mean={sum(rates)/len(rates):.1f}%")
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for r in missing:
            print(f"  - {r['repo_id']}")
    if disc:
        print(f"\nDISCREPANCIES ({len(disc)}):")
        for r in disc:
            print(f"  - {r['repo_id']}: {r['old_pass_rate']:.1f}% -> {r['pass_rate']:.1f}%")


if __name__ == "__main__":
    main()
