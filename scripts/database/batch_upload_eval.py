#!/usr/bin/env python3
"""Batch upload eval job results to Supabase + HuggingFace.

Supports two modes:
1. Explicit job dirs: upload specific completed jobs
2. Auto-detect overlong: scan a jobs directory for timed-out jobs

Usage:
    # Upload specific job dirs (normal eval)
    python scripts/database/batch_upload_eval.py \
        jobs/terminal_bench_2_model_A_20260407_* \
        jobs/dev_set_v2_model_B_20260407_*

    # Upload as overlong (timed-out jobs)
    python scripts/database/batch_upload_eval.py --overlong \
        jobs/terminal_bench_2_slow_model_*

    # Auto-detect overlong jobs in a directory
    python scripts/database/batch_upload_eval.py --auto-detect-overlong \
        --jobs-dir /path/to/jobs

    # Parallel upload with HF traces
    python scripts/database/batch_upload_eval.py -p 8 jobs/swebench_*

    # Skip HF upload (DB only)
    python scripts/database/batch_upload_eval.py --skip-hf jobs/swebench_*

    # Dry run
    python scripts/database/batch_upload_eval.py --dry-run jobs/swebench_*
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_db_path = _repo_root / "database" / "unified_db"
if str(_db_path) not in sys.path:
    sys.path.insert(0, str(_db_path))

DEFAULT_JOBS_DIR = _repo_root / "jobs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_iso(ts_str):
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def derive_hf_repo_id(job_dir):
    """Derive HF repo ID from job directory name."""
    name = Path(job_dir).name
    sanitized = name.replace("@", "-").replace(" ", "-")
    return f"DCAgent3/{sanitized}-traces"


def get_job_info(job_dir):
    """Extract basic info from a job directory."""
    job_dir = Path(job_dir)
    result_file = job_dir / "result.json"
    if not result_file.exists():
        return None

    try:
        d = json.load(open(result_file))
    except (json.JSONDecodeError, OSError):
        return None

    n_total = d.get("n_total_trials", 0)
    n_trials = d.get("stats", {}).get("n_trials", 0)
    finished = d.get("finished_at") is not None

    # Accuracy
    accuracy = None
    for eval_data in d.get("stats", {}).get("evals", {}).values():
        metrics = eval_data.get("metrics", [])
        if metrics and isinstance(metrics, list):
            mr = metrics[0].get("mean_reward")
            if mr is not None:
                accuracy = mr

    # Model from meta.env
    model = None
    meta_env = job_dir / "meta.env"
    if meta_env.exists():
        with open(meta_env) as f:
            for line in f:
                if line.startswith("MODEL="):
                    model = line.strip().split("=", 1)[1]
                    break

    # Benchmark from trial source
    benchmark = None
    for entry in sorted(job_dir.iterdir()):
        if entry.is_dir() and "__" in entry.name:
            tr = entry / "result.json"
            if tr.exists():
                try:
                    td = json.load(open(tr))
                    source = td.get("source", "")
                    if source:
                        for pfx in ["DCAgent3_", "DCAgent2_", "DCAgent_"]:
                            if source.startswith(pfx):
                                source = source[len(pfx):]
                                break
                        benchmark = source
                except (json.JSONDecodeError, OSError):
                    pass
                break

    model_short = (model.split("/")[-1] if model and "/" in model else model) or job_dir.name

    return {
        "dir": job_dir,
        "model": model or job_dir.name,
        "model_short": model_short,
        "benchmark": benchmark or "unknown",
        "n_trials": n_trials,
        "n_total": n_total,
        "finished": finished,
        "accuracy": accuracy,
    }


def detect_overlong_jobs(jobs_dir, min_elapsed_h=20, benchmark_filter=None):
    """Find jobs that timed out (no finished_at, elapsed > min_elapsed_h)."""
    results = []
    jobs_dir = Path(jobs_dir)

    for entry in sorted(jobs_dir.iterdir()):
        if not entry.is_dir():
            continue
        info = get_job_info(entry)
        if not info or info["finished"] or info["n_trials"] == 0:
            continue

        # Check elapsed
        try:
            d = json.load(open(entry / "result.json"))
        except:
            continue
        started = parse_iso(d.get("started_at"))
        if not started:
            continue

        latest = None
        for te in entry.iterdir():
            if not te.is_dir() or "__" not in te.name:
                continue
            tr = te / "result.json"
            if tr.exists():
                try:
                    td = json.load(open(tr))
                    fa = td.get("finished_at")
                    if fa:
                        ft = parse_iso(fa)
                        if ft and (latest is None or ft > latest):
                            latest = ft
                except:
                    pass

        if latest is None:
            continue
        elapsed_h = (latest - started).total_seconds() / 3600
        if elapsed_h < min_elapsed_h:
            continue

        if benchmark_filter and benchmark_filter.lower() not in info["benchmark"].lower():
            continue

        info["elapsed_h"] = elapsed_h
        results.append(info)

    # Deduplicate: same model+benchmark, keep best progress
    seen = {}
    for r in results:
        key = (r["model"], r["benchmark"])
        if key not in seen or r["n_trials"] > seen[key]["n_trials"]:
            seen[key] = r
    return sorted(seen.values(), key=lambda r: (r["benchmark"], r["model"]))


def upload_one(job_dir, benchmark_name=None, skip_hf=False, is_overlong=False):
    """Upload a single job. Returns (success, label, message)."""
    from hpc.launch_utils import sync_eval_to_database

    job_dir = Path(job_dir)
    label = job_dir.name
    hf_repo_id = None if skip_hf else derive_hf_repo_id(job_dir)

    try:
        # is_overlong only passed if True; older sync_eval_to_database (PR clone) doesn't accept it
        extra = {}
        if is_overlong:
            extra["is_overlong"] = True
        result = sync_eval_to_database(
            job_dir=job_dir,
            error_mode="skip_on_error",
            benchmark_name=benchmark_name,
            register_benchmark=True,
            forced_update=True,
            hf_repo_id=hf_repo_id,
            hf_token=os.environ.get("HF_TOKEN"),
            **extra,
        )
        if result.get("success"):
            job_id = result.get("job_id")
            n_trials = result.get("n_trials_uploaded", 0)
            hf_url = result.get("hf_dataset_url", "n/a")
            return True, label, f"job_id={job_id}, trials={n_trials}, hf={hf_url}"
        else:
            return False, label, result.get("error", "unknown")
    except Exception as e:
        return False, label, str(e)


def run_uploads(jobs_to_upload, n_workers, skip_hf, is_overlong):
    """Run uploads sequentially or in parallel."""
    n_workers = min(n_workers, len(jobs_to_upload))

    if n_workers <= 1:
        success = failed = 0
        for job in jobs_to_upload:
            ok, label, msg = upload_one(
                job["dir"], job.get("benchmark"), skip_hf, is_overlong)
            print(f"  [{'OK' if ok else 'FAIL'}] {label}: {msg}")
            if ok:
                success += 1
            else:
                failed += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        success = failed = 0

        def _worker(job):
            return upload_one(job["dir"], job.get("benchmark"), skip_hf, is_overlong)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker, j): j for j in jobs_to_upload}
            for future in as_completed(futures):
                ok, label, msg = future.result()
                print(f"  [{'OK' if ok else 'FAIL'}] {label}: {msg}")
                if ok:
                    success += 1
                else:
                    failed += 1

    print(f"\nDone: {success} uploaded, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Batch upload eval results to Supabase + HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode
    parser.add_argument("job_dirs", nargs="*", type=Path,
                        help="Job directories to upload (glob-friendly)")
    parser.add_argument("--auto-detect-overlong", action="store_true",
                        help="Auto-detect overlong jobs (>20h elapsed, incomplete)")
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR,
                        help="Jobs directory for --auto-detect-overlong mode")

    # Upload options
    parser.add_argument("--overlong", action="store_true",
                        help="Mark uploaded jobs as overlong (is_overlong=true)")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HuggingFace upload (DB only)")
    parser.add_argument("--force", action="store_true",
                        help="Override quality gates (low accuracy, incomplete trials)")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                        help="Number of parallel upload workers (default: 1)")

    # Filters (for auto-detect mode)
    parser.add_argument("--benchmark", "-b", type=str, default=None,
                        help="Filter by benchmark substring")
    parser.add_argument("--min-elapsed", type=float, default=20,
                        help="Min elapsed hours for overlong detection (default: 20)")

    # Safety
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be uploaded without uploading")

    args = parser.parse_args()

    if args.force:
        os.environ["EVAL_UPLOAD_FORCE"] = "1"

    if args.auto_detect_overlong:
        # Auto-detect mode
        print(f"Scanning {args.jobs_dir} for overlong jobs (>{args.min_elapsed}h elapsed, incomplete)...")
        jobs_to_upload = detect_overlong_jobs(args.jobs_dir, args.min_elapsed, args.benchmark)
        is_overlong = True
        print(f"Found {len(jobs_to_upload)} overlong model/benchmark pairs\n")
    elif args.job_dirs:
        # Explicit job dirs mode
        jobs_to_upload = []
        for jd in args.job_dirs:
            jd = jd.resolve()
            if not jd.exists():
                print(f"  [SKIP] {jd}: directory not found")
                continue
            info = get_job_info(jd)
            if info is None:
                print(f"  [SKIP] {jd}: no result.json")
                continue
            jobs_to_upload.append(info)
        is_overlong = args.overlong
        print(f"Found {len(jobs_to_upload)} job(s) to upload\n")
    else:
        parser.print_help()
        return

    if not jobs_to_upload:
        print("Nothing to upload.")
        return

    # Display table
    hdr_elapsed = "Elapsed" if args.auto_detect_overlong else ""
    print(f"  {'Model':<55} {'Benchmark':<25} {'Progress':>10} {'Acc':>7} {hdr_elapsed:>8}")
    print("  " + "-" * 110)
    for r in jobs_to_upload:
        acc = f"{r['accuracy']:.1%}" if r['accuracy'] is not None else "N/A"
        elapsed = f"{r.get('elapsed_h', 0):.1f}h" if args.auto_detect_overlong else ""
        print(f"  {r['model_short']:<55} {r['benchmark']:<25} {r['n_trials']:>4}/{r['n_total']:<4} {acc:>7} {elapsed:>8}")

    mode = "overlong" if is_overlong else "normal"
    hf = "skip" if args.skip_hf else "enabled"
    print(f"\n  Mode: {mode} | HF: {hf} | Workers: {args.parallel}")

    if args.dry_run:
        print("\n  Dry run — pass without --dry-run to upload.")
        return

    print()
    run_uploads(jobs_to_upload, args.parallel, args.skip_hf, is_overlong)


if __name__ == "__main__":
    main()
