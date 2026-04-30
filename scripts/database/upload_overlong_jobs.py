#!/usr/bin/env python3
"""Detect and upload overlong eval jobs (hit SLURM 24h time limit).

Scans a jobs directory for incomplete runs that elapsed >20h, checks if the
model/benchmark pair already has a Finished record in Supabase, and uploads
missing ones with is_overlong=true.

Usage:
    # Dry run (default) — show what would be uploaded
    python scripts/database/upload_overlong_jobs.py

    # Actually upload
    python scripts/database/upload_overlong_jobs.py --upload

    # Filter to specific benchmark
    python scripts/database/upload_overlong_jobs.py --upload --benchmark terminal_bench_2

    # Custom jobs dir
    python scripts/database/upload_overlong_jobs.py --upload --jobs-dir /path/to/jobs
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


def _preset_dataset_prefixes():
    """Run-tag prefixes derived from the listener's PRESETS dict.

    Run dirs are named ``<safe-dataset>_<safe-model>_<date>_<time>`` where the
    safe-dataset name is the last path segment of the HF dataset id with
    ``-`` -> ``_``. Reading PRESETS keeps this list in sync as new presets
    are added (instead of a hardcoded list that drifts).
    """
    try:
        from eval.unified_eval_listener import PRESETS
    except Exception:  # listener importable in any env that ships the repo
        return []

    seen = []
    for cfg in PRESETS.values():
        # PRESETS uses 'datasets' (plural list); fall back to 'dataset' for safety.
        datasets = cfg.get("datasets") or ([cfg["dataset"]] if cfg.get("dataset") else [])
        for ds in datasets:
            short = ds.split("/", 1)[-1] if "/" in ds else ds
            short = short.replace("-", "_")
            if short and short not in seen:
                seen.append(short)
    return seen


_PRESET_PREFIXES = _preset_dataset_prefixes()


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


def detect_overlong_jobs(jobs_dir, min_elapsed_h=20, benchmark_filter=None):
    """Find jobs that timed out (no finished_at, elapsed > min_elapsed_h)."""
    results = []
    jobs_dir = Path(jobs_dir)

    for entry in sorted(jobs_dir.iterdir()):
        if not entry.is_dir():
            continue
        rf = entry / "result.json"
        if not rf.exists():
            continue
        try:
            d = json.load(open(rf))
        except (json.JSONDecodeError, OSError):
            continue

        if d.get("finished_at") is not None:
            continue
        n_total = d.get("n_total_trials", 0)
        n_trials = d.get("stats", {}).get("n_trials", 0)
        if n_trials == 0:
            continue

        started = parse_iso(d.get("started_at"))
        if not started:
            continue

        # Find latest trial finish time
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
                except (json.JSONDecodeError, OSError):
                    pass

        if latest is None:
            continue
        elapsed_h = (latest - started).total_seconds() / 3600
        if elapsed_h < min_elapsed_h:
            continue

        # Parse model from meta.env
        model = None
        meta_env = entry / "meta.env"
        if meta_env.exists():
            with open(meta_env) as f:
                for line in f:
                    if line.startswith("MODEL="):
                        model = line.strip().split("=", 1)[1]
                        break

        # Derive benchmark from trial source field
        benchmark = None
        for te in sorted(entry.iterdir()):
            if not te.is_dir() or "__" not in te.name:
                continue
            tr = te / "result.json"
            if tr.exists():
                try:
                    td = json.load(open(tr))
                    source = td.get("source", "")
                    if source:
                        for org_prefix in ["DCAgent2_", "DCAgent_"]:
                            if source.startswith(org_prefix):
                                source = source[len(org_prefix):]
                                break
                        benchmark = source
                    break
                except (json.JSONDecodeError, OSError):
                    pass

        # Fallback: infer from dir name. The prefix list comes from the
        # listener's PRESETS dict, so any new preset auto-stays-in-sync.
        # Match longest first so e.g. "swebench_verified_random_100_folders"
        # wins over a shorter conflicting prefix.
        prefixes_long_first = sorted(_PRESET_PREFIXES, key=len, reverse=True)
        if not benchmark:
            dirname = entry.name
            for prefix in prefixes_long_first:
                if dirname.startswith(prefix):
                    benchmark = prefix
                    break

        if not model:
            dirname = entry.name
            for prefix in prefixes_long_first:
                token = prefix + "_"
                if dirname.startswith(token):
                    rest = dirname[len(token):]
                    parts = rest.rsplit("_", 2)
                    if len(parts) >= 3 and len(parts[-1]) == 6 and len(parts[-2]) == 8:
                        model = "_".join(parts[:-2])
                    break

        if benchmark_filter and benchmark_filter.lower() not in (benchmark or "").lower():
            continue

        # Extract accuracy
        accuracy = None
        for eval_data in d.get("stats", {}).get("evals", {}).values():
            metrics = eval_data.get("metrics", [])
            if metrics and isinstance(metrics, list):
                mr = metrics[0].get("mean_reward")
                if mr is not None:
                    accuracy = mr

        results.append({
            "dir": entry,
            "model": model or entry.name,
            "model_short": (model.split("/")[-1] if model and "/" in model else model) or entry.name,
            "benchmark": benchmark or "unknown",
            "n_trials": n_trials,
            "n_total": n_total,
            "elapsed_h": elapsed_h,
            "accuracy": accuracy,
        })

    # Deduplicate: same model+benchmark, keep best progress
    seen = {}
    for r in results:
        key = (r["model"], r["benchmark"])
        if key not in seen or r["n_trials"] > seen[key]["n_trials"]:
            seen[key] = r
    return sorted(seen.values(), key=lambda r: (r["benchmark"], r["model"]))


def check_db_exists(model_name, benchmark_name):
    """Check if model/benchmark pair already has a Finished job in DB."""
    try:
        from unified_db.utils import get_model_by_name, get_benchmark_by_name, get_supabase_client

        model = get_model_by_name(model_name)
        if not model:
            return False, None

        benchmark = get_benchmark_by_name(benchmark_name)
        if not benchmark:
            return False, None

        client = get_supabase_client()
        resp = (
            client.table("sandbox_jobs")
            .select("id,job_status,is_overlong")
            .eq("model_id", model["id"])
            .eq("benchmark_id", benchmark["id"])
            .eq("job_status", "Finished")
            .limit(1)
            .execute()
        )

        if resp.data:
            return True, resp.data[0]
        return False, None
    except Exception as e:
        print(f"  [warn] DB check failed for {model_name}/{benchmark_name}: {e}")
        return False, None


def derive_hf_repo_id(job_dir):
    """Derive HF repo ID from job directory name."""
    name = Path(job_dir).name
    sanitized = name.replace("@", "-").replace(" ", "-")
    return f"DCAgent2/{sanitized}-traces"


def upload_job(job_dir, benchmark_name, skip_hf=False):
    """Upload a single overlong job."""
    from hpc.launch_utils import sync_eval_to_database

    hf_repo_id = None if skip_hf else derive_hf_repo_id(job_dir)

    result = sync_eval_to_database(
        job_dir=job_dir,
        error_mode="skip_on_error",
        benchmark_name=benchmark_name,
        register_benchmark=True,
        forced_update=True,
        is_overlong=True,
        hf_repo_id=hf_repo_id,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    return result


def _upload_one(args_tuple):
    """Worker function for parallel uploads."""
    r, skip_hf = args_tuple
    label = f"{r['model_short']} / {r['benchmark']}"
    try:
        result = upload_job(r["dir"], r["benchmark"], skip_hf=skip_hf)
        if result.get("success"):
            job_id = result.get("job_id")
            n_trials = result.get("n_trials_uploaded", 0)
            hf_url = result.get("hf_dataset_url", "n/a")
            return True, label, f"job_id={job_id}, trials={n_trials}, hf={hf_url}"
        else:
            return False, label, result.get("error", "unknown")
    except Exception as e:
        return False, label, str(e)


def main():
    parser = argparse.ArgumentParser(description="Detect and upload overlong eval jobs")
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR)
    parser.add_argument("--upload", action="store_true", help="Actually upload (default: dry run)")
    parser.add_argument("--benchmark", "-b", type=str, default=None, help="Filter by benchmark")
    parser.add_argument("--min-elapsed", type=float, default=20, help="Min elapsed hours to consider overlong")
    parser.add_argument("--skip-db-check", action="store_true", help="Skip checking DB for existing records")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HuggingFace upload (DB only)")
    parser.add_argument("--force", action="store_true", help="Override quality gates")
    parser.add_argument("--parallel", "-p", type=int, default=1, help="Number of parallel upload workers (default: 1)")
    args = parser.parse_args()

    if args.force:
        os.environ["EVAL_UPLOAD_FORCE"] = "1"

    print(f"Scanning {args.jobs_dir} for overlong jobs (>{args.min_elapsed}h elapsed, incomplete)...")
    overlong = detect_overlong_jobs(args.jobs_dir, args.min_elapsed, args.benchmark)
    print(f"Found {len(overlong)} overlong model/benchmark pairs\n")

    if not overlong:
        return

    # Display table
    print(f"{'Model':<55} {'Benchmark':<20} {'Progress':>12} {'Elapsed':>8} {'Accuracy':>8} {'Action':>10}")
    print("-" * 120)

    to_upload = []
    for r in overlong:
        acc_str = f"{r['accuracy']:.1%}" if r['accuracy'] is not None else "N/A"
        progress = f"{r['n_trials']}/{r['n_total']}"

        if not args.skip_db_check:
            exists, existing = check_db_exists(r["model"], r["benchmark"])
            if exists:
                is_ol = existing.get("is_overlong", False) if existing else False
                action = "skip (exists" + (", overlong" if is_ol else "") + ")"
            else:
                action = "UPLOAD" if args.upload else "would upload"
                to_upload.append(r)
        else:
            action = "UPLOAD" if args.upload else "would upload"
            to_upload.append(r)

        print(f"{r['model_short']:<55} {r['benchmark']:<20} {progress:>12} {r['elapsed_h']:>6.1f}h {acc_str:>8} {action:>10}")

    print(f"\n{len(to_upload)} jobs to upload, {len(overlong) - len(to_upload)} already in DB")

    if not args.upload:
        print("\nDry run — pass --upload to actually upload.")
        return

    if not to_upload:
        print("\nNothing to upload.")
        return

    n_workers = min(args.parallel, len(to_upload))
    hf_label = "skip" if args.skip_hf else "enabled"
    print(f"\nUploading {len(to_upload)} overlong jobs (workers={n_workers}, hf={hf_label})...\n")

    if n_workers <= 1:
        # Sequential
        success = 0
        failed = 0
        for r in to_upload:
            ok, label, msg = _upload_one((r, args.skip_hf))
            status = "OK" if ok else "FAILED"
            print(f"  [{status}] {label}: {msg}")
            if ok:
                success += 1
            else:
                failed += 1
    else:
        # Parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        success = 0
        failed = 0
        work = [(r, args.skip_hf) for r in to_upload]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_upload_one, w): w[0]["model_short"] for w in work}
            for future in as_completed(futures):
                ok, label, msg = future.result()
                status = "OK" if ok else "FAILED"
                print(f"  [{status}] {label}: {msg}")
                if ok:
                    success += 1
                else:
                    failed += 1

    print(f"\nDone: {success} uploaded, {failed} failed")


if __name__ == "__main__":
    main()
