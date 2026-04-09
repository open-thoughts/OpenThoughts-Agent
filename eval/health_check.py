#!/usr/bin/env python3
"""Health check for running eval jobs — catches silent failures early.

Checks per job:
  1. vLLM alive (no OOM/crash/segfault in log)
  2. Progress rate (trials/hour, stall detection)
  3. Error distribution (invalid errors vs threshold)
  4. Daytona errors (auth failures, rate limits)
  5. Stuck trials (started but no result.json for >2h)

Usage:
    python eval/health_check.py              # check all running jobs
    python eval/health_check.py -b tb2       # filter by benchmark
    python eval/health_check.py --json       # machine-readable output
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_DIR / "eval" / "logs"
EXTRA_LOG_DIRS = [
    REPO_DIR / "eval" / "local" / "logs",
    REPO_DIR / "eval" / "MBZ" / "logs",
    REPO_DIR / "eval" / "local" / "mbz" / "logs",
    REPO_DIR / "eval" / "local" / "m2" / "logs",
]
DEFAULT_JOBS_DIR = REPO_DIR / "jobs"

VALID_ERROR_TYPES = {
    "AgentTimeoutError", "ContextLengthExceededError",
    "SummarizationTimeout", "SummarizationTimeoutError", "BadRequestError",
}

# vLLM crash signatures
VLLM_CRASH_PATTERNS = [
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemoryError", re.IGNORECASE),
    re.compile(r"torch\.cuda\.OutOfMemoryError"),
    re.compile(r"Segmentation fault"),
    re.compile(r"SIGKILL"),
    re.compile(r"SIGTERM"),
    re.compile(r"RuntimeError.*CUDA"),
    re.compile(r"NCCL error"),
    re.compile(r"Process.*killed"),
]

# Daytona error patterns in data log
DAYTONA_ERROR_PATTERNS = [
    re.compile(r"Bearer token is invalid"),
    re.compile(r"DaytonaError"),
    re.compile(r"DaytonaRateLimitError"),
    re.compile(r"daytona.*unauthorized", re.IGNORECASE),
    re.compile(r"daytona.*403", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Reused helpers from check_progress.py
# ---------------------------------------------------------------------------

def get_running_jobs():
    result = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "--format=%.10i %.15j %.8T %.20S", "--noheader"],
        capture_output=True, text=True,
    )
    jobs = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 4:
            jobs.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()))
    return jobs


def _find_log(jid, job_name, prefix, ext):
    """Find a log file across all log directories."""
    all_log_dirs = [LOGS_DIR] + [d for d in EXTRA_LOG_DIRS if d.exists()]
    candidates = []
    for log_dir in all_log_dirs:
        candidates.append(log_dir / f"{prefix}_{jid}.{ext}")
    for c in candidates:
        if c.exists():
            return c
    return None


def parse_eval_log(jid, job_name):
    """Parse the data log to extract model, bench, run_tag, error_threshold."""
    all_log_dirs = [LOGS_DIR] + [d for d in EXTRA_LOG_DIRS if d.exists()]
    candidates = []
    for log_dir in all_log_dirs:
        candidates.append(log_dir / f"{job_name}_{jid}.out")
    if job_name.startswith("eval_dp_"):
        for log_dir in all_log_dirs:
            candidates.append(log_dir / f"eval_dp_{jid}.out")
    elif job_name.startswith("eval_"):
        for log_dir in all_log_dirs:
            candidates.append(log_dir / f"eval_{jid}.out")
    if job_name.startswith("res_dp_"):
        for log_dir in all_log_dirs:
            candidates.append(log_dir / f"res_dp_{jid}.out")
    elif job_name.startswith("res_"):
        for log_dir in all_log_dirs:
            candidates.append(log_dir / f"res_{jid}.out")
    # Also try data_<JID>.out (MBZ naming)
    for log_dir in all_log_dirs:
        candidates.append(log_dir / f"data_{jid}.out")

    model = bench = run_tag = None
    num_shards = 0
    error_threshold = 10  # default
    for log in candidates:
        if log.exists():
            with open(log) as f:
                for line in f:
                    if line.startswith("Model: "):
                        model = line.strip()[7:]
                    elif line.startswith("Dataset: "):
                        bench = line.strip()[9:]
                    elif line.startswith("Run tag: "):
                        run_tag = line.strip()[9:]
                    elif line.startswith("Error threshold: "):
                        try:
                            error_threshold = int(line.strip().split(": ", 1)[1])
                        except (ValueError, IndexError):
                            pass
                    elif "total shards)" in line:
                        try:
                            num_shards = int(line.split("total shards")[0].split(",")[-1].strip())
                        except (ValueError, IndexError):
                            pass
            break
    return model, bench, run_tag, num_shards, error_threshold


def get_progress_single(run_tag, jobs_dir):
    """Get progress from a single job dir. Returns dict with detailed info."""
    if not run_tag:
        return None
    job_dir = jobs_dir / run_tag
    rf = job_dir / "result.json"
    if not rf.exists():
        return None
    try:
        with open(rf) as f:
            d = json.load(f)
        completed = d.get("stats", {}).get("n_trials")
        total = d.get("n_total_trials")
        finished = d.get("finished_at") is not None
        started_at = d.get("started_at")

        # Parse exception stats
        invalid_trials = set()
        exception_counts = defaultdict(int)
        evals = d.get("stats", {}).get("evals", {})
        accuracy = None
        for eval_data in evals.values():
            for error_type, trial_names in eval_data.get("exception_stats", {}).items():
                if isinstance(trial_names, list):
                    exception_counts[error_type] += len(trial_names)
                    if error_type not in VALID_ERROR_TYPES:
                        invalid_trials.update(trial_names)
            metrics = eval_data.get("metrics", [])
            if metrics and isinstance(metrics, list):
                mr = metrics[0].get("mean_reward")
                if mr is not None:
                    accuracy = mr

        # Count on-disk completed trials from reward_stats
        n_on_disk = None
        for eval_data in evals.values():
            rs = eval_data.get("reward_stats", {}).get("reward", {})
            if rs:
                n_on_disk = sum(len(v) for v in rs.values() if isinstance(v, list))
                break
        if n_on_disk is None:
            n_on_disk = completed

        return {
            "completed": completed,
            "total": total,
            "n_on_disk": n_on_disk,
            "n_invalid_errors": len(invalid_trials),
            "finished": finished,
            "accuracy": accuracy,
            "started_at": started_at,
            "exception_counts": dict(exception_counts),
        }
    except Exception:
        return None


def format_elapsed(start_str):
    try:
        if start_str == "N/A":
            return "-", 0
        st = datetime.fromisoformat(start_str.replace("T", " "))
        delta = datetime.now() - st
        secs = int(delta.total_seconds())
        hours, rem = divmod(secs, 3600)
        mins, _ = divmod(rem, 60)
        return f"{hours}h{mins:02d}m", secs
    except Exception:
        return "?", 0


# ---------------------------------------------------------------------------
# Health check functions
# ---------------------------------------------------------------------------

def check_vllm_health(jid):
    """Check vLLM log for crashes/OOM. Returns (status, detail)."""
    vllm_log = _find_log(jid, "", "vllm", "log")
    if not vllm_log:
        return "UNKNOWN", "no vllm log found"

    # Read last 200 lines for recent state
    try:
        with open(vllm_log) as f:
            lines = f.readlines()
    except Exception as e:
        return "UNKNOWN", f"cannot read log: {e}"

    if not lines:
        return "UNKNOWN", "empty vllm log"

    # Check for crash patterns in entire log
    ansi_strip = re.compile(r"\x1b\[[0-9;]*m")
    crash_found = []
    for line in lines:
        clean = ansi_strip.sub("", line)
        for pat in VLLM_CRASH_PATTERNS:
            if pat.search(clean):
                crash_found.append(pat.pattern)
                break

    if crash_found:
        # Deduplicate
        unique = list(dict.fromkeys(crash_found))
        return "CRITICAL", f"crash detected: {', '.join(unique[:3])}"

    # Check recent activity — look for HTTP 200 or throughput lines in last 100 lines
    tail = lines[-100:]
    last_ok_time = None
    last_activity_time = None
    # Strip ANSI escape codes for parsing
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    for line in reversed(tail):
        line = ansi_re.sub("", line)
        # Parse timestamp from vLLM log: "INFO 04-07 13:10:50"
        m = re.search(r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m:
            if last_activity_time is None:
                try:
                    ts_str = m.group(1)
                    now = datetime.now()
                    month, day = ts_str.split(" ")[0].split("-")
                    time_part = ts_str.split(" ")[1]
                    last_activity_time = datetime(
                        now.year, int(month), int(day),
                        *[int(x) for x in time_part.split(":")]
                    )
                except Exception:
                    pass
        if "200 OK" in line and last_ok_time is None:
            # HTTP OK lines may not have a date stamp — use nearby throughput timestamp
            # or fall back to file mtime
            m2 = re.search(r"INFO (\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if m2:
                try:
                    ts_str = m2.group(1)
                    now = datetime.now()
                    month, day = ts_str.split(" ")[0].split("-")
                    time_part = ts_str.split(" ")[1]
                    last_ok_time = datetime(
                        now.year, int(month), int(day),
                        *[int(x) for x in time_part.split(":")]
                    )
                except Exception:
                    pass
            else:
                # No timestamp on this line — use last_activity_time if already found,
                # otherwise use file mtime as proxy
                if last_activity_time:
                    last_ok_time = last_activity_time
                else:
                    try:
                        mtime = os.path.getmtime(vllm_log)
                        last_ok_time = datetime.fromtimestamp(mtime)
                    except Exception:
                        pass
            break  # found the most recent 200 OK

    now = datetime.now()
    if last_ok_time:
        age_secs = (now - last_ok_time).total_seconds()
        if age_secs < 300:
            return "OK", f"last 200 OK {int(age_secs)}s ago"
        elif age_secs < 1800:
            return "OK", f"last 200 OK {int(age_secs / 60)}m ago"
        else:
            return "WARN", f"no 200 OK in {int(age_secs / 60)}m"
    elif last_activity_time:
        age_secs = (now - last_activity_time).total_seconds()
        if age_secs < 600:
            return "OK", f"active {int(age_secs)}s ago (no recent requests)"
        else:
            return "WARN", f"last log activity {int(age_secs / 60)}m ago"
    else:
        # Fall back to file modification time
        try:
            mtime = os.path.getmtime(vllm_log)
            age_secs = now.timestamp() - mtime
            if age_secs < 600:
                return "OK", f"log updated {int(age_secs)}s ago"
            else:
                return "WARN", f"log not updated in {int(age_secs / 60)}m"
        except Exception:
            return "UNKNOWN", "cannot parse vllm timestamps"


def check_progress_rate(progress_info, elapsed_secs):
    """Check trial throughput. Returns (status, detail)."""
    if not progress_info:
        return "UNKNOWN", "no progress data"

    completed = progress_info.get("completed")
    total = progress_info.get("total")

    if completed is None or total is None:
        return "UNKNOWN", "no progress data"

    if elapsed_secs < 600:  # < 10 min, still starting
        if completed == 0:
            return "OK", "starting up"
        return "OK", f"{completed}/{total}"

    if elapsed_secs > 0:
        rate = completed / (elapsed_secs / 3600)
    else:
        rate = 0

    if completed >= total:
        return "OK", f"complete ({completed}/{total})"

    if rate < 0.5 and elapsed_secs > 3600:
        return "CRITICAL", f"{rate:.1f} trials/hr — effectively stuck"
    elif rate < 1 and elapsed_secs > 1800:
        return "WARN", f"{rate:.1f} trials/hr — very slow"
    else:
        return "OK", f"{rate:.1f} trials/hr ({completed}/{total})"


def check_error_distribution(progress_info, error_threshold):
    """Check invalid error count vs threshold. Returns (status, detail)."""
    if not progress_info:
        return "UNKNOWN", "no data"

    n_invalid = progress_info.get("n_invalid_errors", 0)

    if n_invalid >= error_threshold:
        return "CRITICAL", f"{n_invalid}/{error_threshold} invalid errors — threshold hit"
    elif n_invalid >= error_threshold * 0.5:
        return "WARN", f"{n_invalid}/{error_threshold} invalid errors — near threshold"
    else:
        return "OK", f"{n_invalid}/{error_threshold} invalid errors"


def check_daytona_errors(jid, job_name, progress_info):
    """Check for Daytona auth/rate-limit errors. Returns (status, detail)."""
    # Check exception_stats from result.json
    daytona_count = 0
    if progress_info:
        exc_counts = progress_info.get("exception_counts", {})
        for key in ["DaytonaError", "DaytonaRateLimitError"]:
            daytona_count += exc_counts.get(key, 0)

    # Also grep the data log for auth errors
    data_log = _find_log(jid, "", "data", "out")
    log_auth_errors = 0
    if data_log and data_log.exists():
        try:
            with open(data_log) as f:
                for line in f:
                    for pat in DAYTONA_ERROR_PATTERNS:
                        if pat.search(line):
                            log_auth_errors += 1
                            break
        except Exception:
            pass

    total_errors = daytona_count + log_auth_errors

    if total_errors == 0:
        return "OK", "0 errors"
    elif log_auth_errors > 0 and "Bearer token" in str(log_auth_errors):
        return "CRITICAL", f"auth token invalid ({log_auth_errors} in log)"
    elif total_errors > 20:
        return "WARN", f"{daytona_count} in results, {log_auth_errors} in log"
    elif total_errors > 5:
        return "WARN", f"{total_errors} total ({daytona_count} result + {log_auth_errors} log)"
    else:
        return "OK", f"{total_errors} total"


def check_stuck_trials(run_tag, jobs_dir, max_age_hours=2):
    """Check for trial dirs without result.json. Returns (status, detail)."""
    if not run_tag:
        return "UNKNOWN", "no run_tag"
    job_dir = jobs_dir / run_tag
    if not job_dir.exists():
        return "UNKNOWN", "job dir not found"

    stuck_count = 0
    oldest_stuck_hours = 0
    total_trial_dirs = 0
    now = time.time()

    try:
        for entry in os.scandir(job_dir):
            if not entry.is_dir() or "__" not in entry.name:
                continue
            total_trial_dirs += 1
            result_path = Path(entry.path) / "result.json"
            if not result_path.exists():
                # Check age via directory mtime (trial start ~ dir creation)
                try:
                    age_hours = (now - entry.stat().st_mtime) / 3600
                except Exception:
                    age_hours = 0
                if age_hours > max_age_hours:
                    stuck_count += 1
                    oldest_stuck_hours = max(oldest_stuck_hours, age_hours)
    except Exception as e:
        return "UNKNOWN", f"scan error: {e}"

    if stuck_count == 0:
        return "OK", f"0 stuck (of {total_trial_dirs} dirs)"
    elif stuck_count > 5:
        return "WARN", f"{stuck_count} trials stuck >{max_age_hours}h (oldest {oldest_stuck_hours:.1f}h)"
    else:
        return "WARN", f"{stuck_count} trial(s) stuck >{max_age_hours}h"


# ---------------------------------------------------------------------------
# Main health check
# ---------------------------------------------------------------------------

def run_health_check(jobs_dir, benchmark_filter=None, json_output=False):
    all_jobs = get_running_jobs()
    if not all_jobs:
        if json_output:
            print(json.dumps({"jobs": [], "summary": "no jobs"}))
        else:
            print("No jobs in queue.")
        return

    running = [(jid, name, state, start) for jid, name, state, start in all_jobs if state == "RUNNING"]
    pending_count = sum(1 for _, _, state, _ in all_jobs if state == "PENDING")

    if not running:
        if json_output:
            print(json.dumps({"jobs": [], "pending": pending_count, "summary": "no running jobs"}))
        else:
            print(f"No running jobs. ({pending_count} pending)")
        return

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []
    counts = {"healthy": 0, "warning": 0, "critical": 0, "unknown": 0}

    for jid, name, state, start_time in running:
        model, bench, run_tag, num_shards, error_threshold = parse_eval_log(jid, name)

        m_short = model.split("/")[-1] if model and "/" in model else (model or "?")
        b_short = bench.split("/")[-1] if bench and "/" in bench else (bench or "?")

        if benchmark_filter:
            bf = benchmark_filter.lower()
            if bf not in (bench or "").lower() and bf not in b_short.lower():
                continue

        elapsed_str, elapsed_secs = format_elapsed(start_time)

        # Get progress data (handle shards)
        progress_info = None
        if run_tag:
            if num_shards > 1:
                # Aggregate shards — simplified for health check
                progress_info = get_progress_single(run_tag, jobs_dir)
                if not progress_info:
                    # Try first shard
                    progress_info = get_progress_single(f"{run_tag}_shard0", jobs_dir)
            else:
                progress_info = get_progress_single(run_tag, jobs_dir)

        # Run all checks
        vllm_status, vllm_detail = check_vllm_health(jid)
        prog_status, prog_detail = check_progress_rate(progress_info, elapsed_secs)
        err_status, err_detail = check_error_distribution(progress_info, error_threshold)
        dayt_status, dayt_detail = check_daytona_errors(jid, name, progress_info)
        stuck_status, stuck_detail = check_stuck_trials(run_tag, jobs_dir)

        checks = {
            "vLLM": (vllm_status, vllm_detail),
            "Progress": (prog_status, prog_detail),
            "Errors": (err_status, err_detail),
            "Daytona": (dayt_status, dayt_detail),
            "Trials": (stuck_status, stuck_detail),
        }

        # Overall severity
        statuses = [s for s, _ in checks.values()]
        if "CRITICAL" in statuses:
            overall = "critical"
        elif "WARN" in statuses:
            overall = "warning"
        elif all(s == "UNKNOWN" for s in statuses):
            overall = "unknown"
        else:
            overall = "healthy"
        counts[overall] += 1

        completed = progress_info["completed"] if progress_info else None
        total = progress_info["total"] if progress_info else None
        pct = f"{completed}/{total} ({completed*100//total}%)" if completed is not None and total else "?"

        results.append({
            "jid": jid,
            "model": m_short,
            "bench": b_short,
            "run_tag": run_tag,
            "progress": pct,
            "elapsed": elapsed_str,
            "overall": overall,
            "checks": checks,
        })

    if json_output:
        # Machine-readable
        out = {
            "timestamp": now_str,
            "jobs": [],
            "pending": pending_count,
            "summary": counts,
        }
        for r in results:
            out["jobs"].append({
                "jid": r["jid"],
                "model": r["model"],
                "bench": r["bench"],
                "run_tag": r["run_tag"],
                "progress": r["progress"],
                "elapsed": r["elapsed"],
                "overall": r["overall"],
                "checks": {k: {"status": s, "detail": d} for k, (s, d) in r["checks"].items()},
            })
        print(json.dumps(out, indent=2))
        return

    # Pretty print
    sep = "=" * 74
    print(f"\nHEALTH CHECK — {now_str}")
    print(sep)

    for r in results:
        status_icon = {"healthy": " ", "warning": "!", "critical": "X", "unknown": "?"}[r["overall"]]
        print(f"[{status_icon}] JOB {r['jid']} | {r['model']} | {r['bench']} | {r['progress']} | {r['elapsed']}")

        for check_name, (status, detail) in r["checks"].items():
            if status == "CRITICAL":
                marker = "XX"
            elif status == "WARN":
                marker = "!!"
            elif status == "OK":
                marker = "ok"
            else:
                marker = "??"
            print(f"    {check_name:10s} [{marker}] {detail}")

        print()

    print(sep)
    total = sum(counts.values())
    parts = []
    if counts["healthy"]:
        parts.append(f"{counts['healthy']} healthy")
    if counts["warning"]:
        parts.append(f"{counts['warning']} warning")
    if counts["critical"]:
        parts.append(f"{counts['critical']} critical")
    if counts["unknown"]:
        parts.append(f"{counts['unknown']} unknown")
    summary = " | ".join(parts)
    pending_str = f" | {pending_count} pending" if pending_count else ""
    print(f"SUMMARY: {total} jobs | {summary}{pending_str}\n")


def main():
    global LOGS_DIR

    parser = argparse.ArgumentParser(description="Health check for running eval jobs.")
    parser.add_argument("--benchmark", "-b", type=str, default=None,
                        help="Filter by benchmark substring")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR,
                        help=f"Jobs directory (default: {DEFAULT_JOBS_DIR})")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR,
                        help=f"Logs directory (default: {LOGS_DIR})")
    args = parser.parse_args()

    LOGS_DIR = args.logs_dir

    run_health_check(args.jobs_dir, benchmark_filter=args.benchmark, json_output=args.json)


if __name__ == "__main__":
    main()
