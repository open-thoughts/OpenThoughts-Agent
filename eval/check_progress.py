#!/usr/bin/env python3
"""Check progress of all running eval jobs on Jupiter.

Usage:
    python check_progress.py              # grouped text output (default)
    python check_progress.py --live       # rich live dashboard
    python check_progress.py --live -i 3  # live with 3s refresh
    python check_progress.py --sort elapsed
    python check_progress.py --jobs-dir /path/to/other/jobs  # override jobs dir
"""

import argparse
import subprocess
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_DIR / "eval" / "logs"
EXTRA_LOG_DIRS = [
    REPO_DIR / "eval" / "local" / "logs",        # legacy v6 default
    REPO_DIR / "eval" / "MBZ" / "logs",           # legacy MBZ v4
    REPO_DIR / "eval" / "local" / "mbz" / "logs", # legacy MBZ v6
    REPO_DIR / "eval" / "local" / "m2" / "logs",  # legacy M2
]
DEFAULT_JOBS_DIR = REPO_DIR / "jobs"


# ---------------------------------------------------------------------------
# Data collection helpers (unchanged logic, refactored for reuse)
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


def parse_eval_log(jid, job_name):
    """Try multiple log naming patterns. SLURM %x is captured at submit time,
    so renamed jobs (eval_dp -> eval_dp_v2) need fallback to original name."""
    # Search primary log dir + all extra log dirs
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
    model = bench = run_tag = None
    num_shards = 0
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
                    elif "total shards)" in line:
                        try:
                            num_shards = int(line.split("total shards")[0].split(",")[-1].strip())
                        except (ValueError, IndexError):
                            pass
            break
    return model, bench, run_tag, num_shards


VALID_ERROR_TYPES = {
    "AgentTimeoutError", "ContextLengthExceededError",
    "SummarizationTimeout", "SummarizationTimeoutError", "BadRequestError",
}


def get_progress_single(run_tag, jobs_dir):
    """Get progress from a single (non-DP) job dir."""
    if not run_tag:
        return None, None, None, None, None, None
    job_dir = jobs_dir / run_tag
    rf = job_dir / "result.json"
    if not rf.exists():
        return None, None, None, None, None, None
    try:
        with open(rf) as f:
            d = json.load(f)
        completed = d.get("stats", {}).get("n_trials", None)
        total = d.get("n_total_trials", None)
        finished = d.get("finished_at") is not None
        invalid_trials = set()
        evals = d.get("stats", {}).get("evals", {})
        # Extract accuracy from metrics
        accuracy = None
        for eval_data in evals.values():
            for error_type, trial_names in eval_data.get("exception_stats", {}).items():
                if error_type not in VALID_ERROR_TYPES and isinstance(trial_names, list):
                    invalid_trials.update(trial_names)
            metrics = eval_data.get("metrics", [])
            if metrics and isinstance(metrics, list):
                mr = metrics[0].get("mean_reward")
                if mr is not None:
                    accuracy = mr
        # Use reward_stats to count completed trials (avoids expensive dir scan)
        n_on_disk = None
        for eval_data in evals.values():
            rs = eval_data.get("reward_stats", {}).get("reward", {})
            if rs:
                n_on_disk = sum(len(v) for v in rs.values() if isinstance(v, list))
                break
        if n_on_disk is None:
            # Fallback: count from stats
            n_on_disk = completed
        return completed, total, len(invalid_trials), finished, n_on_disk, accuracy
    except Exception:
        return None, None, None, None, None, None


def get_progress(run_tag, num_shards, jobs_dir):
    """Get progress, aggregating across shards for DP jobs."""
    if not run_tag:
        return None, None, None, None, None, None
    if num_shards > 1:
        total_completed = 0
        total_total = 0
        total_errors = 0
        total_on_disk = 0
        all_finished = True
        found_any = False
        all_accuracies = []
        for shard_idx in range(num_shards):
            shard_tag = f"{run_tag}_shard{shard_idx}"
            c, t, e, fin, od, acc = get_progress_single(shard_tag, jobs_dir)
            if c is not None:
                found_any = True
                total_completed += c
                total_total += (t or 0)
                total_errors += (e or 0)
                total_on_disk += (od or 0)
                if acc is not None:
                    all_accuracies.append(acc)
                if not fin:
                    all_finished = False
        if found_any:
            avg_acc = (sum(all_accuracies) / len(all_accuracies)) if all_accuracies else None
            return total_completed, total_total, total_errors, all_finished, total_on_disk, avg_acc
        return None, None, None, None, None, None
    else:
        return get_progress_single(run_tag, jobs_dir)


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
# Unified data collection
# ---------------------------------------------------------------------------

def collect_job_data(jobs_dir):
    """Collect all job data into structured dicts.

    Returns (running_data: list[dict], pending_jobs: list[tuple]).
    """
    all_jobs = get_running_jobs()
    if not all_jobs:
        return [], []

    running_raw = [(jid, name, state, start) for jid, name, state, start in all_jobs if state == "RUNNING"]
    pending = [(jid, name, state, start) for jid, name, state, start in all_jobs if state == "PENDING"]

    running_data = []
    for jid, name, state, start_time in running_raw:
        model, bench, run_tag, num_shards = parse_eval_log(jid, name)
        completed, total, invalid_errors, finished, n_on_disk, accuracy = get_progress(run_tag, num_shards, jobs_dir)

        if finished:
            status = "done"
        elif completed is not None and total and completed >= total:
            status = "retry"
        else:
            status = "active"

        elapsed, elapsed_secs = format_elapsed(start_time)
        m_short = model.split("/")[-1] if model and "/" in model else (model or "?")
        b_short = bench.split("/")[-1] if bench and "/" in bench else (bench or "?")

        is_resume = name.startswith("res_")
        tags = []
        if is_resume:
            tags.append("RES")
        if num_shards > 1:
            tags.append(f"{num_shards}x DP")

        # Progress percentage (based on disk for ground truth)
        if n_on_disk is not None and total and total > 0:
            progress_pct = n_on_disk / total
        elif completed is not None and total and total > 0:
            progress_pct = completed / total
        else:
            progress_pct = 0.0

        running_data.append({
            "jid": jid,
            "job_name": name,
            "model": m_short,
            "model_full": model or "?",
            "bench": b_short,
            "bench_full": bench or "?",
            "run_tag": run_tag,
            "num_shards": num_shards,
            "completed": completed,
            "total": total,
            "n_on_disk": n_on_disk,
            "n_invalid_errors": invalid_errors,
            "accuracy": accuracy,
            "finished": finished,
            "elapsed": elapsed,
            "elapsed_secs": elapsed_secs,
            "status": status,
            "tags": tags,
            "progress_pct": progress_pct,
            "is_resume": is_resume,
        })

    return running_data, pending


def group_by_benchmark(jobs, sort_key="progress"):
    """Group jobs by benchmark, sort groups alphabetically, sort jobs within groups.

    sort_key: 'progress' (default), 'elapsed', 'model', 'errors'
    """
    groups = defaultdict(list)
    for job in jobs:
        groups[job["bench"]].append(job)

    # Sort within each group
    for bench in groups:
        if sort_key == "elapsed":
            groups[bench].sort(key=lambda j: j["elapsed_secs"], reverse=True)
        elif sort_key == "model":
            groups[bench].sort(key=lambda j: j["model"].lower())
        elif sort_key == "errors":
            groups[bench].sort(key=lambda j: (j["n_invalid_errors"] or 0), reverse=True)
        else:  # progress (default)
            groups[bench].sort(key=lambda j: j["progress_pct"], reverse=True)

    # Return as OrderedDict-like sorted by benchmark name
    return dict(sorted(groups.items()))


def get_pending_reasons():
    """Get pending job reasons from squeue."""
    result = subprocess.run(
        ["squeue", "-u", os.environ["USER"], "--format=%.10i %.8T %R", "--noheader"],
        capture_output=True, text=True,
    )
    reasons = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.split(None, 2)
        if len(parts) >= 3 and parts[1].strip() == "PENDING":
            reasons[parts[0].strip()] = parts[2].strip()
    return reasons


# ---------------------------------------------------------------------------
# Default text mode (enhanced with grouping)
# ---------------------------------------------------------------------------

def print_default(running_data, pending, sort_key="progress"):
    if not running_data and not pending:
        print("No jobs in queue.")
        return

    if running_data:
        groups = group_by_benchmark(running_data, sort_key)

        # Count statuses
        status_counts = defaultdict(int)
        for j in running_data:
            status_counts[j["status"]] += 1

        print(f"\nRUNNING ({len(running_data)}):")

        header = f"  {'JID':>8s}  {'Progress':>10s}  {'On Disk':>8s}  {'Acc':>6s}  {'Errors':>6s}  {'Status':>6s}  {'Elapsed':>8s}  Model"
        sep = "  " + "-" * (len(header) - 2)

        for bench_name, jobs in groups.items():
            # Per-benchmark summary
            n_active = sum(1 for j in jobs if j["status"] == "active")
            n_retry = sum(1 for j in jobs if j["status"] == "retry")
            n_done = sum(1 for j in jobs if j["status"] == "done")
            parts = []
            if n_active:
                parts.append(f"{n_active} active")
            if n_retry:
                parts.append(f"{n_retry} retry")
            if n_done:
                parts.append(f"{n_done} done")
            status_str = ", ".join(parts) if parts else "0 jobs"

            print(f"\n  === {bench_name} ({len(jobs)} jobs: {status_str}) ===")
            print(header)
            print(sep)

            for j in jobs:
                progress = f"{j['completed']}/{j['total']}" if j["completed"] is not None else "-"
                on_disk = f"{j['n_on_disk']}/{j['total']}" if j["n_on_disk"] is not None and j["total"] else "-"
                acc = f"{j['accuracy']:.1%}" if j["accuracy"] is not None else "-"
                errors = str(j["n_invalid_errors"]) if j["n_invalid_errors"] is not None else "-"
                tag_str = f" [{', '.join(j['tags'])}]" if j["tags"] else ""
                print(f"  {j['jid']:>8s}  {progress:>10s}  {on_disk:>8s}  {acc:>6s}  {errors:>6s}  {j['status']:>6s}  {j['elapsed']:>8s}  {j['model']}{tag_str}")

    if pending:
        reasons = get_pending_reasons()
        print(f"\n  PENDING ({len(pending)}):")
        print(f"  {'JID':>8s}  Reason")
        print("  " + "-" * 38)
        for jid, name, state, start in pending:
            reason = reasons.get(jid, "?")
            print(f"  {jid:>8s}  {reason}")

    # Summary line
    status_counts = defaultdict(int)
    for j in running_data:
        status_counts[j["status"]] += 1
    parts = []
    for s in ["active", "retry", "done"]:
        if status_counts[s]:
            parts.append(f"{status_counts[s]} {s}")
    status_detail = f" ({', '.join(parts)})" if parts else ""
    print(f"\n  Total: {len(running_data)} running{status_detail}, {len(pending)} pending\n")


# ---------------------------------------------------------------------------
# Rich Live dashboard
# ---------------------------------------------------------------------------

def run_live_dashboard(jobs_dir, interval, sort_key="progress", compact=False,
                       benchmark_filter=None, page_mode=False):
    from rich.console import Console, Group
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar

    console = Console()
    page_idx = [0]  # mutable for closure

    STATUS_STYLES = {
        "active": "green",
        "active_res": "purple",
        "retry": "yellow",
        "done": "dim",
    }

    def _job_style(j):
        """Get style key for a job: distinguish resume (cyan) from fresh (green) for active jobs."""
        if j["status"] == "active" and j["is_resume"]:
            return "active_res"
        return j["status"]

    def make_progress_bar(pct, style_key):
        """Create a colored progress bar."""
        color = STATUS_STYLES.get(style_key, "white")
        bar = ProgressBar(total=100, completed=int(pct * 100), width=10,
                          complete_style=color, finished_style=color)
        return bar

    def render():
        running_data, pending = collect_job_data(jobs_dir)
        now = datetime.now().strftime("%H:%M:%S")

        renderables = []

        # Header
        n_fresh = sum(1 for j in running_data if j["status"] == "active" and not j["is_resume"])
        n_resume = sum(1 for j in running_data if j["status"] == "active" and j["is_resume"])
        n_retry = sum(1 for j in running_data if j["status"] == "retry")
        n_done = sum(1 for j in running_data if j["status"] == "done")
        header_parts = [
            f"[bold]EVAL DASHBOARD[/bold]",
            f"[green]{n_fresh} fresh[/green]" if n_fresh else None,
            f"[purple]{n_resume} resume[/purple]" if n_resume else None,
            f"[yellow]{n_retry} retry[/yellow]" if n_retry else None,
            f"[dim]{n_done} done[/dim]" if n_done else None,
            f"{len(pending)} pending" if pending else None,
            f"[dim]Updated {now}[/dim]",
        ]
        header_text = "  |  ".join(p for p in header_parts if p)
        renderables.append(Text.from_markup(header_text))
        renderables.append(Text(""))

        if not running_data and not pending:
            renderables.append(Text("No jobs in queue.", style="dim"))
            return Group(*renderables)

        # Group by benchmark
        groups = group_by_benchmark(running_data, sort_key)

        # Apply benchmark filter
        if benchmark_filter:
            groups = {k: v for k, v in groups.items()
                      if benchmark_filter.lower() in k.lower()}

        # Page mode: show one benchmark at a time, rotating each refresh
        if page_mode and groups:
            bench_names = list(groups.keys())
            idx = page_idx[0] % len(bench_names)
            selected = bench_names[idx]
            groups = {selected: groups[selected]}
            page_idx[0] += 1
            renderables.append(Text.from_markup(
                f"[dim]Page {idx + 1}/{len(bench_names)}[/dim]"))
            renderables.append(Text(""))

        for bench_name, jobs in groups.items():
            # Per-benchmark stats
            b_fresh = sum(1 for j in jobs if j["status"] == "active" and not j["is_resume"])
            b_resume = sum(1 for j in jobs if j["status"] == "active" and j["is_resume"])
            b_retry = sum(1 for j in jobs if j["status"] == "retry")
            b_done = sum(1 for j in jobs if j["status"] == "done")
            b_errors = sum(j["n_invalid_errors"] or 0 for j in jobs)
            avg_pct = sum(j["progress_pct"] for j in jobs) / len(jobs) if jobs else 0

            status_parts = []
            if b_fresh:
                status_parts.append(f"[green]{b_fresh} fresh[/green]")
            if b_resume:
                status_parts.append(f"[purple]{b_resume} resume[/purple]")
            if b_retry:
                status_parts.append(f"[yellow]{b_retry} retry[/yellow]")
            if b_done:
                status_parts.append(f"[dim]{b_done} done[/dim]")
            error_str = f"  [red]{b_errors} err[/red]" if b_errors > 10 else (f"  {b_errors} err" if b_errors else "")

            title = (f"[bold cyan]{bench_name}[/bold cyan]  "
                     f"{len(jobs)} jobs | {' | '.join(status_parts)}{error_str} | "
                     f"avg {avg_pct:.0%}")

            # In compact mode, only show active jobs in the table
            display_jobs = [j for j in jobs if j["status"] == "active"] if compact else jobs
            n_hidden = len(jobs) - len(display_jobs)

            table = Table(
                show_header=True, header_style="bold", box=None,
                padding=(0, 1), expand=True,
            )
            table.add_column("JID", style="dim", width=7, justify="right")
            table.add_column("Progress", width=8, justify="right")
            if not compact:
                table.add_column("Bar", width=12)
            table.add_column("Disk", width=8, justify="right")
            table.add_column("Acc", width=6, justify="right")
            table.add_column("Err", width=4, justify="right")
            table.add_column("Status", width=6, justify="right")
            table.add_column("Elapsed", width=6, justify="right")
            table.add_column("Model", no_wrap=False, ratio=1)

            for j in display_jobs:
                skey = _job_style(j)
                style = STATUS_STYLES.get(skey, "")
                progress = f"{j['completed']}/{j['total']}" if j["completed"] is not None else "-"
                on_disk = f"{j['n_on_disk']}/{j['total']}" if j["n_on_disk"] is not None and j["total"] else "-"
                acc = f"{j['accuracy']:.1%}" if j["accuracy"] is not None else "-"
                errors = str(j["n_invalid_errors"]) if j["n_invalid_errors"] is not None else "-"
                error_style = "red bold" if (j["n_invalid_errors"] or 0) > 10 else ""
                tag_str = f" [{', '.join(j['tags'])}]" if j["tags"] else ""
                model_display = j["model_full"]

                row = [
                    j["jid"],
                    Text(progress, style=style),
                ]
                if not compact:
                    row.append(make_progress_bar(j["progress_pct"], skey))
                row.extend([
                    Text(on_disk, style=style),
                    Text(acc, style="cyan" if j["accuracy"] is not None else style),
                    Text(errors, style=error_style or style),
                    Text("resume" if skey == "active_res" else j["status"], style=style),
                    Text(j["elapsed"], style=style),
                    Text(f"{model_display}{tag_str}", style=style),
                ])
                table.add_row(*row)

            renderables.append(Text.from_markup(title))
            if display_jobs:
                renderables.append(table)
            if n_hidden:
                renderables.append(Text.from_markup(
                    f"  [dim]... {n_hidden} done/retry job(s) hidden (use without --compact to show)[/dim]"))
            renderables.append(Text(""))

        # Pending section
        if pending:
            reasons = get_pending_reasons()
            pending_lines = []
            for jid, name, state, start in pending:
                reason = reasons.get(jid, "?")
                pending_lines.append(f"  {jid}  {reason}")
            pending_text = "\n".join(pending_lines[:10])
            if len(pending) > 10:
                pending_text += f"\n  ... and {len(pending) - 10} more"
            renderables.append(Panel(
                pending_text,
                title=f"[dim]PENDING ({len(pending)})[/dim]",
                border_style="dim",
                expand=False,
            ))

        # Footer
        tags = []
        if compact:
            tags.append("compact")
        if page_mode:
            tags.append("paging")
        if benchmark_filter:
            tags.append(f"filter: {benchmark_filter}")
        extra = f"  |  {', '.join(tags)}" if tags else ""
        renderables.append(Text.from_markup(
            f"[dim]Ctrl+C to exit  |  Refreshing every {interval}s  |  "
            f"Sort: {sort_key}{extra}[/dim]"
        ))

        return Group(*renderables)

    try:
        with Live(render(), console=console, refresh_per_second=1,
                  vertical_overflow="ellipsis") as live:
            while True:
                time.sleep(interval)
                live.update(render())
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check progress of running eval jobs on Jupiter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s                        # grouped text output
  %(prog)s --live                 # rich live dashboard
  %(prog)s --live -i 3            # 3s refresh interval
  %(prog)s --sort elapsed         # sort by elapsed time
  %(prog)s --sort errors          # sort by error count
  %(prog)s --live --compact       # live, hide done/retry + no bar (fits more)
  %(prog)s --live -b tb2          # live, only terminal_bench_2
  %(prog)s --live --page          # live, rotate one benchmark per refresh
""",
    )
    parser.add_argument(
        "--live", "-w", action="store_true",
        help="Launch Rich live dashboard (auto-refreshing)",
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=5,
        help="Refresh interval in seconds for --live mode (default: 5, min: 3)",
    )
    parser.add_argument(
        "--sort", "-s", choices=["progress", "elapsed", "model", "errors"],
        default="progress",
        help="Sort order within benchmark groups (default: progress)",
    )
    parser.add_argument(
        "--compact", "-c", action="store_true",
        help="In --live mode, hide done/retry jobs and progress bar (fits more rows)",
    )
    parser.add_argument(
        "--benchmark", "-b", type=str, default=None,
        help="Filter to benchmarks matching this substring (e.g. 'tb2', 'dev_set')",
    )
    parser.add_argument(
        "--page", "-p", action="store_true",
        help="In --live mode, show one benchmark per page, rotating each refresh",
    )
    parser.add_argument(
        "--jobs-dir", type=Path, default=DEFAULT_JOBS_DIR,
        help=f"Path to eval jobs directory (default: {DEFAULT_JOBS_DIR})",
    )
    parser.add_argument(
        "--logs-dir", type=Path, default=LOGS_DIR,
        help=f"Path to eval logs directory (default: {LOGS_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global LOGS_DIR
    LOGS_DIR = args.logs_dir

    if args.live:
        interval = max(3, args.interval)
        run_live_dashboard(args.jobs_dir, interval, sort_key=args.sort,
                           compact=args.compact, benchmark_filter=args.benchmark,
                           page_mode=args.page)
    else:
        running_data, pending = collect_job_data(args.jobs_dir)
        if args.benchmark:
            running_data = [j for j in running_data
                           if args.benchmark.lower() in j["bench"].lower()]
        print_default(running_data, pending, sort_key=args.sort)


if __name__ == "__main__":
    main()
