#!/usr/bin/env python3
"""Recompute Harbor metrics for jobs in the Supabase database.

Samples N jobs from sandbox_jobs (where hf_traces_link is not null),
reconstructs trial-level data from the stats JSONB, applies the new
mean_drop_ei / accuracy_drop_ei metrics alongside the original metrics,
and writes a comparison table.

Usage:
    source hpc/dotenv/tacc.env   # or export SUPABASE_URL + SUPABASE_ANON_KEY

    # Sample 10 random jobs, write comparison table
    python scripts/harbor/recompute_metrics_from_db.py --n 10

    # All jobs with traces
    python scripts/harbor/recompute_metrics_from_db.py --all

    # Custom output directory
    python scripts/harbor/recompute_metrics_from_db.py --n 20 \
        --output-dir /Users/benjaminfeuer/Documents/notes/results_recompute

Required environment variables:
    SUPABASE_URL
    SUPABASE_ANON_KEY
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Resolve imports
# ---------------------------------------------------------------------------

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_db_path = _repo_root / "database" / "unified_db"
if str(_db_path) not in sys.path:
    sys.path.insert(0, str(_db_path))

# ---------------------------------------------------------------------------
# Default drop exceptions (must match harbor/metrics/drop_ei.py)
# ---------------------------------------------------------------------------

DEFAULT_DROP_EXCEPTIONS: frozenset[str] = frozenset(
    [
        "AgentEnvironmentTimeoutError",
        "DaytonaError",
        "DaytonaRateLimitError",
        "DaytonaNotFoundError",
        "EnvironmentStartTimeoutError",
        "SandboxBuildFailedError",
        "PodmanHPCTimeoutError",
        "PodmanHPCCommandError",
        "ApptainerTimeoutError",
        "ApptainerCommandError",
    ]
)


# ---------------------------------------------------------------------------
# Reconstruct trial data from stats JSONB
# ---------------------------------------------------------------------------

def _parse_task_name_from_trial(trial_name: str) -> str:
    """Extract task name from trial_name format: task_name__short_uuid."""
    # Trial names are "{task_name[:32]}__{ShortUUID(7)}"
    match = re.match(r"^(.+)__\w+$", trial_name)
    if match:
        return match.group(1)
    return trial_name


def reconstruct_trials_from_stats(
    stats: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct per-evals-key trial data from stats JSONB.

    Returns: {evals_key: [{"trial_name": str, "task_name": str,
                            "reward": float|None, "exception_type": str|None}, ...]}
    """
    result: dict[str, list[dict[str, Any]]] = {}

    evals = stats.get("evals", {})
    for evals_key, eval_data in evals.items():
        trials: dict[str, dict[str, Any]] = {}  # trial_name -> info

        # Extract from reward_stats: {reward_key: {value: [trial_names]}}
        reward_stats = eval_data.get("reward_stats", {})
        for reward_key, value_map in reward_stats.items():
            if not isinstance(value_map, dict):
                continue
            for reward_value_str, trial_names in value_map.items():
                if not isinstance(trial_names, list):
                    continue
                try:
                    reward_value = float(reward_value_str)
                except (ValueError, TypeError):
                    reward_value = 0.0
                for tn in trial_names:
                    if tn not in trials:
                        trials[tn] = {
                            "trial_name": tn,
                            "task_name": _parse_task_name_from_trial(tn),
                            "reward": None,
                            "exception_type": None,
                        }
                    trials[tn]["reward"] = reward_value

        # Extract from exception_stats: {exception_type: [trial_names]}
        exception_stats = eval_data.get("exception_stats", {})
        for exc_type, trial_names in exception_stats.items():
            if not isinstance(trial_names, list):
                continue
            for tn in trial_names:
                if tn not in trials:
                    trials[tn] = {
                        "trial_name": tn,
                        "task_name": _parse_task_name_from_trial(tn),
                        "reward": None,
                        "exception_type": None,
                    }
                trials[tn]["exception_type"] = exc_type

        result[evals_key] = list(trials.values())

    return result


# ---------------------------------------------------------------------------
# Metric computation (standalone, no Harbor imports needed)
# ---------------------------------------------------------------------------

def compute_mean(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Flat mean over all trials (None/error → 0)."""
    values = []
    for t in trials:
        r = t.get("reward")
        values.append(r if r is not None else 0.0)
    if not values:
        return {"mean_reward": 0.0, "mean_reward_count": 0}
    return {
        "mean_reward": sum(values) / len(values),
        "mean_reward_count": len(values),
    }


def compute_drop_ei(
    trials: list[dict[str, Any]],
    n_attempts: int,
    drop_exceptions: frozenset[str] = DEFAULT_DROP_EXCEPTIONS,
    floor_rewards: bool = False,
) -> dict[str, Any]:
    """Compute mean (or accuracy) after dropping errored trials + incomplete tasks."""
    prefix = "accuracy_drop_ei" if floor_rewards else "mean_drop_ei"

    # Step 1: filter errored trials
    valid = []
    n_trials_dropped = 0
    for t in trials:
        if t.get("exception_type") and t["exception_type"] in drop_exceptions:
            n_trials_dropped += 1
            continue
        valid.append(t)

    # Step 2: group by task
    task_trials: dict[str, list] = defaultdict(list)
    for t in valid:
        task_trials[t["task_name"]].append(t)

    # Step 3: drop incomplete tasks
    n_tasks_dropped = 0
    complete: dict[str, list] = {}
    for task_name, tlist in task_trials.items():
        if len(tlist) >= n_attempts:
            complete[task_name] = tlist
        else:
            n_tasks_dropped += 1

    # Step 4: compute
    values = []
    for tlist in complete.values():
        for t in tlist:
            r = t.get("reward")
            v = r if r is not None else 0.0
            if floor_rewards:
                v = math.floor(v)
            values.append(v)

    mean_val = sum(values) / len(values) if values else 0.0
    return {
        f"{prefix}_reward": mean_val,
        f"{prefix}_reward_count": len(values),
        f"{prefix}_tasks_dropped": n_tasks_dropped,
        f"{prefix}_trials_dropped": n_trials_dropped,
    }


# ---------------------------------------------------------------------------
# Supabase query
# ---------------------------------------------------------------------------

def _fetch_batch(client, offset: int, batch_size: int) -> list[dict]:
    """Fetch a single batch from sandbox_jobs."""
    return (
        client.table("sandbox_jobs")
        .select(
            "id, job_name, n_trials, n_rep_eval, metrics, stats, hf_traces_link, "
            "agent_id, model_id, benchmark_id, created_at"
        )
        .not_.is_("hf_traces_link", "null")
        .eq("job_status", "Finished")
        .order("created_at", desc=True)
        .range(offset, offset + batch_size - 1)
        .execute()
    ).data or []


def fetch_jobs(n: int | None, all_jobs: bool = False, batch_size: int = 200) -> list[dict]:
    """Fetch jobs from sandbox_jobs where hf_traces_link is not null."""
    from database.unified_db.config import get_default_client

    client = get_default_client()

    if not all_jobs and n is not None:
        # For sampling, fetch enough in batches then sample
        target = max(n * 3, 100)
        jobs: list[dict] = []
        offset = 0
        while len(jobs) < target:
            batch = _fetch_batch(client, offset, batch_size)
            if not batch:
                break
            jobs.extend(batch)
            offset += batch_size
            if len(batch) < batch_size:
                break
        if len(jobs) > n:
            jobs = random.sample(jobs, n)
        return jobs

    # Fetch all, paginated
    jobs = []
    offset = 0
    while True:
        batch = _fetch_batch(client, offset, batch_size)
        if not batch:
            break
        jobs.extend(batch)
        print(f"  Fetched {len(jobs)} jobs so far...")
        offset += batch_size
        if len(batch) < batch_size:
            break

    return jobs


def fetch_name_lookups(
    client, agent_ids: set, model_ids: set, benchmark_ids: set
) -> tuple[dict, dict, dict, dict]:
    """Batch-fetch agent, model, benchmark names and benchmark groups.

    Returns:
        (agent_names, model_names, benchmark_names, benchmark_group)
        where benchmark_group maps benchmark_id -> canonical group name
        (resolving duplicate_of chains).
    """
    agent_names: dict[str, str] = {}
    model_names: dict[str, str] = {}
    benchmark_names: dict[str, str] = {}
    benchmark_group: dict[str, str] = {}

    if agent_ids:
        resp = (
            client.table("agents")
            .select("id, name")
            .in_("id", list(agent_ids))
            .execute()
        )
        for row in resp.data or []:
            agent_names[row["id"]] = row["name"]

    if model_ids:
        resp = (
            client.table("models")
            .select("id, name")
            .in_("id", list(model_ids))
            .execute()
        )
        for row in resp.data or []:
            model_names[row["id"]] = row["name"]

    if benchmark_ids:
        # Fetch all benchmarks (not just the ones we need) to resolve
        # duplicate_of chains where the canonical parent may not be in our set
        resp = (
            client.table("benchmarks")
            .select("id, name, duplicate_of")
            .execute()
        )
        all_benchmarks = {row["id"]: row for row in (resp.data or [])}

        for row in all_benchmarks.values():
            benchmark_names[row["id"]] = row["name"]

        # Resolve duplicate_of to canonical name
        def _resolve_group(bid: str) -> str:
            visited: set[str] = set()
            current = bid
            while current in all_benchmarks:
                if current in visited:
                    break  # cycle guard
                visited.add(current)
                parent = all_benchmarks[current].get("duplicate_of")
                if parent and parent in all_benchmarks:
                    current = parent
                else:
                    break
            return all_benchmarks[current]["name"] if current in all_benchmarks else bid

        for bid in benchmark_ids:
            benchmark_group[bid] = _resolve_group(bid)

    return agent_names, model_names, benchmark_names, benchmark_group


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _extract_original_metric(metrics: list[dict] | None, name: str) -> float | None:
    """Extract a named metric value from the DB metrics array."""
    if not metrics:
        return None
    for m in metrics:
        if m.get("name") == name:
            v = m.get("value")
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
    return None


def run(
    n: int | None,
    all_jobs: bool,
    output_dir: Path,
    verbose: bool = False,
) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        print("Warning: tqdm not installed, using basic progress", file=sys.stderr)
        tqdm = None  # type: ignore[assignment]

    # Fetch jobs
    print("Fetching jobs from Supabase...")
    jobs = fetch_jobs(n, all_jobs=all_jobs)
    if not jobs:
        print("No jobs found with hf_traces_link.")
        return
    print(f"Processing {len(jobs)} jobs")

    # Batch-fetch names
    from database.unified_db.config import get_default_client

    client = get_default_client()
    agent_ids = {j["agent_id"] for j in jobs if j.get("agent_id")}
    model_ids = {j["model_id"] for j in jobs if j.get("model_id")}
    benchmark_ids = {j["benchmark_id"] for j in jobs if j.get("benchmark_id")}
    agent_names, model_names, benchmark_names, benchmark_group = fetch_name_lookups(
        client, agent_ids, model_ids, benchmark_ids
    )

    # Process each job
    rows_out: list[dict[str, Any]] = []
    n_reviewed = 0
    iterator = tqdm(jobs, desc="Recomputing metrics") if tqdm else jobs

    for job in iterator:
        job_name = job.get("job_name", "unknown")
        stats = job.get("stats")
        if not stats or not isinstance(stats, dict):
            if verbose:
                print(f"  Skipping {job_name}: no stats")
            continue

        n_attempts_db = job.get("n_rep_eval") or 1
        original_metrics = job.get("metrics") or []

        # Reconstruct trials from stats
        evals_trials = reconstruct_trials_from_stats(stats)
        if not evals_trials:
            if verbose:
                print(f"  Skipping {job_name}: no trial data in stats")
            continue

        # Compute new metrics for each evals_key
        for evals_key, trials in evals_trials.items():
            if not trials:
                continue

            # Infer n_attempts from data: use the most common per-task
            # trial count as a cross-check against the DB value.
            task_counts: dict[str, int] = defaultdict(int)
            for t in trials:
                task_counts[t["task_name"]] += 1
            if task_counts:
                from collections import Counter
                count_freq = Counter(task_counts.values())
                inferred_n_attempts = count_freq.most_common(1)[0][0]
            else:
                inferred_n_attempts = 1

            # Use the larger of DB value and inferred value — the DB
            # sometimes records 1 when the job actually ran multiple attempts.
            n_attempts = max(n_attempts_db, inferred_n_attempts)
            if verbose and n_attempts != n_attempts_db:
                print(
                    f"  {job_name}: n_attempts corrected from {n_attempts_db} "
                    f"(DB) to {n_attempts} (inferred from trial data)"
                )

            mean_result = compute_mean(trials)
            mean_drop_ei = compute_drop_ei(
                trials, n_attempts, floor_rewards=False
            )
            accuracy_drop_ei = compute_drop_ei(
                trials, n_attempts, floor_rewards=True
            )

            # Extract original values
            orig_accuracy = _extract_original_metric(original_metrics, "accuracy")
            orig_stderr = _extract_original_metric(original_metrics, "accuracy_stderr")

            n_reviewed += 1

            # Only save rows where at least 1 trial or task was dropped
            total_dropped = (
                mean_drop_ei["mean_drop_ei_trials_dropped"]
                + mean_drop_ei["mean_drop_ei_tasks_dropped"]
            )
            if total_dropped == 0:
                continue

            row = {
                "job_name": job_name,
                "evals_key": evals_key,
                "agent": agent_names.get(job.get("agent_id"), ""),
                "model": model_names.get(job.get("model_id"), ""),
                "benchmark": benchmark_names.get(job.get("benchmark_id"), ""),
                "benchmark_group": benchmark_group.get(job.get("benchmark_id"), benchmark_names.get(job.get("benchmark_id"), "")),
                "n_trials": job.get("n_trials", 0),
                "n_attempts_db": n_attempts_db,
                "n_attempts_inferred": inferred_n_attempts,
                "n_attempts_used": n_attempts,
                "hf_traces_link": job.get("hf_traces_link", ""),
                "created_at": job.get("created_at", ""),
                # Original metrics from DB
                "orig_accuracy": orig_accuracy,
                "orig_accuracy_stderr": orig_stderr,
                # Recomputed: flat mean (should match orig_accuracy)
                "recomputed_mean_reward": mean_result["mean_reward"],
                "recomputed_mean_reward_count": mean_result["mean_reward_count"],
                # New: mean_drop_ei
                "mean_drop_ei_reward": mean_drop_ei["mean_drop_ei_reward"],
                "mean_drop_ei_reward_count": mean_drop_ei["mean_drop_ei_reward_count"],
                "mean_drop_ei_tasks_dropped": mean_drop_ei["mean_drop_ei_tasks_dropped"],
                "mean_drop_ei_trials_dropped": mean_drop_ei["mean_drop_ei_trials_dropped"],
                # New: accuracy_drop_ei
                "accuracy_drop_ei_reward": accuracy_drop_ei["accuracy_drop_ei_reward"],
                "accuracy_drop_ei_reward_count": accuracy_drop_ei["accuracy_drop_ei_reward_count"],
                "accuracy_drop_ei_tasks_dropped": accuracy_drop_ei["accuracy_drop_ei_tasks_dropped"],
                "accuracy_drop_ei_trials_dropped": accuracy_drop_ei["accuracy_drop_ei_trials_dropped"],
            }
            rows_out.append(row)

    print(f"\nReviewed {n_reviewed} evals across {len(jobs)} jobs")
    print(f"Found {len(rows_out)} with dropped trials/tasks")

    if not rows_out:
        print("No jobs had infrastructure errors or incomplete tasks — nothing to write.")
        return

    # Group rows by benchmark_group
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows_out:
        grouped[r["benchmark_group"]].append(r)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Full CSV (all rows)
    csv_path = run_dir / "all_benchmarks.csv"
    fieldnames = list(rows_out[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nWrote CSV: {csv_path}")

    # Per-benchmark-group outputs
    try:
        from scipy.stats import spearmanr
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        has_plot_deps = True
    except ImportError as exc:
        print(f"Skipping plots (missing dependency: {exc})")
        has_plot_deps = False

    for group_name, group_rows in sorted(grouped.items()):
        safe_name = re.sub(r"[^\w\-.]", "_", group_name)

        # Per-group CSV
        group_csv = run_dir / f"{safe_name}.csv"
        with open(group_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(group_rows)

        # Per-group Markdown
        group_md = run_dir / f"{safe_name}.md"
        with open(group_md, "w") as f:
            f.write(f"# {group_name} — Metrics Recomputation\n\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            bm_variants = sorted({r["benchmark"] for r in group_rows})
            if len(bm_variants) > 1:
                f.write(f"Benchmark variants: {', '.join(bm_variants)}\n")
            f.write(f"Jobs: {len(group_rows)}\n\n")

            f.write(
                "| Job | Model | N | "
                "Orig Acc | Mean | Mean Drop EI | Acc Drop EI | "
                "Trials Drop | Tasks Drop |\n"
            )
            f.write(
                "|-----|-------|---|"
                "---------|------|--------------|-------------|"
                "------------|------------|\n"
            )
            for r in group_rows:
                orig = f"{r['orig_accuracy']:.4f}" if r["orig_accuracy"] is not None else "N/A"
                f.write(
                    f"| {r['job_name'][:40]} "
                    f"| {r['model'][:25]} "
                    f"| {r['recomputed_mean_reward_count']} "
                    f"| {orig} "
                    f"| {r['recomputed_mean_reward']:.4f} "
                    f"| {r['mean_drop_ei_reward']:.4f} "
                    f"| {r['accuracy_drop_ei_reward']:.4f} "
                    f"| {r['mean_drop_ei_trials_dropped']} "
                    f"| {r['mean_drop_ei_tasks_dropped']} "
                    f"|\n"
                )

            # Delta analysis
            f.write("\n## Delta Analysis\n\n")
            deltas = []
            for r in group_rows:
                if r["orig_accuracy"] is not None:
                    delta = r["mean_drop_ei_reward"] - r["orig_accuracy"]
                    deltas.append((r["job_name"], r["model"], delta,
                                   r["mean_drop_ei_trials_dropped"],
                                   r["mean_drop_ei_tasks_dropped"]))
            if deltas:
                f.write("| Job | Model | Delta | Trials Drop | Tasks Drop |\n")
                f.write("|-----|-------|-------|-------------|------------|\n")
                for name, model, delta, td, tkd in sorted(deltas, key=lambda x: -abs(x[2])):
                    sign = "+" if delta >= 0 else ""
                    f.write(f"| {name[:40]} | {model[:25]} | {sign}{delta:.4f} | {td} | {tkd} |\n")

        # Per-group scatterplot
        if has_plot_deps and len(group_rows) >= 3:
            means = [r["recomputed_mean_reward"] for r in group_rows]
            means_drop = [r["mean_drop_ei_reward"] for r in group_rows]

            rho, pval = spearmanr(means, means_drop)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(means, means_drop, alpha=0.7, edgecolors="k", linewidths=0.5)

            x = np.array(means)
            y = np.array(means_drop)
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), "r--", linewidth=1.5,
                    label=f"fit: y={coeffs[0]:.3f}x+{coeffs[1]:.4f}")

            hi = max(x.max(), y.max()) * 1.05
            ax.plot([0, hi], [0, hi], "k:", alpha=0.4, label="y=x")

            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Mean Reward (original)")
            ax.set_ylabel("Mean Drop EI Reward")
            ax.set_title(f"{group_name}\nSpearman rho={rho:.3f}, p={pval:.3g}, n={len(group_rows)}")
            ax.legend()
            ax.set_aspect("equal", adjustable="datalim")
            fig.tight_layout()

            plot_path = run_dir / f"{safe_name}.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)

        print(f"  [{group_name}] {len(group_rows)} jobs — wrote {safe_name}.{{csv,md,png}}")

    # Global correlation
    if has_plot_deps and len(rows_out) >= 3:
        means = [r["recomputed_mean_reward"] for r in rows_out]
        means_drop = [r["mean_drop_ei_reward"] for r in rows_out]
        rho, pval = spearmanr(means, means_drop)
        print(f"\nGlobal Spearman (Mean vs Mean Drop EI): rho={rho:.4f}, p={pval:.4g}")

        fig, ax = plt.subplots(figsize=(8, 6))
        # Color by benchmark group
        group_names_sorted = sorted(grouped.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(group_names_sorted), 1)))
        for i, gname in enumerate(group_names_sorted):
            grows = grouped[gname]
            gx = [r["recomputed_mean_reward"] for r in grows]
            gy = [r["mean_drop_ei_reward"] for r in grows]
            ax.scatter(gx, gy, alpha=0.7, edgecolors="k", linewidths=0.3,
                       color=colors[i % len(colors)], label=gname[:30], s=40)

        x = np.array(means)
        y = np.array(means_drop)
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "r--", linewidth=1.5)
        hi = max(x.max(), y.max()) * 1.05
        ax.plot([0, hi], [0, hi], "k:", alpha=0.4)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Mean Reward (original)")
        ax.set_ylabel("Mean Drop EI Reward")
        ax.set_title(f"All Benchmarks — Spearman rho={rho:.3f}, p={pval:.3g}")
        ax.legend(fontsize=7, loc="upper left", ncol=2)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()

        plot_path = run_dir / "all_benchmarks.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Wrote global scatterplot: {plot_path}")

    print(f"\nDone! {len(rows_out)} evals across {len(grouped)} benchmark groups → {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute Harbor metrics from Supabase stats and compare with originals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of jobs to sample (default: 10)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_jobs",
        help="Process all jobs (ignore --n)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/benjaminfeuer/Documents/notes/results_recompute",
        help="Output directory for comparison tables",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    run(
        n=args.n,
        all_jobs=args.all_jobs,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
