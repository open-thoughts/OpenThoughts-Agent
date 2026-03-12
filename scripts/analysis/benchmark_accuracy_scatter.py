#!/usr/bin/env python3
"""
Fetch sandbox_jobs from Supabase and render:
  1. Scatterplot of accuracy over time for a single benchmark.
  2. Radar plot comparing selected models across ALL benchmarks.

Usage:
    # Scatter only
    python scripts/analysis/benchmark_accuracy_scatter.py \
        --benchmark-id b94dfab2-c438-4c32-ba29-23e46d566763

    # Radar comparing models across all benchmarks
    python scripts/analysis/benchmark_accuracy_scatter.py \
        --benchmark-id b94dfab2-c438-4c32-ba29-23e46d566763 \
        --radar-models "Qwen3-32B" "GPT-4o" "Claude-3.5-Sonnet"

Requires: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY in environment
(or in the file pointed to by DC_AGENT_SECRET_ENV).
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

# Load secrets from DC_AGENT_SECRET_ENV if available
SECRET_ENV_PATH = os.environ.get("DC_AGENT_SECRET_ENV")
if SECRET_ENV_PATH and os.path.isfile(SECRET_ENV_PATH):
    load_dotenv(SECRET_ENV_PATH)


FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")


def create_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)


def get_benchmark_name(client: Client, benchmark_id: str) -> str:
    resp = client.table("benchmarks").select("name").eq("id", benchmark_id).execute()
    if not resp.data:
        raise ValueError(f"Benchmark {benchmark_id} not found")
    return resp.data[0]["name"]


def get_all_benchmarks(client: Client) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({benchmark_id: canonical_name}, {canonical_name: canonical_name}).

    Handles the ``duplicate_of`` column: if benchmark B has
    ``duplicate_of = A``, then B's canonical name is A's name and any
    scores recorded against B count toward A.

    Returns:
        id_to_canonical: {benchmark_id: canonical_benchmark_name}
        canonical_names: {canonical_name: canonical_name} (deduplicated set as dict)
    """
    resp = client.table("benchmarks").select("id, name, duplicate_of").execute()
    raw = {b["id"]: b for b in resp.data}

    # Build id -> canonical name, resolving duplicate_of chains
    id_to_canonical: dict[str, str] = {}
    for b in resp.data:
        target = b
        # Follow duplicate_of (one hop is enough; guard against loops)
        seen: set[str] = {b["id"]}
        while target.get("duplicate_of") and target["duplicate_of"] in raw:
            if target["duplicate_of"] in seen:
                break
            seen.add(target["duplicate_of"])
            target = raw[target["duplicate_of"]]
        id_to_canonical[b["id"]] = target["name"]

    canonical_names = {n: n for n in set(id_to_canonical.values())}
    return id_to_canonical, canonical_names


def extract_accuracy_from_metrics(metrics_data) -> Optional[float]:
    """Extract accuracy from the metrics JSONB field.

    Handles two shapes:
      - list of {"name": "accuracy", "value": 0.15} dicts
      - dict with "accuracy", "acc", or "score" key
    """
    if not metrics_data:
        return None
    try:
        if isinstance(metrics_data, str):
            metrics = json.loads(metrics_data)
        else:
            metrics = metrics_data

        if isinstance(metrics, list):
            for item in metrics:
                if isinstance(item, dict):
                    name = item.get("name", "").lower()
                    if "accuracy" in name and "stderr" not in name:
                        return float(item["value"])

        elif isinstance(metrics, dict):
            for key in ("accuracy", "acc", "score"):
                if key in metrics:
                    return float(metrics[key])
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        pass
    return None


def extract_accuracy_from_stats(stats_data) -> Optional[float]:
    """Extract accuracy from the stats JSONB field.

    Stats shape: {"evals": {"<key>": {"metrics": [{"mean": 0.15}], ...}}}
    Uses the first evals entry's first metric mean value.
    """
    if not stats_data:
        return None
    try:
        if isinstance(stats_data, str):
            stats = json.loads(stats_data)
        else:
            stats = stats_data

        evals = stats.get("evals", {})
        if not evals:
            return None

        # Take the first evals entry
        first_eval = next(iter(evals.values()))
        metrics_list = first_eval.get("metrics", [])
        if metrics_list and isinstance(metrics_list[0], dict):
            mean_val = metrics_list[0].get("mean")
            if mean_val is not None:
                return float(mean_val)
    except (json.JSONDecodeError, TypeError, KeyError, ValueError, StopIteration):
        pass
    return None


def extract_accuracy(job: dict) -> Optional[float]:
    """Extract accuracy from a job row, trying metrics first then stats."""
    acc = extract_accuracy_from_metrics(job.get("metrics"))
    if acc is not None:
        return acc
    return extract_accuracy_from_stats(job.get("stats"))


def fetch_jobs(client: Client, benchmark_id: str) -> list[dict]:
    """Fetch all sandbox_jobs for the given benchmark_id, paginating."""
    jobs = []
    batch_size = 1000
    offset = 0
    while True:
        resp = (
            client.table("sandbox_jobs")
            .select("id, created_at, job_name, metrics, stats, model_id")
            .eq("benchmark_id", benchmark_id)
            .order("created_at", desc=False)
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        if not resp.data:
            break
        jobs.extend(resp.data)
        if len(resp.data) < batch_size:
            break
        offset += batch_size
    return jobs


def fetch_all_jobs(client: Client) -> list[dict]:
    """Fetch all sandbox_jobs across all benchmarks, paginating."""
    jobs = []
    batch_size = 1000
    offset = 0
    while True:
        resp = (
            client.table("sandbox_jobs")
            .select("id, created_at, job_name, metrics, stats, model_id, benchmark_id")
            .order("created_at", desc=False)
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        if not resp.data:
            break
        jobs.extend(resp.data)
        if len(resp.data) < batch_size:
            break
        offset += batch_size
    return jobs


def get_model_names(client: Client, model_ids: set[str]) -> dict[str, str]:
    """Map model UUIDs to names."""
    if not model_ids:
        return {}
    names = {}
    batch = list(model_ids)
    chunk_size = 100
    for i in range(0, len(batch), chunk_size):
        chunk = batch[i : i + chunk_size]
        resp = client.table("models").select("id, name").in_("id", chunk).execute()
        for row in resp.data:
            names[row["id"]] = row["name"]
    return names


def resolve_model_ids_by_name(
    client: Client, model_names: list[str]
) -> dict[str, str]:
    """Find model IDs by exact name match.

    Returns {model_id: model_name} for all matches.
    """
    matched: dict[str, str] = {}
    for name in model_names:
        resp = client.table("models").select("id, name").eq("name", name).execute()
        if resp.data:
            for row in resp.data:
                matched[row["id"]] = row["name"]
        else:
            print(f"  WARNING: No model found with exact name '{name}'")
    return matched


def plot_scatter(
    points: list[dict],
    model_names: dict[str, str],
    benchmark_name: str,
    output_path: str,
    label_points: bool = False,
):
    """Render and save the accuracy-over-time scatterplot."""
    unique_models = sorted(set(p["model_id"] for p in points))
    cmap = plt.cm.get_cmap("tab20", max(len(unique_models), 1))
    model_color = {mid: cmap(i) for i, mid in enumerate(unique_models)}

    fig, ax = plt.subplots(figsize=(14, 7))

    for mid in unique_models:
        model_pts = [p for p in points if p["model_id"] == mid]
        dates = [p["date"] for p in model_pts]
        accs = [p["accuracy"] for p in model_pts]
        label = model_names.get(mid, mid[:12])
        ax.scatter(dates, accs, color=model_color[mid], label=label, s=40, alpha=0.8, edgecolors="k", linewidths=0.3)

    if label_points:
        for p in points:
            label = model_names.get(p["model_id"], p["job_name"][:20])
            ax.annotate(label, (p["date"], p["accuracy"]), fontsize=5, alpha=0.6, rotation=30)

    ax.set_xlabel("Date Added")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy over Time — {benchmark_name}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, alpha=0.3)

    if len(unique_models) > 10:
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        fig.subplots_adjust(right=0.75)
    else:
        ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved scatter plot to {output_path}")
    plt.close(fig)


def plot_radar(
    model_scores: dict[str, dict[str, float]],
    benchmark_names: list[str],
    model_display_names: dict[str, str],
    output_path: str,
):
    """Render a radar plot comparing models across benchmarks.

    Args:
        model_scores: {model_id: {benchmark_name: best_accuracy}}
        benchmark_names: ordered list of benchmark names to include
        model_display_names: {model_id: display_name}
        output_path: where to save the figure
    """
    N = len(benchmark_names)
    angles = np.linspace(0, 2 * math.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    # Compute dynamic y-axis range from data
    all_values = [
        scores[bn]
        for scores in model_scores.values()
        for bn in benchmark_names
        if bn in scores
    ]
    max_val = max(all_values) if all_values else 1.0
    y_max = min(round(max_val + 0.1, 1), 1.0)
    y_min = max(round(min(all_values) - 0.1, 1), 0.0) if all_values else 0.0
    # Nice tick spacing
    tick_step = 0.1 if (y_max - y_min) <= 0.5 else 0.2
    y_ticks = np.arange(
        math.ceil(y_min / tick_step) * tick_step,
        y_max + tick_step / 2,
        tick_step,
    )
    y_ticks = np.round(y_ticks, 2)

    # Wrap long benchmark names so labels don't overlap the chart
    wrapped_names = []
    for bn in benchmark_names:
        if len(bn) > 14:
            mid = len(bn) // 2
            # Find nearest space or hyphen to split
            best = mid
            for offset in range(mid):
                for pos in (mid + offset, mid - offset):
                    if 0 < pos < len(bn) and bn[pos] in (" ", "-", "_"):
                        best = pos
                        break
                else:
                    continue
                break
            wrapped_names.append(bn[:best + 1].rstrip() + "\n" + bn[best + 1:].lstrip())
        else:
            wrapped_names.append(bn)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
              "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
    cmap_colors = colors[:max(len(model_scores), 1)]

    for i, (mid, scores) in enumerate(sorted(model_scores.items(), key=lambda x: model_display_names.get(x[0], ""))):
        values = [scores[bn] for bn in benchmark_names]
        values += values[:1]  # close
        label = model_display_names.get(mid, mid[:12])
        color = cmap_colors[i % len(cmap_colors)]
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=color,
                markersize=5, markeredgecolor="white", markeredgewidth=0.5)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), wrapped_names, fontsize=9, fontweight="medium")
    # Push labels outward so they don't overlap the data area
    ax.set_rlabel_position(30)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:.1f}" if v < 1 else "1.0" for v in y_ticks], fontsize=7, color="#555555")
    ax.set_title("Model Comparison Across Benchmarks", y=1.12, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(-0.15, -0.08),
              ncol=min(len(model_scores), 3), frameon=True, fancybox=True,
              shadow=False, edgecolor="#cccccc")
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_linewidth(1.2)
    ax.spines["polar"].set_color("#333333")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved radar plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Scatterplot of accuracy over time, and radar plot across benchmarks."
    )
    parser.add_argument("--benchmark-id", required=True, help="UUID of the benchmark for the scatter plot.")
    parser.add_argument("--output", default=None, help="Output path for scatter plot.")
    parser.add_argument("--label-points", action="store_true", help="Label scatter points with model name.")
    parser.add_argument(
        "--radar-models",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Exact model names for radar plot (e.g. 'Qwen/Qwen3-32B' 'allenai/SERA-32B'). "
             "Only benchmarks with results for ALL listed models are included.",
    )
    parser.add_argument("--radar-output", default=None, help="Output path for radar plot.")
    args = parser.parse_args()

    client = create_supabase_client()

    # ---- Scatter plot ----
    benchmark_name = get_benchmark_name(client, args.benchmark_id)
    print(f"Benchmark: {benchmark_name} ({args.benchmark_id})")

    jobs = fetch_jobs(client, args.benchmark_id)
    print(f"Fetched {len(jobs)} jobs")

    model_ids = set()
    points = []
    for job in jobs:
        acc = extract_accuracy(job)
        if acc is None:
            continue
        dt = datetime.fromisoformat(job["created_at"].replace("Z", "+00:00"))
        points.append({
            "date": dt,
            "accuracy": acc,
            "job_name": job.get("job_name", ""),
            "model_id": job.get("model_id", ""),
        })
        if job.get("model_id"):
            model_ids.add(job["model_id"])

    if not points:
        print("No jobs with valid accuracy found for scatter plot.")
    else:
        print(f"Plotting {len(points)} points with valid accuracy")
        model_names = get_model_names(client, model_ids)

        scatter_path = args.output
        if not scatter_path:
            os.makedirs(FIGURES_DIR, exist_ok=True)
            safe_name = benchmark_name.replace(" ", "_").replace("/", "_")
            scatter_path = os.path.join(FIGURES_DIR, f"{safe_name}_accuracy.png")

        plot_scatter(points, model_names, benchmark_name, scatter_path, args.label_points)

    # ---- Radar plot ----
    if args.radar_models:
        print(f"\n=== Radar Plot ===")
        print(f"Matching models for: {args.radar_models}")

        matched_models = resolve_model_ids_by_name(client, args.radar_models)
        if not matched_models:
            print("No models matched the given names.")
            sys.exit(1)

        for mid, mname in sorted(matched_models.items(), key=lambda x: x[1]):
            print(f"  Matched: {mname} ({mid[:8]}...)")

        id_to_canonical, canonical_names = get_all_benchmarks(client)
        n_raw = len(id_to_canonical)
        n_canonical = len(canonical_names)
        if n_raw != n_canonical:
            print(f"Resolved {n_raw} benchmark rows to {n_canonical} canonical benchmarks (duplicates unified)")
        print(f"Fetching jobs across {n_canonical} canonical benchmarks...")

        all_jobs = fetch_all_jobs(client)
        print(f"Fetched {len(all_jobs)} total jobs")

        # Build {model_id: {canonical_benchmark_name: best_accuracy}}
        # for matched models only, unifying duplicate benchmarks
        matched_ids = set(matched_models.keys())
        model_bench_scores: dict[str, dict[str, float]] = defaultdict(dict)

        for job in all_jobs:
            mid = job.get("model_id")
            bid = job.get("benchmark_id")
            if mid not in matched_ids or bid not in id_to_canonical:
                continue
            acc = extract_accuracy(job)
            if acc is None:
                continue
            bname = id_to_canonical[bid]
            # Keep the highest score in case of multiple runs / duplicates
            if bname not in model_bench_scores[mid] or acc > model_bench_scores[mid][bname]:
                model_bench_scores[mid][bname] = acc

        # Only include benchmarks where ALL matched models have results
        all_bench_names = set()
        for scores in model_bench_scores.values():
            all_bench_names.update(scores.keys())

        common_benchmarks = sorted(
            bn for bn in all_bench_names
            if all(bn in model_bench_scores.get(mid, {}) for mid in matched_ids)
        )

        if not common_benchmarks:
            print("No benchmarks have results for all specified models.")
            sys.exit(1)

        # Report which benchmarks were dropped
        dropped = sorted(all_bench_names - set(common_benchmarks))
        if dropped:
            print(f"Dropped benchmarks (missing results for some models): {', '.join(dropped)}")
        print(f"Radar axes ({len(common_benchmarks)}): {', '.join(common_benchmarks)}")

        radar_path = args.radar_output
        if not radar_path:
            os.makedirs(FIGURES_DIR, exist_ok=True)
            radar_path = os.path.join(FIGURES_DIR, "model_radar.png")

        plot_radar(model_bench_scores, common_benchmarks, matched_models, radar_path)


if __name__ == "__main__":
    main()
