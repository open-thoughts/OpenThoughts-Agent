#!/usr/bin/env python3
"""
Aggregate runtime statistics from evaluation trace result.json files.

The script walks through the provided evaltraces directory, looks for
`trace_jobs/<timestamp>/<task>/result.json` files, extracts the
`agent_execution.started_at` and `agent_execution.finished_at`
timestamps, reports summary statistics on the runtimes, and tracks the
prevalence of companion `exception.txt` files anywhere beneath each task
directory. It also reports quantiles for each major stage
(`overall`, `environment_setup`, `agent_setup`, `agent_execution`, `verifier`) and
exception frequency segmented by agent execution quantiles. The paths of
tasks that fall in the highest agent_execution quantile and contain
exceptions are written to `agent_execution_top_quantile_exceptions.txt`
by default.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_ROOT = Path("/Users/benjaminfeuer/Documents/evaltraces")
STAGES = ("environment_setup", "agent_setup", "agent_execution", "verifier")
STAGE_OUTPUT_ORDER = ("overall",) + STAGES


@dataclass
class TaskMetrics:
    path: Path
    durations: Dict[str, float]
    exception_present: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute runtime quantiles and mean from trace job result files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root directory containing eval trace data (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--quantiles",
        "-q",
        type=float,
        nargs="*",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Quantile values to report (between 0 and 1).",
    )
    parser.add_argument(
        "--top-quantile-output",
        type=Path,
        default=Path("agent_execution_top_quantile_exceptions.txt"),
        help="File path where the script writes exception task paths for the top agent_execution quantile.",
    )
    return parser.parse_args()


def find_result_files(root: Path) -> Iterable[Path]:
    """Yield task-level result.json files located under trace_jobs trees."""
    for path in root.glob("**/trace_jobs/**/result.json"):
        parts = path.parts
        try:
            idx = parts.index("trace_jobs")
        except ValueError:
            continue
        # Require at least two directory levels beneath trace_jobs (e.g. timestamp/task/result.json)
        if len(parts) - idx - 1 >= 3:
            yield path


def parse_timestamp(value: str, path: Path, field: str) -> datetime | None:
    if not value:
        print(f"[warn] Missing value for {field} in {path}", file=sys.stderr)
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        print(f"[warn] Failed to parse {field}='{value}' in {path}: {exc}", file=sys.stderr)
        return None


def compute_stage_duration(payload: Dict[str, object], stage: str, path: Path) -> float | None:
    window = payload.get(stage)
    if not isinstance(window, dict):
        return None

    started = parse_timestamp(window.get("started_at"), path, f"{stage}.started_at")
    finished = parse_timestamp(window.get("finished_at"), path, f"{stage}.finished_at")
    if not started or not finished:
        return None

    delta = (finished - started).total_seconds()
    if delta < 0:
        print(f"[warn] Negative runtime ({delta}s) in {path} for {stage}, skipping.", file=sys.stderr)
        return None

    return delta


def compute_overall_duration(payload: Dict[str, object], path: Path) -> float | None:
    started = parse_timestamp(payload.get("started_at"), path, "started_at")
    finished = parse_timestamp(payload.get("finished_at"), path, "finished_at")
    if not started or not finished:
        return None

    delta = (finished - started).total_seconds()
    if delta < 0:
        print(f"[warn] Negative runtime ({delta}s) in {path} for overall duration, skipping.", file=sys.stderr)
        return None

    return delta


def extract_task_durations(path: Path) -> Dict[str, float] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] Failed to read {path}: {exc}", file=sys.stderr)
        return None

    durations: Dict[str, float] = {}
    for stage in STAGES:
        duration = compute_stage_duration(payload, stage, path)
        if duration is not None:
            durations[stage] = duration

    overall = compute_overall_duration(payload, path)
    if overall is not None:
        durations["overall"] = overall

    return durations


def compute_quantiles(values: List[float], points: Iterable[float]) -> Dict[float, float]:
    if not values:
        return {}

    sorted_values = sorted(values)
    n = len(sorted_values)
    result: Dict[float, float] = {}

    for q in points:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Quantile must be between 0 and 1, got {q}")
        if n == 1 or q in (0.0, 1.0):
            index = 0 if q == 0.0 else n - 1
            result[q] = sorted_values[index]
            continue

        position = q * (n - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        result[q] = sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    return result


def normalise_quantile_edges(points: Iterable[float]) -> List[float]:
    cleaned = sorted({p for p in points if 0.0 <= p <= 1.0})
    if not cleaned:
        cleaned = [0.0, 1.0]
    if not math.isclose(cleaned[0], 0.0, abs_tol=1e-9):
        cleaned = [0.0] + cleaned
    if not math.isclose(cleaned[-1], 1.0, abs_tol=1e-9):
        cleaned = cleaned + [1.0]
    return cleaned


def exception_frequency_by_quantile(
    samples: List[Tuple[float, bool]], points: Iterable[float]
) -> List[Dict[str, float]]:
    if not samples:
        return []

    edges = normalise_quantile_edges(points)
    bins: List[Dict[str, float]] = [
        {"lower": edges[i], "upper": edges[i + 1], "count": 0, "exceptions": 0}
        for i in range(len(edges) - 1)
    ]

    sorted_samples = sorted(samples, key=lambda item: item[0])
    n = len(sorted_samples)
    bin_idx = 0
    current_upper = bins[bin_idx]["upper"]

    for idx, (_, has_exception) in enumerate(sorted_samples, start=1):
        fraction = idx / n
        while fraction > current_upper and bin_idx < len(bins) - 1:
            bin_idx += 1
            current_upper = bins[bin_idx]["upper"]

        bins[bin_idx]["count"] += 1
        if has_exception:
            bins[bin_idx]["exceptions"] += 1

    return bins


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()

    if not root.exists():
        print(f"[error] Root directory {root} does not exist.", file=sys.stderr)
        return 2

    metrics: List[TaskMetrics] = []
    total_results = 0
    exception_dirs = 0
    for result_file in find_result_files(root):
        total_results += 1
        job_dir = result_file.parent
        has_exception = any(job_dir.glob("**/exception.txt"))
        if has_exception:
            exception_dirs += 1
        durations = extract_task_durations(result_file)
        if durations is None:
            continue
        metrics.append(
            TaskMetrics(path=result_file, durations=durations, exception_present=has_exception)
        )

    if not metrics:
        print("[error] No valid runtimes found.", file=sys.stderr)
        return 1

    stage_values: Dict[str, List[float]] = {stage: [] for stage in STAGE_OUTPUT_ORDER}
    agent_samples: List[Tuple[float, bool, TaskMetrics]] = []
    for record in metrics:
        for stage, duration in record.durations.items():
            stage_values[stage].append(duration)
        agent_duration = record.durations.get("agent_execution")
        if agent_duration is not None:
            agent_samples.append((agent_duration, record.exception_present, record))

    print(
        f"Evaluated {len(metrics)} task results (from {total_results} result files) under {root}"
    )
    if total_results:
        exception_rate = exception_dirs / total_results
        print(
            f"Exceptions: {exception_dirs}/{total_results} directories "
            f"({exception_rate:.2%})"
        )

    top_exception_paths: List[Path] = []
    if agent_samples:
        bins = exception_frequency_by_quantile(
            [(duration, has_exception) for duration, has_exception, _ in agent_samples],
            args.quantiles,
        )
        print("Exception frequency by agent_execution quantile:")
        for bucket in bins:
            count = bucket["count"]
            exceptions = bucket["exceptions"]
            rate = exceptions / count if count else 0.0
            print(
                f"  {bucket['lower']:.2%}-{bucket['upper']:.2%}: "
                f"{exceptions}/{count} tasks ({rate:.2%})"
            )
        edges = normalise_quantile_edges(args.quantiles)
        sorted_samples = sorted(agent_samples, key=lambda item: item[0])
        n = len(sorted_samples)
        if n:
            last_bin_index = len(edges) - 2
            bin_idx = 0
            current_upper = edges[bin_idx + 1]
            for idx, sample in enumerate(sorted_samples, start=1):
                fraction = idx / n
                while fraction > current_upper and bin_idx < last_bin_index:
                    bin_idx += 1
                    current_upper = edges[bin_idx + 1]
                duration, has_exception, record = sample
                if bin_idx == last_bin_index and has_exception:
                    top_exception_paths.append(record.path)
    else:
        print("Exception frequency by agent_execution quantile: no agent_execution data")

    quantile_points = sorted({q for q in args.quantiles if 0.0 <= q <= 1.0})
    if not quantile_points:
        quantile_points = [0.0, 1.0]

    print("Timings by stage:")
    for stage in STAGE_OUTPUT_ORDER:
        values = stage_values[stage]
        if not values:
            print(f"  {stage}: no data")
            continue

        count = len(values)
        mean_seconds = sum(values) / count
        quantiles = compute_quantiles(values, quantile_points)
        print(f"  {stage} (n={count}, mean={mean_seconds:.2f} seconds)")
        for q in quantile_points:
            if q in quantiles:
                print(f"    {q:.2%}: {quantiles[q]:.2f} seconds")

    if top_exception_paths and args.top_quantile_output:
        output_path = args.top_quantile_output.expanduser()
        try:
            with output_path.open("w", encoding="utf-8") as handle:
                for path in top_exception_paths:
                    handle.write(f"{path}\n")
            print(
                f"Wrote {len(top_exception_paths)} exception paths from top agent_execution quantile to {output_path}"
            )
        except OSError as exc:
            print(
                f"[error] Failed to write top quantile exception paths to {output_path}: {exc}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
