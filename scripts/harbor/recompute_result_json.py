#!/usr/bin/env python3
"""Recompute result.json for an existing Harbor job directory.

Walks trial subdirectories, loads each trial's result.json, rebuilds
JobStats, applies metrics (including new drop_ei aggregators), and
writes an updated result.json.

Usage:
    python scripts/harbor/recompute_result_json.py /path/to/job/dir

    # Override metrics from a YAML config
    python scripts/harbor/recompute_result_json.py /path/to/job/dir \
        --metrics-config hpc/harbor_yaml/eval/eval_ctx32k.yaml

    # Dry run — show what would be written without modifying anything
    python scripts/harbor/recompute_result_json.py /path/to/job/dir --dry-run

    # Accept a run root (auto-resolves to the job dir inside trace_jobs/)
    python scripts/harbor/recompute_result_json.py /path/to/trace_runs/my-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4

# ---------------------------------------------------------------------------
# Resolve Harbor + repo imports
# ---------------------------------------------------------------------------

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from harbor.metrics.base import BaseMetric
    from harbor.metrics.context import MetricContext
    from harbor.metrics.factory import MetricFactory
    from harbor.metrics.mean import Mean
    from harbor.models.job.config import JobConfig
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.metric.config import MetricConfig
    from harbor.models.trial.paths import TrialPaths
    from harbor.models.trial.result import TrialResult
except ImportError as exc:
    print(f"Error: Could not import Harbor modules: {exc}", file=sys.stderr)
    print("Make sure Harbor is installed (pip install -e /path/to/harbor)", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Job directory resolution (same logic as manual_db_eval_push.py)
# ---------------------------------------------------------------------------

_TRIAL_DIR_PATTERN = re.compile(r"^.+__\w+$")


def _looks_like_job_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    for child in d.iterdir():
        if child.is_dir() and _TRIAL_DIR_PATTERN.match(child.name):
            return True
    return False


def resolve_job_dir(user_path: Path) -> Path:
    """Walk down from user-provided path to find the actual Harbor job directory."""
    if _looks_like_job_dir(user_path):
        return user_path

    trace_jobs = user_path / "trace_jobs"
    if trace_jobs.is_dir():
        for child in sorted(trace_jobs.iterdir()):
            if _looks_like_job_dir(child):
                return child

    for child in sorted(user_path.iterdir()):
        if _looks_like_job_dir(child):
            return child

    return user_path


# ---------------------------------------------------------------------------
# Safe JSON serialization (mirrors Harbor's _safe_model_dump_json)
# ---------------------------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def safe_model_dump_json(model, **dump_kwargs) -> str:
    data = model.model_dump(**dump_kwargs)
    return json.dumps(data, indent=4, default=_json_default)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_trial_results(job_dir: Path, verbose: bool = False) -> list[TrialResult]:
    """Load all trial result.json files from a job directory."""
    results: list[TrialResult] = []
    skipped = 0

    for trial_dir in sorted(job_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        if not _TRIAL_DIR_PATTERN.match(trial_dir.name):
            continue

        trial_paths = TrialPaths(trial_dir)
        if not trial_paths.result_path.exists():
            skipped += 1
            if verbose:
                print(f"  Skipping {trial_dir.name} (no result.json)")
            continue

        try:
            trial_result = TrialResult.model_validate_json(
                trial_paths.result_path.read_text()
            )
            results.append(trial_result)
        except Exception as exc:
            skipped += 1
            if verbose:
                print(f"  Skipping {trial_dir.name} (parse error: {exc})")

    print(f"Loaded {len(results)} trial results ({skipped} skipped)")
    return results


def build_metrics(
    job_dir: Path,
    metrics_config_path: Path | None = None,
) -> tuple[list[BaseMetric], int]:
    """Build metric instances from job config or override YAML.

    Returns (metrics_list, n_attempts).
    """
    n_attempts = 1
    metrics: list[BaseMetric] = []

    # Try loading from override config first
    if metrics_config_path is not None:
        import yaml

        raw = yaml.safe_load(metrics_config_path.read_text())
        n_attempts = raw.get("n_attempts", 1)
        for m in raw.get("metrics", []):
            mc = MetricConfig.model_validate(m)
            metrics.append(MetricFactory.create_metric(mc.type, **mc.kwargs))
        if metrics:
            return metrics, n_attempts

    # Try loading from the job's own config.json
    config_path = job_dir / "config.json"
    if config_path.exists():
        try:
            job_config = JobConfig.model_validate_json(config_path.read_text())
            n_attempts = job_config.n_attempts
            for mc in job_config.metrics:
                metrics.append(MetricFactory.create_metric(mc.type, **mc.kwargs))
        except Exception as exc:
            print(f"Warning: Could not parse config.json: {exc}")

    # Default: just Mean
    if not metrics:
        metrics = [Mean()]

    return metrics, n_attempts


def recompute(
    job_dir: Path,
    metrics_config_path: Path | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> JobResult:
    """Recompute result.json for a job directory."""
    print(f"Job directory: {job_dir}")

    # Load trial results
    trial_results = load_trial_results(job_dir, verbose=verbose)
    if not trial_results:
        print("Error: No trial results found.", file=sys.stderr)
        sys.exit(1)

    # Build stats
    stats = JobStats.from_trial_results(trial_results)

    # Build metrics
    metrics_list, n_attempts = build_metrics(job_dir, metrics_config_path)
    print(f"Metrics: {[type(m).__name__ for m in metrics_list]}")
    print(f"n_attempts: {n_attempts}")

    # Compute metrics per evals_key
    # Build rewards map and trial results map keyed by evals_key
    rewards_map: dict[str, list] = defaultdict(list)
    trials_map: dict[str, list[TrialResult]] = defaultdict(list)

    for tr in trial_results:
        agent_name = tr.agent_info.name
        model_name = tr.agent_info.model_info.name if tr.agent_info.model_info else None
        dataset_name = tr.source or "adhoc"
        evals_key = JobStats.format_agent_evals_key(agent_name, model_name, dataset_name)

        trials_map[evals_key].append(tr)
        if tr.verifier_result is not None:
            rewards_map[evals_key].append(tr.verifier_result.rewards)
        else:
            rewards_map[evals_key].append(None)

    # Apply metrics
    for evals_key in stats.evals:
        rewards = rewards_map.get(evals_key, [])
        relevant_trials = trials_map.get(evals_key, [])
        ctx = MetricContext(trial_results=relevant_trials, n_attempts=n_attempts)

        stats.evals[evals_key].metrics = [
            metric.compute(rewards, context=ctx) for metric in metrics_list
        ]

    # Try to preserve id and timestamps from existing result.json
    existing_result_path = job_dir / "result.json"
    job_id = uuid4()
    started_at = datetime.now()
    finished_at = datetime.now()

    if existing_result_path.exists():
        try:
            existing = json.loads(existing_result_path.read_text())
            if "id" in existing:
                from uuid import UUID
                job_id = UUID(existing["id"])
            if "started_at" in existing and existing["started_at"]:
                started_at = datetime.fromisoformat(existing["started_at"])
            if "finished_at" in existing and existing["finished_at"]:
                finished_at = datetime.fromisoformat(existing["finished_at"])
        except Exception:
            pass

    job_result = JobResult(
        id=job_id,
        started_at=started_at,
        finished_at=finished_at,
        n_total_trials=len(trial_results),
        stats=stats,
    )

    # Print summary
    print(f"\nResults summary:")
    print(f"  Total trials: {stats.n_trials}")
    print(f"  Total errors: {stats.n_errors}")
    for evals_key, eval_stats in stats.evals.items():
        print(f"\n  [{evals_key}]")
        print(f"    Trials: {eval_stats.n_trials}, Errors: {eval_stats.n_errors}")
        for i, m in enumerate(eval_stats.metrics):
            print(f"    Metric {i}: {json.dumps(m, indent=6)}")

    # Write
    if dry_run:
        print("\nDRY RUN — not writing result.json")
    else:
        output = safe_model_dump_json(job_result, exclude={"trial_results"})
        existing_result_path.write_text(output)
        print(f"\nWrote {existing_result_path}")

    return job_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute result.json for an existing Harbor job directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "job_dir",
        help="Path to the job directory (or a parent — auto-resolves to the job dir)",
    )
    parser.add_argument(
        "--metrics-config",
        default=None,
        help="Path to a YAML config with metrics/n_attempts to override the job's config.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    raw_path = Path(args.job_dir).expanduser().resolve()
    if not raw_path.exists():
        print(f"Error: Path does not exist: {raw_path}", file=sys.stderr)
        sys.exit(1)

    job_dir = resolve_job_dir(raw_path)
    if job_dir != raw_path:
        print(f"[resolve] Using job dir: {job_dir}")

    metrics_config_path = None
    if args.metrics_config:
        metrics_config_path = Path(args.metrics_config).expanduser().resolve()
        if not metrics_config_path.exists():
            print(f"Error: Metrics config not found: {metrics_config_path}", file=sys.stderr)
            sys.exit(1)

    recompute(
        job_dir=job_dir,
        metrics_config_path=metrics_config_path,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
