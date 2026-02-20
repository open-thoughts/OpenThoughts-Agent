#!/usr/bin/env python3
"""Count unique Dockerfiles (snapshots) from a task dataset.

This script answers the question: "If I run a Harbor job on this dataset with
the Daytona backend and auto_snapshot=True, how many container snapshots would
be created?"

Harbor's Daytona backend uses content-based hashing of Dockerfiles to determine
snapshot identity. Tasks with identical Dockerfile content share a snapshot,
while tasks with different Dockerfiles each require a separate snapshot build.

Usage:
    # From a local dataset directory
    python -m scripts.harbor.count_snapshots_from_tasks --local-dataset /path/to/tasks

    # From a Harbor registry dataset
    python -m scripts.harbor.count_snapshots_from_tasks --registry-dataset mlfoundations/code_contests

    # Show detailed hash breakdown
    python -m scripts.harbor.count_snapshots_from_tasks --local-dataset /path/to/tasks --verbose

    # Limit to specific tasks (glob patterns supported)
    python -m scripts.harbor.count_snapshots_from_tasks --local-dataset /path/to/tasks \
        --task-names "code_contests-*" --exclude "code_contests-100*"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from harbor.constants import TASK_CACHE_DIR
from harbor.models.job.config import LocalDatasetConfig, RegistryDatasetConfig
from harbor.models.registry import RemoteRegistryInfo
from harbor.tasks.client import TaskClient
from harbor.utils.container_cache import (
    DockerfileStats,
    analyze_task_dockerfiles,
)


def load_tasks_from_local_dataset(
    dataset_path: Path,
    task_names: list[str] | None = None,
    exclude_task_names: list[str] | None = None,
    n_tasks: int | None = None,
) -> list[Path]:
    """Load task directories from a local dataset."""
    config = LocalDatasetConfig(
        path=dataset_path,
        task_names=task_names,
        exclude_task_names=exclude_task_names,
        n_tasks=n_tasks,
    )
    # Get task configs which contain resolved paths
    task_configs = config.get_task_configs(disable_verification=True)
    return [tc.path for tc in task_configs]


def load_tasks_from_registry_dataset(
    dataset_name: str,
    version: str | None = None,
    task_names: list[str] | None = None,
    exclude_task_names: list[str] | None = None,
    n_tasks: int | None = None,
    download_dir: Path | None = None,
) -> list[Path]:
    """Load task directories from a Harbor registry dataset.

    Downloads tasks to local cache if not already present.
    """
    config = RegistryDatasetConfig(
        registry=RemoteRegistryInfo(),  # Uses default Harbor registry
        name=dataset_name,
        version=version,
        task_names=task_names,
        exclude_task_names=exclude_task_names,
        n_tasks=n_tasks,
        download_dir=download_dir,
    )

    # Get task configs (this will download tasks if needed)
    task_configs = config.get_task_configs(disable_verification=True)

    # Download the actual task directories
    client = TaskClient()
    task_ids = [tc.get_task_id() for tc in task_configs if tc.is_git_task()]
    local_ids = [tc.path for tc in task_configs if not tc.is_git_task()]

    if task_ids:
        downloaded_paths = client.download_tasks(
            task_ids=task_ids,
            overwrite=False,
            output_dir=download_dir or TASK_CACHE_DIR,
        )
        return downloaded_paths + local_ids

    return local_ids


def print_stats(stats: DockerfileStats, verbose: bool = False) -> None:
    """Print Dockerfile statistics."""
    print("\n" + "=" * 60)
    print("DOCKERFILE / SNAPSHOT ANALYSIS")
    print("=" * 60)

    print(f"\nTotal tasks:              {stats.total_tasks}")
    print(f"Tasks with Dockerfile:    {stats.tasks_with_dockerfile}")
    print(f"Tasks without Dockerfile: {stats.tasks_without_dockerfile}")

    print(f"\n{'=' * 60}")
    print(f"UNIQUE DOCKERFILES (SNAPSHOTS): {stats.unique_hashes}")
    print(f"{'=' * 60}")

    if stats.unique_hashes == 0:
        print("\nNo Dockerfiles found in tasks.")
        return

    # Calculate reuse statistics
    max_reuse = max(stats.hash_counts.values())
    avg_reuse = stats.tasks_with_dockerfile / stats.unique_hashes if stats.unique_hashes > 0 else 0
    single_use = sum(1 for count in stats.hash_counts.values() if count == 1)

    print(f"\nSnapshot reuse statistics:")
    print(f"  Average tasks per snapshot: {avg_reuse:.1f}")
    print(f"  Maximum tasks per snapshot: {max_reuse}")
    print(f"  Single-use snapshots:       {single_use} ({100*single_use/stats.unique_hashes:.1f}%)")

    if verbose:
        print(f"\n{'=' * 60}")
        print("HASH BREAKDOWN (sorted by frequency)")
        print("=" * 60)

        for i, (hash_val, count) in enumerate(stats.hash_counts.most_common(), 1):
            snapshot_name = f"harbor__{hash_val}__snapshot"
            print(f"  {i:3d}. {snapshot_name}: {count} task(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Count unique Dockerfiles (snapshots) from a task dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset source (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--local-dataset",
        type=Path,
        metavar="PATH",
        help="Path to local task dataset directory",
    )
    source_group.add_argument(
        "--registry-dataset",
        type=str,
        metavar="NAME",
        help="Name of dataset in Harbor registry (e.g., mlfoundations/code_contests)",
    )

    # Filtering options
    parser.add_argument(
        "--task-names",
        type=str,
        nargs="+",
        help="Only include tasks matching these patterns (glob supported)",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        dest="exclude_task_names",
        help="Exclude tasks matching these patterns (glob supported)",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        help="Maximum number of tasks to analyze",
    )

    # Registry options
    parser.add_argument(
        "--version",
        type=str,
        help="Dataset version (for registry datasets)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Directory to cache downloaded tasks",
    )

    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed hash breakdown",
    )

    args = parser.parse_args()

    # Load tasks
    print("Loading tasks...")
    try:
        if args.local_dataset:
            task_dirs = load_tasks_from_local_dataset(
                dataset_path=args.local_dataset.expanduser().resolve(),
                task_names=args.task_names,
                exclude_task_names=args.exclude_task_names,
                n_tasks=args.n_tasks,
            )
        else:
            task_dirs = load_tasks_from_registry_dataset(
                dataset_name=args.registry_dataset,
                version=args.version,
                task_names=args.task_names,
                exclude_task_names=args.exclude_task_names,
                n_tasks=args.n_tasks,
                download_dir=args.download_dir,
            )
    except Exception as e:
        print(f"Error loading tasks: {e}", file=sys.stderr)
        sys.exit(1)

    if not task_dirs:
        print("No tasks found matching the specified criteria.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(task_dirs)} tasks. Analyzing Dockerfiles...")

    # Analyze Dockerfiles
    stats = analyze_task_dockerfiles(task_dirs)

    # Print results
    print_stats(stats, verbose=args.verbose)

    # Summary for scripting
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"With auto_snapshot=True, this dataset would create {stats.unique_hashes} snapshot(s).")

    if stats.tasks_without_dockerfile > 0:
        print(f"\nNote: {stats.tasks_without_dockerfile} task(s) have no Dockerfile and would use")
        print("      docker_image from task config or fail to start.")


if __name__ == "__main__":
    main()
