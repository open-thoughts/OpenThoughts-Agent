"""Helpers for limiting task and trace directories."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional


def _copy_common_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, destination / item.name)


def limit_tasks_directory(
    source_dir: str | Path,
    limit: Optional[int],
    *,
    dataset_prefix: str = "limited_tasks",
) -> str:
    """Return a directory containing at most ``limit`` task subdirectories."""
    if limit is None or limit <= 0:
        return str(source_dir)

    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Task directory does not exist: {source_path}")

    task_dirs = sorted((d for d in source_path.iterdir() if d.is_dir()), key=lambda p: p.name)
    if len(task_dirs) <= limit:
        return str(source_path)

    dest_path = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_"))
    _copy_common_files(source_path, dest_path)

    for task_dir in task_dirs[:limit]:
        shutil.copytree(task_dir, dest_path / task_dir.name)

    print(
        f"[limit_tasks_directory] Limited {source_path} to {limit} tasks (from {len(task_dirs)}) -> {dest_path}"
    )
    return str(dest_path)


def limit_trace_jobs_directory(
    source_dir: str | Path,
    limit: Optional[int],
    *,
    dataset_prefix: str = "limited_traces",
) -> str:
    """Return a directory containing at most ``limit`` trace trial subdirectories."""
    if limit is None or limit <= 0:
        return str(source_dir)

    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Trace jobs directory does not exist: {source_path}")

    trial_dirs = sorted(
        (d for d in source_path.iterdir() if d.is_dir() and (d / "agent").exists()),
        key=lambda p: p.name,
    )
    if len(trial_dirs) <= limit:
        return str(source_path)

    dest_path = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_"))
    _copy_common_files(source_path, dest_path)

    for trial_dir in trial_dirs[:limit]:
        shutil.copytree(trial_dir, dest_path / trial_dir.name)

    print(
        f"[limit_trace_jobs_directory] Limited {source_path} to {limit} trials (from {len(trial_dirs)}) -> {dest_path}"
    )
    return str(dest_path)


__all__ = ["limit_tasks_directory", "limit_trace_jobs_directory"]
