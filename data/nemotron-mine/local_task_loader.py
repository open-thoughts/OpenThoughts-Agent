#!/usr/bin/env python3
"""
Load harbor-format tasks from local directories when HF is not available.

Provides load_local_tasks_as_dicts() as a drop-in alternative for
load_harbor_tasks_as_dicts() that reads from data/benchmark_tasks_by_dataset/.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark_tasks_by_dataset"


def load_local_tasks_as_dicts(
    dataset_name: str = None,
    task_dir: str = None,
    limit: int = 10_000,
) -> List[Dict[str, str]]:
    """Load harbor tasks from local directories.

    Args:
        dataset_name: Name of dataset in benchmark_tasks_by_dataset/
                      (e.g., "exercism", "bigcodebench"). If None, loads from all.
        task_dir: Direct path to a tasks directory (overrides dataset_name).
        limit: Max tasks to load.

    Returns:
        List of dicts with keys: instruction, test_sh, dockerfile,
        task_toml, metadata, test_files_json, task_dir_name.
    """
    if task_dir:
        root = Path(task_dir)
    elif dataset_name:
        root = BENCHMARK_DIR / dataset_name
    else:
        root = BENCHMARK_DIR

    if not root.exists():
        raise FileNotFoundError(f"Task directory not found: {root}")

    tasks = []

    # Find all task directories (contain instruction.md)
    if (root / "instruction.md").exists():
        # root itself is a task dir
        task_dirs = [root]
    else:
        # root contains subdirectories with tasks
        task_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and (d / "instruction.md").exists()
        )

    # If root is a dataset collection (e.g., benchmark_tasks_by_dataset/),
    # recurse into subdirectories
    if not task_dirs:
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                for d in sorted(sub.iterdir()):
                    if d.is_dir() and (d / "instruction.md").exists():
                        task_dirs.append(d)

    for task_dir_path in task_dirs[:limit]:
        task = _read_task_dir(task_dir_path)
        if task:
            tasks.append(task)

    return tasks


def _read_task_dir(task_dir: Path) -> Optional[Dict[str, str]]:
    """Read a single task directory into a dict."""
    try:
        instruction = _read_file(task_dir / "instruction.md")
        if not instruction:
            return None

        # Read Dockerfile
        dockerfile = ""
        for df_path in [
            task_dir / "environment" / "Dockerfile",
            task_dir / "Dockerfile",
        ]:
            if df_path.exists():
                dockerfile = df_path.read_text(encoding="utf-8")
                break

        # Read test.sh
        test_sh = ""
        for ts_path in [
            task_dir / "tests" / "test.sh",
            task_dir / "test.sh",
        ]:
            if ts_path.exists():
                test_sh = ts_path.read_text(encoding="utf-8")
                break

        # Read task.toml
        task_toml = _read_file(task_dir / "task.toml")

        # Read metadata
        metadata = _read_file(task_dir / "metadata.json") or "{}"

        # Read test files
        test_files = {}
        tests_dir = task_dir / "tests"
        if tests_dir.exists():
            for f in tests_dir.rglob("*"):
                if f.is_file() and f.name != "test.sh":
                    rel = str(f.relative_to(tests_dir))
                    try:
                        test_files[rel] = f.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        pass

        return {
            "task_dir_name": task_dir.name,
            "instruction": instruction,
            "test_sh": test_sh,
            "dockerfile": dockerfile,
            "task_toml": task_toml,
            "metadata": metadata,
            "test_files_json": json.dumps(test_files),
        }
    except Exception:
        return None


def _read_file(path: Path) -> str:
    """Read file if it exists, return empty string otherwise."""
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ""
    return ""


def list_available_datasets() -> List[str]:
    """List locally available datasets."""
    if not BENCHMARK_DIR.exists():
        return []
    return sorted(
        d.name for d in BENCHMARK_DIR.iterdir()
        if d.is_dir()
    )


if __name__ == "__main__":
    print("Available local datasets:")
    for name in list_available_datasets():
        tasks = load_local_tasks_as_dicts(dataset_name=name, limit=3)
        print(f"  {name}: {len(tasks)} tasks (showing 3)")
        if tasks:
            print(f"    Example: {tasks[0]['task_dir_name']}")
            print(f"    Instruction: {tasks[0]['instruction'][:80]}...")
