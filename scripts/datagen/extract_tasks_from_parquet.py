#!/usr/bin/env python3
"""
Extract Harbor-compatible task folders from a task parquet snapshot.

This is a thin wrapper around `scripts.harbor.tasks_parquet_converter`
so you can download a parquet (e.g. from Hugging Face) and materialize
the canonical `task-xxxx` directory structure without launching traces.

Supports two HuggingFace dataset formats:
1. Parquet with `task_binary` column: Tasks are compressed in tar archives, needs extraction
2. Raw task directories: Tasks are already in directory form with instruction.md files

The script auto-detects the format and handles accordingly.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from hpc.hf_utils import is_raw_tasks_directory
from scripts.harbor import tasks_parquet_converter as tpc

# Lazy import to avoid requiring google.cloud unless needed.
download_hf_dataset = None


def copy_raw_tasks(source_dir: Path, output_dir: Path, on_exist: str = "skip") -> int:
    """Copy raw task directories to output directory.

    Args:
        source_dir: Directory containing raw task folders
        output_dir: Destination directory
        on_exist: How to handle existing directories (skip, overwrite, error)

    Returns:
        Number of tasks copied
    """
    task_dirs = tpc.find_tasks(source_dir, recursive=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for task_dir in task_dirs:
        # Preserve relative path structure
        rel_path = task_dir.relative_to(source_dir)
        dest_dir = output_dir / rel_path

        if dest_dir.exists():
            if on_exist == "skip":
                continue
            elif on_exist == "error":
                raise FileExistsError(f"Target exists: {dest_dir}")
            elif on_exist == "overwrite":
                shutil.rmtree(dest_dir)

        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(task_dir, dest_dir)
        copied += 1

    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Harbor tasks from a parquet file.")
    parser.add_argument(
        "--parquet",
        required=True,
        help=(
            "Path to the task parquet file to extract, a directory containing parquet files, "
            "or a Hugging Face dataset repo ID. If a repo ID is provided, the script downloads "
            "it before extraction. Also supports raw task directories (auto-detected)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where extracted task folders will be written.",
    )
    parser.add_argument(
        "--on_exist",
        choices=("skip", "overwrite", "error"),
        default="error",
        help="How to handle existing task folders (default: error).",
    )
    parser.add_argument(
        "--tasks_revision",
        default=None,
        help="Optional revision/commit to download when --parquet references a Hugging Face repo.",
    )
    parser.add_argument(
        "--parquet_name",
        default=None,
        help="Optional parquet filename to select when the source contains multiple parquet files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print actions without extracting anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parquet_input = args.parquet
    source_dir: Path | None = None  # For raw task directories
    parquet_path: Path | None = None  # For parquet extraction

    candidate_path = Path(parquet_input).expanduser()
    if candidate_path.exists():
        # Local path exists
        if candidate_path.is_dir():
            # Check if it's raw tasks or contains parquet files
            if is_raw_tasks_directory(candidate_path):
                print(f"[extract] Detected raw task directory: {candidate_path}")
                source_dir = candidate_path
            else:
                # Look for parquet files
                parquet_files = sorted(candidate_path.rglob("*.parquet"))
                if not parquet_files:
                    raise FileNotFoundError(
                        f"No parquet files or raw tasks found under directory: {candidate_path}"
                    )
                if args.parquet_name:
                    matching = [p for p in parquet_files if p.name == args.parquet_name]
                    if not matching:
                        raise FileNotFoundError(
                            f"Could not find parquet named '{args.parquet_name}' under {candidate_path}"
                        )
                    parquet_path = matching[0]
                else:
                    parquet_path = parquet_files[0]
        else:
            parquet_path = candidate_path.resolve()
    else:
        # Not a local path - treat as HuggingFace repo
        global download_hf_dataset
        if download_hf_dataset is None:
            try:
                from data.commons import download_hf_dataset as _download_hf_dataset  # type: ignore
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    f"Cannot resolve '{parquet_input}' as a local file and failed to import data.commons "
                    f"to download Hugging Face repos: {exc}"
                ) from exc
            download_hf_dataset = _download_hf_dataset

        print(f"[extract] Treating '{parquet_input}' as a Hugging Face dataset repo; downloading snapshot...")
        snapshot_dir = Path(
            download_hf_dataset(parquet_input, revision=args.tasks_revision)
        )

        # Check if it's raw tasks or parquet with task_binary
        if is_raw_tasks_directory(snapshot_dir):
            print(f"[extract] Detected raw task directory (no extraction needed): {snapshot_dir}")
            source_dir = snapshot_dir
        else:
            # Look for parquet files with task_binary column
            parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(
                    f"No parquet files or raw tasks found under downloaded repo '{parquet_input}' (path: {snapshot_dir})"
                )
            if args.parquet_name:
                matching = [p for p in parquet_files if p.name == args.parquet_name]
                if not matching:
                    available = "\n  - ".join(str(p.relative_to(snapshot_dir)) for p in parquet_files[:20])
                    raise FileNotFoundError(
                        f"Could not find parquet named '{args.parquet_name}' in repo '{parquet_input}'. "
                        f"Available examples:\n  - {available}"
                    )
                parquet_path = matching[0]
            else:
                parquet_path = parquet_files[0]
            print(f"[extract] Using parquet file: {parquet_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()

    # Early exit if output already exists
    if (
        args.on_exist == "error"
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        print(
            "[extract] Output directory already contains data and on_exist=error; "
            "assuming tasks have already been extracted. Exiting gracefully."
        )
        return

    if args.dry_run:
        if source_dir:
            print(f"[extract] DRY RUN: Would copy raw tasks from {source_dir} to {output_dir}")
        else:
            print(f"[extract] DRY RUN: Would extract parquet {parquet_path} to {output_dir}")
        return

    # Perform extraction or copy
    if source_dir:
        # Raw task directory - copy to output
        print(f"[extract] Copying raw tasks from: {source_dir}")
        print(f"[extract] Output directory: {output_dir} (on_exist={args.on_exist})")
        num_copied = copy_raw_tasks(source_dir, output_dir, on_exist=args.on_exist)
        print(f"[extract] Done. Copied {num_copied} task directories.")
    else:
        # Parquet with task_binary - extract
        assert parquet_path is not None
        print(f"[extract] Extracting parquet: {parquet_path}")
        print(f"[extract] Output directory: {output_dir} (on_exist={args.on_exist})")
        tpc.from_parquet(str(parquet_path), str(output_dir), on_exist=args.on_exist)
        print("[extract] Done.")


if __name__ == "__main__":
    main()
