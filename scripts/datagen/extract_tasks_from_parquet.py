#!/usr/bin/env python3
"""
Extract Harbor-compatible task folders from a task parquet snapshot.

This is a thin wrapper around `scripts.harbor.tasks_parquet_converter`
so you can download a parquet (e.g. from Hugging Face) and materialize
the canonical `task-xxxx` directory structure without launching traces.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.harbor import tasks_parquet_converter as tpc

# Lazy import to avoid requiring google.cloud unless needed.
download_hf_dataset = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Harbor tasks from a parquet file.")
    parser.add_argument(
        "--parquet",
        required=True,
        help=(
            "Path to the task parquet file to extract, a directory containing parquet files, "
            "or a Hugging Face dataset repo ID. If a repo ID is provided, the script downloads "
            "it before extraction."
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
    parquet_path: Path

    candidate_path = Path(parquet_input).expanduser()
    if candidate_path.exists():
        if candidate_path.is_dir():
            parquet_files = sorted(candidate_path.rglob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found under directory: {candidate_path}")
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
        parquet_files = sorted(snapshot_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under downloaded repo '{parquet_input}' (path: {snapshot_dir})"
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

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    print(f"[extract] Parquet file: {parquet_path}")
    print(f"[extract] Output directory: {output_dir} (on_exist={args.on_exist})")

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
        print("[extract] DRY RUN: skipping extraction step.")
        return

    tpc.from_parquet(str(parquet_path), str(output_dir), on_exist=args.on_exist)
    print("[extract] Done.")


if __name__ == "__main__":
    main()
