#!/usr/bin/env python3
"""
Load a Hugging Face dataset repository, group rows by task, and keep only the latest episode per task.

Episodes are expected to look like "episode-0", "episode-1", etc. The numeric suffix determines recency.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

DEFAULT_REPO_ID = "DCAgent/DCAgent_dev_set_71_tasks_mlfoundations-dev_freelancer-projects-sandboxes-tracesd6ddf5a0"
EPISODE_PATTERN = re.compile(r"episode-(\d+)$")


def episode_index(episode: Optional[str]) -> Optional[int]:
    """Return the numeric episode index extracted from an episode string."""
    if episode is None:
        return None
    match = EPISODE_PATTERN.fullmatch(episode.strip())
    if not match:
        return None
    return int(match.group(1))


def keep_latest_per_task(split: Dataset, split_name: str) -> Dataset:
    """Reduce a split so that only the latest episode per task remains."""
    latest: Dict[str, Tuple[int, Dict]] = {}
    skipped_no_task = 0
    skipped_no_episode = 0

    for row in split:
        task = row.get("task")
        if task is None:
            skipped_no_task += 1
            continue

        ep_value = row.get("episode")
        ep_idx = episode_index(ep_value)
        if ep_idx is None:
            skipped_no_episode += 1
            continue

        current = latest.get(task)
        if current is None or ep_idx > current[0]:
            # Make a shallow copy so we do not keep references to HF internals.
            latest[task] = (ep_idx, dict(row))

    if skipped_no_task:
        logging.warning(
            "Split '%s': skipped %d rows missing 'task'",
            split_name,
            skipped_no_task,
        )
    if skipped_no_episode:
        logging.warning(
            "Split '%s': skipped %d rows with invalid or missing 'episode'",
            split_name,
            skipped_no_episode,
        )

    ordered_records = sorted(
        ((task, ep_idx, record) for task, (ep_idx, record) in latest.items()),
        key=lambda item: (item[0], item[1]),
    )
    reduced_records: List[Dict] = [record for _, _, record in ordered_records]

    logging.info(
        "Split '%s': reduced from %d to %d rows across %d unique tasks",
        split_name,
        len(split),
        len(reduced_records),
        len(reduced_records),
    )

    return Dataset.from_list(reduced_records)


def save_jsonl(records: Iterable[Dict], output_path: str) -> None:
    """Save a collection of dictionaries as JSON Lines."""
    with open(output_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a Hugging Face dataset and keep only the most recent episode per task."
        )
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repo ID to load (default: {DEFAULT_REPO_ID}). Ignored when --job-dir is provided.",
    )
    parser.add_argument(
        "--job-dir",
        dest="job_dir",
        help=(
            "Path to a local Harbor job directory containing traces (as used by make_and_upload_trace_dataset.py). "
            "When provided, traces are read locally instead of from the Hub."
        ),
    )
    parser.add_argument(
        "--split",
        help="Optional dataset split to process (e.g. 'train'). If omitted, every split is processed.",
    )
    parser.add_argument(
        "--save-to-disk",
        metavar="PATH",
        help="Optional directory to write the reduced dataset(s) via Dataset.save_to_disk().",
    )
    parser.add_argument(
        "--output-jsonl",
        metavar="PATH",
        help="Optional path to write the reduced records as JSON Lines. Only valid when a single split is processed.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )

    # Load from local Harbor job directory if provided, else from Hub repo_id
    if args.job_dir:
        try:
            # Import lazily to avoid a hard dependency when not needed
            from harbor.utils.traces_utils import export_traces  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "--job-dir was provided, but Harbor is not available. Install Harbor (pip install -e ../harbor) "
                f"or ensure it's on PYTHONPATH. Import error: {exc}"
            )

        logging.info("Exporting local traces from '%s'", args.job_dir)
        ds = export_traces(
            root=args.job_dir,
            recursive=True,
            episodes="all",
            to_sharegpt=False,
            repo_id=None,
            push=False,
            verbose=False,
            success_filter=None,
        )
        if isinstance(ds, Dataset):
            dataset_dict = DatasetDict({"default": ds})
        else:
            dataset_dict = ds  # assumed DatasetDict
        # When reading from a local job dir, --split is ignored unless the export returns multiple splits
        if args.split and args.split not in dataset_dict:
            logging.warning("--split '%s' not found in exported dataset; proceeding with all splits.", args.split)
    else:
        logging.info("Loading dataset '%s'", args.repo_id)
        if args.split:
            dataset = load_dataset(args.repo_id, split=args.split)
            dataset_dict = DatasetDict({args.split: dataset})
        else:
            loaded = load_dataset(args.repo_id)
            if isinstance(loaded, Dataset):
                dataset_dict = DatasetDict({"default": loaded})
            else:
                dataset_dict = loaded

    reduced_splits: Dict[str, Dataset] = {}
    for split_name, split in dataset_dict.items():
        reduced_splits[split_name] = keep_latest_per_task(split, split_name)

    if args.save_to_disk:
        logging.info("Saving reduced dataset to '%s'", args.save_to_disk)
        DatasetDict(reduced_splits).save_to_disk(args.save_to_disk)

    if args.output_jsonl:
        if len(reduced_splits) != 1:
            raise ValueError(
                "--output-jsonl can only be used when processing a single split."
            )
        split_dataset = next(iter(reduced_splits.values()))
        save_jsonl(split_dataset, args.output_jsonl)

    if not args.save_to_disk and not args.output_jsonl:
        for split_name, split_dataset in reduced_splits.items():
            print(f"# Split '{split_name}' ({len(split_dataset)} rows)")
            for row in split_dataset:
                print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
