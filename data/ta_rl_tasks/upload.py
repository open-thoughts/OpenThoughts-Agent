#!/usr/bin/env python3
"""Upload previously generated TA RL datasets to Hugging Face."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sandboxes.tasks_parquet_converter import convert_to_parquet

OUTPUT_ROOT = ROOT / "data" / "ta_rl_tasks" / "out"


MODEL_CONFIGS = {
    "gpt-5-mini": {
        "dataset_prefix": "freelancer_ta_rl_mini",
        "upload_repo": "DCAgent/freelancer-projects-sandboxes-ta-rl-gpt-5-mini",
    },
    "gpt-5-nano": {
        "dataset_prefix": "freelancer_ta_rl_nano",
        "upload_repo": "DCAgent/freelancer-projects-sandboxes-ta-rl-gpt-5-nano",
    },
}


def _has_final_tests(task_dir: Path) -> bool:
    tests_dir = task_dir / "tests"
    return (tests_dir / "test.sh").exists() and (tests_dir / "config.json").exists()


def _list_tasks(prefix_dir: Path) -> Iterable[Path]:
    pattern = f"{prefix_dir.name}-*"
    yield from sorted(prefix_dir.glob(pattern))


def upload_dataset(
    *,
    model: str,
    min_completed: int,
    force: bool,
) -> bool:
    config = MODEL_CONFIGS[model]
    dataset_dir = OUTPUT_ROOT / config["dataset_prefix"]
    if not dataset_dir.exists():
        logging.error("Dataset directory %s does not exist; aborting", dataset_dir)
        return False

    tasks = list(_list_tasks(dataset_dir))
    if not tasks:
        logging.error("Dataset directory %s is empty; aborting", dataset_dir)
        return False

    incomplete: List[Path] = [p for p in tasks if not _has_final_tests(p)]
    completed = len(tasks) - len(incomplete)
    if completed < min_completed:
        logging.warning(
            "Model %s has %s completed tasks (< %s); skipping upload",
            model,
            completed,
            min_completed,
        )
        if incomplete:
            logging.info(
                "First incomplete tasks: %s",
                ", ".join(p.name for p in incomplete[:10]),
            )
        return False

    if incomplete:
        logging.warning(
            "Model %s has %s incomplete tasks but meets minimum threshold; proceeding",
            model,
            len(incomplete),
        )

    marker = dataset_dir / ".uploaded"
    if marker.exists() and not force:
        logging.info("Model %s already uploaded (marker found); skipping", model)
        return False

    logging.info("Uploading %s to %s", dataset_dir, config["upload_repo"])
    tmp_dir = convert_to_parquet(str(dataset_dir))
    try:
        parquet_dir = Path(tmp_dir)
        api = HfApi()
        create_repo(
            repo_id=config["upload_repo"],
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(parquet_dir),
            repo_id=config["upload_repo"],
            repo_type="dataset",
            delete_patterns="*",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    marker.write_text("uploaded", encoding="utf-8")
    logging.info("Upload complete for %s", model)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload TA RL datasets to HF")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which model's dataset to upload",
    )
    parser.add_argument(
        "--min-completed",
        type=int,
        default=9_999,
        help="Minimum number of completed tasks required before uploading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if .uploaded marker exists",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    models = MODEL_CONFIGS.keys() if args.model == "all" else [args.model]
    any_uploaded = False
    for model in models:
        success = upload_dataset(
            model=model,
            min_completed=args.min_completed,
            force=args.force,
        )
        any_uploaded = any_uploaded or success

    if not any_uploaded:
        logging.warning("No datasets uploaded (check logs above)")


if __name__ == "__main__":
    main()
