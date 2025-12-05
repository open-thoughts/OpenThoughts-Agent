#!/usr/bin/env python3
"""BaseDataGenerator integration for the Nemotron-PrismMath pipeline."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.commons import (
    BaseDataGenerator,
    GenerationContext,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    finalize_dataset_output,
    subsample_tasks_directory,
    upload_tasks_to_hf,
)
from data.nemo_prism_math.generate import create_sandboxed_tasks


def _count_task_directories(root: Path) -> int:
    return sum(1 for entry in root.iterdir() if entry.is_dir())


class NemoPrismMathGenerator(BaseDataGenerator):
    """BaseDataGenerator-compatible wrapper for Nemotron-PrismMath."""

    parser_description = "Generate Nemotron-PrismMath sandboxes dataset"
    default_target_repo: Optional[str] = "mlfoundations-dev/Nemotron-PrismMath-sandboxes-1"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-prefix",
            type=str,
            default="Nemotron-PrismMath",
            help="Prefix for generated task directories.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=10_000,
            help="Maximum number of tasks to materialise.",
        )
        parser.add_argument(
            "--offset",
            type=int,
            default=0,
            help="Dataset row offset to start from.",
        )
        parser.add_argument(
            "--subsample-count",
            type=int,
            default=10_000,
            help="Optional maximum number of tasks to retain (subsample when exceeded).",
        )

    def run_task_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        args = context.args
        limit = self._resolve_limit(request) or getattr(args, "limit", None)
        if limit is None or limit <= 0:
            limit = getattr(args, "limit", 10_000)

        dataset_prefix = getattr(args, "dataset_prefix", "Nemotron-PrismMath")
        temp_dir = Path(create_sandboxed_tasks(limit=limit, offset=getattr(args, "offset", 0)))

        current_root = temp_dir
        temp_paths = {current_root}
        task_count = _count_task_directories(current_root)

        subsample_count = getattr(args, "subsample_count", None)
        if subsample_count and subsample_count > 0 and task_count > subsample_count:
            subsampled = Path(
                subsample_tasks_directory(
                    str(current_root),
                    subsample_count,
                    dataset_prefix=dataset_prefix,
                )
            )
            if subsampled != current_root:
                temp_paths.add(subsampled)
                current_root = subsampled
                task_count = _count_task_directories(current_root)

        final_path = finalize_dataset_output(str(current_root), getattr(args, "output_dir", None))

        uploaded_repo: Optional[str] = None
        if not getattr(args, "no_upload", False):
            target_repo = getattr(args, "target_repo", None) or self.default_target_repo
            if target_repo:
                upload_tasks_to_hf(str(final_path), target_repo)
                uploaded_repo = target_repo
            else:
                raise ValueError(
                    "No target repo specified; pass --target-repo or set default_target_repo."
                )

        for candidate in temp_paths:
            if str(candidate) != str(final_path):
                shutil.rmtree(candidate, ignore_errors=True)

        artifacts = {
            "limit": limit,
            "task_count": task_count,
            "dataset_prefix": dataset_prefix,
        }

        return GenerationResult(
            dataset_path=str(final_path),
            status=GenerationStatus.SUCCESS,
            uploaded_repo=uploaded_repo,
            artifacts=artifacts,
        )


def main() -> None:
    NemoPrismMathGenerator().run()


if __name__ == "__main__":
    main()
