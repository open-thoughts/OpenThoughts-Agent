#!/usr/bin/env python3
"""
Generate STAQC dataset in sandboxes style (BaseDataGenerator adapter).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from data.commons import (
    BaseDataGenerator,
    GenerationContext,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    finalize_dataset_output,
    generate_tasks_from_questions,
    subsample_tasks_directory,
    upload_tasks_to_hf,
)
from data.staqc.generate import load_staqc_subset


DEFAULT_SUBSETS: List[str] = [
    "man_python",
    "mca_python",
    "sca_python",
    "man_sql",
    "mca_sql",
    "sca_sql",
]


class StaqcGenerator(BaseDataGenerator):
    parser_description = "Generate STAQC sandboxes dataset"
    default_target_repo = "mlfoundations-dev/staqc-sandboxes"

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--subsets",
            nargs="*",
            default=DEFAULT_SUBSETS,
            help="STAQC subsets to include.",
        )
        parser.add_argument(
            "--subsample_count",
            type=int,
            default=10_000,
            help="Number of tasks to retain after subsampling.",
        )

    def run_task_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        args = context.args
        subsets = args.subsets if getattr(args, "subsets", None) else DEFAULT_SUBSETS
        limit = self._resolve_limit(request)

        print(f"[1/2] Loading STAQC subsets: {', '.join(subsets)}")
        questions: List[str] = []
        for subset in subsets:
            questions.extend(load_staqc_subset(subset))

        if limit and limit > 0:
            questions = questions[:limit]

        print(f"[2/2] Generating tasks from {len(questions)} questions")
        temp_tasks_dir = Path(generate_tasks_from_questions(questions, "staqc"))
        sample_cap = getattr(args, "subsample_count", 10_000)
        subsample_count = min(sample_cap, len(questions)) if questions else sample_cap
        subsampled_dataset_dir = subsample_tasks_directory(str(temp_tasks_dir), subsample_count)
        output_path = finalize_dataset_output(subsampled_dataset_dir, getattr(args, "output_dir", None))
        dataset_path = str(output_path)

        shutil.rmtree(temp_tasks_dir, ignore_errors=True)

        uploaded_repo: Optional[str] = None
        target_repo = getattr(args, "target_repo", None) or self.default_target_repo
        if not getattr(args, "no_upload", False) and target_repo:
            upload_tasks_to_hf(dataset_path, target_repo)
            uploaded_repo = target_repo
        else:
            print(f"Skipping upload. Dataset available at: {dataset_path}")

        return GenerationResult(
            dataset_path=dataset_path,
            status=GenerationStatus.SUCCESS,
            uploaded_repo=uploaded_repo,
        )


def main() -> None:
    StaqcGenerator().run()


if __name__ == "__main__":
    main()

