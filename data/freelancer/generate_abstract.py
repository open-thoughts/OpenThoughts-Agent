#!/usr/bin/env python3
"""
Generate Freelancer projects dataset in sandboxes style (BaseDataGenerator adapter).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from data.commons import (
    BaseDataGenerator,
    GenerationContext,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    finalize_dataset_output,
    generate_tasks_from_questions,
    upload_tasks_to_hf,
    upsample_tasks_directory,
)
from data.freelancer.generate import scrape_freelancer_projects


class FreelancerGenerator(BaseDataGenerator):
    parser_description = "Generate Freelancer sandboxes dataset"
    default_target_repo = "mlfoundations-dev/freelancer-projects-sandboxes"

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--use_discovery",
            action="store_true",
            help="Enable full project discovery before scraping.",
        )
        parser.add_argument(
            "--max_projects",
            type=int,
            default=None,
            help="Maximum number of projects to scrape (defaults to all discovered).",
        )
        parser.add_argument(
            "--upsample_count",
            type=int,
            default=10_000,
            help="Target number of tasks after upsampling.",
        )
        parser.add_argument(
            "--repo_id",
            type=str,
            default=None,
            help="Override Hugging Face repository ID for upload.",
        )
        parser.add_argument(
            "--private",
            action="store_true",
            help="Upload dataset as a private repository.",
        )

    def run_task_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        args = context.args
        limit = self._resolve_limit(request)
        max_projects = args.max_projects if getattr(args, "max_projects", None) else None
        if limit and limit > 0:
            max_projects = min(max_projects, limit) if max_projects else limit

        print("[1/3] Scraping Freelancer project descriptions")
        questions = scrape_freelancer_projects(
            use_discovery=getattr(args, "use_discovery", False),
            max_projects=max_projects,
            use_all_categories=True,
        )
        if limit and limit > 0:
            questions = questions[:limit]

        print(f"[2/3] Generating {len(questions)} tasks")
        temp_tasks_dir = Path(generate_tasks_from_questions(questions, "freelancer"))
        target_count = getattr(args, "upsample_count", 10_000)
        upsampled_dir = upsample_tasks_directory(str(temp_tasks_dir), target_count)
        output_path = finalize_dataset_output(upsampled_dir, getattr(args, "output_dir", None))
        dataset_path = str(output_path)

        shutil.rmtree(temp_tasks_dir, ignore_errors=True)

        print("[3/3] Uploading dataset (if enabled)")
        uploaded_repo: Optional[str] = None
        target_repo = (
            getattr(args, "repo_id", None)
            or getattr(args, "target_repo", None)
            or self.default_target_repo
        )
        if not getattr(args, "no_upload", False) and target_repo:
            upload_tasks_to_hf(
                dataset_path=dataset_path,
                repo_id=target_repo,
                private=getattr(args, "private", False),
            )
            uploaded_repo = target_repo
        else:
            print(f"Skipping upload. Dataset available at: {dataset_path}")

        return GenerationResult(
            dataset_path=dataset_path,
            status=GenerationStatus.SUCCESS,
            uploaded_repo=uploaded_repo,
        )


def main() -> None:
    FreelancerGenerator().run()


if __name__ == "__main__":
    main()

