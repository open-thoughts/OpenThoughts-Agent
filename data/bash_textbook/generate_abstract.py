#!/usr/bin/env python3
"""BaseDataGenerator integration for the Bash Textbook pipeline."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional

if __package__ in (None, ""):
    # Allow executing the script directly via `python data/bash_textbook/generate_abstract.py`.
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.bash_textbook.generate import (
    create_url_dataset,
    download_pdfs,
    expand_and_extract_pages,
    extract_text_from_page_bytes,
    get_all_page_counts,
    upsample_dataset,
)
from data.commons import (
    BaseDataGenerator,
    GenerationContext,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    finalize_dataset_output,
    generate_tasks_from_questions,
    subsample_tasks_directory,
    upsample_tasks_directory,
    upload_tasks_to_hf,
)
from data.completions import generate_questions_from_text


DEFAULT_BASH_URLS: List[str] = [
    "https://www.tldp.org/LDP/Bash-Beginners-Guide/Bash-Beginners-Guide.pdf",
    "https://www.gnu.org/software/bash/manual/bash.pdf",
    "https://www.tldp.org/LDP/abs/abs-guide.pdf",
    "https://linuxcommand.org/tlcl.pdf",
    "https://www.freecodecamp.org/news/content/images/2022/03/Bash-Scripting-Tutorial.pdf",
    "https://www.cyberciti.biz/media/new/faq/2009/11/bash_reference_manual.pdf",
]


def _count_task_directories(root: Path) -> int:
    return sum(1 for entry in root.iterdir() if entry.is_dir())


class BashTextbookGenerator(BaseDataGenerator):
    """BaseDataGenerator-compatible wrapper around the Bash Textbook pipeline."""

    parser_description = "Generate Bash Textbook sandboxes dataset"
    default_target_repo: Optional[str] = "DCAgent/bash_textbook_tasks"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-prefix",
            type=str,
            default="bash_textbook",
            help="Prefix to use for generated task directories.",
        )
        parser.add_argument(
            "--target-samples",
            type=int,
            default=10_000,
            help="Number of tasks to keep after up/down sampling.",
        )
        parser.add_argument(
            "--num-repeat",
            type=int,
            default=1,
            help="Number of completions to request per extracted page.",
        )
        parser.add_argument(
            "--max-pages",
            type=int,
            default=None,
            help="Optional hard cap on the number of extracted pages processed.",
        )
        parser.add_argument(
            "--url",
            action="append",
            dest="urls",
            help="Additional PDF URL to include. Can be passed multiple times.",
        )

    def _resolve_urls(self, args: argparse.Namespace) -> List[str]:
        urls: List[str] = list(DEFAULT_BASH_URLS)
        extra: Optional[Iterable[str]] = getattr(args, "urls", None)
        if extra:
            urls.extend(url for url in extra if url)
        return urls

    def run_task_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        args = context.args
        limit = self._resolve_limit(request)

        target_samples = limit if limit is not None else getattr(args, "target_samples", 10_000)
        if target_samples is None or target_samples <= 0:
            target_samples = 10_000
        target_samples = max(1, target_samples)

        urls = self._resolve_urls(args)
        url_dataset = create_url_dataset(urls)
        pdf_dataset = download_pdfs(url_dataset)
        page_counts = get_all_page_counts(pdf_dataset)
        pages_dataset = expand_and_extract_pages(page_counts)

        max_pages = getattr(args, "max_pages", None)
        if max_pages is not None and max_pages > 0:
            page_count = min(max_pages, len(pages_dataset))
            pages_dataset = pages_dataset.select(range(page_count))

        text_dataset = extract_text_from_page_bytes(pages_dataset)

        upsample_floor = max(target_samples, len(text_dataset))
        repeated_dataset = upsample_dataset(text_dataset, upsample_floor)

        questions = generate_questions_from_text(
            repeated_dataset,
            num_repeat=getattr(args, "num_repeat", 1),
        )

        if limit is not None and limit > 0:
            questions = questions[:limit]

        tasks_path = generate_tasks_from_questions(
            questions,
            dataset_prefix=getattr(args, "dataset_prefix", "bash_textbook"),
        )

        temp_paths = {Path(tasks_path)}
        tasks_root = Path(tasks_path)
        prefix = getattr(args, "dataset_prefix", "bash_textbook")

        current_count = _count_task_directories(tasks_root)
        if current_count < target_samples:
            upsampled = Path(upsample_tasks_directory(str(tasks_root), target_samples, dataset_prefix=prefix))
            if upsampled != tasks_root:
                temp_paths.add(upsampled)
                tasks_root = upsampled

        if target_samples > 0:
            subsampled = Path(subsample_tasks_directory(str(tasks_root), target_samples, dataset_prefix=prefix))
            if subsampled != tasks_root:
                temp_paths.add(subsampled)
                tasks_root = subsampled

        output_dir = getattr(args, "output_dir", None)
        final_path = finalize_dataset_output(str(tasks_root), output_dir)

        uploaded_repo: Optional[str] = None
        if not getattr(args, "no_upload", False):
            target_repo = getattr(args, "target_repo", None) or self.default_target_repo
            if target_repo:
                upload_tasks_to_hf(final_path, target_repo)
                uploaded_repo = target_repo
            else:
                raise ValueError(
                    "No target repo specified; pass --target-repo or set default_target_repo."
                )

        for candidate in temp_paths:
            if str(candidate) != str(final_path):
                shutil.rmtree(candidate, ignore_errors=True)

        artifacts = {
            "urls": urls,
            "target_samples": target_samples,
            "num_questions": len(questions),
            "output_dir": str(final_path),
        }

        return GenerationResult(
            dataset_path=str(final_path),
            status=GenerationStatus.SUCCESS,
            uploaded_repo=uploaded_repo,
            artifacts=artifacts,
        )


def main() -> None:
    BashTextbookGenerator().run()


if __name__ == "__main__":
    main()
