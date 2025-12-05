#!/usr/bin/env python3
"""
Trace generator entrypoint for the GSM8K Terminal Bench dataset.

This script ensures that a sandbox tasks dataset is available locally (pulling
from Hugging Face if needed) and then reuses the shared trace orchestration
logic provided by `BaseDataGenerator`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from data.commons import download_hf_dataset
from data.generation.base import BaseDataGenerator
from data.generation.schemas import (
    GenerationContext,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    InputValidationError,
)


DEFAULT_TASKS_REPO = "mlfoundations-dev/gsm8k-terminal-bench"


class GSM8KTerminalBenchTraceGenerator(BaseDataGenerator):
    """Generate traces for the GSM8K Terminal Bench sandbox dataset."""

    parser_description = "Generate traces for the GSM8K Terminal Bench dataset"

    def __init__(self) -> None:
        super().__init__()
        # Default to traces-only flow; callers can override with --stage.
        self.parser.set_defaults(stage="traces")

    def add_arguments(self, parser) -> None:  # type: ignore[override]
        super().add_arguments(parser)

        dataset_group = parser.add_argument_group("GSM8K Terminal Bench Tasks")
        dataset_group.add_argument(
            "--tasks-repo",
            type=str,
            default=DEFAULT_TASKS_REPO,
            help="Hugging Face repository containing the sandbox tasks "
            "(default: %(default)s)",
        )
        dataset_group.add_argument(
            "--tasks-revision",
            type=str,
            default=None,
            help="Optional revision/commit hash for the tasks repository",
        )
        dataset_group.add_argument(
            "--local-tasks-path",
            type=str,
            default=None,
            help="Use a pre-existing directory of sandbox tasks instead of "
            "downloading from Hugging Face",
        )
        dataset_group.add_argument(
            "--copy-to-output",
            action="store_true",
            help="Copy the resolved tasks dataset into --output-dir when the "
            "tasks stage runs. By default, the cached dataset is reused.",
        )

    def run_task_generation(  # type: ignore[override]
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        tasks_path = self._resolve_tasks_dataset(request, context)
        output_dir = getattr(context.args, "output_dir", None)

        if getattr(context.args, "copy_to_output", False) and output_dir:
            output_dir_path = Path(output_dir).resolve()
            if output_dir_path.exists() and any(output_dir_path.iterdir()):
                raise RuntimeError(
                    f"Output directory '{output_dir_path}' is not empty; "
                    "remove existing contents or omit --copy-to-output."
                )
            output_dir_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(tasks_path, output_dir_path, dirs_exist_ok=True)
            final_path = output_dir_path
        else:
            final_path = tasks_path

        request.tasks_input = str(final_path)
        return GenerationResult(
            dataset_path=str(final_path),
            status=GenerationStatus.SUCCESS,
        )

    def run_trace_generation(  # type: ignore[override]
        self,
        request: GenerationRequest,
        context: GenerationContext,
        task_result: Optional[GenerationResult],
    ) -> GenerationResult:
        tasks_path = self._resolve_tasks_dataset(request, context)
        request.tasks_input = str(tasks_path)
        return super().run_trace_generation(request, context, task_result)

    def _resolve_tasks_dataset(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> Path:
        """Ensure there is a local tasks dataset and return its path."""
        metadata = request.metadata or {}

        cached_path = metadata.get("resolved_tasks_path")
        if cached_path:
            cached_dir = Path(cached_path)
            if cached_dir.exists():
                return cached_dir

        explicit_tasks = request.tasks_input or getattr(context.args, "tasks_input", None)
        if explicit_tasks:
            tasks_dir = Path(explicit_tasks)
            if not tasks_dir.exists():
                raise InputValidationError(
                    f"Provided tasks input path does not exist: {tasks_dir}"
                )
            metadata["resolved_tasks_path"] = str(tasks_dir.resolve())
            request.metadata = metadata
            return tasks_dir

        local_override = getattr(context.args, "local_tasks_path", None)
        if local_override:
            tasks_dir = Path(local_override).expanduser().resolve()
            if not tasks_dir.exists():
                raise InputValidationError(
                    f"--local-tasks-path not found: {tasks_dir}"
                )
            metadata["resolved_tasks_path"] = str(tasks_dir)
            request.metadata = metadata
            return tasks_dir

        repo_id = getattr(context.args, "tasks_repo", DEFAULT_TASKS_REPO)
        if not repo_id:
            raise InputValidationError(
                "No tasks dataset source provided; supply --tasks-input, "
                "--local-tasks-path, or --tasks-repo."
            )

        revision = getattr(context.args, "tasks_revision", None)
        print(f"[gsm8k-terminal-traces] Downloading tasks dataset: {repo_id}")
        dataset_path = download_hf_dataset(repo_id, revision=revision)
        tasks_dir = Path(dataset_path).resolve()
        if not tasks_dir.exists():
            raise RuntimeError(
                f"Downloaded dataset path missing or inaccessible: {tasks_dir}"
            )

        metadata["resolved_tasks_path"] = str(tasks_dir)
        request.metadata = metadata
        return tasks_dir


def main() -> None:
    generator = GSM8KTerminalBenchTraceGenerator()
    generator.run()


if __name__ == "__main__":
    main()
