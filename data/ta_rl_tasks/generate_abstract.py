#!/usr/bin/env python3
"""
Generate TA RL tasks with SFT traces (BaseDataGenerator adapter).

This generator takes existing Harbor tasks and their traces, extracts solution scripts
from traces, and uses an LLM to generate RL test harnesses (tests/test.sh and tests/config.json)
following SWEBench conventions.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download

from data.generation import (
    BaseDataGenerator,
    GenerationContext,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
)
from data.commons import finalize_dataset_output
from data.generation.engines import InferenceEngine
from data.ta_rl_tasks.utils import (
    commands_from_trace,
    generate_tests,
    read_text,
    write_metadata,
)
from scripts.sandboxes.tasks_parquet_converter import from_parquet

LOGGER = logging.getLogger(__name__)

# Concurrency tuning
DEFAULT_MAX_CONCURRENCY = 256
MAX_GENERATION_RETRIES = 3
_THREAD_LOCAL = threading.local()


def _normalize_key(value: str) -> str:
    """Normalize task identifier from trace metadata."""
    name = Path(value.strip()).name
    return name.split("__", 1)[0]


def _build_trace_lookup(traces: Dataset) -> Dict[str, Dict[str, any]]:
    """Build a lookup dictionary mapping task identifiers to trace rows."""
    lookup: Dict[str, Dict[str, any]] = {}
    for row in traces:
        trial_name = row.get("trial_name") or row.get("task_path") or row.get("path")
        if not isinstance(trial_name, str) or not trial_name.strip():
            LOGGER.debug("Skipping trace row lacking task identifier: %s", sorted(row.keys()))
            continue
        key = _normalize_key(trial_name)
        if key in lookup:
            LOGGER.debug("Duplicate trace detected for %s; keeping the first occurrence", key)
            continue
        lookup[key] = row
    LOGGER.info("Indexed %s traces", len(lookup))
    return lookup


def _discover_task_dirs(tasks_root: Path) -> List[Path]:
    """Discover all task directories in the tasks root."""
    return sorted({path.parent for path in tasks_root.rglob("instruction.md")})


def _write_solution(dest_dir: Path, *, commands: List[str], trace: Dict[str, any]) -> Dict[str, any]:
    """Write solution script from extracted commands."""
    if not commands:
        return {}

    solution_dir = dest_dir / "solution"
    solution_dir.mkdir(exist_ok=True)

    body = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        f"# Commands reconstructed from trace '{trace.get('trial_name')}'",
    ]
    body.append(f"# Trace success: {trace.get('success')} | command lines: {len(commands)}")
    body.append("")
    body.extend(commands)
    body_text = "\n".join(body)
    if not body_text.endswith("\n"):
        body_text += "\n"

    script_path = solution_dir / "solve.sh"
    script_path.write_text(body_text, encoding="utf-8")
    os.chmod(script_path, 0o755)

    return write_metadata(dest_dir, trace, len(commands))


def _thread_local_engine(engine: InferenceEngine):
    """Get or create thread-local engine instance."""
    cached_engine = getattr(_THREAD_LOCAL, "engine", None)
    if cached_engine is None or cached_engine is not engine:
        _THREAD_LOCAL.engine = engine
    return _THREAD_LOCAL.engine


def _run_generation_with_retries(
    *,
    dest_dir: Path,
    instruction: str,
    solution_script: str,
    trace_meta: Dict[str, any] | None,
    engine: InferenceEngine,
) -> bool:
    """Run test generation with retries."""
    last_exc: Exception | None = None
    for attempt in range(1, MAX_GENERATION_RETRIES + 1):
        try:
            thread_engine = _thread_local_engine(engine)
            generate_tests(
                dest_dir=dest_dir,
                instruction=instruction,
                solution_script=solution_script,
                trace_metadata=[trace_meta] if trace_meta else [],
                engine=thread_engine,
            )
            return True
        except Exception as exc:  # pragma: no cover - depends on external service
            LOGGER.warning(
                "LLM test generation attempt %s/%s failed for %s: %s",
                attempt,
                MAX_GENERATION_RETRIES,
                dest_dir.name,
                exc,
            )
            last_exc = exc
    LOGGER.error(
        "Skipping %s after %s failed attempts: %s",
        dest_dir.name,
        MAX_GENERATION_RETRIES,
        last_exc,
    )
    return False


def _hydrate_and_generate(
    *,
    idx: int,
    task_dir: Path,
    dest_dir: Path,
    trace_row: Dict[str, any],
    engine: InferenceEngine,
) -> bool:
    """Hydrate task with solution and generate RL test harness."""
    commands = commands_from_trace(trace_row)
    trace_meta = _write_solution(dest_dir, commands=commands, trace=trace_row)

    instruction = read_text(dest_dir / "instruction.md")
    solution_script = read_text(dest_dir / "solution" / "solve.sh")
    if not instruction or not solution_script:
        LOGGER.warning(
            "Missing instruction or solution in %s; skipping test generation",
            dest_dir,
        )
        return False

    return _run_generation_with_retries(
        dest_dir=dest_dir,
        instruction=instruction,
        solution_script=solution_script,
        trace_meta=trace_meta,
        engine=engine,
    )


def _has_final_tests(dest_dir: Path) -> bool:
    """Check if task has complete RL test harness."""
    tests_dir = dest_dir / "tests"
    return (tests_dir / "test.sh").exists() and (tests_dir / "config.json").exists()


class TARLTaskGenerator(BaseDataGenerator):
    """Generate RL test harnesses for existing Harbor tasks using their traces."""

    parser_description = "Generate TA RL tasks with RL test harnesses from Harbor tasks and traces"
    default_target_repo = None  # Must be specified via CLI

    def requires_engine_for_stage(self, stage: str) -> bool:
        """TA RL task generation always requires an inference engine for test generation."""
        return True  # Always requires engine for LLM-based test generation

    def add_arguments(self, parser) -> None:
        """Add TA RL task-specific arguments."""
        parser.add_argument(
            "--tasks-repo",
            type=str,
            required=True,
            help="Hugging Face repository ID for source tasks (e.g., 'mlfoundations-dev/freelancer-projects-sandboxes')",
        )
        parser.add_argument(
            "--traces-repo",
            type=str,
            required=True,
            help="Hugging Face repository ID for source traces (e.g., 'mlfoundations-dev/freelancer-projects-sandboxes-traces-terminus-2')",
        )
        parser.add_argument(
            "--traces-split",
            type=str,
            default="train",
            help="Dataset split to use for traces (default: 'train')",
        )
        parser.add_argument(
            "--max-concurrency",
            type=int,
            default=None,
            help=f"Maximum concurrent test generations (default: {DEFAULT_MAX_CONCURRENCY}, or TA_RL_MAX_CONCURRENCY env var)",
        )
        parser.add_argument(
            "--min-completed-tasks",
            type=int,
            default=0,
            help="Minimum number of completed tasks required (default: 0, no minimum)",
        )

    def _resolve_concurrency(self, override: Optional[int]) -> int:
        """Resolve concurrency limit from args or environment."""
        if override is not None:
            return max(1, override)

        env_value = os.environ.get("TA_RL_MAX_CONCURRENCY")
        if env_value:
            try:
                return max(1, int(env_value))
            except ValueError:
                LOGGER.warning(
                    "Invalid TA_RL_MAX_CONCURRENCY value '%s'; falling back to default %s",
                    env_value,
                    DEFAULT_MAX_CONCURRENCY,
                )
        return DEFAULT_MAX_CONCURRENCY

    def run_task_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        """Generate RL test harnesses for tasks."""
        args = context.args
        engine = context.engine

        if engine is None:
            raise ValueError(
                "TA RL task generation requires an inference engine. "
                "Specify --engine (e.g., openai, anthropic, vllm_local) and --model."
            )

        tasks_repo = getattr(args, "tasks_repo", None)
        traces_repo = getattr(args, "traces_repo", None)
        traces_split = getattr(args, "traces_split", "train")
        max_concurrency = self._resolve_concurrency(getattr(args, "max_concurrency", None))
        min_completed = getattr(args, "min_completed_tasks", 0)

        if not tasks_repo or not traces_repo:
            raise ValueError(
                "Both --tasks-repo and --traces-repo must be specified. "
                "Example: --tasks-repo mlfoundations-dev/freelancer-projects-sandboxes "
                "--traces-repo mlfoundations-dev/freelancer-projects-sandboxes-traces-terminus-2"
            )

        LOGGER.info("Downloading tasks from %s", tasks_repo)
        repo_dir = Path(snapshot_download(repo_id=tasks_repo, repo_type="dataset"))
        parquet_files = sorted(repo_dir.glob("**/*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet artifact found in dataset repo {tasks_repo}")

        tasks_root = Path(tempfile.mkdtemp(prefix="ta_rl_tasks_source_"))
        LOGGER.info("Extracting %s to %s", parquet_files[0], tasks_root)
        from_parquet(str(parquet_files[0]), str(tasks_root))

        LOGGER.info("Loading traces from %s (split=%s)", traces_repo, traces_split)
        traces = load_dataset(traces_repo, split=traces_split)
        trace_lookup = _build_trace_lookup(traces)

        output_dir = request.output_dir
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(tempfile.mkdtemp(prefix="ta_rl_tasks_output_"))

        task_dirs = _discover_task_dirs(tasks_root)
        limit = self._resolve_limit(request)
        if limit and limit > 0:
            task_dirs = task_dirs[:limit]

        LOGGER.info("Processing %s tasks", len(task_dirs))

        # Prepare task specs
        task_specs: List[tuple[int, Path, Dict[str, any], Path]] = []
        target_dirs: List[Path] = []
        dataset_prefix = "ta_rl_task"

        for idx, task_dir in enumerate(task_dirs):
            key = _normalize_key(task_dir.name)
            trace_row = trace_lookup.get(key)
            if trace_row is None:
                LOGGER.debug("No trace found for %s; skipping", key)
                continue

            dest_dir = output_dir / f"{dataset_prefix}-{idx:05d}"
            target_dirs.append(dest_dir)

            # Skip if already complete
            if dest_dir.exists():
                if _has_final_tests(dest_dir):
                    LOGGER.debug("Task %s already generated; skipping", dest_dir.name)
                    continue
                LOGGER.info(
                    "Task %s exists but incomplete; removing stale contents",
                    dest_dir.name,
                )
                shutil.rmtree(dest_dir, ignore_errors=True)

            try:
                shutil.copytree(task_dir, dest_dir)
            except OSError as exc:
                LOGGER.error("Failed to copy %s -> %s: %s", task_dir, dest_dir, exc)
                continue

            task_specs.append((idx, task_dir, trace_row, dest_dir))

        LOGGER.info(
            "Generating tests with max_concurrency=%s over %s tasks",
            max_concurrency,
            len(task_specs),
        )

        # Generate tests in parallel
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(
                    _hydrate_and_generate,
                    idx=idx,
                    task_dir=task_dir,
                    dest_dir=dest_dir,
                    trace_row=trace_row,
                    engine=engine,
                ): dest_dir
                for idx, task_dir, trace_row, dest_dir in task_specs
            }

            for future in as_completed(futures):
                dest_dir = futures[future]
                try:
                    success = future.result()
                    if success:
                        completed_count += 1
                except Exception as exc:  # pragma: no cover - depends on external service
                    LOGGER.error(
                        "LLM test generation failed for %s: %s",
                        dest_dir.name,
                        exc,
                    )
                    # Continue processing other tasks instead of failing completely

        incomplete = [path for path in target_dirs if not _has_final_tests(path)]
        completed = len(target_dirs) - len(incomplete)

        if completed < min_completed:
            raise RuntimeError(
                f"Only {completed} tasks completed (minimum required: {min_completed}). "
                f"Incomplete tasks: {len(incomplete)}"
            )

        if incomplete:
            LOGGER.warning(
                "Dataset has %s incomplete tasks but meets minimum threshold (%s >= %s); proceeding",
                len(incomplete),
                completed,
                min_completed,
            )

        # Cleanup
        shutil.rmtree(tasks_root, ignore_errors=True)

        # Finalize output directory
        final_output = finalize_dataset_output(output_dir, request.output_dir)

        uploaded_repo: Optional[str] = None
        target_repo = (
            getattr(args, "target_repo", None)
            or request.target_repo
            or self.default_target_repo
        )
        if not getattr(args, "no_upload", False) and target_repo:
            from data.commons import upload_tasks_to_hf

            upload_tasks_to_hf(
                dataset_path=str(final_output),
                repo_id=target_repo,
                private=getattr(args, "private", False),
            )
            uploaded_repo = target_repo
        else:
            LOGGER.info(f"Skipping upload. Dataset available at: {final_output}")

        return GenerationResult(
            dataset_path=str(final_output),
            status=GenerationStatus.SUCCESS,
            uploaded_repo=uploaded_repo,
            artifacts={
                "completed_tasks": completed,
                "incomplete_tasks": len(incomplete),
                "total_tasks": len(target_dirs),
            },
        )


def main() -> None:
    """Entry point for CLI execution."""
    TARLTaskGenerator().run()


if __name__ == "__main__":
    main()

