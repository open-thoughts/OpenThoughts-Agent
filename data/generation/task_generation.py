"""Utilities for running multiple Daytona-backed task generations with concurrency."""

from __future__ import annotations

import copy
import logging
import asyncio
import importlib
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from data.commons import finalize_dataset_output

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskGenerationSpec:
    """Describe a single teacher generation run."""

    seed: int
    output_dir: Path
    dataset_prefix: str
    jobs_dir: Path
    dataset_index: Optional[int] = None


@dataclass
class TaskGenerationResult:
    """Result payload for a single generation run."""

    spec: TaskGenerationSpec
    success: bool
    dataset_dir: Optional[Path] = None
    artifacts: Optional[dict] = None
    exception_path: Optional[Path] = None
    error: Optional[BaseException] = None


class DaytonaTaskGenerationExecutor:
    """Execute multiple Daytona-backed task generations with bounded concurrency."""

    def __init__(
        self,
        base_args: Optional[object],
        max_concurrent: int = 1,
        generate_fn: Optional[Callable[[object], Tuple[Path, dict]]] = None,
    ) -> None:
        if generate_fn is None:
            raise ValueError("generate_fn must be provided")
        self._base_args = base_args
        self._max_concurrent = max(1, max_concurrent)
        self._generate_fn = generate_fn

    def run(self, specs: Iterable[TaskGenerationSpec]) -> List[TaskGenerationResult]:
        """Run the configured generation function for each spec with bounded concurrency."""

        specs_list = list(specs)
        if not specs_list:
            return []

        results: List[TaskGenerationResult] = []
        with ThreadPoolExecutor(max_workers=self._max_concurrent) as executor:
            future_map = {
                executor.submit(self._run_single, spec): spec for spec in specs_list
            }
            for future in as_completed(future_map):
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    spec = future_map[future]
                    LOGGER.exception("Unhandled exception while generating seed %s", spec.seed)
                    results.append(
                        TaskGenerationResult(
                            spec=spec,
                            success=False,
                            error=exc,
                            exception_path=self._write_exception(
                                spec.output_dir,
                                f"Unhandled exception while generating seed {spec.seed}: {exc}",
                            ),
                        )
                    )
                else:
                    results.append(result)
        return results

    def _run_single(self, spec: TaskGenerationSpec) -> TaskGenerationResult:
        args = copy.deepcopy(self._base_args)

        # Clean output directory.
        if spec.output_dir.exists():
            for child in spec.output_dir.iterdir():
                if child.is_dir():
                    self._remove_tree(child)
                else:
                    child.unlink()
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        spec.jobs_dir.mkdir(parents=True, exist_ok=True)

        # Populate per-run arguments.
        args.seed = spec.seed
        args.output = str(spec.output_dir)
        args.dataset_prefix = spec.dataset_prefix
        args.teacher_jobs_dir = str(spec.jobs_dir)
        args.dataset_index = spec.dataset_index

        try:
            dataset_dir, artifacts = self._generate_fn(args)
            final_dir = finalize_dataset_output(dataset_dir, args.output)
            return TaskGenerationResult(
                spec=spec,
                success=True,
                dataset_dir=final_dir,
                artifacts=artifacts,
            )
        except Exception as exc:
            LOGGER.exception("Generation failed for seed %s", spec.seed)
            exception_path = self._write_exception(
                spec.output_dir,
                f"Generation failed for seed {spec.seed}: {exc}",
            )
            return TaskGenerationResult(
                spec=spec,
                success=False,
                error=exc,
                exception_path=exception_path,
            )
        finally:
            self._shutdown_litellm()

    @staticmethod
    def _write_exception(output_dir: Path, message: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        exception_path = output_dir / "exception.txt"
        exception_path.write_text(message + "\n", encoding="utf-8")
        return exception_path

    @staticmethod
    def _remove_tree(path: Path) -> None:
        for child in path.iterdir():
            if child.is_dir():
                DaytonaTaskGenerationExecutor._remove_tree(child)
            else:
                child.unlink()
        path.rmdir()

    @staticmethod
    def _shutdown_litellm(timeout: float = 5.0) -> None:
        try:
            litellm = importlib.import_module("litellm")
        except ModuleNotFoundError:
            return

        shutdown_fn = getattr(litellm, "shutdown", None)
        if shutdown_fn is None:
            return

        try:
            sig = inspect.signature(shutdown_fn)
            result = shutdown_fn(timeout=timeout) if "timeout" in sig.parameters else shutdown_fn()
            if inspect.isawaitable(result):
                asyncio.run(result)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            LOGGER.debug("LiteLLM shutdown raised %r", exc)
