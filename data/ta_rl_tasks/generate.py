"""Generate RL tasks with SFT traces and TA."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX platforms
    resource = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download

from data.ta_rl_tasks.utils import (
    commands_from_trace,
    create_engine,
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

# Completion policy
MIN_COMPLETED_TASKS = 9_999


# Repositories and naming
TASKS_REPO = "mlfoundations-dev/freelancer-projects-sandboxes"
TRACES_REPO = "mlfoundations-dev/freelancer-projects-sandboxes-traces-terminus-2"

MODEL_CONFIGS = [
    {
        "model": "gpt-5-mini",
        "dataset_prefix": "freelancer_ta_rl_mini",
        "upload_repo": "DCAgent/freelancer-projects-sandboxes-ta-rl-gpt-5-mini",
    },
    {
        "model": "gpt-5-nano",
        "dataset_prefix": "freelancer_ta_rl_nano",
        "upload_repo": "DCAgent/freelancer-projects-sandboxes-ta-rl-gpt-5-nano",
    },
]

DEFAULT_MODEL_CONFIG = MODEL_CONFIGS[0]

ROOT_OUTPUT_DIR = ROOT / "data" / "ta_rl_tasks" / "out"
ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_tasks(
    repo_id: str = TASKS_REPO,
    *,
    dataset_prefix: str,
) -> Path:
    LOGGER.info("Downloading tasks dataset from %s", repo_id)
    repo_dir = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
    parquet_files = sorted(repo_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet artifact found in dataset repo {repo_id}")

    output_dir = Path(tempfile.mkdtemp(prefix=f"{dataset_prefix}_source_"))
    LOGGER.info("Extracting %s to %s", parquet_files[0], output_dir)
    from_parquet(str(parquet_files[0]), str(output_dir))
    return output_dir


def discover_task_dirs(tasks_root: Path) -> List[Path]:
    return sorted({path.parent for path in tasks_root.rglob("instruction.md")})


def load_traces(repo_id: str = TRACES_REPO, *, split: str = "train") -> Dataset:
    LOGGER.info("Loading traces dataset %s (split=%s)", repo_id, split)
    return load_dataset(repo_id, split=split)


def _normalize_key(value: str) -> str:
    name = Path(value.strip()).name
    return name.split("__", 1)[0]


def build_trace_lookup(traces: Dataset) -> Dict[str, Dict[str, any]]:
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


def write_solution(dest_dir: Path, *, commands: List[str], trace: Dict[str, any]) -> Dict[str, any]:
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


def _resolve_concurrency(override: Optional[int]) -> int:
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


def _thread_local_engine(model_name: str):
    cached_model = getattr(_THREAD_LOCAL, "model_name", None)
    engine = getattr(_THREAD_LOCAL, "engine", None)
    if engine is None or cached_model != model_name:
        engine = create_engine(model_name=model_name)
        _THREAD_LOCAL.engine = engine
        _THREAD_LOCAL.model_name = model_name
    return engine


def _run_generation_with_retries(
    *,
    dest_dir: Path,
    instruction: str,
    solution_script: str,
    trace_meta: Dict[str, any] | None,
    model_name: str,
) -> bool:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_GENERATION_RETRIES + 1):
        try:
            engine = _thread_local_engine(model_name)
            generate_tests(
                dest_dir=dest_dir,
                instruction=instruction,
                solution_script=solution_script,
                trace_metadata=[trace_meta] if trace_meta else [],
                engine=engine,
            )
            return True
        except Exception as exc:  # pragma: no cover - depends on external service
            LOGGER.warning(
                "LLM test generation attempt %s/%s failed for %s (model=%s): %s",
                attempt,
                MAX_GENERATION_RETRIES,
                dest_dir.name,
                model_name,
                exc,
            )
            last_exc = exc
    LOGGER.error(
        "Skipping %s after %s failed attempts (model=%s): %s",
        dest_dir.name,
        MAX_GENERATION_RETRIES,
        model_name,
        last_exc,
    )
    return False


def _hydrate_and_generate(
    *,
    idx: int,
    task_dir: Path,
    dest_dir: Path,
    trace_row: Dict[str, any],
    model_name: str,
) -> bool:
    commands = commands_from_trace(trace_row)
    trace_meta = write_solution(dest_dir, commands=commands, trace=trace_row)

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
        model_name=model_name,
    )


def generate_dataset(
    *,
    max_tasks: Optional[int] = None,
    model_name: Optional[str] = None,
    dataset_prefix: Optional[str] = None,
    max_concurrency: Optional[int] = None,
    output_root: Optional[Path] = None,
) -> Tuple[Path, List[Path]]:
    _bump_nofile_limit()
    selected_model = model_name or DEFAULT_MODEL_CONFIG["model"]
    selected_prefix = dataset_prefix or DEFAULT_MODEL_CONFIG["dataset_prefix"]

    base_output_dir = output_root or (ROOT_OUTPUT_DIR / selected_prefix)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    tasks_root = download_tasks(dataset_prefix=selected_prefix)
    trace_lookup = build_trace_lookup(load_traces())

    output_root = Path(tempfile.mkdtemp(prefix=f"{selected_prefix}_"))
    LOGGER.info(
        "Writing hydrated tasks for %s (%s) to %s",
        selected_model,
        selected_prefix,
        output_root,
    )

    task_specs: List[Tuple[int, Path, Dict[str, any], Path]] = []
    target_dirs: List[Path] = []
    for idx, task_dir in enumerate(discover_task_dirs(tasks_root)):
        if max_tasks is not None and idx >= max_tasks:
            break

        key = _normalize_key(task_dir.name)
        trace_row = trace_lookup.get(key)
        if trace_row is None:
            LOGGER.debug("No trace found for %s; skipping", key)
            continue

        dest_dir = base_output_dir / f"{selected_prefix}-{idx:05d}"
        target_dirs.append(dest_dir)
        if dest_dir.exists():
            tests_dir = dest_dir / "tests"
            test_sh = tests_dir / "test.sh"
            config_json = tests_dir / "config.json"
            if test_sh.exists() and config_json.exists():
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

    concurrency = _resolve_concurrency(max_concurrency)
    LOGGER.info(
        "Generating tests with max_concurrency=%s over %s tasks",
        concurrency,
        len(task_specs),
    )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _hydrate_and_generate,
                idx=idx,
                task_dir=task_dir,
                dest_dir=dest_dir,
                trace_row=trace_row,
                model_name=selected_model,
            ): dest_dir
            for idx, task_dir, trace_row, dest_dir in task_specs
        }

        for future in as_completed(futures):
            dest_dir = futures[future]
            try:
                success = future.result()
            except Exception as exc:  # pragma: no cover - depends on external service
                LOGGER.error(
                    "LLM test generation failed for %s (model=%s): %s",
                    dest_dir.name,
                    selected_model,
                    exc,
                )
                executor.shutdown(cancel_futures=True)
                raise
            else:
                if not success:
                    LOGGER.warning(
                        "Task %s was skipped after repeated failures (model=%s)",
                        dest_dir.name,
                        selected_model,
                    )

    shutil.rmtree(tasks_root, ignore_errors=True)
    return base_output_dir, target_dirs


def upload_dataset(dataset_dir: Path, *, upload_repo: str) -> bool:
    from data.commons import upload_tasks_to_hf  # imported lazily to avoid heavy deps

    marker = dataset_dir / ".uploaded"
    if marker.exists():
        LOGGER.info("Dataset %s already marked as uploaded; skipping", dataset_dir)
        return False

    LOGGER.info("Uploading dataset at %s to %s", dataset_dir, upload_repo)
    upload_tasks_to_hf(str(dataset_dir), upload_repo)
    marker.write_text("uploaded", encoding="utf-8")
    return True


def main(max_tasks: Optional[int] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for config in MODEL_CONFIGS:
        LOGGER.info("Starting generation for %s", config["model"])
        dataset_dir, target_dirs = generate_dataset(
            max_tasks=max_tasks,
            model_name=config["model"],
            dataset_prefix=config["dataset_prefix"],
            max_concurrency=None,
            output_root=ROOT_OUTPUT_DIR / config["dataset_prefix"],
        )

        incomplete = [path for path in target_dirs if not _has_final_tests(path)]
        completed = len(target_dirs) - len(incomplete)
        if completed < MIN_COMPLETED_TASKS:
            LOGGER.warning(
                "Dataset %s has only %s completed tasks (< %s); skipping upload",
                dataset_dir,
                completed,
                MIN_COMPLETED_TASKS,
            )
            if incomplete:
                LOGGER.info(
                    "First five incomplete tasks: %s",
                    ", ".join(path.name for path in incomplete[:5]),
                )
            continue

        if incomplete:
            LOGGER.warning(
                "Dataset %s has %s incomplete tasks but meets minimum threshold (%s >= %s); proceeding with upload",
                dataset_dir,
                len(incomplete),
                completed,
                MIN_COMPLETED_TASKS,
            )

        upload_dataset(dataset_dir, upload_repo=config["upload_repo"])


def _bump_nofile_limit(target: int = 4096) -> None:
    if resource is None:  # pragma: no cover - non-POSIX
        return

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):  # pragma: no cover - defensive
        return

    if soft >= target:
        return

    new_soft = min(target, hard if hard != resource.RLIM_INFINITY else target)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        LOGGER.info("Raised RLIMIT_NOFILE from %s to %s", soft, new_soft)
    except (ValueError, OSError) as exc:  # pragma: no cover - best effort
        LOGGER.warning("Failed to raise RLIMIT_NOFILE: %s", exc)


def _has_final_tests(dest_dir: Path) -> bool:
    tests_dir = dest_dir / "tests"
    return (tests_dir / "test.sh").exists() and (tests_dir / "config.json").exists()


if __name__ == "__main__":
    parsed_max = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(max_tasks=parsed_max)
