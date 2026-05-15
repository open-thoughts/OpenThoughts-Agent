#!/usr/bin/env python3
"""
Validate Daytona-buildability of tasks stored in a Hugging Face dataset (parquet
with columns 'path' and 'task_binary'), then upload only the tasks that build
successfully back to the Hub.

Workflow:
1) Download HF dataset and extract task directories to a local path
2) Optionally take a random subset (--sample_size N [--sample_seed S]) and/or
   a deterministic prefix (--limit N) of the discovered tasks
3) Run any subset of these stages, in order, via --stages (default: 'daytona,harbor'):
   - 'daytona': build each task's environment/Dockerfile in Daytona
       up to 5 retries with exponential backoff (2,4,8,16,32s) on failure;
       runs up to 32 validations in parallel (configurable)
   - 'harbor': Harbor smoke test (terminus-2, gpt-5-nano, max_episodes=1).
       --filter_successful swaps in GPT-5-Codex with 160 episodes and requires reward > 0.0
   - 'oracle': run the oracle agent against solution/solve.sh, requiring reward=1.0
4) Copy surviving tasks into a staging directory, then either upload to HF or
   skip the upload (--skip_upload) and just leave them on disk
   - use --final_output_dir to relocate the staged successes directory after upload

Notes:
- Requires 'daytona' to be importable and a working Daytona backend.
- Requires 'datasets', 'pyarrow', and 'huggingface_hub' for IO.
- Auth to HF via HF_TOKEN env var or pass --token.

Example:
    python OpenThoughts-Agent/scripts/daytona/validate_and_upload_from_hf.py \
        --repo_id org/dataset \
        --revision main \
        --extract_dir ./data/tmp_tasks \
        --oracle_check \
        --timeout 600 \
        --target_repo org/validated-dataset

Inspect failures:
    export SEARCHPATH=./experiments/codeforces-GLM-4.6-traces-32ep-32k-1-2-4-dv/trace_jobs/chunk_000/2025-11-13__16-23-33
    ls -lh "$SEARCHPATH" | wc -l
    find "$SEARCHPATH" -type f -name "exception.txt" | wc -l
    find "$SEARCHPATH" -type f -name "exception.txt" -print0 \
        | tee >(while IFS= read -r -d '' file; do \
            grep -o "harbor.trial.trial.AgentTimeoutError:" "$file"; \
          done | wc -l > ./agenttimeoutcount.txt) \
        | tr '\0' '\n'
    echo "agent timeout error count:"
    cat ./agenttimeoutcount.txt
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset

from scripts.harbor import tasks_parquet_converter as tpc
from data.commons import upload_tasks_to_hf

CONSOLE = Console()

STAGE_DAYTONA = "daytona"
STAGE_HARBOR = "harbor"
STAGE_ORACLE = "oracle"
ALL_STAGES = (STAGE_DAYTONA, STAGE_HARBOR, STAGE_ORACLE)

PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=None),
    TextColumn("{task.completed}/{task.total}"),
    TextColumn("[green]✓ {task.fields[success]}"),
    TextColumn("[red]✗ {task.fields[failed]}"),
    TextColumn("Rem: {task.fields[remaining]}"),
    TextColumn("{task.fields[rate]:.2f} task/s"),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


# ---- Daytona imports (lazy within worker as well) ----
from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources
from daytona.common.errors import DaytonaError


def _extract_hf_dataset(repo_id: str, revision: Optional[str], base_dir: Path) -> Path:
    """Download a HF dataset and extract tasks into base_dir.

    Writes a temporary parquet snapshot and uses the shared converter to extract
    directories preserving the 'path' column.
    """
    # Skip extraction if tasks already exist in target directory
    base_dir.mkdir(parents=True, exist_ok=True)
    existing_tasks = [d for d in base_dir.iterdir() if d.is_dir() and (d / "environment" / "Dockerfile").exists()]
    if existing_tasks:
        CONSOLE.print(f"[cyan][extract][/cyan] Found {len(existing_tasks)} existing tasks in {base_dir}; skipping extraction")
        return base_dir
    
    ds = load_dataset(repo_id, revision=revision)
    # Pick a split (prefer train)
    if hasattr(ds, "keys"):
        split_name = "train" if "train" in ds.keys() else next(iter(ds.keys()))
        split = ds[split_name]
    else:  # Single-split dataset
        split = ds

    tmp_dir = Path(tempfile.mkdtemp(prefix="hf_tasks_"))
    parquet_path = tmp_dir / "tasks.parquet"
    split.to_parquet(str(parquet_path))

    tpc.from_parquet(str(parquet_path), base=str(base_dir), on_exist="overwrite")
    return base_dir


async def _try_build_once(dockerfile: Path, *, cpu: int, memory_gb: int, disk_gb: int, gpu: int, timeout: int) -> bool:
    """Attempt a single Daytona build for a Dockerfile."""
    resources = Resources(cpu=max(cpu, 1), memory=max(memory_gb, 1), disk=max(disk_gb, 1), gpu=max(gpu, 0))
    params = CreateSandboxFromImageParams(image=Image.from_dockerfile(dockerfile), auto_delete_interval=0, resources=resources)

    daytona = AsyncDaytona()
    sandbox = None
    try:
        sandbox = await daytona.create(params=params, timeout=timeout)
        # Successful build and start: consider success immediately.
        return True
    finally:
        try:
            if sandbox is not None:
                await sandbox.delete()
        finally:
            await daytona.close()


def _validate_task_worker(args: Tuple[Path, int, int, int, int, int]) -> Tuple[str, bool]:
    """Worker function for multiprocessing.

    Args tuple: (task_dir, cpu, memory_gb, disk_gb, gpu, timeout)
    Returns: (task_name, success)
    """
    task_dir, cpu, memory_gb, disk_gb, gpu, timeout = args
    name = task_dir.name
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        return name, False

    backoffs = [2, 4, 8, 16, 32]
    for i, delay in enumerate(backoffs):
        try:
            ok = asyncio.run(_try_build_once(dockerfile, cpu=cpu, memory_gb=memory_gb, disk_gb=disk_gb, gpu=gpu, timeout=timeout))
            if ok:
                return name, True
        except (DaytonaError, Exception):  # broad catch: any failure triggers retry
            pass
        if i < len(backoffs) - 1:
            time.sleep(delay)
    return name, False


def _ensure_output_dir_in_dockerfile(dockerfile: Path) -> None:
    """Ensure the staged Dockerfile creates /output for tasks expecting it."""

    if not dockerfile.exists():
        return

    content = dockerfile.read_text().splitlines()
    directive = "RUN mkdir -p /output && chmod 777 /output"
    if any(directive in line for line in content):
        return

    insert_idx = 0
    for idx, line in enumerate(content):
        if line.strip().upper().startswith("WORKDIR"):
            insert_idx = idx + 1
            break

    content.insert(insert_idx, directive)
    dockerfile.write_text("\n".join(content) + "\n")


def _discover_tasks(extracted_root: Path) -> list[Path]:
    tasks: list[Path] = []
    for path in sorted(extracted_root.iterdir()):
        dockerfile = path / "environment" / "Dockerfile"
        if path.is_dir() and dockerfile.exists():
            _ensure_output_dir_in_dockerfile(dockerfile)
            tasks.append(path)
    return tasks


def _copy_successes(task_dirs: Iterable[Path], dest_root: Path) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    for task in task_dirs:
        shutil.copytree(task, dest_root / task.name, dirs_exist_ok=True)
    return dest_root

def _relocate_staged_output(staged_dir: Path, target_dir: Path) -> Path:
    """Move the staged successes directory into target_dir and return the new path."""
    target_dir = target_dir.expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    moved_path = Path(shutil.move(str(staged_dir), str(target_dir)))
    return moved_path


def _run_daytona_validation(
    task_dirs: List[Path],
    *,
    base_dir: Path,
    cpu: int,
    memory_gb: int,
    disk_gb: int,
    gpu: int,
    timeout: int,
    processes: int,
) -> Tuple[List[Path], List[Path]]:
    if not task_dirs:
        return [], []

    name_to_path: Dict[str, Path] = {path.name: path for path in task_dirs}
    worker_args = [
        (task, cpu, memory_gb, disk_gb, gpu, timeout) for task in task_dirs
    ]

    successes: List[Path] = []
    failures: List[Path] = []
    success_count = 0
    failure_count = 0
    total = len(task_dirs)
    start = time.perf_counter()

    progress = Progress(*PROGRESS_COLUMNS, console=CONSOLE, transient=True)
    with progress:
        task_id = progress.add_task(
            "Stage 1: Daytona build validation",
            total=total,
            success=0,
            failed=0,
            remaining=total,
            rate=0.0,
        )
        with Pool(processes=max(1, processes)) as pool:
            for name, ok in pool.imap_unordered(_validate_task_worker, worker_args):
                task_path = name_to_path.get(name, base_dir / name)
                if ok:
                    successes.append(task_path)
                    success_count += 1
                else:
                    failures.append(task_path)
                    failure_count += 1
                completed = success_count + failure_count
                remaining = max(total - completed, 0)
                elapsed = max(time.perf_counter() - start, 1e-6)
                rate = completed / elapsed
                progress.update(
                    task_id,
                    advance=1,
                    success=success_count,
                    failed=failure_count,
                    remaining=remaining,
                    rate=rate,
                )
    return successes, failures


def _import_harbor_components():
    try:
        from harbor.job import Job
        from harbor.models.agent.name import AgentName
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.job.config import (
            AgentConfig,
            EnvironmentConfig,
            JobConfig,
            LocalDatasetConfig,
            RetryConfig,
            VerifierConfig,
        )
        from harbor.orchestrators.base import OrchestratorEvent
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Harbor is required for stage 2 validation. Install it (e.g. pip install -e ../harbor) "
            f"or ensure it's on PYTHONPATH. Import error: {exc}"
        ) from exc

    return {
        "Job": Job,
        "AgentName": AgentName,
        "EnvironmentType": EnvironmentType,
        "JobConfig": JobConfig,
        "AgentConfig": AgentConfig,
        "EnvironmentConfig": EnvironmentConfig,
        "LocalDatasetConfig": LocalDatasetConfig,
        "RetryConfig": RetryConfig,
        "VerifierConfig": VerifierConfig,
        "OrchestratorEvent": OrchestratorEvent,
    }


def _make_retry_config(components: dict) -> object:
    """Create a RetryConfig matching the tracegen YAML (retries transient Daytona errors)."""
    RetryConfig = components["RetryConfig"]
    return RetryConfig(
        max_retries=10,
        include_exceptions={
            "DaytonaRateLimitError",
            "AgentEnvironmentTimeoutError",
        },
        exclude_exceptions={
            "AgentTimeoutError",
            "VerifierTimeoutError",
            "RewardFileNotFoundError",
            "RewardFileEmptyError",
            "VerifierOutputParseError",
            "SandboxBuildFailedError",
            "VerifierRuntimeError",
        },
        wait_multiplier=2.0,
        min_wait_sec=1.0,
        max_wait_sec=90.0,
    )


def _run_harbor_smoke_test(
    dataset_root: Path,
    *,
    agent_timeout: int,
    verifier_timeout: int,
    cpu: int,
    memory_gb: int,
    disk_gb: int,
    concurrency: int,
    filter_successful: bool,
) -> Tuple[List[Path], List[Path], Optional[Path], Dict[str, int]]:
    tasks = _discover_tasks(dataset_root)
    if not tasks:
        return [], [], None, {"total": 0, "infra_ok": 0, "solved": 0}

    components = _import_harbor_components()
    Job = components["Job"]
    AgentName = components["AgentName"]
    EnvironmentType = components["EnvironmentType"]
    JobConfig = components["JobConfig"]
    AgentConfig = components["AgentConfig"]
    EnvironmentConfig = components["EnvironmentConfig"]
    LocalDatasetConfig = components["LocalDatasetConfig"]
    VerifierConfig = components["VerifierConfig"]
    OrchestratorEvent = components["OrchestratorEvent"]

    name_to_path: Dict[str, Path] = {task.name: task for task in tasks}
    jobs_root = Path(tempfile.mkdtemp(prefix="harbor_jobs_"))
    job_config = JobConfig()
    job_config.job_name = f"harbor-validate-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    job_config.jobs_dir = jobs_root
    job_config.n_attempts = 1
    job_config.timeout_multiplier = 1.0
    effective_concurrency = (
        len(tasks) if concurrency is None or concurrency <= 0 else min(concurrency, len(tasks))
    )
    job_config.orchestrator.n_concurrent_trials = max(1, effective_concurrency)
    job_config.orchestrator.quiet = True
    job_config.orchestrator.retry = _make_retry_config(components)
    agent_model = "gpt-5-codex" if filter_successful else "gpt-5-nano"
    agent_kwargs = {"max_episodes": 160} if filter_successful else {"max_episodes": 1}
    agent_kwargs.setdefault("reasoning_effort", "medium")
    job_config.agents = [
        AgentConfig(
            name=AgentName.TERMINUS_2.value,
            model_name=agent_model,
            override_timeout_sec=agent_timeout,
            kwargs=agent_kwargs,
        )
    ]
    job_config.environment = EnvironmentConfig(
        type=EnvironmentType.DAYTONA,
        force_build=True,
        delete=True,
        override_cpus=cpu or None,
        override_memory_mb=(memory_gb * 1024) if memory_gb else None,
        override_storage_mb=(disk_gb * 1024) if disk_gb else None,
    )
    job_config.verifier = VerifierConfig(override_timeout_sec=verifier_timeout)
    job_config.datasets = [LocalDatasetConfig(path=dataset_root)]

    job = Job(job_config)
    job_dir = job.job_dir

    job_label = "Harbor filter" if filter_successful else "Harbor smoke test"
    CONSOLE.print(
        f"[stage2] {job_label} job directory: [underline]{job_dir}[/underline]"
    )

    stage_status: Dict[str, bool] = {}
    success_count = 0
    failure_count = 0
    infra_ok_count = 0
    solved_count = 0
    total = len(tasks)
    start = time.perf_counter()

    progress = Progress(*PROGRESS_COLUMNS, console=CONSOLE, transient=True)

    def _reward_positive(rewards: Optional[Dict[str, float | int]]) -> bool:
        if not rewards:
            return False
        raw_value = rewards.get("reward")
        if raw_value is None:
            return False
        try:
            return float(raw_value) > 0.0
        except (TypeError, ValueError):
            return False

    with progress:
        task_id = progress.add_task(
            "Stage 2: Harbor filter" if filter_successful else "Stage 2: Harbor smoke test",
            total=total,
            success=0,
            failed=0,
            remaining=total,
            rate=0.0,
        )

        async def _progress_hook(hook_event):
            nonlocal success_count, failure_count, infra_ok_count, solved_count
            trial_result = hook_event.result
            rewards = None
            if trial_result is not None and trial_result.verifier_result is not None:
                rewards = trial_result.verifier_result.rewards
            is_infra_ok = trial_result is not None and trial_result.exception_info is None
            is_solved = is_infra_ok and _reward_positive(rewards)
            if is_infra_ok:
                infra_ok_count += 1
            if is_solved:
                solved_count += 1
            is_success = is_solved if filter_successful else is_infra_ok
            stage_status[hook_event.task_name] = bool(is_success)
            if is_success:
                success_count += 1
            else:
                failure_count += 1
            completed = success_count + failure_count
            remaining = max(total - completed, 0)
            elapsed = max(time.perf_counter() - start, 1e-6)
            rate = completed / elapsed
            progress.update(
                task_id,
                advance=1,
                success=success_count,
                failed=failure_count,
                remaining=remaining,
                rate=rate,
            )

        job._orchestrator.add_hook(OrchestratorEvent.TRIAL_COMPLETED, _progress_hook)

        try:
            asyncio.run(job.run())
        except Exception as exc:  # pragma: no cover - dependent on Harbor env
            CONSOLE.print(
                f"[red]Harbor validation encountered an error. Logs retained at {job_dir}.[/red]"
            )
            raise

    passed: List[Path] = []
    failed: List[Path] = []
    for name, path in name_to_path.items():
        outcome = stage_status.get(name)
        if outcome is True:
            passed.append(path)
        else:
            failed.append(path)

    stats = {"total": total, "infra_ok": infra_ok_count, "solved": solved_count}
    return passed, failed, job_dir, stats


def _run_oracle_solution_check(
    dataset_root: Path,
    *,
    agent_timeout: int,
    verifier_timeout: int,
    cpu: int,
    memory_gb: int,
    disk_gb: int,
    concurrency: int,
) -> Tuple[List[Path], List[Path], List[Path], Optional[Path]]:
    tasks = _discover_tasks(dataset_root)
    if not tasks:
        return [], [], [], None

    components = _import_harbor_components()
    Job = components["Job"]
    AgentName = components["AgentName"]
    EnvironmentType = components["EnvironmentType"]
    JobConfig = components["JobConfig"]
    AgentConfig = components["AgentConfig"]
    EnvironmentConfig = components["EnvironmentConfig"]
    LocalDatasetConfig = components["LocalDatasetConfig"]
    VerifierConfig = components["VerifierConfig"]
    OrchestratorEvent = components["OrchestratorEvent"]

    missing_solution: List[Path] = []
    candidates: List[Path] = []
    for task in tasks:
        solve_path = task / "solution" / "solve.sh"
        if solve_path.is_file():
            candidates.append(task)
        else:
            missing_solution.append(task)

    if not candidates:
        return [], [], missing_solution, None

    jobs_root = Path(tempfile.mkdtemp(prefix="harbor_oracle_jobs_"))
    job_config = JobConfig()
    job_config.job_name = f"harbor-oracle-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    job_config.jobs_dir = jobs_root
    job_config.n_attempts = 1
    job_config.timeout_multiplier = 1.0
    effective_concurrency = (
        len(candidates) if concurrency is None or concurrency <= 0 else min(concurrency, len(candidates))
    )
    job_config.orchestrator.n_concurrent_trials = max(1, effective_concurrency)
    job_config.orchestrator.quiet = True
    job_config.orchestrator.retry = _make_retry_config(components)
    job_config.agents = [
        AgentConfig(
            name=AgentName.ORACLE.value,
            override_timeout_sec=agent_timeout,
        )
    ]
    job_config.environment = EnvironmentConfig(
        type=EnvironmentType.DAYTONA,
        force_build=True,
        delete=True,
        override_cpus=cpu or None,
        override_memory_mb=(memory_gb * 1024) if memory_gb else None,
        override_storage_mb=(disk_gb * 1024) if disk_gb else None,
    )
    job_config.verifier = VerifierConfig(override_timeout_sec=verifier_timeout)
    job_config.datasets = [
        LocalDatasetConfig(
            path=dataset_root,
            task_names=sorted(task.name for task in candidates),
        )
    ]

    job = Job(job_config)
    job_dir = job.job_dir

    CONSOLE.print(
        f"[stage3] Oracle job directory: [underline]{job_dir}[/underline]"
    )

    stage_status: Dict[str, bool] = {}
    success_count = 0
    failure_count = 0
    total = len(candidates)
    start = time.perf_counter()

    progress = Progress(*PROGRESS_COLUMNS, console=CONSOLE, transient=True)

    def _reward_is_one(rewards: Optional[Dict[str, float | int]]) -> bool:
        if not rewards:
            return False
        raw_value = rewards.get("reward")
        if raw_value is None:
            return False
        try:
            return float(raw_value) == 1.0
        except (TypeError, ValueError):
            return False

    with progress:
        task_id = progress.add_task(
            "Stage 3: Oracle solution check",
            total=total,
            success=0,
            failed=0,
            remaining=total,
            rate=0.0,
        )

        async def _progress_hook(hook_event):
            nonlocal success_count, failure_count
            trial_result = hook_event.result
            rewards = None
            if trial_result is not None and trial_result.verifier_result is not None:
                rewards = trial_result.verifier_result.rewards
            is_success = (
                trial_result is not None
                and trial_result.exception_info is None
                and _reward_is_one(rewards)
            )
            stage_status[hook_event.task_name] = bool(is_success)
            if is_success:
                success_count += 1
            else:
                failure_count += 1
            completed = success_count + failure_count
            remaining = max(total - completed, 0)
            elapsed = max(time.perf_counter() - start, 1e-6)
            rate = completed / elapsed
            progress.update(
                task_id,
                advance=1,
                success=success_count,
                failed=failure_count,
                remaining=remaining,
                rate=rate,
            )

        job._orchestrator.add_hook(OrchestratorEvent.TRIAL_COMPLETED, _progress_hook)

        try:
            asyncio.run(job.run())
        except Exception:
            CONSOLE.print(
                f"[red]Oracle validation encountered an error. Logs retained at {job_dir}.[/red]"
            )
            raise

    passed: List[Path] = []
    failed: List[Path] = []
    for candidate in candidates:
        outcome = stage_status.get(candidate.name)
        if outcome is True:
            passed.append(candidate)
        else:
            failed.append(candidate)

    return passed, failed, missing_solution, job_dir


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Daytona builds for HF task dataset and upload successes")
    p.add_argument("--repo_id", required=True, help="HF dataset repo id (org/name)")
    p.add_argument("--revision", default=None, help="Optional dataset revision/commit")
    p.add_argument("--extract_dir", default=None, help="Directory to extract tasks to (default: temp)")
    p.add_argument("--processes", type=int, default=32, help="Parallel processes for validation (default: 32)")
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--memory_gb", type=int, default=4)
    p.add_argument("--disk_gb", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--timeout", type=int, default=600, help="Per-build timeout seconds")
    p.add_argument("--target_repo", default=None, help="Target HF repo for upload (default: same as --repo_id)")
    p.add_argument("--private", action="store_true", help="Create target repo as private")
    p.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env)")
    p.add_argument("--keep_failed_dir", default=None, help="If set, copy failed tasks here for inspection")
    p.add_argument(
        "--final_output_dir",
        default=None,
        help="If set, move the successful staged tasks into this directory after upload",
    )
    p.add_argument("--limit", type=int, default=None, help="Optionally limit number of tasks validated")
    p.add_argument(
        "--harbor_concurrency",
        type=int,
        default=8,
        help="Concurrent Harbor trials during stage 2 smoke testing (default: 8)",
    )
    p.add_argument(
        "--filter_successful",
        action="store_true",
        help="Use GPT-5-Codex (160 episodes) in stage 2 and filter on reward > 0.0",
    )
    p.add_argument(
        "--stages",
        default=None,
        help=(
            "Comma-separated stages to run, in order. Choices: "
            f"{','.join(ALL_STAGES)}. Default: 'daytona,harbor' (plus 'oracle' "
            "if --oracle_check is set). Overrides --oracle_check / --oracle_check_only."
        ),
    )
    p.add_argument(
        "--skip_upload",
        action="store_true",
        help="Skip the final HF upload (staged successes are kept on disk)",
    )
    p.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="If set, take a random sample of N tasks from the input dataset before any stage runs",
    )
    p.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Seed for --sample_size (default: nondeterministic)",
    )
    oracle_group = p.add_mutually_exclusive_group()
    oracle_group.add_argument(
        "--oracle_check",
        action="store_true",
        help="(Back-compat) Append oracle stage to the default stages",
    )
    oracle_group.add_argument(
        "--oracle_check_only",
        action="store_true",
        help="(Back-compat) Equivalent to --stages oracle",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _resolve_stages(args: argparse.Namespace) -> List[str]:
    if args.stages:
        requested = [s.strip().lower() for s in args.stages.split(",") if s.strip()]
        invalid = [s for s in requested if s not in ALL_STAGES]
        if invalid:
            raise SystemExit(
                f"Unknown stage(s): {invalid}. Valid choices: {list(ALL_STAGES)}"
            )
        # Preserve user order; drop dupes.
        seen: Set[str] = set()
        ordered = []
        for s in requested:
            if s not in seen:
                ordered.append(s)
                seen.add(s)
        return ordered
    if args.oracle_check_only:
        return [STAGE_ORACLE]
    if args.oracle_check:
        return [STAGE_DAYTONA, STAGE_HARBOR, STAGE_ORACLE]
    return [STAGE_DAYTONA, STAGE_HARBOR]


def _save_failures_for_inspection(
    failures: Iterable[Path],
    keep_failed_dir: Optional[str],
    stage_name: str,
    fallback_source_root: Optional[Path] = None,
) -> None:
    failures = list(failures)
    if not keep_failed_dir or not failures:
        return
    dest_root = Path(keep_failed_dir) / stage_name
    dest_root.mkdir(parents=True, exist_ok=True)
    for task_dir in failures:
        source = task_dir
        if fallback_source_root is not None:
            candidate = fallback_source_root / task_dir.name
            if candidate.exists():
                source = candidate
        shutil.copytree(source, dest_root / task_dir.name, dirs_exist_ok=True)
    CONSOLE.print(f"[{stage_name}] Copied failures to {dest_root}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    stages = _resolve_stages(args)
    extract_dir = Path(args.extract_dir) if args.extract_dir else Path(tempfile.mkdtemp(prefix="tasks_extracted_"))

    CONSOLE.print(f"[bold cyan][extract][/bold cyan] Repo: {args.repo_id} rev={args.revision or '<latest>'}")
    CONSOLE.print(f"[bold cyan][extract][/bold cyan] Target directory: {extract_dir}")
    CONSOLE.print(f"[bold cyan][stages][/bold cyan] Running: {', '.join(stages) if stages else '(none — upload-only)'}")
    _extract_hf_dataset(args.repo_id, args.revision, extract_dir)

    tasks = _discover_tasks(extract_dir)
    CONSOLE.print(f"[cyan][validate][/cyan] Found {len(tasks)} candidate tasks")

    if args.sample_size is not None and args.sample_size > 0 and args.sample_size < len(tasks):
        rng = random.Random(args.sample_seed)
        tasks = sorted(rng.sample(tasks, args.sample_size), key=lambda p: p.name)
        seed_note = f" (seed={args.sample_seed})" if args.sample_seed is not None else ""
        CONSOLE.print(f"[cyan][validate][/cyan] Random sample: {len(tasks)} tasks{seed_note}")

    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]
        CONSOLE.print(f"[cyan][validate][/cyan] After --limit: {len(tasks)} tasks")

    if not tasks:
        CONSOLE.print("[yellow][validate] No tasks found; exiting[/yellow]")
        return

    # Stage 1 (Daytona) is special: it works directly on extract_dir, then we
    # copy survivors into staged. If stage 1 is not requested, we copy all
    # selected tasks into staged so downstream stages and upload have a single
    # working directory.
    if STAGE_DAYTONA in stages:
        successes, failures = _run_daytona_validation(
            tasks,
            base_dir=extract_dir,
            cpu=args.cpu,
            memory_gb=args.memory_gb,
            disk_gb=args.disk_gb,
            gpu=args.gpu,
            timeout=args.timeout,
            processes=max(1, int(args.processes)),
        )
        CONSOLE.print(
            f"[bold][daytona][/bold] Success: [green]{len(successes)}[/green]  Fail: [red]{len(failures)}[/red]"
        )
        _save_failures_for_inspection(failures, args.keep_failed_dir, "daytona")
        survivors = successes
    else:
        survivors = list(tasks)

    if not survivors:
        CONSOLE.print("[yellow][daytona] No survivors; exiting[/yellow]")
        return

    staged = Path(tempfile.mkdtemp(prefix="tasks_success_"))
    _copy_successes(survivors, staged)

    if STAGE_HARBOR in stages:
        successes, failures, harbor_job_dir, harbor_stats = _run_harbor_smoke_test(
            staged,
            agent_timeout=args.timeout,
            verifier_timeout=args.timeout,
            cpu=args.cpu,
            memory_gb=args.memory_gb,
            disk_gb=args.disk_gb,
            concurrency=args.harbor_concurrency,
            filter_successful=bool(args.filter_successful),
        )
        mode_label = "filter" if args.filter_successful else "smoke test"
        n = max(1, harbor_stats["total"])
        infra_rate = harbor_stats["infra_ok"] / n
        solved_rate = harbor_stats["solved"] / n
        CONSOLE.print(
            f"[bold][harbor][/bold] {mode_label}: "
            f"Success: [green]{len(successes)}[/green]  Fail: [red]{len(failures)}[/red]  "
            f"infra_ok: {harbor_stats['infra_ok']}/{harbor_stats['total']} ({infra_rate:.3f})  "
            f"solved: {harbor_stats['solved']}/{harbor_stats['total']} ({solved_rate:.3f})"
        )
        _save_failures_for_inspection(failures, args.keep_failed_dir, "harbor", fallback_source_root=extract_dir)
        for task_dir in failures:
            shutil.rmtree(task_dir, ignore_errors=True)
        if harbor_job_dir is not None:
            # Always retain the Harbor job dir (per-trial agent trajectories + verifier outputs).
            # batch_validate_from_md.sh greps this path from the log to sync the traces into
            # the per-dataset output dir. Never auto-delete on success — those traces are the
            # whole point of running the smoke test.
            CONSOLE.print(
                f"[harbor] Harbor job logs retained at [underline]{harbor_job_dir}[/underline]"
            )
        if not successes:
            CONSOLE.print("[yellow][harbor] No tasks passed Harbor stage; exiting[/yellow]")
            return

    if STAGE_ORACLE in stages:
        successes, failures, missing, oracle_job_dir = _run_oracle_solution_check(
            staged,
            agent_timeout=args.timeout,
            verifier_timeout=args.timeout,
            cpu=args.cpu,
            memory_gb=args.memory_gb,
            disk_gb=args.disk_gb,
            concurrency=args.harbor_concurrency,
        )
        CONSOLE.print(
            "[bold][oracle][/bold] Success: "
            f"[green]{len(successes)}[/green]  "
            f"Fail: [red]{len(failures)}[/red]  "
            f"Missing: [yellow]{len(missing)}[/yellow]"
        )
        all_failures = list(dict.fromkeys(failures + missing))
        _save_failures_for_inspection(all_failures, args.keep_failed_dir, "oracle", fallback_source_root=extract_dir)
        for task_dir in all_failures:
            shutil.rmtree(task_dir, ignore_errors=True)
        if oracle_job_dir is not None:
            CONSOLE.print(
                f"[oracle] Harbor job logs retained at [underline]{oracle_job_dir}[/underline]"
            )
        if missing:
            missing_names = ", ".join(sorted(path.name for path in missing))
            CONSOLE.print(
                f"[yellow][oracle] Missing solution/solve.sh for: {missing_names}[/yellow]"
            )
        if not successes:
            CONSOLE.print("[yellow][oracle] No tasks passed oracle validation; exiting[/yellow]")
            return

    final_tasks = _discover_tasks(staged)
    if not final_tasks:
        CONSOLE.print("[yellow][upload] No tasks remain after staging; exiting[/yellow]")
        return

    if args.skip_upload:
        CONSOLE.print(
            f"[bold magenta][upload][/bold magenta] Skipped (--skip_upload). "
            f"{len(final_tasks)} staged tasks at [underline]{staged}[/underline]"
        )
    else:
        target_repo = args.target_repo or args.repo_id
        token = args.token or os.environ.get("HF_TOKEN")
        CONSOLE.print(f"[bold magenta][upload][/bold magenta] Uploading {len(final_tasks)} tasks to {target_repo}")
        upload_tasks_to_hf(str(staged), target_repo, private=bool(args.private), token=token)
        CONSOLE.print(f"[bold magenta][upload][/bold magenta] Done: https://huggingface.co/datasets/{target_repo}")

    if args.final_output_dir:
        try:
            staged = _relocate_staged_output(staged, Path(args.final_output_dir))
        except Exception as exc:  # pragma: no cover - filesystem dependent
            CONSOLE.print(
                f"[red][output] Failed to move staged successes to {args.final_output_dir}: {exc}[/red]"
            )
            raise
        CONSOLE.print(
            f"[bold magenta][output][/bold magenta] Moved staged successes to {staged}"
        )


if __name__ == "__main__":
    main()
