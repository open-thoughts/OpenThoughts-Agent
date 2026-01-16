from __future__ import annotations

"""
Composable entrypoint to run a Harbor dataset (as you would via the Harbor CLI) and return a
Hugging Face Dataset of extracted traces, without invoking subprocesses.

Assumes `harbor` is importable (e.g., installed via `pip install -e .`).
"""

import asyncio
import hashlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional
import re
import importlib
import inspect
import os
from pydantic import BaseModel

from harbor.job import Job
from harbor.models.agent.name import AgentName
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import JobConfig, LocalDatasetConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, VerifierConfig
from harbor.utils.traces_utils import export_traces as _export_traces
from data.gcs_cache import gcs_cache
from scripts.harbor.job_config_utils import (
    ensure_trailing_dataset,
    set_job_metadata,
    set_local_dataset,
)

logger = logging.getLogger(__name__)


def _compute_dataset_checksum(dataset_path: Path) -> str:
    """
    Compute a checksum for a dataset directory based on all task files.
    This creates a unique identifier for the dataset contents.

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        SHA256 hex digest of the dataset contents
    """
    hasher = hashlib.sha256()

    # Get all files in sorted order for deterministic hashing
    all_files = sorted(dataset_path.rglob('*'))

    for file_path in all_files:
        if file_path.is_file():
            # Add relative path to hash
            rel_path = file_path.relative_to(dataset_path)
            hasher.update(str(rel_path).encode())

            # Add file contents to hash
            try:
                with open(file_path, 'rb') as f:
                    # Read in chunks to handle large files
                    while chunk := f.read(8192):
                        hasher.update(chunk)
            except Exception:
                # For files we can't read, just hash the path
                pass

    return hasher.hexdigest()


def _find_job_by_checksum(jobs_dir: Path, checksum: str) -> Optional[str]:
    """
    Search the jobs directory for any job that has the same dataset checksum.

    Args:
        jobs_dir: Path to the jobs directory
        checksum: The dataset checksum to search for

    Returns:
        Job name (directory name) if found, None otherwise
    """
    if not jobs_dir.exists():
        return None

    for job_folder in jobs_dir.iterdir():
        try:
            if not job_folder.is_dir():
                continue

            checksum_file = job_folder / ".dataset_checksum"
            if checksum_file.exists():
                
                existing_checksum = checksum_file.read_text().strip()
                if existing_checksum == checksum:
                    return job_folder.name
        except Exception as e:
            print(f"Job issue {e}")
            continue

    return None


def _save_dataset_checksum(job_dir: Path, checksum: str) -> None:
    """
    Save the dataset checksum to the job directory.

    Args:
        job_dir: Path to the job directory
        checksum: The dataset checksum to save
    """
    checksum_file = job_dir / ".dataset_checksum"
    checksum_file.write_text(checksum)


def _cleanup_empty_agent_trials(job_dir: Path) -> int:
    """
    Clean up trial directories that have empty agent folders or missing config files.
    These trials likely failed early and should be retried.

    Args:
        job_dir: Path to the job directory

    Returns:
        Number of trial directories removed
    """
    if not job_dir.exists():
        return 0

    removed_count = 0
    for trial_dir in job_dir.iterdir():
        if not trial_dir.is_dir():
            continue

        # Skip special files like .dataset_checksum
        if trial_dir.name.startswith('.'):
            continue

        # Check if trial config exists - if not, this trial is corrupted
        config_path = trial_dir / "config.json"
        if not config_path.exists():
            logger.info(f"Removing trial with missing config.json: {trial_dir.name}")
            try:
                shutil.rmtree(trial_dir)
                removed_count += 1
                continue
            except Exception as e:
                logger.warning(f"Failed to remove {trial_dir.name}: {e}")
                continue

        agent_dir = trial_dir / "agent"

        # Check if agent directory exists and is empty
        if agent_dir.exists() and agent_dir.is_dir():
            # Check if directory is empty (no files, only empty subdirectories if any)
            has_content = False
            try:
                for item in agent_dir.rglob('*'):
                    if item.is_file():
                        has_content = True
                        break
            except Exception:
                # If we can't read the directory, assume it has content
                has_content = True

            if not has_content:
                logger.info(f"Removing trial with empty agent directory: {trial_dir.name}")
                try:
                    shutil.rmtree(trial_dir)
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {trial_dir.name}: {e}")

    return removed_count

def _shutdown_litellm(timeout: float = 5.0) -> None:
    """Best-effort LiteLLM shutdown to avoid dangling tasks/sessions."""
    try:
        litellm = importlib.import_module("litellm")
    except ModuleNotFoundError:
        return

    shutdown_fn = getattr(litellm, "shutdown", None)
    if shutdown_fn is None:
        return

    try:
        result = shutdown_fn(timeout=timeout) if "timeout" in inspect.signature(shutdown_fn).parameters else shutdown_fn()  # type: ignore[arg-type]
        if inspect.isawaitable(result):
            asyncio.run(result)
    except Exception as exc:  # pragma: no cover - defensive cleanup
        print(f"[run_dataset_to_traces] Warning: LiteLLM shutdown raised {exc!r}")

def _is_pydantic_model(x) -> bool:
    return isinstance(x, BaseModel)


def _merge_verifier_timeout_into_config(config: JobConfig, verifier_timeout_sec: float | None) -> None:
    if verifier_timeout_sec is None:
        return
    v = getattr(config, "verifier", None)
    if v is None:
        config.verifier = VerifierConfig(override_timeout_sec=verifier_timeout_sec)
        return
    if _is_pydantic_model(v):
        config.verifier = v.model_copy(update={"override_timeout_sec": verifier_timeout_sec})
    elif isinstance(v, dict):
        nv = dict(v); nv["override_timeout_sec"] = verifier_timeout_sec
        config.verifier = nv
    else:
        raise TypeError(f"Unsupported verifier type on JobConfig: {type(v)}")


def _merge_disable_verification_into_config(
    config: JobConfig, disable_verification: bool
) -> None:
    if not disable_verification:
        return

    verifier = getattr(config, "verifier", None)
    if verifier is None:
        config.verifier = VerifierConfig(disable=True)
        return

    if _is_pydantic_model(verifier):
        config.verifier = verifier.model_copy(update={"disable": True})
        return

    if isinstance(verifier, dict):
        updated = dict(verifier)
        updated["disable"] = True
        config.verifier = updated
        return

    if hasattr(verifier, "disable"):
        verifier.disable = True
        return

    raise TypeError(f"Unsupported verifier type on JobConfig: {type(verifier)}")

def _merge_agent_timeout_into_config(config: JobConfig, agent_timeout_sec: float | None) -> None:
    if agent_timeout_sec is None:
        return
    agents = getattr(config, "agents", None)
    if not agents:
        config.agents = [AgentConfig(override_timeout_sec=agent_timeout_sec)]
        return
    merged = []
    for a in agents:
        if _is_pydantic_model(a):
            merged.append(a.model_copy(update={"override_timeout_sec": agent_timeout_sec}))
        elif isinstance(a, dict):
            na = dict(a); na["override_timeout_sec"] = agent_timeout_sec
            merged.append(na)
        else:
            raise TypeError(f"Unsupported agent type on JobConfig.agents: {type(a)}")
    config.agents = merged

def force_resume(
    existing_job_dir: str,
    recursive: bool | str = True,
    episodes: str = "last",
    to_sharegpt: bool = False,
    repo_id: Optional[str] = None,
    push: bool = False,
    export_filter: Optional[str] = None,  # success|failure|None
    verbose: bool = False,
) -> Dataset:
    existing_job_dir = Path(existing_job_dir)
    config = JobConfig.model_validate_json((existing_job_dir / "config.json").read_text())
    job = Job(config)

    try:
        asyncio.run(job.run())
    finally:
        _shutdown_litellm(timeout=5.0)

    ds = _export_traces(
        root=job_dir,
        recursive=bool(recursive),
        episodes=episodes,
        to_sharegpt=to_sharegpt,
        repo_id=repo_id if push else None,
        push=push,
        verbose=verbose,
        success_filter=export_filter,
        include_instruction=True,
        include_verifier_output=True,
    )
    return ds

@gcs_cache()
def run_dataset_to_traces(
    dataset_path: Path | str | None = None,
    *,
    job_config: JobConfig | None = None,
    job_name: Optional[str] = None,
    jobs_dir: Path | str = Path("jobs"),
    recursive: bool | str = True,
    # Trials/execution
    n_attempts: int = 1,
    timeout_multiplier: float = 1.0,
    n_concurrent: int = 4,
    quiet: bool = True,
    log_progress: bool | str = False,
    # Agent
    agent_name: AgentName | None = None,
    agent_import_path: Optional[str] = None,
    model_name: Optional[str] = None,
    agent_kwargs: Optional[dict[str, str]] = None,
    # Environment
    env_type: EnvironmentType = EnvironmentType.DAYTONA,
    force_build: bool | str = True,
    delete_env: bool | str = True,
    env_kwargs: Optional[dict[str, str]] = None,
    override_cpus: int | None = None,
    override_memory_mb: int | None = None,
    override_storage_mb: int | None = None,
    # Export
    episodes: str = "last",
    to_sharegpt: bool = False,
    push: bool = False,
    repo_id: Optional[str] = None,
    export_filter: Optional[str] = None,  # success|failure|None
    verbose: bool = False,
    agent_timeout_sec: float | None = None,
    verifier_timeout_sec: float | None = None,
    disable_verification: bool = False,
):
    """Run a dataset of tasks and return a HF Dataset of episode traces.

    Prefer supplying ``job_config`` (a pre-validated Harbor ``JobConfig``) so runtime
    settings like agents, environments, and orchestration are defined declaratively.
    Legacy keyword overrides remain for compatibility but are gradually being removed.
    """
    config_from_spec: JobConfig | None = None
    if job_config is not None:
        config_from_spec = job_config.model_copy(deep=True)

        config_dataset_path = ensure_trailing_dataset(config_from_spec)
        if dataset_path is None:
            dataset_path = config_dataset_path
        else:
            dataset_path = Path(dataset_path).expanduser().resolve()
            if dataset_path.resolve() != config_dataset_path.resolve():
                raise ValueError(
                    "dataset_path argument does not match dataset path in Harbor job config"
                )

        config_from_spec = set_local_dataset(config_from_spec, dataset_path)
        config_from_spec = set_job_metadata(
            config_from_spec,
            job_name=job_name or config_from_spec.job_name,
            jobs_dir=config_from_spec.jobs_dir,
        )
        job_name = config_from_spec.job_name
        jobs_dir = Path(config_from_spec.jobs_dir)

        orchestrator_cfg = config_from_spec.orchestrator
        n_attempts = config_from_spec.n_attempts
        timeout_multiplier = config_from_spec.timeout_multiplier
        n_concurrent = orchestrator_cfg.n_concurrent_trials
        quiet = orchestrator_cfg.quiet

        if config_from_spec.agents:
            agent_spec = config_from_spec.agents[0]
            agent_name = agent_spec.name
            agent_import_path = agent_spec.import_path
            model_name = agent_spec.model_name
            agent_kwargs = dict(agent_spec.kwargs or {})
            override_agent_timeout = agent_spec.override_timeout_sec
        else:
            agent_spec = AgentConfig()
            agent_name = None
            agent_import_path = None
            model_name = None
            agent_kwargs = {}
            override_agent_timeout = None

        if agent_timeout_sec is None:
            agent_timeout_sec = override_agent_timeout

        env_cfg = config_from_spec.environment
        env_type = env_cfg.type
        force_build = env_cfg.force_build
        delete_env = env_cfg.delete
        env_kwargs = dict(env_cfg.kwargs or {})
        override_cpus = env_cfg.override_cpus
        override_memory_mb = env_cfg.override_memory_mb
        override_storage_mb = env_cfg.override_storage_mb

        verifier_timeout_sec = verifier_timeout_sec or getattr(
            config_from_spec.verifier, "override_timeout_sec", None
        )
        if disable_verification:
            config_from_spec.verifier.disable = True
    else:
        if dataset_path is None:
            raise ValueError("dataset_path is required when job_config is not provided")
        dataset_path = Path(dataset_path)
        jobs_dir = Path(jobs_dir)

    if isinstance(force_build, str):
        force_build = force_build.lower() not in {"", "false", "0", "no"}
    if isinstance(delete_env, str):
        delete_env = delete_env.lower() not in {"", "false", "0", "no"}
    if isinstance(recursive, str):
        recursive = recursive.lower() not in {"", "false", "0", "no"}
    if isinstance(log_progress, str):
        log_progress = log_progress.lower() not in {"", "false", "0", "no"}

    if disable_verification:
        logger.info("[run_dataset_to_traces] Verification disabled; Harbor tests will be skipped.")

    # Compute checksum of the dataset directory
    logger.info(f"Computing checksum for dataset: {dataset_path}")
    dataset_checksum = _compute_dataset_checksum(dataset_path)
    logger.info(f"Dataset checksum: {dataset_checksum[:16]}...")

    # Check if a job with this checksum already exists
    try:
        existing_job_name = _find_job_by_checksum(jobs_dir, dataset_checksum)
    except:
        existing_job_name = None
        
    if existing_job_name and not job_name:
        existing_job_dir = jobs_dir / existing_job_name
        config_path = existing_job_dir / "config.json"
        with open(config_path, 'r') as file:
            data = json.load(file)
        
        if data['agents'][0]['model_name'] != model_name or data['agents'][0]['kwargs'] != agent_kwargs or data['agents'][0]['name'] != agent_name:
            logger.warning(f"Existing job '{existing_job_name}' has different model name, kwargs, or name")
            logger.warning(f" → Creating new job")
            existing_job_name = None    
        else:
            logger.info(f"✓ Found existing job with matching checksum: {existing_job_name}")
            logger.info(f"→ Resuming job: {existing_job_name}")
            job_name = existing_job_name

            # Clean up trials with empty agent directories before resuming
            existing_job_dir = jobs_dir / existing_job_name
            removed_count = _cleanup_empty_agent_trials(existing_job_dir)
            if removed_count > 0:
                logger.info(f"  Cleaned up {removed_count} trial(s) with empty agent directories")
    elif existing_job_name and job_name and existing_job_name != job_name:
        logger.warning(f"Existing job '{existing_job_name}' has the same dataset checksum")
        logger.warning(f"  But you specified job_name='{job_name}'. Proceeding with your specified name.")
    else:
        if not existing_job_name:
            logger.info(f"✗ No existing job found with matching checksum")
            logger.info(f"→ Creating new job")


    if existing_job_name:
        existing_job_dir = jobs_dir / existing_job_name
        config = JobConfig.model_validate_json((existing_job_dir / "config.json").read_text())

        # Update orchestrator concurrency if it changed to keep Harbor validation happy
        desired_n_concurrent = int(n_concurrent)
        orchestrator_cfg = getattr(config, "orchestrator", None)
        old_n_concurrent = None
        if orchestrator_cfg is not None:
            old_n_concurrent = getattr(orchestrator_cfg, "n_concurrent_trials", None)
            if old_n_concurrent != desired_n_concurrent:
                logger.info(
                    f"  Updating n_concurrent_trials: {old_n_concurrent} → {desired_n_concurrent}"
                )
                orchestrator_cfg.n_concurrent_trials = desired_n_concurrent
                (existing_job_dir / "config.json").write_text(config.model_dump_json(indent=4))
        else:
            logger.warning("Existing Harbor config missing orchestrator settings; leaving unchanged.")

        _merge_agent_timeout_into_config(config, agent_timeout_sec)
        _merge_verifier_timeout_into_config(config, verifier_timeout_sec)
        _merge_disable_verification_into_config(config, disable_verification)
        job = Job(config)
    else:
        if config_from_spec is not None:
            config = config_from_spec
        else:
            if not force_build and env_type == EnvironmentType.DAYTONA:
                raise RuntimeError(
                    "Daytona environments require pre-built Harbor environments. "
                    "Pass force_build=True (default) or pre-build via Harbor tooling."
                )

            env_label = env_type.value if isinstance(env_type, EnvironmentType) else str(env_type)
            print(
                f"[run_dataset_to_traces] env={env_label} force_build={force_build} delete_env={delete_env}"
            )

            env_kwargs_dict: dict[str, str] = dict(env_kwargs or {})

            resolved_override_cpus = override_cpus
            if resolved_override_cpus is None:
                candidate = env_kwargs_dict.pop("override_cpus", None) or env_kwargs_dict.pop("cpu", None)
                if candidate is not None:
                    resolved_override_cpus = int(candidate)
            if resolved_override_cpus is None:
                env_val = os.environ.get("SANDBOX_CPU")
                if env_val:
                    try:
                        resolved_override_cpus = int(env_val)
                    except ValueError:
                        logger.warning(f"[run_dataset_to_traces] Ignoring invalid SANDBOX_CPU: {env_val!r}")

            resolved_override_memory_mb = override_memory_mb
            if resolved_override_memory_mb is None:
                candidate = env_kwargs_dict.pop("override_memory_mb", None)
                if candidate is not None:
                    resolved_override_memory_mb = int(candidate)
            if resolved_override_memory_mb is None:
                candidate = env_kwargs_dict.pop("memory_mb", None)
                if candidate is not None:
                    resolved_override_memory_mb = int(candidate)
            if resolved_override_memory_mb is None:
                candidate = env_kwargs_dict.pop("memory_gb", None)
                if candidate is not None:
                    resolved_override_memory_mb = int(candidate) * 1024
            if resolved_override_memory_mb is None:
                env_val = os.environ.get("SANDBOX_MEMORY_GB")
                if env_val:
                    try:
                        resolved_override_memory_mb = int(env_val) * 1024
                    except ValueError:
                        logger.warning(f"[run_dataset_to_traces] Ignoring invalid SANDBOX_MEMORY_GB: {env_val!r}")

            resolved_override_storage_mb = override_storage_mb
            if resolved_override_storage_mb is None:
                candidate = env_kwargs_dict.pop("override_storage_mb", None)
                if candidate is not None:
                    resolved_override_storage_mb = int(candidate)
            if resolved_override_storage_mb is None:
                candidate = env_kwargs_dict.pop("storage_mb", None)
                if candidate is not None:
                    resolved_override_storage_mb = int(candidate)
            if resolved_override_storage_mb is None:
                candidate = env_kwargs_dict.pop("storage_gb", None) or env_kwargs_dict.pop("disk_gb", None)
                if candidate is not None:
                    resolved_override_storage_mb = int(candidate) * 1024
            if resolved_override_storage_mb is None:
                env_val = os.environ.get("SANDBOX_DISK_GB")
                if env_val:
                    try:
                        resolved_override_storage_mb = int(env_val) * 1024
                    except ValueError:
                        logger.warning(f"[run_dataset_to_traces] Ignoring invalid SANDBOX_DISK_GB: {env_val!r}")

            # Build a JobConfig programmatically
            config = JobConfig()
            if job_name:
                config.job_name = job_name
            config.jobs_dir = Path(jobs_dir)
            config.n_attempts = int(n_attempts)
            config.timeout_multiplier = float(timeout_multiplier)

            # Orchestrator
            config.orchestrator.n_concurrent_trials = int(n_concurrent)
            config.orchestrator.quiet = bool(quiet)

            # Agents
            if agent_name is not None or agent_import_path is not None or model_name is not None:
                config.agents = [
                    AgentConfig(
                        name=agent_name,
                        import_path=agent_import_path,
                        model_name=model_name,
                        kwargs=agent_kwargs or {},
                    )
                ]

            # Dataset of tasks
            config.datasets = [LocalDatasetConfig(path=dataset_path)]
            config.tasks = []

            # Environment
            config.environment = EnvironmentConfig(
                type=env_type,
                force_build=bool(force_build),
                delete=bool(delete_env),
                override_cpus=resolved_override_cpus,
                override_memory_mb=resolved_override_memory_mb,
                override_storage_mb=resolved_override_storage_mb,
                kwargs=env_kwargs_dict,
            )

        _merge_agent_timeout_into_config(config, agent_timeout_sec)
        _merge_verifier_timeout_into_config(config, verifier_timeout_sec)
        _merge_disable_verification_into_config(config, disable_verification)

        job = Job(config)
        job.job_dir.mkdir(parents=True, exist_ok=True)
        _save_dataset_checksum(job.job_dir, dataset_checksum)

    try:
        asyncio.run(job.run())
    finally:
        _shutdown_litellm(timeout=5.0)
    # Export traces from the produced job directory
    job_dir = job.job_dir  # derived from config; stable
    if export_filter not in (None, "success", "failure"):
        raise ValueError("export_filter must be one of: None, 'success', 'failure'")

    ds = _export_traces(
        root=job_dir,
        recursive=bool(recursive),
        episodes=episodes,
        to_sharegpt=to_sharegpt,
        repo_id=repo_id if push else None,
        push=push,
        verbose=verbose,
        success_filter=export_filter,
        include_instruction=True,
        include_verifier_output=True,
    )
    return _finalize_trace_dataset(ds)

def only_export_traces(
    job_dir: Path | str,
    *,
    episodes: str = "last",
    to_sharegpt: bool = False,
    repo_id: Optional[str] = None,
    recursive: bool = False,
    push: bool = False,
    verbose: bool = False,
    success_filter: Optional[str] = None,
):
    """Export traces from a job directory."""
    ds = _export_traces(
        root=job_dir,
        recursive=recursive,
        episodes=episodes,
        to_sharegpt=to_sharegpt,
        repo_id=repo_id if push else None,
        push=push,
        verbose=verbose,
        success_filter=success_filter,
        include_instruction=True,
        include_verifier_output=True,
    )
    return _finalize_trace_dataset(ds)

_BASH_JOB_CONTROL_WARNING = (
    "bash: initialize_job_control: no job control in background: Bad file descriptor"
)


def _sanitize_bash_warnings(dataset):
    """Strip bash job control warnings from trace datasets to avoid agent confusion."""
    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        return dataset

    def _clean_value(value):
        if isinstance(value, str):
            cleaned = value.replace(f"{_BASH_JOB_CONTROL_WARNING}\n", "")
            cleaned = cleaned.replace(f"\n{_BASH_JOB_CONTROL_WARNING}", "")
            return cleaned.replace(_BASH_JOB_CONTROL_WARNING, "")
        if isinstance(value, list):
            return [_clean_value(item) for item in value]
        if isinstance(value, dict):
            return {key: _clean_value(val) for key, val in value.items()}
        return value

    def _sanitize_record(record):
        return {key: _clean_value(val) for key, val in record.items()}

    if isinstance(dataset, DatasetDict):
        return DatasetDict({k: _sanitize_bash_warnings(v) for k, v in dataset.items()})
    if isinstance(dataset, Dataset):
        return dataset.map(_sanitize_record, load_from_cache_file=False)
    return dataset


def _finalize_trace_dataset(dataset):
    """Apply final cleanup/formatting before returning a dataset."""
    dataset = _sanitize_bash_warnings(dataset)
    dataset = _sanitize_surrogates(dataset)
    dataset = _ensure_sharegpt_conversations(dataset)
    return dataset


_SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")


def _sanitize_surrogates(dataset):
    """Replace Unicode surrogate code points with spaces to keep PyArrow happy."""
    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        return dataset

    if isinstance(dataset, DatasetDict):
        return DatasetDict({k: v.map(_strip_surrogates_from_record, load_from_cache_file=False) for k, v in dataset.items()})
    if isinstance(dataset, Dataset):
        return dataset.map(_strip_surrogates_from_record, load_from_cache_file=False)
    return dataset


def _strip_surrogates_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _strip_surrogates(value) for key, value in record.items()}


def _strip_surrogates(value: Any) -> Any:
    if isinstance(value, str):
        return _SURROGATE_PATTERN.sub(" ", value)
    if isinstance(value, list):
        return [_strip_surrogates(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_surrogates(val) for key, val in value.items()}
    return value

def _ensure_sharegpt_conversations(dataset):
    """Guarantee conversation columns conform to ShareGPT expectations."""
    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        return dataset

    def _map_record(record):
        conversations = record.get("conversations")
        if not isinstance(conversations, list):
            return record
        return {"conversations": _squash_system_turns(conversations)}

    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {
                split: ds.map(_map_record, load_from_cache_file=False)
                for split, ds in dataset.items()
            }
        )
    if isinstance(dataset, Dataset):
        return dataset.map(_map_record, load_from_cache_file=False)
    return dataset


def _squash_system_turns(conversations):
    """
    Merge consecutive system messages into user turns so the final transcript alternates
    user/assistant as expected by ShareGPT loaders.
    """
    if not isinstance(conversations, list):
        return conversations

    cleaned: list[dict[str, Any]] = []
    system_buffer: list[str] = []

    def _drain_buffer() -> Optional[str]:
        nonlocal system_buffer
        if not system_buffer:
            return None
        text = "\n\n".join(piece for piece in system_buffer if piece)
        system_buffer = []
        return text

    def _merge_buffer_with_user(content: str) -> str:
        prefix = _drain_buffer()
        if not prefix:
            return content
        if content:
            return f"{prefix}\n\n{content}"
        return prefix

    for message in conversations:
        role = message.get("role")
        content = message.get("content") or ""

        if role == "system":
            system_buffer.append(content)
            continue

        if role == "user":
            merged_content = _merge_buffer_with_user(content)
            cleaned.append({**message, "role": "user", "content": merged_content})
            continue

        buffered = _drain_buffer()
        if buffered:
            cleaned.append({"role": "user", "content": buffered})
        cleaned.append(dict(message))

    leftover = _drain_buffer()
    if leftover:
        if cleaned and cleaned[-1].get("role") == "user":
            existing = cleaned[-1].get("content") or ""
            cleaned[-1]["content"] = f"{existing}\n\n{leftover}" if existing else leftover
        else:
            cleaned.append({"role": "user", "content": leftover})

    if cleaned and cleaned[0].get("role") != "user":
        cleaned.insert(0, {"role": "user", "content": ""})

    return cleaned

def _compute_hdf5_checksum(hdf5_path: Path) -> str:
    """
    Compute a checksum for an HDF5 file - much faster than directory checksums.

    Args:
        hdf5_path: Path to the HDF5 file

    Returns:
        SHA256 hex digest of the file contents
    """
    hasher = hashlib.sha256()

    with open(hdf5_path, 'rb') as f:
        # Read in chunks to handle large files
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


@gcs_cache()
def run_dataset_to_traces_hdf5(
    hdf5_path: Path | str,
    *,
    job_name: Optional[str] = None,
    jobs_dir: Path | str = Path("jobs"),
    recursive: bool | str = True,
    # Trials/execution
    n_attempts: int = 1,
    timeout_multiplier: float = 1.0,
    n_concurrent: int = 4,
    quiet: bool = True,
    log_progress: bool | str = False,
    # Agent
    agent_name: AgentName | None = None,
    agent_import_path: Optional[str] = None,
    model_name: Optional[str] = None,
    agent_kwargs: Optional[dict[str, str]] = None,
    # Environment
    env_type: EnvironmentType = EnvironmentType.DAYTONA,
    force_build: bool | str = True,
    delete_env: bool | str = True,
    env_kwargs: Optional[dict[str, str]] = None,
    # Export
    episodes: str = "last",
    to_sharegpt: bool = False,
    push: bool = False,
    repo_id: Optional[str] = None,
    export_filter: Optional[str] = None,  # success|failure|None
    verbose: bool = False,
):
    """Run a dataset from HDF5 format and return a HF Dataset of episode traces.

    This is optimized for large datasets:
    1. Computes checksum on the HDF5 file (much faster than directory tree)
    2. Only extracts to directory right before running harbor
    3. Reuses existing job runs based on HDF5 checksum

    Parameters mirror run_dataset_to_traces but accept an HDF5 file instead of directory.
    """
    hdf5_path = Path(hdf5_path)
    jobs_dir = Path(jobs_dir)

    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    # Compute checksum of the HDF5 file (much faster than directory)
    logger.info(f"Computing checksum for HDF5 file: {hdf5_path}")
    dataset_checksum = _compute_hdf5_checksum(hdf5_path)
    logger.info(f"HDF5 checksum: {dataset_checksum[:16]}...")

    # Check if a job with this checksum already exists
    try:
        existing_job_name = _find_job_by_checksum(jobs_dir, dataset_checksum)
    except Exception as e:
        logger.warning(f"Error finding job by checksum: {e}")
        existing_job_name = None

    if existing_job_name and not job_name:
        existing_job_dir = jobs_dir / existing_job_name
        config_path = existing_job_dir / "config.json"
        with open(config_path, 'r') as file:
            data = json.load(file)

        if data['agents'][0]['model_name'] != model_name or data['agents'][0]['kwargs'] != agent_kwargs or data['agents'][0]['name'] != agent_name:
            logger.warning(f"Existing job '{existing_job_name}' has different model name, kwargs, or name")
            logger.warning(f" → Creating new job")
            existing_job_name = None
        else:
            logger.info(f"✓ Found existing job with matching checksum: {existing_job_name}")
            logger.info(f"→ Resuming job: {existing_job_name}")
            job_name = existing_job_name

            # Clean up trials with empty agent directories before resuming
            existing_job_dir = jobs_dir / existing_job_name
            removed_count = _cleanup_empty_agent_trials(existing_job_dir)
            if removed_count > 0:
                logger.info(f"  Cleaned up {removed_count} trial(s) with empty agent directories")
    elif existing_job_name and job_name and existing_job_name != job_name:
        logger.warning(f"Existing job '{existing_job_name}' has the same dataset checksum")
        logger.warning(f"  But you specified job_name='{job_name}'. Proceeding with your specified name.")
    else:
        if not existing_job_name:
            logger.info(f"✗ No existing job found with matching checksum")
            logger.info(f"→ Creating new job")

    if existing_job_name:
        config = JobConfig.model_validate_json((existing_job_dir / "config.json").read_text())

        # Update config with new parameters (only orchestrator settings, agent config must match)
        old_n_concurrent = config.orchestrator.n_concurrent_trials
        config.orchestrator.n_concurrent_trials = int(n_concurrent)
        if old_n_concurrent != n_concurrent:
            logger.info(f"  Updating n_concurrent_trials: {old_n_concurrent} → {n_concurrent}")
            # Save updated config so Harbor's validation passes
            (existing_job_dir / "config.json").write_text(config.model_dump_json(indent=4))

        job = Job(config)
    else:
        # Extract HDF5 to persistent directory for harbor to use (enables resuming)
        # Use a deterministic path based on HDF5 file location
        logger.info(f"Extracting HDF5 to persistent directory for job execution...")
        from data.commons import extract_hdf5_to_task_paths

        # Create persistent extraction directory next to the HDF5 file
        persistent_extract_dir = hdf5_path.parent / f"{hdf5_path.stem}_extracted"

        # Always extract when creating a new job (clean up old extraction if it exists)
        if persistent_extract_dir.exists():
            logger.info(f"Cleaning up old extraction directory: {persistent_extract_dir}")
            shutil.rmtree(persistent_extract_dir)

        logger.info(f"Extracting to: {persistent_extract_dir}")
        extracted_paths = extract_hdf5_to_task_paths(str(hdf5_path), output_dir=str(persistent_extract_dir))
        logger.info(f"Extracted {len(extracted_paths)} tasks")

        dataset_path = persistent_extract_dir

        if isinstance(force_build, str):
            force_build = force_build.lower() not in {"", "false", "0", "no"}
        if isinstance(delete_env, str):
            delete_env = delete_env.lower() not in {"", "false", "0", "no"}
        if isinstance(recursive, str):
            recursive = recursive.lower() not in {"", "false", "0", "no"}
        if isinstance(log_progress, str):
            log_progress = log_progress.lower() not in {"", "false", "0", "no"}

        if not force_build and env_type == EnvironmentType.DAYTONA:
            raise RuntimeError(
                "Daytona environments require pre-built sandboxes. "
                "Pass force_build=True (default) or pre-build via sandboxes tooling."
            )

        env_label = env_type.value if isinstance(env_type, EnvironmentType) else str(env_type)
        print(
            f"[run_dataset_to_traces_hdf5] env={env_label} force_build={force_build} delete_env={delete_env}"
        )

        # Build a JobConfig programmatically
        config = JobConfig()
        if job_name:
            config.job_name = job_name
        config.jobs_dir = Path(jobs_dir)
        config.n_attempts = int(n_attempts)
        config.timeout_multiplier = float(timeout_multiplier)

        # Orchestrator
        config.orchestrator.n_concurrent_trials = int(n_concurrent)
        config.orchestrator.quiet = bool(quiet)

        # Agents
        if agent_name is not None or agent_import_path is not None or model_name is not None:
            config.agents = [
                AgentConfig(
                    name=agent_name,
                    import_path=agent_import_path,
                    model_name=model_name,
                    kwargs=agent_kwargs or {},
                )
            ]
        # Environment
        config.environment = EnvironmentConfig(
            type=env_type, force_build=bool(force_build), delete=bool(delete_env), kwargs=env_kwargs or {}
        )

        # Dataset of tasks
        config.datasets = [LocalDatasetConfig(path=dataset_path)]
        config.tasks = []

        # Create job
        job = Job(config)
        # Environment
        config.environment = EnvironmentConfig(
            type=env_type,
            force_build=bool(force_build),
            delete=bool(delete_env),
            kwargs=env_kwargs or {},
        )

        # Save the dataset checksum to the job directory (for future lookups)
        job.job_dir.mkdir(parents=True, exist_ok=True)
        _save_dataset_checksum(job.job_dir, dataset_checksum)

    try:
        asyncio.run(job.run())
    finally:
        _shutdown_litellm(timeout=5.0)

    # Export traces from the produced job directory
    job_dir = job.job_dir  # derived from config; stable
    if export_filter not in (None, "success", "failure"):
        raise ValueError("export_filter must be one of: None, 'success', 'failure'")

    ds = _export_traces(
        root=job_dir,
        recursive=bool(recursive),
        episodes=episodes,
        to_sharegpt=to_sharegpt,
        repo_id=repo_id if push else None,
        push=push,
        verbose=verbose,
        success_filter=export_filter,
        include_instruction=True,
        include_verifier_output=True,
    )
    return _finalize_trace_dataset(ds)


__all__ = ["run_dataset_to_traces", "run_dataset_to_traces_hdf5"]


if __name__ == "__main__":
    import argparse

    def _parse_kv_list(pairs: list[str] | None) -> dict[str, str]:
        out: dict[str, str] = {}
        if not pairs:
            return out
        for item in pairs:
            if "=" not in item:
                raise SystemExit(f"Invalid key=value: {item}")
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
        return out

    parser = argparse.ArgumentParser(description="Run a Harbor dataset and export traces (no subprocesses)")
    parser.add_argument("--dataset-path", required=True, type=Path, help="Path to directory containing task subdirectories")
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--n-attempts", type=int, default=1)
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--n-concurrent", type=int, default=4)
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--progress-log", action="store_true", default=False)

    parser.add_argument("--agent-name", type=str, default=None, help=f"Agent name enum (e.g., {', '.join([a.value for a in AgentName])})")
    parser.add_argument("--agent-import-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("-ak", "--agent-kwarg", action="append", default=None, help="Agent kwarg key=value (repeat)")

    parser.add_argument("--env", type=str, choices=[e.value for e in EnvironmentType], default=EnvironmentType.DOCKER.value)
    parser.add_argument("--no-force-build", dest="force_build", action="store_false", default=True)
    parser.add_argument("--no-delete-env", dest="delete_env", action="store_false", default=True)
    parser.add_argument("-ek", "--env-kwarg", action="append", default=None, help="Environment kwarg key=value (repeat)")
    parser.add_argument("--override-cpus", type=int, default=None, help="Override sandbox CPU count.")
    parser.add_argument("--override-memory-mb", type=int, default=None, help="Override sandbox memory in MB.")
    parser.add_argument("--override-storage-mb", type=int, default=None, help="Override sandbox storage in MB.")

    parser.add_argument("--episodes", type=str, choices=["all", "last"], default="last")
    parser.add_argument("--sharegpt", action="store_true", default=False)
    parser.add_argument("--push", action="store_true", default=False)
    parser.add_argument("--repo", type=str, default=None)
    parser.add_argument("--filter", type=str, choices=["success", "failure", "none"], default="none")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--agent-timeout-sec", type=float, default=None, help="Override Harbor agent timeout for each trial (seconds)")
    parser.add_argument("--verifier-timeout-sec", type=float, default=None, help="Override Harbor verifier timeout for each trial (seconds)")

    args = parser.parse_args()

    agent_name_val = None
    if args.agent_name:
        try:
            agent_name_val = AgentName(args.agent_name)
        except ValueError:
            raise SystemExit(f"Invalid --agent-name: {args.agent_name}")

    env_type_val = EnvironmentType(args.env)
    agent_kwargs_dict = _parse_kv_list(args.agent_kwarg)
    env_kwargs_dict = _parse_kv_list(args.env_kwarg)
    export_filter = None if args.filter == "none" else args.filter

    ds = run_dataset_to_traces(
        dataset_path=args.dataset_path,
        job_name=args.job_name,
        jobs_dir=args.jobs_dir,
        recursive=args.recursive,
        n_attempts=args.n_attempts,
        timeout_multiplier=args.timeout_multiplier,
        n_concurrent=args.n_concurrent,
        quiet=args.quiet,
        log_progress=args.progress_log,
        agent_name=agent_name_val,
        agent_import_path=args.agent_import_path,
        model_name=args.model_name,
        agent_kwargs=agent_kwargs_dict,
        env_type=env_type_val,
        force_build=args.force_build,
        delete_env=args.delete_env,
        env_kwargs=env_kwargs_dict,
        override_cpus=args.override_cpus,
        override_memory_mb=args.override_memory_mb,
        override_storage_mb=args.override_storage_mb,
        episodes=args.episodes,
        to_sharegpt=args.sharegpt,
        push=args.push,
        repo_id=args.repo,
        export_filter=export_filter,
        verbose=args.verbose,
        agent_timeout_sec=args.agent_timeout_sec,
        verifier_timeout_sec=args.verifier_timeout_sec,
    )

    logger.info(f"Exported {len(ds)} rows.")
