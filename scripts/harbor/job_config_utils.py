from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from harbor.models.job.config import JobConfig, LocalDatasetConfig


def load_job_config(config_path: Path | str) -> JobConfig:
    """Load a Harbor ``JobConfig`` from YAML or JSON."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Harbor job config not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        config = JobConfig.model_validate(yaml.safe_load(path.read_text()))
    elif suffix == ".json":
        config = JobConfig.model_validate_json(path.read_text())
    else:
        raise ValueError(
            f"Unsupported Harbor job config format '{path.suffix}'. "
            "Expected one of: .yaml, .yml, .json."
        )

    return _normalize_job_config_agent_kwargs(config)


def dump_job_config(config: JobConfig, output_path: Path | str) -> Path:
    """Persist ``config`` to ``output_path`` as YAML."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    # ``mode="json"`` ensures Pydantic converts Path objects and other
    # non-YAML-native types (e.g. UUID) into plain serialisable values.
    serialisable = config.model_dump(mode="json")
    path.write_text(yaml.safe_dump(serialisable, sort_keys=False))
    return path


def set_local_dataset(config: JobConfig, dataset_path: Path | str) -> JobConfig:
    """Return a copy of ``config`` pointing at ``dataset_path`` for local datasets."""

    path = Path(dataset_path).expanduser().resolve()
    updated = config.model_copy(deep=True)
    updated.datasets = [LocalDatasetConfig(path=path)]
    updated.tasks = []
    return updated


def set_job_metadata(
    config: JobConfig,
    *,
    job_name: str | None = None,
    jobs_dir: Path | str | None = None,
) -> JobConfig:
    """Return a copy of ``config`` with updated job metadata."""

    updated = config.model_copy(deep=True)
    if job_name is not None:
        updated.job_name = job_name
    if jobs_dir is not None:
        updated.jobs_dir = Path(jobs_dir).expanduser().resolve()
    return updated


def ensure_trailing_dataset(config: JobConfig) -> Path:
    """Extract the first local dataset path from ``config``."""

    for dataset in config.datasets:
        if isinstance(dataset, LocalDatasetConfig):
            return Path(dataset.path).expanduser().resolve()
    raise ValueError(
        "Harbor job config must include at least one local dataset with a `path` entry."
    )


def update_agent_kwargs(config: JobConfig, kwargs: dict | None) -> JobConfig:
    """Merge ``kwargs`` into the first agent definition."""

    if not kwargs:
        return config

    updated = config.model_copy(deep=True)
    if not updated.agents:
        raise ValueError("Harbor job config must define at least one agent.")

    merged = dict(updated.agents[0].kwargs or {})
    merged.update(kwargs)
    merged = normalize_trajectory_kwargs(merged)
    updated.agents[0].kwargs = merged
    return updated


def overwrite_agent_fields(
    config: JobConfig,
    *,
    name: str | None = None,
    import_path: str | None = None,
    model_name: str | None = None,
    override_timeout_sec: float | None = None,
) -> JobConfig:
    """Overwrite basic agent metadata on the first agent entry."""

    updated = config.model_copy(deep=True)
    if not updated.agents:
        raise ValueError("Harbor job config must define at least one agent.")

    agent = updated.agents[0]
    if name is not None:
        agent.name = name
    if import_path is not None:
        agent.import_path = import_path
    if model_name is not None:
        agent.model_name = model_name
    if override_timeout_sec is not None:
        agent.override_timeout_sec = override_timeout_sec
    updated.agents[0] = agent
    return updated


def extend_agent_kwargs_lists(
    config: JobConfig,
    keys: Iterable[str],
    *,
    value: str,
) -> JobConfig:
    """Append ``value`` to list-valued agent kwargs for each key."""

    updated = config.model_copy(deep=True)
    if not updated.agents:
        raise ValueError("Harbor job config must define at least one agent.")

    agent = updated.agents[0]
    for key in keys:
        existing = agent.kwargs.get(key)
        if existing is None:
            agent.kwargs[key] = [value]
        elif isinstance(existing, list):
            agent.kwargs[key] = [*existing, value]
        else:
            agent.kwargs[key] = [existing, value]
    updated.agents[0] = agent
    return updated


def normalize_trajectory_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Convert legacy ``trajectory_configs`` into the singular ``trajectory_config``."""

    if kwargs is None:
        return {}

    normalized = dict(kwargs)
    legacy = normalized.pop("trajectory_configs", None)
    current = normalized.get("trajectory_config")

    def _ensure_mapping(value: Any, label: str) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(
                f"{label} must be a mapping (received {type(value).__name__})."
            )
        return value

    merged: dict[str, Any] = {}
    legacy_map = _ensure_mapping(legacy, "trajectory_configs")
    if legacy_map:
        merged.update(legacy_map)
    current_map = _ensure_mapping(current, "trajectory_config")
    if current_map:
        merged.update(current_map)

    if merged:
        normalized["trajectory_config"] = merged
    elif current is None:
        normalized.pop("trajectory_config", None)

    return normalized


def _normalize_job_config_agent_kwargs(config: JobConfig) -> JobConfig:
    """Apply ``normalize_trajectory_kwargs`` to all agent entries."""

    updated = config.model_copy(deep=True)
    normalized_agents = []
    for agent in updated.agents:
        agent_copy = agent.model_copy(deep=True)
        agent_copy.kwargs = normalize_trajectory_kwargs(agent_copy.kwargs)
        normalized_agents.append(agent_copy)
    updated.agents = normalized_agents
    return updated
