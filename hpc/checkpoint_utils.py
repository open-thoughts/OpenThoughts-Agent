"""Checkpoint and model pre-download utilities for HPC clusters.

This module provides utilities for pre-downloading HuggingFace models and datasets
on login nodes before submitting SLURM jobs. This is necessary for clusters where
compute nodes do not have internet access (e.g., Perlmutter, JSC clusters).

Usage:
    from hpc.checkpoint_utils import (
        pre_download_model,
        pre_download_dataset,
        pre_download_artifacts,
        needs_pre_download,
    )

    # Check if pre-download is needed for this cluster
    if needs_pre_download(hpc):
        model_path = pre_download_model("Qwen/Qwen3-8B")
        dataset_path = pre_download_dataset("mlfoundations/dataset")
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Log prefix for consistent messaging
LOG_PREFIX = "[checkpoint]"


def log_info(msg: str) -> None:
    """Log info message with prefix."""
    print(f"{LOG_PREFIX} {msg}", flush=True)


def log_warn(msg: str) -> None:
    """Log warning message with prefix."""
    print(f"{LOG_PREFIX} WARNING: {msg}", file=sys.stderr, flush=True)


def log_error(msg: str) -> None:
    """Log error message with prefix."""
    print(f"{LOG_PREFIX} ERROR: {msg}", file=sys.stderr, flush=True)


@dataclass
class DownloadResult:
    """Result of a pre-download operation."""

    original_path: str  # Original HF repo ID or local path
    local_path: str  # Resolved local path after download
    was_downloaded: bool  # True if downloaded, False if already cached or local
    repo_type: str  # "model" or "dataset"


def needs_pre_download(hpc: Any) -> bool:
    """Check if the HPC cluster requires pre-downloading artifacts.

    Pre-downloading is needed when compute nodes don't have internet access.
    This is determined by the `internet_node` field in the HPC configuration.

    Args:
        hpc: HPC configuration object with `internet_node` attribute.

    Returns:
        True if pre-download is needed, False otherwise.
    """
    # If hpc is None, assume we need pre-download to be safe
    if hpc is None:
        return True

    # Check the internet_node attribute
    internet_node = getattr(hpc, "internet_node", None)
    if internet_node is None:
        # Default to needing pre-download if not specified
        return True

    # If internet_node is True, compute nodes have internet - no pre-download needed
    # If internet_node is False, compute nodes don't have internet - pre-download needed
    return not internet_node


def is_huggingface_repo(path_or_repo: str) -> bool:
    """Check if a string looks like a HuggingFace repo ID.

    Args:
        path_or_repo: String that could be a HF repo ID or local path.

    Returns:
        True if it looks like a HF repo ID (org/repo format), False otherwise.
    """
    # Local paths
    if os.path.exists(path_or_repo):
        return False

    # Absolute or relative paths
    if path_or_repo.startswith("/") or path_or_repo.startswith("./") or path_or_repo.startswith("../"):
        return False

    # HF repo IDs have format "org/repo" or just "repo"
    # They don't contain path separators beyond the single /
    parts = path_or_repo.split("/")
    if len(parts) > 2:
        return False

    # Check for common path patterns
    if any(part in [".", ".."] for part in parts):
        return False

    return True


def get_cache_dir() -> str:
    """Get the HuggingFace cache directory.

    Checks environment variables in order:
    1. HF_HUB_CACHE
    2. HF_HOME (appends /hub)
    3. Default ~/.cache/huggingface/hub

    Returns:
        Path to HuggingFace cache directory.
    """
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    elif os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    else:
        return os.path.expanduser("~/.cache/huggingface/hub")


def pre_download_model(
    model_path: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    force: bool = False,
) -> DownloadResult:
    """Pre-download a HuggingFace model to local cache.

    If the path is already a local directory, returns it unchanged.
    If it's a HuggingFace repo ID, downloads it using snapshot_download.

    Args:
        model_path: HuggingFace model repo ID (e.g., "Qwen/Qwen3-8B") or local path.
        revision: Optional git revision (branch, tag, or commit hash).
        token: Optional HuggingFace token for private repos.
        force: If True, re-download even if cached.

    Returns:
        DownloadResult with local path and download status.

    Raises:
        ValueError: If the model cannot be found or downloaded.
    """
    # Check if it's a local path
    if os.path.isdir(model_path):
        log_info(f"Model is local path: {model_path}")
        return DownloadResult(
            original_path=model_path,
            local_path=os.path.abspath(model_path),
            was_downloaded=False,
            repo_type="model",
        )

    # Check if it looks like a HF repo
    if not is_huggingface_repo(model_path):
        # Might be a path that doesn't exist yet - return as-is
        log_warn(f"Model path doesn't exist and doesn't look like HF repo: {model_path}")
        return DownloadResult(
            original_path=model_path,
            local_path=model_path,
            was_downloaded=False,
            repo_type="model",
        )

    # Download from HuggingFace
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
    except ImportError:
        raise ImportError("huggingface_hub is required for pre-downloading. Install with: pip install huggingface_hub")

    log_info(f"Downloading model: {model_path}")
    if revision:
        log_info(f"  Revision: {revision}")

    try:
        # Get token from environment if not provided
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        local_path = snapshot_download(
            repo_id=model_path,
            repo_type="model",
            revision=revision,
            token=token,
            force_download=force,
        )
        log_info(f"Downloaded model to: {local_path}")
        return DownloadResult(
            original_path=model_path,
            local_path=local_path,
            was_downloaded=True,
            repo_type="model",
        )
    except RepositoryNotFoundError:
        raise ValueError(f"Model repository not found: {model_path}")
    except HFValidationError as e:
        raise ValueError(f"Invalid model repository: {model_path} - {e}")
    except Exception as e:
        raise ValueError(f"Failed to download model {model_path}: {e}")


def pre_download_dataset(
    dataset_path: str,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    force: bool = False,
) -> DownloadResult:
    """Pre-download a HuggingFace dataset to local cache.

    If the path is already a local directory, returns it unchanged.
    If it's a HuggingFace repo ID, downloads it using snapshot_download.

    Args:
        dataset_path: HuggingFace dataset repo ID or local path.
        revision: Optional git revision (branch, tag, or commit hash).
        token: Optional HuggingFace token for private repos.
        force: If True, re-download even if cached.

    Returns:
        DownloadResult with local path and download status.

    Raises:
        ValueError: If the dataset cannot be found or downloaded.
    """
    # Check if it's a local path
    if os.path.isdir(dataset_path):
        log_info(f"Dataset is local path: {dataset_path}")
        return DownloadResult(
            original_path=dataset_path,
            local_path=os.path.abspath(dataset_path),
            was_downloaded=False,
            repo_type="dataset",
        )

    # Check if it looks like a HF repo
    if not is_huggingface_repo(dataset_path):
        log_warn(f"Dataset path doesn't exist and doesn't look like HF repo: {dataset_path}")
        return DownloadResult(
            original_path=dataset_path,
            local_path=dataset_path,
            was_downloaded=False,
            repo_type="dataset",
        )

    # Download from HuggingFace
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError
    except ImportError:
        raise ImportError("huggingface_hub is required for pre-downloading. Install with: pip install huggingface_hub")

    log_info(f"Downloading dataset: {dataset_path}")
    if revision:
        log_info(f"  Revision: {revision}")

    try:
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        local_path = snapshot_download(
            repo_id=dataset_path,
            repo_type="dataset",
            revision=revision,
            token=token,
            force_download=force,
        )
        log_info(f"Downloaded dataset to: {local_path}")
        return DownloadResult(
            original_path=dataset_path,
            local_path=local_path,
            was_downloaded=True,
            repo_type="dataset",
        )
    except RepositoryNotFoundError:
        raise ValueError(f"Dataset repository not found: {dataset_path}")
    except HFValidationError as e:
        raise ValueError(f"Invalid dataset repository: {dataset_path} - {e}")
    except Exception as e:
        raise ValueError(f"Failed to download dataset {dataset_path}: {e}")


def pre_download_artifacts(
    model_path: Optional[str] = None,
    dataset_paths: Optional[List[str]] = None,
    hpc: Optional[Any] = None,
    force: bool = False,
    skip_if_internet: bool = True,
) -> Dict[str, DownloadResult]:
    """Pre-download all artifacts (model and datasets) for a training job.

    This is a convenience function that downloads both model and datasets
    in one call, with proper logging and error handling.

    Args:
        model_path: HuggingFace model repo ID or local path.
        dataset_paths: List of HuggingFace dataset repo IDs or local paths.
        hpc: HPC configuration object. If provided and has internet_node=True,
            pre-download will be skipped unless force=True.
        force: If True, download even if cached or on internet-enabled cluster.
        skip_if_internet: If True and hpc.internet_node=True, skip pre-download.

    Returns:
        Dict mapping artifact names to DownloadResult objects:
        - "model": Model download result (if model_path provided)
        - "dataset_0", "dataset_1", etc.: Dataset download results

    Raises:
        ValueError: If any artifact cannot be downloaded.
    """
    results: Dict[str, DownloadResult] = {}

    # Check if pre-download is needed
    if skip_if_internet and not force and hpc is not None:
        if not needs_pre_download(hpc):
            cluster_name = getattr(hpc, "name", "unknown")
            log_info(f"Cluster {cluster_name} has internet on compute nodes - skipping pre-download")
            # Return empty results but don't fail
            if model_path:
                results["model"] = DownloadResult(
                    original_path=model_path,
                    local_path=model_path,
                    was_downloaded=False,
                    repo_type="model",
                )
            if dataset_paths:
                for i, ds_path in enumerate(dataset_paths):
                    results[f"dataset_{i}"] = DownloadResult(
                        original_path=ds_path,
                        local_path=ds_path,
                        was_downloaded=False,
                        repo_type="dataset",
                    )
            return results

    cluster_name = getattr(hpc, "name", "unknown") if hpc else "unknown"
    log_info(f"Pre-downloading artifacts for cluster: {cluster_name}")

    # Download model
    if model_path:
        results["model"] = pre_download_model(model_path, force=force)

    # Download datasets
    if dataset_paths:
        for i, ds_path in enumerate(dataset_paths):
            results[f"dataset_{i}"] = pre_download_dataset(ds_path, force=force)

    # Summary
    downloaded_count = sum(1 for r in results.values() if r.was_downloaded)
    total_count = len(results)
    log_info(f"Pre-download complete: {downloaded_count}/{total_count} artifacts downloaded")

    return results


def ensure_model_available(
    model_path: str,
    hpc: Optional[Any] = None,
) -> str:
    """Ensure a model is available locally, downloading if necessary.

    This is a simple wrapper that returns the local path to a model,
    downloading it first if on a no-internet cluster.

    Args:
        model_path: HuggingFace model repo ID or local path.
        hpc: HPC configuration object.

    Returns:
        Local path to the model (either original path or downloaded location).

    Raises:
        ValueError: If the model cannot be found or downloaded.
    """
    if hpc is not None and not needs_pre_download(hpc):
        # Cluster has internet - return path as-is
        return model_path

    result = pre_download_model(model_path)
    return result.local_path


def ensure_dataset_available(
    dataset_path: str,
    hpc: Optional[Any] = None,
) -> str:
    """Ensure a dataset is available locally, downloading if necessary.

    Args:
        dataset_path: HuggingFace dataset repo ID or local path.
        hpc: HPC configuration object.

    Returns:
        Local path to the dataset.

    Raises:
        ValueError: If the dataset cannot be found or downloaded.
    """
    if hpc is not None and not needs_pre_download(hpc):
        return dataset_path

    result = pre_download_dataset(dataset_path)
    return result.local_path


def parse_dataset_list(dataset_arg: Union[str, List[str], None]) -> List[str]:
    """Parse a dataset argument that may be a JSON list string or actual list.

    Handles formats like:
    - '["dataset1", "dataset2"]' (JSON string from CLI)
    - ["dataset1", "dataset2"] (actual list)
    - "dataset1,dataset2" (comma-separated string)
    - "dataset1" (single dataset)
    - None (returns empty list)

    Args:
        dataset_arg: Dataset argument in various formats.

    Returns:
        List of dataset paths/repo IDs.
    """
    if dataset_arg is None:
        return []

    if isinstance(dataset_arg, list):
        return dataset_arg

    # Try JSON parsing first
    import json
    try:
        parsed = json.loads(dataset_arg)
        if isinstance(parsed, list):
            return parsed
        return [str(parsed)]
    except json.JSONDecodeError:
        pass

    # Try comma-separated
    if "," in dataset_arg:
        return [ds.strip() for ds in dataset_arg.split(",") if ds.strip()]

    # Single dataset
    return [dataset_arg] if dataset_arg.strip() else []


# Convenience aliases for backward compatibility
download_model = pre_download_model
download_dataset = pre_download_dataset


__all__ = [
    # Core download functions
    "pre_download_model",
    "pre_download_dataset",
    "pre_download_artifacts",
    # Convenience functions
    "ensure_model_available",
    "ensure_dataset_available",
    # Utility functions
    "needs_pre_download",
    "is_huggingface_repo",
    "get_cache_dir",
    "parse_dataset_list",
    # Data classes
    "DownloadResult",
    # Aliases
    "download_model",
    "download_dataset",
]
