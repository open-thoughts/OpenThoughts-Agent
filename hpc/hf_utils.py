"""HuggingFace utilities for HPC launchers.

This module provides common utilities for working with HuggingFace Hub:
- Repository ID validation and sanitization
- Dataset path detection
- HF repo ID derivation for eval uploads
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Optional

# Default HuggingFace org for auto-derived repo IDs (override with env var)
DEFAULT_HF_ORG = "DCAgent"
HF_ORG_ENV_VAR = "DCAGENT_HF_ORG"


def is_hf_dataset_path(path: str) -> bool:
    """Check if path looks like a HuggingFace dataset identifier.

    HF identifiers have format: org/repo-name or username/repo-name
    They contain exactly one "/" and no path separators like "./" or "../"

    Args:
        path: Path string to check

    Returns:
        True if path appears to be an HF dataset identifier
    """
    if not path:
        return False

    # Must contain exactly one "/"
    if path.count("/") != 1:
        return False

    # Must not look like a filesystem path
    if path.startswith(("./", "../", "/", "~")):
        return False

    # Must not contain backslashes (Windows paths)
    if "\\" in path:
        return False

    # Both parts must be non-empty
    parts = path.split("/")
    if not all(p.strip() for p in parts):
        return False

    return True


def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    """Sanitize a HuggingFace repo_id to comply with naming rules.

    Keeps org prefix (e.g. 'mlfoundations-dev/') and cleans up the rest.
    Used when deriving HF dataset repo names from job names or model paths.

    Args:
        repo_id: The repository ID to sanitize (e.g., 'org/some-name').
        max_length: Maximum allowed length for the full repo_id.

    Returns:
        Sanitized repo_id that complies with HuggingFace naming rules.
    """

    def collapse(value: str) -> str:
        prev = None
        while value != prev:
            prev = value
            value = value.replace("--", "-").replace("..", ".")
        return value

    org, name = repo_id.split("/", 1) if "/" in repo_id else (None, repo_id)
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = collapse(name).strip("-.")
    if not name:
        name = "repo"
    limit = max_length - (len(org) + 1 if org else 0)
    if len(name) > limit > 8:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        keep = max(1, limit - len(digest))
        base = name[:keep].rstrip("-.") or "r"
        name = collapse(f"{base}{digest}").strip("-.")
    if name[0] in "-.":
        name = f"r{name[1:]}"
    if name[-1] in "-.":
        name = f"{name[:-1]}0"
    return f"{org}/{name}" if org else name


def derive_default_hf_repo_id(job_name: str) -> str:
    """Derive default HF repo ID from job name.

    Used by both local and HPC eval runners to auto-derive an HF repo ID
    when --upload_to_database is set but --upload_hf_repo is not provided.

    The org defaults to "DCAgent" but can be overridden via the
    DCAGENT_HF_ORG environment variable.

    Args:
        job_name: Name of the eval job (used as the repo name)

    Returns:
        HF repo ID in format "<org>/<job_name>"
    """
    org = os.environ.get(HF_ORG_ENV_VAR, DEFAULT_HF_ORG)
    return f"{org}/{job_name}"


def resolve_dataset_path(
    path_or_repo: str,
    *,
    verbose: bool = True,
) -> str:
    """Resolve a dataset path, downloading from HuggingFace if needed.

    Handles both local filesystem paths and HuggingFace dataset identifiers.
    Used by both eval and datagen launchers to resolve --tasks_input_path.

    Args:
        path_or_repo: Either a local path or HF dataset identifier (e.g., "org/repo")
        verbose: Whether to print status messages

    Returns:
        Resolved local filesystem path (absolute)
    """
    from pathlib import Path

    if is_hf_dataset_path(path_or_repo):
        # It's an HF dataset identifier - download it
        from huggingface_hub import snapshot_download

        if verbose:
            print(f"[hf_utils] Downloading HF dataset: {path_or_repo}")
        local_path = snapshot_download(repo_id=path_or_repo, repo_type="dataset")
        if verbose:
            print(f"[hf_utils] Downloaded to: {local_path}")
        return local_path
    else:
        # It's a local path - resolve relative to PROJECT_ROOT
        from hpc.launch_utils import resolve_repo_path

        resolved = resolve_repo_path(path_or_repo)
        return str(resolved)


def is_raw_tasks_directory(snapshot_dir) -> bool:
    """Check if a directory contains raw task folders (not parquet with task_binary).

    Raw task directories have subdirectories with instruction.md files,
    rather than parquet files with task_binary columns that need extraction.

    Used to auto-detect HuggingFace dataset format:
    - Parquet with task_binary: needs extraction via tasks_parquet_converter
    - Raw task dirs: can be used directly or copied

    Args:
        snapshot_dir: Path to directory to check (str or Path)

    Returns:
        True if directory contains raw task folders, False if it has parquet with task_binary
    """
    from pathlib import Path

    snapshot_dir = Path(snapshot_dir)

    # Check if there are any parquet files
    parquet_files = list(snapshot_dir.rglob("*.parquet"))
    if parquet_files:
        # Has parquet files - check if they have task_binary column
        try:
            import pyarrow.parquet as pq
            for pf in parquet_files[:1]:  # Check first parquet
                table = pq.read_table(pf)
                if "task_binary" in table.column_names:
                    return False  # Needs extraction
        except Exception:
            pass

    # Check for raw task directories (subdirs with instruction.md)
    # Look for any instruction.md file recursively
    instruction_files = list(snapshot_dir.rglob("instruction.md"))
    if instruction_files:
        return True  # Already raw tasks

    return False


def resolve_hf_repo_id(
    explicit_repo: Optional[str],
    upload_to_database: bool,
    job_name: str,
) -> Optional[str]:
    """Resolve HF repo ID for eval upload.

    If explicit_repo is provided, use it.
    If upload_to_database is True but no explicit repo, auto-derive from job_name.
    Otherwise return None.

    Used by both local and HPC eval runners to determine the HF repo ID.

    Args:
        explicit_repo: Explicitly specified HF repo ID (--upload_hf_repo)
        upload_to_database: Whether database upload is enabled
        job_name: Name of the eval job (used as repo name if auto-deriving)

    Returns:
        Sanitized HF repo ID, or None if HF upload should be skipped
    """
    if explicit_repo:
        return sanitize_hf_repo_id(explicit_repo)

    if upload_to_database:
        # Auto-derive HF repo ID: <org>/<job_name>
        derived_repo = derive_default_hf_repo_id(job_name)
        return sanitize_hf_repo_id(derived_repo)

    return None


__all__ = [
    "DEFAULT_HF_ORG",
    "HF_ORG_ENV_VAR",
    "is_hf_dataset_path",
    "is_raw_tasks_directory",
    "sanitize_hf_repo_id",
    "derive_default_hf_repo_id",
    "resolve_dataset_path",
    "resolve_hf_repo_id",
]
