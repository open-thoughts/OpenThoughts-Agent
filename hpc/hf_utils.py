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
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote

# Default HuggingFace org for auto-derived repo IDs (override with env var)
DEFAULT_HF_ORG = "DCAgent"
HF_ORG_ENV_VAR = "DCAGENT_HF_ORG"
HF_SELECTOR_REVISION_SEPARATOR = "@"
HF_SELECTOR_SUBDIR_SEPARATOR = "::"


@dataclass(frozen=True)
class HfDatasetSelector:
    """A Hugging Face dataset repo with an optional revision and subdirectory."""

    repo_id: str
    revision: str | None = None
    subdir: str | None = None

    def canonical(self) -> str:
        revision = (
            f"{HF_SELECTOR_REVISION_SEPARATOR}{self.revision}" if self.revision else ""
        )
        subdir = f"{HF_SELECTOR_SUBDIR_SEPARATOR}{self.subdir}" if self.subdir else ""
        return f"{self.repo_id}{revision}{subdir}"

    def cache_name(self) -> str:
        """Return a reversible cache key containing repo, subdirectory, and revision."""
        components = (
            ("repo", self.repo_id),
            ("subdir", self.subdir),
            ("revision", self.revision),
        )
        encoded = (
            (name, quote(value, safe="-._"))
            for name, value in components
            if value is not None
        )
        return "__".join(f"{name}-{len(value)}-{value}" for name, value in encoded)


def parse_hf_dataset_selector(value: str) -> HfDatasetSelector | None:
    """Parse ``org/repo[@revision][::subdir]`` dataset selectors."""
    if not value or value.startswith(("./", "../", "/", "~")) or "\\" in value:
        return None

    repo_revision, separator, subdir = value.partition(HF_SELECTOR_SUBDIR_SEPARATOR)
    repo_id, revision_separator, revision = repo_revision.partition(
        HF_SELECTOR_REVISION_SEPARATOR
    )
    if repo_id.count("/") != 1 or not all(part.strip() for part in repo_id.split("/")):
        return None
    if separator and (
        not subdir or subdir.startswith("/") or ".." in subdir.split("/")
    ):
        return None
    if revision_separator and not revision:
        return None

    return HfDatasetSelector(
        repo_id=repo_id,
        revision=revision if revision_separator else None,
        subdir=subdir if separator else None,
    )


def is_hf_dataset_path(path: str) -> bool:
    """Check if path looks like a HuggingFace dataset identifier.

    Supports bare ``org/repo`` identifiers and pinned subdirectory selectors in
    the form ``org/repo@revision::subdir``.

    Args:
        path: Path string to check

    Returns:
        True if path appears to be an HF dataset identifier
    """
    return parse_hf_dataset_selector(path) is not None


def resolve_hf_dataset_selector(value: str) -> HfDatasetSelector:
    """Resolve a selector revision to an immutable Hub commit."""
    selector = parse_hf_dataset_selector(value)
    if selector is None:
        raise ValueError(f"Invalid Hugging Face dataset selector: {value!r}")
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(selector.repo_id, revision=selector.revision)
    return HfDatasetSelector(selector.repo_id, info.sha, selector.subdir)


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


def _download_object_store_dataset(uri: str, *, verbose: bool = True) -> str:
    """Recursively fetch a gs://|s3:// dataset snapshot to a local temp dir.

    Uses fsspec (gcsfs/s3fs) with default credential discovery — on iris workers
    that is workload-identity for gs:// and the injected AWS_* creds for s3://.
    Makes ZERO HuggingFace calls, so it is safe under HF_HUB_OFFLINE=1. Returns
    the local directory (containing the mirrored parquet / task tree), which the
    caller feeds to the same raw-vs-parquet handling as an HF snapshot.
    """
    import tempfile
    from pathlib import Path

    import fsspec

    proto = uri.split("://", 1)[0]
    fs = fsspec.filesystem("gcs" if proto == "gs" else "s3")
    dest = Path(tempfile.mkdtemp(prefix="ds_precached_"))
    if verbose:
        print(
            f"[hf_utils] Fetching pre-cached dataset from {uri} -> {dest} (offline, no HF)"
        )
    # Trailing slash + recursive => copy the directory tree, preserving layout.
    fs.get(uri.rstrip("/") + "/", str(dest) + "/", recursive=True)
    files = [p for p in dest.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError(
            f"Pre-cached dataset URI {uri} resolved to no files under {dest}; "
            "the mirror is empty or the URI is wrong."
        )
    if verbose:
        print(f"[hf_utils] Fetched {len(files)} files from {uri}")
    return str(dest)


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

    if path_or_repo.startswith(("gs://", "s3://")):
        # Pre-cached dataset snapshot in object storage (offline path — the iris
        # launcher mirrored the HF dataset to the region-local GCS bucket and
        # rewrote --dataset_path to it). Fetch from the store WITHOUT touching HF
        # (so HF_HUB_OFFLINE=1 is honored), then let the caller's raw-vs-parquet
        # detection + convert_parquet_to_tasks run on the local snapshot as usual.
        return _download_object_store_dataset(path_or_repo, verbose=verbose)

    if is_hf_dataset_path(path_or_repo):
        # It's an HF dataset identifier - download it
        from huggingface_hub import snapshot_download

        selector = parse_hf_dataset_selector(path_or_repo)
        assert selector is not None

        if verbose:
            print(f"[hf_utils] Downloading HF dataset: {path_or_repo}")
        allow_patterns = [f"{selector.subdir}/**"] if selector.subdir else None
        local_path = snapshot_download(
            repo_id=selector.repo_id,
            repo_type="dataset",
            revision=selector.revision,
            allow_patterns=allow_patterns,
        )
        if selector.subdir:
            local_path = os.path.join(local_path, selector.subdir)
            if not os.path.isdir(local_path):
                raise FileNotFoundError(
                    f"Dataset selector {path_or_repo} did not resolve a subdirectory"
                )
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
