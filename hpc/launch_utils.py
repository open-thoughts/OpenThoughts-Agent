"""
Utility helpers shared across HPC launch entry points.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from .job_name_ignore_list import JOB_NAME_IGNORE_KEYS


def sanitize_repo_for_job(repo_id: str) -> str:
    """Return a filesystem-safe representation of a repo identifier."""

    safe = re.sub(r"[^A-Za-z0-9._\-]+", "-", repo_id.strip())
    safe = safe.strip("-_")
    return safe or "consolidate"


def sanitize_repo_component(value: Optional[str]) -> Optional[str]:
    """Extract the meaningful suffix from trace repositories (traces-<slug>)."""

    if not value:
        return None
    match = re.search(r"traces-([A-Za-z0-9._\-]+)", value)
    return match.group(1) if match else None


def get_job_name(cli_args: Mapping[str, Any]) -> str:
    """Derive a stable job name from user-provided CLI arguments."""

    job_type = str(cli_args.get("job_type", "train") or "train").lower()
    if job_type == "consolidate":
        repo_id = (
            cli_args.get("consolidate_repo_id")
            or cli_args.get("consolidate_base_repo")
            or "consolidate"
        )
        job_name = f"{sanitize_repo_for_job(str(repo_id))}_consolidate"
        if len(job_name) > 96:
            job_name = job_name[:96]
        return job_name

    job_name_components: list[str] = []
    job_name_suffix: Optional[str] = None

    for key, value in cli_args.items():
        if not isinstance(value, (str, int, float)):
            continue
        if value == "None" or key in JOB_NAME_IGNORE_KEYS:
            continue

        if key == "seed":
            try:
                if float(value) == 42:
                    continue
            except (TypeError, ValueError):
                pass

        if key not in {"dataset", "model_name_or_path"}:
            job_name_components.append(str(key).replace("_", "-"))

        value_str = str(value)
        if value_str == "Qwen/Qwen2.5-32B-Instruct":
            job_name_suffix = "_32B"
        elif value_str == "Qwen/Qwen2.5-14B-Instruct":
            job_name_suffix = "_14B"
        elif value_str == "Qwen/Qwen2.5-3B-Instruct":
            job_name_suffix = "_3B"
        elif value_str == "Qwen/Qwen2.5-1.5B-Instruct":
            job_name_suffix = "_1.5B"
        else:
            job_name_components.append(value_str.split("/")[-1])

    job_name = "_".join(job_name_components)
    job_name = (
        job_name.replace("/", "_")
        .replace("?", "")
        .replace("*", "")
        .replace("{", "")
        .replace("}", "")
        .replace(":", "")
        .replace('"', "")
        .replace(" ", "_")
    )
    if job_name_suffix:
        job_name += job_name_suffix

    if len(job_name) > 96:
        print("Truncating job name to less than HF limit of 96 characters...")
        job_name = "_".join(
            "-".join(segment[:4] for segment in chunk.split("-"))
            for chunk in job_name.split("_")
        )
        if len(job_name) > 96:
            raise ValueError(
                f"Job name {job_name} is still too long (96 characters) after truncation. "
                "Try renaming the dataset or providing a shorter YAML config."
            )

    return job_name


__all__ = ["get_job_name", "sanitize_repo_for_job", "sanitize_repo_component"]
