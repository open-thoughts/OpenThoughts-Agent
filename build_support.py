"""Helpers for setuptools dynamic metadata (extras, etc.)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


DATA_EXTRAS = [
    "click",
    "bs4",
    "rapidfuzz",
]

LLAMAFACTORY_SUBMODULE = Path("sft/llamafactory")
LLAMAFACTORY_EXTRAS = ["hf-kernels", "liger-kernel", "deepspeed", "bitsandbytes"]


def _maybe_sync_llamafactory(repo_root: Path) -> None:
    """Ensure the LLaMA-Factory submodule exists before referencing it."""

    llama_dir = (repo_root / LLAMAFACTORY_SUBMODULE).resolve()
    if llama_dir.exists():
        return

    if shutil.which("git") is None:
        raise RuntimeError(
            "git is required to update the sft/llamafactory submodule when installing the 'sft' extra"
        )

    cmd = [
        "git",
        "-C",
        str(repo_root),
        "submodule",
        "update",
        "--init",
        "--remote",
        LLAMAFACTORY_SUBMODULE.as_posix(),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - setup-time guard
        raise RuntimeError(
            "Failed to sync sft/llamafactory submodule. Run the command manually and retry installation."
        ) from exc


def resolve_llamafactory_requirement() -> str:
    """Build the direct-reference requirement pointing at the submodule path."""

    repo_root = Path(__file__).resolve().parent
    if os.environ.get("OT_AGENT_SKIP_SFT_SYNC", "0") != "1":
        _maybe_sync_llamafactory(repo_root)

    llama_dir = (repo_root / LLAMAFACTORY_SUBMODULE).resolve()
    extras = ",".join(LLAMAFACTORY_EXTRAS)
    return f"llamafactory[{extras}] @ {llama_dir.as_uri()}"
