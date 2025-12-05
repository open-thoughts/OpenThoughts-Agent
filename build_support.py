"""
Helpers for setuptools dynamic metadata (extras, etc.).
"""

from __future__ import annotations

from pathlib import Path

DATA_EXTRAS = [
    "click",
    "bs4",
    "rapidfuzz",
]


def resolve_llamafactory_requirement() -> str:
    """
    Build a direct-reference requirement pointing at the local LLaMA Factory checkout.

    The directory is expected to be provided via the git submodule
    `sft/llamafactory`. We intentionally do not fail if the directory is
    missing so that base installs (without `[sft]`) still succeed; attempting to
    install the `sft` extra will fail later if the path is absent.
    """

    repo_root = Path(__file__).resolve().parent
    llama_dir = (repo_root / "sft" / "llamafactory").resolve()
    return f"llamafactory @ {llama_dir.as_uri()}"
