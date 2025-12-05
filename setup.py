from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_support import DATA_EXTRAS, resolve_llamafactory_requirement


def compute_extras() -> dict[str, list[str]]:
    extras: dict[str, list[str]] = {
        "data": DATA_EXTRAS,
    }

    extras["sft"] = [resolve_llamafactory_requirement()]
    return extras


if __name__ == "__main__":
    setup(extras_require=compute_extras())
