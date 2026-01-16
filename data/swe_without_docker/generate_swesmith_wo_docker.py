#!/usr/bin/env python3
"""
Generate SWE-Smith tasks with plain Ubuntu Docker.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.swesmith.generate_with_plain_docker import create_sandboxed_tasks
from data.commons import upload_tasks_to_hf, subsample_tasks_directory


def main() -> str:
    path = str(create_sandboxed_tasks(limit=10_000, offset=0))
    sampled_path = subsample_tasks_directory(path, 10_000)
    upload_tasks_to_hf(sampled_path, "DCAgent/exp-swd-swesmith-wo-docker")
    return path


if __name__ == "__main__":
    print(main())
