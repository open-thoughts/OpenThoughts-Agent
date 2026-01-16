#!/usr/bin/env python3
"""
Generate R2E-Gym tasks with pre-built Docker images (standard).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.r2egym.utils import load_r2egym_instances, add_r2egym_verifier_tests_to_instances
from data.commons import generate_tasks_to_hdf5, extract_hdf5_to_task_paths, upload_tasks_to_hf


def main() -> str:
    instances = load_r2egym_instances()
    tuples = add_r2egym_verifier_tests_to_instances(instances)
    # Unzip tuples into separate lists
    instructions, metadata, solutions, test_sh, test_py, task_toml, dockerfiles = zip(*tuples)
    hdf5_path = generate_tasks_to_hdf5(
        instructions=list(instructions),
        metadata=list(metadata),
        solutions=list(solutions),
        test_sh=list(test_sh),
        test_py=list(test_py),
        task_toml=list(task_toml),
        dockerfiles=list(dockerfiles),
        dataset_prefix="r2egym",
    )
    extracted_paths = extract_hdf5_to_task_paths(hdf5_path)
    extracted_dir = str(extracted_paths[0]).rsplit('/', 1)[0]
    upload_tasks_to_hf(extracted_dir, "DCAgent/exp-swd-r2egym-standard")
    return extracted_dir


if __name__ == "__main__":
    print(main())
