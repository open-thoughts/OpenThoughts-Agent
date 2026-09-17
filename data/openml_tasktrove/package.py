"""Package only verified tasks using the repository's existing converter."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts.harbor.tasks_parquet_converter import find_tasks, to_parquet

from .verify import file_hashes


def run(tasks_dir, output_path):
    tasks_dir, output_path = Path(tasks_dir), Path(output_path)
    tasks = find_tasks(tasks_dir, recursive=True)
    if not tasks:
        raise ValueError("No accepted tasks to package")
    for task in tasks:
        receipt = json.loads((task / "verification.json").read_text())
        result = receipt["result"]
        if result["status"] != "accepted" or not all(
            result[k] is True
            for k in ("oracle_passed", "empty_failed", "deterministic")
        ):
            raise ValueError(f"Task has not passed verification: {task}")
        if receipt["files"] != file_hashes(task):
            raise ValueError(f"Task changed after verification: {task}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    to_parquet(tasks_dir, output_path, tasks, compression="gz")
    parquet = pq.ParquetFile(output_path)
    if parquet.schema_arrow.names != [
        "path",
        "task_binary",
    ] or parquet.metadata.num_rows != len(tasks):
        raise ValueError("Unexpected TaskTrove Parquet schema or task count")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.tasks_dir, args.output)


if __name__ == "__main__":
    main()
