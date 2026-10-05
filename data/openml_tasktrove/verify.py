"""Gate tasks on regeneration, real Harbor no-op, and real Harbor oracle trials."""

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harbor.models.task.task import Task

from .common import (
    DATA_FILES,
    canonical_json,
    canonicalize,
    load_dataset,
    sha256,
    write_csv,
)
from .generate import render_task

SUMMARY_FIELDS = [
    "openml_task_id",
    "openml_task_type_id",
    "openml_dataset_id",
    "dataset_version",
    "task_id",
    "status",
    "reason",
    "split_seed",
    "train_rows",
    "test_rows",
    "train_sha256",
    "test_sha256",
    "hidden_labels_sha256",
    "oracle_passed",
    "empty_failed",
    "deterministic",
]


def file_hashes(task):
    return {
        p.relative_to(task).as_posix(): sha256(p)
        for p in sorted(task.rglob("*"))
        if p.is_file() and p.name != "verification.json"
    }


def row_ids(path):
    with path.open(encoding="utf-8", newline="") as stream:
        ids = [r["row_id"] for r in csv.DictReader(stream)]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate IDs in {path}")
    return set(ids)


def check_determinism(task, source_cache=None):
    metadata = json.loads((task / "provenance.json").read_text())
    df = load_dataset(metadata, source_cache)
    canonical, _, _ = canonicalize(df, metadata["target"])
    source_ids = {r["row_id"] for r in canonical}
    with tempfile.TemporaryDirectory(prefix="openml-regenerate-") as temporary:
        # Exclude generated fields so provenance is reproduced exactly.
        row = {
            k: v
            for k, v in metadata.items()
            if k
            not in {
                "task_id",
                "task_type",
                "metric",
                "train_rows",
                "test_rows",
                "missing_target_rows",
            }
        }
        first = render_task(row, df, Path(temporary) / "one")
        second = render_task(row, df.iloc[::-1], Path(temporary) / "two")
        if file_hashes(first) != file_hashes(second) or file_hashes(
            first
        ) != file_hashes(task):
            raise ValueError("Regenerated task files differ")
        for candidate in (task, first, second):
            train = row_ids(candidate / DATA_FILES[0])
            test = row_ids(candidate / DATA_FILES[1])
            if train & test or train | test != source_ids:
                raise ValueError("Split IDs do not partition the canonical source")
            if test != row_ids(candidate / DATA_FILES[2]):
                raise ValueError("Test and label IDs differ")
            if train != row_ids(first / DATA_FILES[0]) or test != row_ids(
                first / DATA_FILES[1]
            ):
                raise ValueError("Split membership changed")


def harbor_trial(task, agent, jobs_dir, harbor="harbor"):
    name = f"{task.name}-{agent}"
    subprocess.run(
        [
            harbor,
            "run",
            "-p",
            str(task.resolve()),
            "-a",
            agent,
            "--jobs-dir",
            str(jobs_dir.resolve()),
            "--job-name",
            name,
        ],
        check=True,
    )
    # Harbor CLI can exit zero for failed trials; inspect the actual trial result.
    result_paths = list((jobs_dir / name).glob("*/attempts/000/result.json"))
    if len(result_paths) != 1:
        raise ValueError(f"Expected one Harbor trial result for {name}")
    result = json.loads(result_paths[0].read_text())
    if result["exception_info"] is not None or result["verifier_result"] is None:
        raise ValueError(f"Harbor trial failed: {result['exception_info']}")
    return result["verifier_result"]["rewards"]


def _verify_task(task, accepted, rejected, source_cache, jobs_dir, harbor):
    metadata = json.loads((task / "provenance.json").read_text())
    result = {field: metadata.get(field, "") for field in SUMMARY_FIELDS}
    result.update(
        status="rejected",
        reason="",
        oracle_passed=False,
        empty_failed=False,
        deterministic=False,
        train_sha256=sha256(task / DATA_FILES[0]),
        test_sha256=sha256(task / DATA_FILES[1]),
        hidden_labels_sha256=sha256(task / DATA_FILES[2]),
    )
    try:
        Task(task)
        check_determinism(task, source_cache)
        result["deterministic"] = True
        empty = harbor_trial(task, "nop", jobs_dir, harbor)
        result["empty_failed"] = empty.get("valid") == 0 and all(
            value == 0 for value in empty.values()
        )
        oracle = harbor_trial(task, "oracle", jobs_dir, harbor)
        result["oracle_passed"] = oracle.get("valid") == 1 and (
            oracle.get("accuracy", 0) > 0
            if metadata["metric"] == "accuracy"
            else oracle.get("rmse", -1) >= 0
        )
        if not result["empty_failed"] or not result["oracle_passed"]:
            raise ValueError(f"Verification gates failed: nop={empty}, oracle={oracle}")
        result["status"] = "accepted"
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    destination = (
        accepted if result["status"] == "accepted" else rejected
    ) / task.name
    shutil.copytree(task, destination)
    receipt = {"result": result, "files": file_hashes(destination)}
    (destination / "verification.json").write_text(canonical_json(receipt) + "\n")
    return result


def run(
    tasks_dir,
    output_dir=None,
    source_cache=None,
    jobs_dir=None,
    harbor="harbor",
    workers=1,
):
    tasks_dir = Path(tasks_dir)
    output_dir = Path(output_dir) if output_dir else tasks_dir.parent
    accepted, rejected = output_dir / "accepted", output_dir / "rejected"
    accepted.mkdir(parents=True, exist_ok=False)
    rejected.mkdir(parents=True, exist_ok=False)
    jobs_dir = Path(jobs_dir) if jobs_dir else output_dir / "jobs"
    tasks = sorted(path.parent for path in tasks_dir.glob("*/instruction.md"))
    if not tasks:
        raise ValueError("No generated tasks found")
    if workers < 1:
        raise ValueError("workers must be positive")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _verify_task,
                task,
                accepted,
                rejected,
                source_cache,
                jobs_dir,
                harbor,
            ): task
            for task in tasks
        }
        for future in as_completed(futures):
            results.append(future.result())
            results.sort(key=lambda result: result["task_id"])
            write_csv(output_dir / "pipeline_summary.csv", results, SUMMARY_FIELDS)
    return accepted, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--jobs-dir", type=Path)
    parser.add_argument("--harbor", default="harbor")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    _, results = run(
        args.tasks_dir,
        args.output_dir,
        args.source_cache,
        args.jobs_dir,
        args.harbor,
        args.workers,
    )
    if any(r["status"] != "accepted" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
