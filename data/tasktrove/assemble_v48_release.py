#!/usr/bin/env python3
"""Validate and assemble the TaskTrove v4.8 release payload."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq
from harbor_config.models.task.config import TaskConfig

from data.tasktrove.build_storage_repair import (
    MAX_BATCH_ROWS,
    MIN_TASKS,
    REQUIRED_MEMBERS,
    TASK_SCHEMA,
    file_sha256,
    read_task,
)
from data.tasktrove.build_v48_repairs import MEMORY_MB, SOURCE_REVISION

SOURCE_VERSION = "4.7"
TARGET_VERSION = "4.8"
MAX_IMAGES = 20


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def validate_dataset(path: Path, entry: dict[str, object]) -> set[str]:
    if file_sha256(path) != entry["output_sha256"]:
        raise ValueError(f"output hash mismatch: {path}")
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != TASK_SCHEMA:
        raise ValueError(f"unexpected schema: {path}")
    if parquet.metadata.num_rows != entry["output_rows"]:
        raise ValueError(f"output row count mismatch: {path}")
    if parquet.metadata.num_rows < MIN_TASKS:
        raise ValueError(f"source fell below {MIN_TASKS} tasks: {path}")
    if any(
        parquet.metadata.row_group(index).num_rows > MAX_BATCH_ROWS
        for index in range(parquet.metadata.num_row_groups)
    ):
        raise ValueError(f"oversized row group: {path}")

    paths: set[str] = set()
    images: set[str] = set()
    shell_hashes: set[str] = set()
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
        for row in batch.to_pylist():
            task_path = row["path"]
            if task_path in paths:
                raise ValueError(f"duplicate task path: {task_path}")
            paths.add(task_path)
            files = read_task(row["task_binary"])
            if not REQUIRED_MEMBERS <= files.keys():
                raise ValueError(f"incomplete task: {task_path}")
            config = TaskConfig.model_validate_toml(files["task.toml"].decode())
            if entry["memory_mb"] is not None:
                if config.environment.memory_mb != MEMORY_MB:
                    raise ValueError(f"wrong memory requirement: {task_path}")
            for name, content in files.items():
                if not name.endswith(".sh"):
                    continue
                digest = hashlib.sha256(content).hexdigest()
                if digest in shell_hashes:
                    continue
                checked = subprocess.run(
                    ["bash", "-n"], input=content, capture_output=True, check=False
                )
                if checked.returncode:
                    raise ValueError(
                        f"invalid shell script {name}: "
                        + checked.stderr.decode(errors="replace")
                    )
                shell_hashes.add(digest)
            images.add(hashlib.sha256(files["environment/Dockerfile"]).hexdigest())
    if len(images) > MAX_IMAGES:
        raise ValueError(f"{path} uses {len(images)} images")
    return images


def assemble(repair_stage: Path, stage: Path) -> None:
    if stage.exists() and any(stage.iterdir()):
        raise ValueError(f"stage must be absent or empty: {stage}")
    source_manifest = json.loads((repair_stage / "manifest.json").read_text())
    if source_manifest["source_revision"] != SOURCE_REVISION:
        raise ValueError("repair stage does not target TaskTrove v4.7")
    stage.mkdir(parents=True, exist_ok=True)
    images: set[str] = set()
    entries = []
    for item in source_manifest["datasets"]:
        source = repair_stage / "datasets" / item["output"] / "tasks.parquet"
        images.update(validate_dataset(source, item))
        relative = Path("datasets") / item["output"] / "tasks.parquet"
        _hardlink_or_copy(source, stage / relative)
        entries.append({**item, "parquet": str(relative)})
    if len(images) > MAX_IMAGES:
        raise ValueError(f"release uses {len(images)} images")
    manifest = {
        "source_repo": "open-thoughts/TaskTrove",
        "source_revision": SOURCE_REVISION,
        "source_version": SOURCE_VERSION,
        "target_version": TARGET_VERSION,
        "datasets": entries,
        "release_unique_images": len(images),
    }
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"assembled {len(entries)} replacements using {len(images)} unique images")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-stage", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.repair_stage, args.stage)


if __name__ == "__main__":
    main()
