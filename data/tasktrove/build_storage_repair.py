#!/usr/bin/env python3
"""Build bounded-memory TaskTrove revisions with larger task storage."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq
from harbor_config.models.task.config import TaskConfig
from huggingface_hub import hf_hub_download

TASKTROVE_REPO = "open-thoughts/TaskTrove"
TASK_SCHEMA = pa.schema([("path", pa.string()), ("task_binary", pa.binary())])
MAX_BATCH_ROWS = 32
MIN_TASKS = 300
REQUIRED_MEMBERS = frozenset(
    {"environment/Dockerfile", "instruction.md", "task.toml", "tests/test.sh"}
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_task(task_binary: bytes) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if (
                not member.name
                or path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != member.name
            ):
                raise ValueError(f"unsafe archive member path: {member.name!r}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are forbidden: {member.name!r}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"unsupported archive member: {member.name!r}")
            if member.name in files:
                raise ValueError(f"duplicate archive member: {member.name!r}")
            extracted = archive.extractfile(member)
            assert extracted is not None
            files[member.name] = extracted.read()
    return files


def write_task(files: dict[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(content))
    return gzip.compress(raw.getvalue(), compresslevel=6, mtime=0)


def task_toml_with_storage(
    task_toml: bytes, expected_storage_mb: int | None, storage_mb: int
) -> bytes:
    source = task_toml.decode("utf-8")
    config = TaskConfig.model_validate_toml(source)
    current = config.environment.storage_mb
    if current != expected_storage_mb:
        raise ValueError(
            f"task declares storage_mb={current}, expected {expected_storage_mb}"
        )
    environment = re.search(r"(?m)^\[environment\][ \t]*(?:#.*)?$", source)
    if environment is None:
        suffix = "" if source.endswith("\n") else "\n"
        transformed = f"{source}{suffix}\n[environment]\nstorage_mb = {storage_mb}\n"
    else:
        next_table = re.search(r"(?m)^\[", source[environment.end() :])
        section_end = (
            len(source)
            if next_table is None
            else environment.end() + next_table.start()
        )
        section = source[environment.end() : section_end]
        declaration = re.search(
            r"(?m)^(storage_mb[ \t]*=[ \t]*)\d+([ \t]*(?:#.*)?)$", section
        )
        if current is None:
            transformed = (
                source[: environment.end()]
                + f"\nstorage_mb = {storage_mb}"
                + source[environment.end() :]
            )
        else:
            if declaration is None:
                raise ValueError("parsed storage_mb has no replaceable declaration")
            start = environment.end() + declaration.start()
            end = environment.end() + declaration.end()
            replacement = f"{declaration.group(1)}{storage_mb}{declaration.group(2)}"
            transformed = source[:start] + replacement + source[end:]
    parsed = TaskConfig.model_validate_toml(transformed)
    if parsed.environment.storage_mb != storage_mb:
        raise ValueError("transformed task has the wrong storage requirement")
    return transformed.encode("utf-8")


def source_parquet(args: argparse.Namespace) -> Path:
    if args.source_parquet is not None:
        return args.source_parquet.resolve()
    return Path(
        hf_hub_download(
            TASKTROVE_REPO,
            f"{args.source_dataset}/tasks.parquet",
            repo_type="dataset",
            revision=args.source_revision,
            local_dir=args.stage / "source",
        )
    )


def validate_output(path: Path, expected_rows: int, storage_mb: int) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != TASK_SCHEMA:
        raise ValueError(f"unexpected output schema: {parquet.schema_arrow}")
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError("output row count changed")
    seen: set[str] = set()
    for group in range(parquet.metadata.num_row_groups):
        if parquet.metadata.row_group(group).num_rows > MAX_BATCH_ROWS:
            raise ValueError("output row group exceeds bounded batch size")
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
        for row in batch.to_pylist():
            path_value = row["path"]
            if path_value in seen:
                raise ValueError(f"duplicate task path: {path_value}")
            seen.add(path_value)
            files = read_task(row["task_binary"])
            if not REQUIRED_MEMBERS <= files.keys():
                raise ValueError(f"incomplete task: {path_value}")
            config = TaskConfig.model_validate_toml(files["task.toml"].decode())
            if config.environment.storage_mb != storage_mb:
                raise ValueError(f"wrong storage requirement: {path_value}")


def build(args: argparse.Namespace) -> None:
    if args.storage_mb <= 0 or args.storage_mb % 1024:
        raise ValueError("storage_mb must be a positive whole number of GiB")
    source = source_parquet(args)
    if file_sha256(source) != args.source_sha256:
        raise ValueError("source Parquet hash mismatch")
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow != TASK_SCHEMA:
        raise ValueError(f"unexpected source schema: {parquet.schema_arrow}")
    if parquet.metadata.num_rows < MIN_TASKS:
        raise ValueError(f"source violates the {MIN_TASKS}-task floor")

    output = args.stage / "datasets" / args.output_dataset / "tasks.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    writer = pq.ParquetWriter(
        output,
        TASK_SCHEMA,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    )
    rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
            transformed_rows = []
            for row in batch.to_pylist():
                original = read_task(row["task_binary"])
                if not REQUIRED_MEMBERS <= original.keys():
                    raise ValueError(f"incomplete task: {row['path']}")
                transformed = dict(original)
                transformed["task.toml"] = task_toml_with_storage(
                    original["task.toml"],
                    args.expected_current_storage_mb,
                    args.storage_mb,
                )
                changed = {
                    name
                    for name in original.keys() | transformed.keys()
                    if original.get(name) != transformed.get(name)
                }
                if changed != {"task.toml"}:
                    raise ValueError(f"unexpected changed members: {sorted(changed)}")
                transformed_rows.append(
                    {"path": row["path"], "task_binary": write_task(transformed)}
                )
            writer.write_table(
                pa.Table.from_pylist(transformed_rows, schema=TASK_SCHEMA)
            )
            rows += len(transformed_rows)
    finally:
        writer.close()

    validate_output(output, rows, args.storage_mb)
    report = {
        "source_repo": TASKTROVE_REPO,
        "source_revision": args.source_revision,
        "source_dataset": args.source_dataset,
        "source_rows": parquet.metadata.num_rows,
        "source_sha256": args.source_sha256,
        "output_dataset": args.output_dataset,
        "output_rows": rows,
        "output_sha256": file_sha256(output),
        "storage_mb": args.storage_mb,
        "output": str(output.relative_to(args.stage)),
    }
    (args.stage / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--storage-mb", type=int, required=True)
    parser.add_argument("--expected-current-storage-mb", type=int)
    parser.add_argument("--source-parquet", type=Path)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
