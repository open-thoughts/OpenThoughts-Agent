#!/usr/bin/env python3
"""Build TaskTrove v4.8 verifier and memory revisions from v4.7."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from harbor_config.models.task.config import TaskConfig
from huggingface_hub import hf_hub_download

from data.tasktrove.build_storage_repair import (
    MAX_BATCH_ROWS,
    MIN_TASKS,
    REQUIRED_MEMBERS,
    TASK_SCHEMA,
    file_sha256,
    read_task,
    task_toml_with_memory,
    write_task,
)

TASKTROVE_REPO = "open-thoughts/TaskTrove"
SOURCE_REVISION = "697a123d73f39114471468536c8c65da29ea9edc"
MEMORY_MB = 4096
ADVERSARIAL_STATIC_VERIFIER_DEFECTS = frozenset({"5k-adversarial-0174"})

ADVERSARIAL_VERIFIER = r"""#!/bin/bash
set -uo pipefail

LOG_DIR=/logs/verifier
REPORT="$LOG_DIR/junit.xml"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR/reward.txt" "$REPORT"
cd /app
export PYTHONPATH="/app:${PYTHONPATH:-}"

if grep -qE '(^|[[:space:]])(from|import)[[:space:]]+django' /tests/test_adversarial.py; then
    settings_file=$(find /app -type f -name settings.py -not -path '*/.venv/*' -print -quit)
    if [[ -n "$settings_file" ]]; then
        settings_module=${settings_file#/app/}
        settings_module=${settings_module%.py}
        export DJANGO_SETTINGS_MODULE=${settings_module//\//.}
    fi
fi

set +e
timeout --signal=TERM --kill-after=15s 690s \
    python3 -m pytest /tests/test_adversarial.py -v --tb=short --junitxml="$REPORT" \
    -p pytester -p aiohttp.pytest_plugin \
    > >(tee "$LOG_DIR/pytest_output.txt") 2>&1
pytest_status=$?
set -e

if [[ $pytest_status -eq 124 || $pytest_status -eq 137 ]]; then
    echo "verifier process-group deadline exceeded" >&2
    exit 124
fi

case "$pytest_status" in
    0)
        if [[ ! -s "$REPORT" ]]; then
            echo "pytest passed without producing a JUnit report" >&2
            exit 42
        fi
        python3 - "$REPORT" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
executed = [
    case for case in root.findall(".//testcase") if case.find("skipped") is None
]
if not executed:
    raise SystemExit("zero non-skipped tests executed")
with open("/logs/verifier/reward.txt", "w", encoding="utf-8") as output:
    output.write("1\n")
PY
        ;;
    1|2)
        # Test failures, target exceptions, and target import/collection failures
        # are consequences of the submitted workspace and are scoreable zeros.
        echo 0 > "$LOG_DIR/reward.txt"
        ;;
    *)
        # Pytest usage/internal/no-test failures do not establish task quality.
        exit "$pytest_status"
        ;;
esac
""".encode()

PHP_EXECUTION_PATTERN = "(Tests: [1-9][0-9]*|OK \\([1-9][0-9]* tests?,)"
PHP_OLD_EXECUTION_GATE = (
    b"if ! grep -Eq 'Tests: [1-9][0-9]*' /logs/verifier/test-stdout.txt; then"
)
PHP_NEW_EXECUTION_GATE = (
    f"if ! grep -Eq '{PHP_EXECUTION_PATTERN}' /logs/verifier/test-stdout.txt; then"
).encode()


@dataclass(frozen=True)
class Revision:
    source: str
    source_sha256: str
    output: str
    memory_mb: int | None = None
    verifier: str | None = None


REVISIONS = (
    Revision(
        "DCAgent__exp_rle_adversarial-v5",
        "da1dc8bbde9d415c39461fdb5558b2d0b6e0927b327381ac1306d580a577f9af",
        "DCAgent__exp_rle_adversarial-v6",
        verifier="adversarial",
    ),
    Revision(
        "laion__exp_rpt_stack-cpp-v3",
        "2f76e1dac738c15553ed51d7dbe59024c3e5885e3a9690ec1a57863eb8ab0ebb",
        "laion__exp_rpt_stack-cpp-v4",
        memory_mb=MEMORY_MB,
    ),
    Revision(
        "laion__exp_rpt_stack-php-large-v8",
        "4469a4eac31ed091d635d6cd0aed4e8c4022dd3a3907a30502b8cc35d25d5219",
        "laion__exp_rpt_stack-php-large-v9",
        memory_mb=MEMORY_MB,
        verifier="php",
    ),
    Revision(
        "laion__exp_rpt_stack-php-v2-v7",
        "45a30e52e868fb2ae8ff630a01de86d9204be1cb65e0eef6cca299ec3703c57f",
        "laion__exp_rpt_stack-php-v2-v8",
        memory_mb=MEMORY_MB,
    ),
)


def patch_php_verifier(test_sh: bytes) -> bytes:
    if test_sh.count(PHP_OLD_EXECUTION_GATE) != 1:
        raise ValueError("PHP verifier execution gate does not match v8")
    return test_sh.replace(PHP_OLD_EXECUTION_GATE, PHP_NEW_EXECUTION_GATE)


def transform_files(files: dict[str, bytes], revision: Revision) -> dict[str, bytes]:
    transformed = dict(files)
    if revision.memory_mb is not None:
        transformed["task.toml"] = task_toml_with_memory(
            files["task.toml"], None, revision.memory_mb
        )
    if revision.verifier == "adversarial":
        transformed["tests/test.sh"] = ADVERSARIAL_VERIFIER
    elif revision.verifier == "php":
        transformed["tests/test.sh"] = patch_php_verifier(files["tests/test.sh"])
    return transformed


def source_path(args: argparse.Namespace, revision: Revision) -> Path:
    if args.source_root is not None:
        return args.source_root / revision.source / "tasks.parquet"
    return Path(
        hf_hub_download(
            TASKTROVE_REPO,
            f"{revision.source}/tasks.parquet",
            repo_type="dataset",
            revision=SOURCE_REVISION,
        )
    )


def should_retain(path: str, revision: Revision) -> bool:
    return not (
        revision.verifier == "adversarial"
        and path in ADVERSARIAL_STATIC_VERIFIER_DEFECTS
    )


def validate_output(path: Path, revision: Revision, expected_rows: int) -> None:
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != TASK_SCHEMA:
        raise ValueError(f"unexpected output schema: {parquet.schema_arrow}")
    if parquet.metadata.num_rows != expected_rows or expected_rows < MIN_TASKS:
        raise ValueError("output row count is invalid")
    seen: set[str] = set()
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
        for row in batch.to_pylist():
            task_path = row["path"]
            if task_path in seen:
                raise ValueError(f"duplicate task path: {task_path}")
            seen.add(task_path)
            files = read_task(row["task_binary"])
            if not REQUIRED_MEMBERS <= files.keys():
                raise ValueError(f"incomplete task: {task_path}")
            config = TaskConfig.model_validate_toml(files["task.toml"].decode())
            if revision.memory_mb is not None:
                if config.environment.memory_mb != revision.memory_mb:
                    raise ValueError(f"wrong memory requirement: {task_path}")
            if revision.verifier == "adversarial":
                if files["tests/test.sh"] != ADVERSARIAL_VERIFIER:
                    raise ValueError(f"wrong adversarial verifier: {task_path}")
            elif revision.verifier == "php":
                test_sh = files["tests/test.sh"]
                if (
                    PHP_NEW_EXECUTION_GATE not in test_sh
                    or PHP_OLD_EXECUTION_GATE in test_sh
                ):
                    raise ValueError(f"wrong PHP execution gate: {task_path}")


def build_revision(args: argparse.Namespace, revision: Revision) -> dict[str, object]:
    source = source_path(args, revision)
    if file_sha256(source) != revision.source_sha256:
        raise ValueError(f"source hash mismatch: {revision.source}")
    parquet = pq.ParquetFile(source)
    if parquet.schema_arrow != TASK_SCHEMA:
        raise ValueError(f"unexpected source schema: {parquet.schema_arrow}")
    output = args.stage / "datasets" / revision.output / "tasks.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    writer = pq.ParquetWriter(
        output, TASK_SCHEMA, compression="zstd", use_dictionary=False
    )
    rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
            transformed_rows = []
            for row in batch.to_pylist():
                if not should_retain(row["path"], revision):
                    continue
                files = read_task(row["task_binary"])
                if not REQUIRED_MEMBERS <= files.keys():
                    raise ValueError(f"incomplete task: {row['path']}")
                transformed = transform_files(files, revision)
                transformed_rows.append(
                    {"path": row["path"], "task_binary": write_task(transformed)}
                )
            if transformed_rows:
                writer.write_table(
                    pa.Table.from_pylist(transformed_rows, schema=TASK_SCHEMA)
                )
                rows += len(transformed_rows)
    finally:
        writer.close()
    validate_output(output, revision, rows)
    return {
        "source": revision.source,
        "source_sha256": revision.source_sha256,
        "source_rows": parquet.metadata.num_rows,
        "output": revision.output,
        "output_rows": rows,
        "output_sha256": file_sha256(output),
        "memory_mb": revision.memory_mb,
        "verifier": revision.verifier,
        "removed_paths": sorted(
            ADVERSARIAL_STATIC_VERIFIER_DEFECTS
            if revision.verifier == "adversarial"
            else ()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    reports = [build_revision(args, revision) for revision in REVISIONS]
    manifest = {
        "source_repo": TASKTROVE_REPO,
        "source_revision": SOURCE_REVISION,
        "datasets": reports,
    }
    (args.stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
