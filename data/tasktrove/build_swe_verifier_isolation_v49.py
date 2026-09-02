#!/usr/bin/env python3
"""Build TaskTrove v4.9 SWE sources with isolated hidden-test installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from harbor_config.models.task.config import TaskConfig
from huggingface_hub import hf_hub_download

from data.nemotron_gym.converters.agent_calendar import _calendar_names_from_prompt
from data.nemotron_gym.verifiers import CALENDAR_VERIFIER_PY
from data.patchers.trusted_test_patch import (
    TRUSTED_TEST_PATCH_INSTALLER,
    trusted_test_patch_command,
)
from data.tasktrove.build_data_quality_batch_a1 import _feasible_calendar
from data.tasktrove.build_storage_repair import (
    MAX_BATCH_ROWS,
    MIN_TASKS,
    REQUIRED_MEMBERS,
    TASK_SCHEMA,
    file_sha256,
    read_task,
    task_toml_with_memory,
    task_toml_with_storage,
    write_task,
)

TASKTROVE_REPO = "open-thoughts/TaskTrove"
SOURCE_REVISION = "35c1139e8932344e9b52b231bca806d95f5d14cb"
INSTALLER_PATH = "tests/install_trusted_test_patch.sh"
SWE_REBENCH_STORAGE_MB = 8192
SWE_REBENCH_MEMORY_MB = 4096
OPENSWE_STORAGE_MB = 4096
EXCLUDED_TASKS = {
    "DCAgent__swe_rebench_v2_patched_oracle": {
        "aallam__openai-kotlin-127",
        "actix__actix-web-2624",
        "act-rules__act-rules.github.io-1277",
        "aio-libs__aiohttp-10551",
        "aio-libs__aiohttp-8636",
        "algebraicjulia__catlab.jl-227",
        "denvercoder1__readme-typing-svg-213",
    },
    "laion__openswe-tasks-patched-v6-oracle-success": {
        "openswe_oss-00715",
    },
}


class UnrecoverableCalendarTask(ValueError):
    """Raised when a calendar task cannot support exact event-name grading."""


@dataclass(frozen=True)
class Revision:
    source: str
    source_sha256: str
    output: str
    verifier: str


REVISIONS = (
    Revision(
        source="DCAgent__swe_rebench_v2_patched_oracle",
        source_sha256="afd827fc1fc5c930736fef88c2b115631bf4a237128188a2cb2f9c3009ac3774",
        output="DCAgent__swe_rebench_v2_patched_oracle-v2",
        verifier="swe_rebench",
    ),
    Revision(
        source="laion__openswe-tasks-patched-v6-oracle-success",
        source_sha256="f9904208d0736ea3e8079dee2dd5633c2711b8d2a8ca3060c82794be2ae9f46f",
        output="laion__openswe-tasks-patched-v7-oracle-success",
        verifier="openswe",
    ),
    Revision(
        source="laion__nemotron-gym-instruction-following-calendar-v2",
        source_sha256="3d89709363b11a28387ccb367163644a1fa6b5aa2ff919aedaa7b63369256864",
        output="laion__nemotron-gym-instruction-following-calendar-v3",
        verifier="calendar",
    ),
)

SWE_REBENCH_PATCH_BLOCK = b"""\
# Apply the hidden test patch (the tests the fix PR introduced)
cd /testbed
if [ -f /tests/test_patch.diff ]; then
    git apply --verbose /tests/test_patch.diff || \\
        git apply --verbose --reject /tests/test_patch.diff || true
fi
"""

OPENSWE_PATCH_BLOCK = b"""\
cd /testbed || exit 0
if [ -s /tests/test_patch.diff ]; then
    git apply --check --allow-empty /tests/test_patch.diff || exit 0
    git apply --verbose --allow-empty /tests/test_patch.diff || exit 0
fi
"""

OPENSWE_SUCCESS_BLOCK = b"""\
if [ "$runner_rc" -eq 0 ] && [ "$junit_rc" -eq 0 ] && [ "$pytest_guard_rc" -eq 0 ]; then
    echo 1 > "$logs_dir/reward.txt"
fi
exit 0
"""


def _replace_once(source: bytes, old: bytes, new: bytes, label: str) -> bytes:
    if source.count(old) != 1:
        raise ValueError(f"{label} does not match the expected source verifier")
    return source.replace(old, new, 1)


def patch_swe_rebench_verifier(test_sh: bytes, base_commit: str) -> bytes:
    """Replace partial hidden-patch application with isolated installation."""
    command = trusted_test_patch_command(base_commit).encode()
    replacement = (
        b"""\
# Restore hidden-test paths from the immutable base before applying the trusted
# patch. Product-code edits made by the agent remain untouched.
if ! """
        + command
        + b"""; then
    exit 1
fi
"""
    )
    return _replace_once(
        test_sh,
        SWE_REBENCH_PATCH_BLOCK,
        replacement,
        "SWE-ReBench hidden-test block",
    )


def patch_openswe_verifier(test_sh: bytes, base_commit: str) -> bytes:
    """Isolate trusted tests and reserve reward zero for executed test failures."""
    transformed = _replace_once(
        test_sh,
        b"# OpenSWE v6 verifier: provision dependencies and score only executed tests.\n",
        b"# OpenSWE v7 verifier: isolate trusted tests and score only executed tests.\n",
        "OpenSWE verifier version",
    )
    transformed = _replace_once(
        transformed,
        b'echo 0 > "$logs_dir/reward.txt"\n',
        b'rm -f "$logs_dir/reward.txt"\n',
        "OpenSWE initial reward",
    )
    transformed = _replace_once(
        transformed,
        b"    exit 0\nfi\nsource /tmp/openswe-setup-environment.sh || exit 0\n",
        b'    exit "$setup_rc"\nfi\nsource /tmp/openswe-setup-environment.sh || exit 1\n',
        "OpenSWE setup failure handling",
    )
    command = trusted_test_patch_command(base_commit).encode()
    patch_block = (
        b"""\
cd /testbed || exit 1
# Restore hidden-test paths from the immutable base before applying the trusted
# patch. Product-code edits made by the agent remain untouched.
if ! """
        + command
        + b"""; then
    exit 1
fi
"""
    )
    transformed = _replace_once(
        transformed,
        OPENSWE_PATCH_BLOCK,
        patch_block,
        "OpenSWE hidden-test block",
    )
    transformed = _replace_once(
        transformed,
        OPENSWE_SUCCESS_BLOCK,
        b"""\
if [ "$runner_rc" -eq 0 ] && [ "$junit_rc" -eq 0 ] && [ "$pytest_guard_rc" -eq 0 ]; then
    echo 1 > "$logs_dir/reward.txt"
else
    echo 0 > "$logs_dir/reward.txt"
fi
exit 0
""",
        "OpenSWE final reward",
    )
    return transformed


def patch_calendar_verifier(files: dict[str, bytes]) -> dict[str, bytes]:
    """Replace the event-local verifier with the complete calendar verifier."""
    instruction = files["instruction.md"].decode("utf-8", errors="replace")
    if "overlap" not in instruction.lower():
        raise ValueError("calendar instruction does not declare the overlap constraint")
    verifier_data = json.loads(files["tests/verifier_data.json"])
    expected = verifier_data.get("expected_events")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("calendar verifier data has no expected events")
    names = _calendar_names_from_prompt(instruction)
    repaired: dict[str, dict[str, object]] = {}
    for key, spec in expected.items():
        if not isinstance(spec, dict):
            raise ValueError(f"calendar event {key!r} is not an object")
        event_id = spec.get("event_id")
        if not isinstance(event_id, int) or isinstance(event_id, bool):
            raise ValueError(f"calendar event {key!r} has an invalid ID")
        event_name = spec.get("event_name") or names.get(event_id)
        if not isinstance(event_name, str) or not event_name.strip():
            raise UnrecoverableCalendarTask(
                f"calendar event {event_id} has no recoverable name"
            )
        repaired[str(event_id)] = {**spec, "event_name": event_name.strip()}
    oracle = _feasible_calendar(repaired)
    if oracle is None:
        raise UnrecoverableCalendarTask(
            "calendar task has no feasible conflict-free schedule"
        )
    transformed = dict(files)
    transformed["tests/verifier.py"] = CALENDAR_VERIFIER_PY.encode()
    transformed["tests/verifier_data.json"] = json.dumps(
        {**verifier_data, "expected_events": repaired},
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode()
    transformed["solution/answer.json"] = json.dumps(
        oracle, ensure_ascii=False, sort_keys=True, indent=2
    ).encode()
    transformed["solution/solve.sh"] = (
        b"#!/bin/bash\nset -eu\ncp /solution/answer.json /app/answer.txt\n"
    )
    return transformed


def patch_paths(patch: bytes) -> set[str]:
    """Return paths named by standard ``diff --git`` headers."""
    paths: set[str] = set()
    fallback_paths: set[str] = set()
    for raw_line in patch.decode("utf-8", errors="replace").splitlines():
        if raw_line.startswith(("--- ", "+++ ")):
            value = raw_line[4:].split("\t", 1)[0]
            if value.startswith('"'):
                fields = shlex.split(value)
                if len(fields) != 1:
                    raise ValueError(f"malformed patch path: {raw_line!r}")
                value = fields[0]
            if value.startswith(("a/", "b/")):
                value = value[2:]
            if value != "/dev/null":
                paths.add(value)
            continue
        if raw_line.startswith(("rename from ", "rename to ")):
            paths.add(raw_line.split(" ", 2)[2])
            continue
        if not raw_line.startswith("diff --git "):
            continue
        payload = raw_line.removeprefix("diff --git ")
        if payload.startswith('"'):
            fields = shlex.split(payload)
            if len(fields) != 2:
                raise ValueError(f"malformed diff header: {raw_line!r}")
        else:
            old, separator, new = payload.partition(" b/")
            if not separator or not old.startswith("a/"):
                raise ValueError(f"malformed diff header: {raw_line!r}")
            fields = [old, "b/" + new]
        fallback_paths.update(path[2:] for path in fields)
    return paths | fallback_paths


def transform_files(files: dict[str, bytes], revision: Revision) -> dict[str, bytes]:
    if revision.verifier == "calendar":
        return patch_calendar_verifier(files)

    config = json.loads(files["tests/config.json"])
    base_commit = str(config.get("base_commit") or "")
    if not base_commit:
        raise ValueError("task has no immutable base commit")
    test_patch = files.get("tests/test_patch.diff", b"")
    overlap = patch_paths(test_patch) & patch_paths(files.get("solution/solve.sh", b""))
    if overlap:
        raise ValueError(f"golden and hidden patches overlap: {sorted(overlap)}")

    transformed = dict(files)
    if revision.verifier == "swe_rebench":
        transformed["tests/test.sh"] = patch_swe_rebench_verifier(
            files["tests/test.sh"], base_commit
        )
        transformed["task.toml"] = task_toml_with_storage(
            task_toml_with_memory(files["task.toml"], None, SWE_REBENCH_MEMORY_MB),
            None,
            SWE_REBENCH_STORAGE_MB,
        )
    elif revision.verifier == "openswe":
        transformed["tests/test.sh"] = patch_openswe_verifier(
            files["tests/test.sh"], base_commit
        )
        transformed["task.toml"] = task_toml_with_storage(
            files["task.toml"], None, OPENSWE_STORAGE_MB
        )
    else:
        raise ValueError(f"unknown verifier family: {revision.verifier}")
    transformed[INSTALLER_PATH] = TRUSTED_TEST_PATCH_INSTALLER.encode()
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


def validate_output(path: Path, revision: Revision, expected_rows: int) -> set[str]:
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != TASK_SCHEMA:
        raise ValueError(f"unexpected output schema: {parquet.schema_arrow}")
    if parquet.metadata.num_rows != expected_rows or expected_rows < MIN_TASKS:
        raise ValueError("output row count is invalid")
    seen: set[str] = set()
    images: set[str] = set()
    shell_hashes: set[str] = set()
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
        for row in batch.to_pylist():
            task_path = row["path"]
            if task_path in seen:
                raise ValueError(f"duplicate task path: {task_path}")
            seen.add(task_path)
            files = read_task(row["task_binary"])
            images.add(hashlib.sha256(files["environment/Dockerfile"]).hexdigest())
            if not REQUIRED_MEMBERS <= files.keys():
                raise ValueError(f"incomplete repaired task: {task_path}")
            if revision.verifier == "calendar":
                if files["tests/verifier.py"] != CALENDAR_VERIFIER_PY.encode():
                    raise ValueError(f"stale calendar verifier remains: {task_path}")
                if not {"solution/answer.json", "solution/solve.sh"} <= files.keys():
                    raise ValueError(f"calendar oracle missing: {task_path}")
                namespace = {"__name__": "tasktrove_calendar_validation"}
                exec(files["tests/verifier.py"], namespace)
                data = json.loads(files["tests/verifier_data.json"])
                oracle = json.loads(files["solution/answer.json"])
                valid, errors = namespace["evaluate_calendar"](
                    data["expected_events"], oracle
                )
                if not valid:
                    raise ValueError(f"calendar oracle fails in {task_path}: {errors}")
                continue
            if INSTALLER_PATH not in files:
                raise ValueError(f"trusted installer missing: {task_path}")
            test_sh = files["tests/test.sh"]
            if (
                b"--reject" in test_sh
                or b"|| true"
                in test_sh.split(b"install_trusted_test_patch.sh", 1)[0][-160:]
            ):
                raise ValueError(f"best-effort hidden patch remains: {task_path}")
            if b"/tests/install_trusted_test_patch.sh" not in test_sh:
                raise ValueError(f"trusted installer is not invoked: {task_path}")
            if revision.verifier == "openswe":
                if b'echo 0 > "$logs_dir/reward.txt"' not in test_sh:
                    raise ValueError(f"OpenSWE has no scoreable-zero path: {task_path}")
                prefix = test_sh.split(b'echo ">>>>> Start Test Output"', 1)[0]
                if b'echo 0 > "$logs_dir/reward.txt"' in prefix:
                    raise ValueError(f"OpenSWE prewrites reward zero: {task_path}")
                config = TaskConfig.model_validate_toml(files["task.toml"].decode())
                if config.environment.storage_mb != OPENSWE_STORAGE_MB:
                    raise ValueError(f"wrong OpenSWE storage: {task_path}")
            elif revision.verifier == "swe_rebench":
                config = TaskConfig.model_validate_toml(files["task.toml"].decode())
                if config.environment.memory_mb != SWE_REBENCH_MEMORY_MB:
                    raise ValueError(f"wrong SWE-ReBench memory: {task_path}")
                if config.environment.storage_mb != SWE_REBENCH_STORAGE_MB:
                    raise ValueError(f"wrong SWE-ReBench storage: {task_path}")
            for content in (test_sh, files[INSTALLER_PATH]):
                digest = str(hash(content))
                if digest in shell_hashes:
                    continue
                checked = subprocess.run(
                    ["bash", "-n"], input=content, capture_output=True, check=False
                )
                if checked.returncode:
                    raise ValueError(
                        f"invalid shell in {task_path}: "
                        + checked.stderr.decode(errors="replace")
                    )
                shell_hashes.add(digest)
    return images


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
        output,
        TASK_SCHEMA,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    )
    rows = 0
    rejected_rows = 0
    test_patch_rows = 0
    try:
        for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
            transformed_rows = []
            for row in batch.to_pylist():
                if row["path"] in EXCLUDED_TASKS.get(revision.source, set()):
                    rejected_rows += 1
                    continue
                files = read_task(row["task_binary"])
                if not REQUIRED_MEMBERS <= files.keys():
                    raise ValueError(f"incomplete task: {row['path']}")
                if files.get("tests/test_patch.diff", b"").strip():
                    test_patch_rows += 1
                try:
                    transformed = transform_files(files, revision)
                except UnrecoverableCalendarTask:
                    rejected_rows += 1
                    continue
                changed = {
                    name
                    for name in files.keys() | transformed.keys()
                    if files.get(name) != transformed.get(name)
                }
                expected_changed = {"tests/test.sh", INSTALLER_PATH}
                if revision.verifier == "swe_rebench":
                    expected_changed.add("task.toml")
                elif revision.verifier == "openswe":
                    expected_changed.add("task.toml")
                elif revision.verifier == "calendar":
                    expected_changed = {
                        "solution/answer.json",
                        "solution/solve.sh",
                        "tests/verifier.py",
                        "tests/verifier_data.json",
                    }
                if changed != expected_changed:
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
    validate_output(output, revision, rows)
    return {
        "source": revision.source,
        "source_sha256": revision.source_sha256,
        "source_rows": parquet.metadata.num_rows,
        "output": revision.output,
        "output_rows": rows,
        "rejected_rows": rejected_rows,
        "output_sha256": file_sha256(output),
        "parquet": str(output.relative_to(args.stage)),
        "verifier": revision.verifier,
        "rows_with_test_patch": test_patch_rows,
        "rows_without_test_patch": rows - test_patch_rows,
        "storage_mb": {
            "swe_rebench": SWE_REBENCH_STORAGE_MB,
            "openswe": OPENSWE_STORAGE_MB,
        }.get(revision.verifier),
        "memory_mb": {
            "swe_rebench": SWE_REBENCH_MEMORY_MB,
        }.get(revision.verifier),
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
        "source_version": "4.8",
        "target_version": "4.9",
        "datasets": reports,
    }
    (args.stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
