"""Build the bounded TaskTrove v4.1 Batch A1 pure-source repairs.

The builder reads the exact TaskTrove v3.42 Parquet in batches of at most 32,
rewrites one source at a time, and emits a deterministic Parquet plus report.
It never uploads or mutates a remote repository.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq
from harbor_config.models.task.config import TaskConfig
from huggingface_hub import hf_hub_download

from data.nemotron_gym.adapter import STANDARD_TEST_SH
from data.nemotron_gym.converters.agent_calendar import _calendar_names_from_prompt
from data.nemotron_gym.verifiers import CALENDAR_VERIFIER_PY


TASKTROVE_REPO = "open-thoughts/TaskTrove"
TASKTROVE_V342_REVISION = "3e96fe6464ce5ab6209e98801caab29b4a1fe87a"
MAX_BATCH_ROWS = 32
MIN_RETAINED_TASKS = 300
CALENDAR_GRANULARITY_MINUTES = 5
TASK_SCHEMA = pa.schema([("path", pa.string()), ("task_binary", pa.binary())])
METHODS2TEST_BLOCK_REASON = (
    "no certifiable >=300-task repair: 0/32 evenly spaced normalized oracles "
    "passed the exact Java/Maven verifier image"
)
RSPEC_BLOCK_REASON = (
    "no packaged oracle and repeated exact-image empty-workspace probes produced "
    "false-positive rewards; final 300-task evenly spaced probe returned reward 1 "
    "for 3 tasks"
)

SOURCE_PATHS = {
    "methods2test": "laion__exp_rpt_methods2test-large-v4/tasks.parquet",
    "rspec": "laion__exp_rpt_stack-rspec-v3/tasks.parquet",
    "calendar": "laion__nemotron-gym-agent-calendar/tasks.parquet",
}
OUTPUT_PATHS = {
    "calendar": "laion__nemotron-gym-agent-calendar-v2/tasks.parquet",
}
CALENDAR_INSTRUCTION_OLD = (
    "The verifier checks duration, time-window, and any natural-language "
    "constraint per event."
)
CALENDAR_INSTRUCTION_NEW = (
    "The verifier checks the exact event IDs and names, duration, every declared "
    "time constraint, uniqueness, omissions, extra events, and overlap."
)
REQUIRED_TASK_MEMBERS = frozenset(
    {"environment/Dockerfile", "instruction.md", "task.toml", "tests/test.sh"}
)
CHANGED_MEMBER_ALLOWLISTS = {
    "calendar": frozenset(
        {
            "instruction.md",
            "solution/answer.json",
            "solution/solve.sh",
            "tests/test.sh",
            "tests/verifier.py",
            "tests/verifier_data.json",
        }
    ),
}
_CALENDAR_NAMESPACE = {"__name__": "tasktrove_calendar_preflight"}
exec(CALENDAR_VERIFIER_PY, _CALENDAR_NAMESPACE)
_EVALUATE_CALENDAR = _CALENDAR_NAMESPACE["evaluate_calendar"]
_CHECK_CALENDAR_CONSTRAINT = _CALENDAR_NAMESPACE["_check_constraint"]


def read_task(task_binary: bytes) -> dict[str, bytes]:
    """Read one gzipped Harbor task without extracting it to disk."""
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
    """Serialize a task deterministically."""
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


def _changed_members(
    original: dict[str, bytes], transformed: dict[str, bytes]
) -> set[str]:
    return {
        name
        for name in original.keys() | transformed.keys()
        if original.get(name) != transformed.get(name)
    }


def _validate_shell(script: bytes, name: str, validated: set[str]) -> None:
    digest = hashlib.sha256(script).hexdigest()
    if digest in validated:
        return
    result = subprocess.run(
        ["bash", "-n"], input=script, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise ValueError(f"bash syntax failed for {name}: {result.stderr.decode()}")
    validated.add(digest)


def _validate_transformed_task(
    source: str,
    original: dict[str, bytes],
    transformed: dict[str, bytes],
    validated_shells: set[str],
) -> None:
    missing = REQUIRED_TASK_MEMBERS - transformed.keys()
    if missing:
        raise ValueError(f"transformed task missing members: {sorted(missing)}")
    changed = _changed_members(original, transformed)
    unexpected = changed - CHANGED_MEMBER_ALLOWLISTS[source]
    if unexpected:
        raise ValueError(f"unexpected changed members: {sorted(unexpected)}")
    TaskConfig.model_validate_toml(transformed["task.toml"].decode("utf-8"))
    for name, content in transformed.items():
        if name.endswith(".sh"):
            _validate_shell(content, name, validated_shells)


def _validate_output_parquet(output_path: Path, expected_rows: int) -> None:
    parquet = pq.ParquetFile(output_path)
    if parquet.schema_arrow != TASK_SCHEMA:
        raise ValueError(f"unexpected output schema: {parquet.schema_arrow}")
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"output row count {parquet.metadata.num_rows} != {expected_rows}"
        )
    paths: set[str] = set()
    for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
        for row in batch.to_pylist():
            path = row["path"]
            if path in paths:
                raise ValueError(f"duplicate output dataset path: {path!r}")
            paths.add(path)
            files = read_task(row["task_binary"])
            if not REQUIRED_TASK_MEMBERS <= files.keys():
                raise ValueError(f"unsafe or incomplete output task: {path!r}")
            TaskConfig.model_validate_toml(files["task.toml"].decode("utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_parquet(source: str, stage: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return Path(
        hf_hub_download(
            TASKTROVE_REPO,
            SOURCE_PATHS[source],
            repo_type="dataset",
            revision=TASKTROVE_V342_REVISION,
            local_dir=stage / "source",
        )
    )


def _time_minutes(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(\d{2}):(\d{2})", value)
    if match is None:
        return None
    hour, minute = map(int, match.groups())
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _feasible_calendar(expected: dict[str, dict]) -> list[dict] | None:
    candidates: dict[int, list[int]] = {}
    for key, spec in expected.items():
        event_id = int(key)
        duration = spec.get("duration")
        minimum = _time_minutes(spec.get("min_time"))
        maximum = _time_minutes(spec.get("max_time"))
        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or duration <= 0
            or minimum is None
            or maximum is None
        ):
            return None
        starts = [
            start
            for start in range(
                minimum,
                maximum - duration + 1,
                CALENDAR_GRANULARITY_MINUTES,
            )
            if _CHECK_CALENDAR_CONSTRAINT(
                spec.get("constraint"), start, start + duration
            )
        ]
        if not starts:
            return None
        candidates[event_id] = starts

    event_ids = sorted(candidates)
    states: dict[int, tuple[int, dict[int, int]]] = {0: (0, {})}
    for mask in range(1 << len(event_ids)):
        state = states.get(mask)
        if state is None:
            continue
        previous_end, placements = state
        for index, event_id in enumerate(event_ids):
            bit = 1 << index
            if mask & bit:
                continue
            start = next(
                (
                    candidate
                    for candidate in candidates[event_id]
                    if candidate >= previous_end
                ),
                None,
            )
            if start is None:
                continue
            end = start + expected[str(event_id)]["duration"]
            next_mask = mask | bit
            existing = states.get(next_mask)
            if existing is not None and existing[0] <= end:
                continue
            states[next_mask] = (end, {**placements, event_id: start})

    complete = states.get((1 << len(event_ids)) - 1)
    if complete is None:
        return None
    placed = complete[1]
    return [
        {
            "event_id": event_id,
            "event_name": expected[str(event_id)]["event_name"],
            "start_time": f"{placed[event_id] // 60:02d}:{placed[event_id] % 60:02d}",
            "duration": expected[str(event_id)]["duration"],
        }
        for event_id in sorted(placed)
    ]


def _calendar_transform(
    files: dict[str, bytes],
) -> tuple[dict[str, bytes] | None, str | None]:
    required = {"instruction.md", "tests/verifier_data.json", "tests/verifier.py"}
    if not required <= files.keys():
        return None, "missing_calendar_files"
    instruction = files["instruction.md"].decode("utf-8", "replace")
    try:
        data = json.loads(files["tests/verifier_data.json"])
    except json.JSONDecodeError:
        return None, "invalid_verifier_data"
    expected = data.get("expected_events")
    if not isinstance(expected, dict) or not expected:
        return None, "missing_expected_events"
    names = _calendar_names_from_prompt(instruction)
    repaired: dict[str, dict] = {}
    for key, spec in expected.items():
        if not isinstance(spec, dict):
            return None, "invalid_expected_event"
        event_id = spec.get("event_id")
        if not isinstance(event_id, int) or isinstance(event_id, bool):
            return None, "invalid_expected_event_id"
        event_name = names.get(event_id)
        if event_name is None:
            return None, "unrecoverable_event_name"
        repaired[str(event_id)] = {**spec, "event_name": event_name}
    oracle = _feasible_calendar(repaired)
    if oracle is None:
        return None, "infeasible_calendar"
    valid, errors = _EVALUATE_CALENDAR(repaired, oracle)
    if not valid:
        return None, "oracle_rejected:" + ",".join(errors)

    out = dict(files)
    out["instruction.md"] = instruction.replace(
        CALENDAR_INSTRUCTION_OLD, CALENDAR_INSTRUCTION_NEW
    ).encode()
    out["tests/test.sh"] = STANDARD_TEST_SH.encode()
    out["tests/verifier.py"] = CALENDAR_VERIFIER_PY.encode()
    out["tests/verifier_data.json"] = json.dumps(
        {**data, "expected_events": repaired},
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode()
    out["solution/answer.json"] = json.dumps(
        oracle, ensure_ascii=False, sort_keys=True, indent=2
    ).encode()
    out["solution/solve.sh"] = (
        "#!/bin/bash\nset -eu\ncp /solution/answer.json /app/answer.txt\n"
    ).encode()
    return out, None


TRANSFORMS = {
    "calendar": _calendar_transform,
}


def build(source: str, stage: Path, source_parquet: Path | None) -> dict[str, object]:
    """Build one repaired source and return its machine-readable report."""
    if stage.exists() and any(stage.iterdir()):
        raise ValueError(f"stage must be absent or empty: {stage}")
    stage.mkdir(parents=True, exist_ok=True)
    if source in {"methods2test", "rspec"}:
        reason = (
            METHODS2TEST_BLOCK_REASON
            if source == "methods2test"
            else RSPEC_BLOCK_REASON
        )
        report = {
            "source": source,
            "source_repo_path": SOURCE_PATHS[source],
            "source_revision": TASKTROVE_V342_REVISION,
            "status": "blocked",
            "reason": reason,
        }
        if source == "methods2test":
            report.update({"probe_tasks": 32, "probe_oracle_passes": 0})
        else:
            report.update(
                {
                    "probe_tasks": 300,
                    "probe_false_positives": 3,
                    "false_positive_paths": [
                        "stack-ruby-2586",
                        "stack-ruby-3589",
                        "stack-ruby-9932",
                    ],
                    "disposition": "retain v3 unchanged and quarantine",
                }
            )
        (stage / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise ValueError(reason)
    source_path = _source_parquet(source, stage, source_parquet)
    output_path = stage / OUTPUT_PATHS[source]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(source_path)
    if parquet.schema_arrow != TASK_SCHEMA:
        raise ValueError(f"unexpected source schema: {parquet.schema_arrow}")
    writer: pq.ParquetWriter | None = None
    rejected: Counter[str] = Counter()
    retained = 0
    source_paths: set[str] = set()
    validated_shells: set[str] = set()
    try:
        for batch in parquet.iter_batches(batch_size=MAX_BATCH_ROWS):
            output_rows: list[dict[str, object]] = []
            for row in batch.to_pylist():
                path = row["path"]
                if not isinstance(path, str) or not path:
                    raise ValueError(f"invalid dataset path: {path!r}")
                if path in source_paths:
                    raise ValueError(f"duplicate source dataset path: {path!r}")
                source_paths.add(path)
                try:
                    files = read_task(row["task_binary"])
                except (tarfile.TarError, ValueError):
                    rejected["unsafe_source_archive"] += 1
                    continue
                transformed, reason = TRANSFORMS[source](files)
                if transformed is None:
                    rejected[reason or "unknown"] += 1
                    continue
                _validate_transformed_task(source, files, transformed, validated_shells)
                task_binary = write_task(transformed)
                if read_task(task_binary) != transformed:
                    raise ValueError(f"serialized task mismatch: {path!r}")
                output_rows.append({"path": path, "task_binary": task_binary})
            if not output_rows:
                continue
            table = pa.Table.from_pylist(
                output_rows,
                schema=TASK_SCHEMA,
            )
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=MAX_BATCH_ROWS)
            retained += len(output_rows)
    finally:
        if writer is not None:
            writer.close()

    if retained < MIN_RETAINED_TASKS:
        raise ValueError(
            f"{source} retained {retained}, below required {MIN_RETAINED_TASKS}"
        )
    _validate_output_parquet(output_path, retained)
    report: dict[str, object] = {
        "source": source,
        "source_repo_path": SOURCE_PATHS[source],
        "source_revision": TASKTROVE_V342_REVISION,
        "source_rows": parquet.metadata.num_rows,
        "source_sha256": _sha256(source_path),
        "output_repo_path": OUTPUT_PATHS[source],
        "retained_rows": retained,
        "dropped_rows": parquet.metadata.num_rows - retained,
        "drop_reasons": dict(sorted(rejected.items())),
        "output_sha256": _sha256(output_path),
        "max_batch_rows": MAX_BATCH_ROWS,
        "validation": {
            "archive_safety": True,
            "bash_syntax": True,
            "changed_member_allowlist": True,
            "harbor_task_config": True,
            "schema_and_unique_paths": True,
        },
    }
    (stage / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCE_PATHS), required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path)
    args = parser.parse_args()
    report = build(args.source, args.stage, args.source_parquet)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
