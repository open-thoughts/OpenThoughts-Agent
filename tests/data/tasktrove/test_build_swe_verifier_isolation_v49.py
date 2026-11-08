import json

import pytest
from harbor_config.models.task.config import TaskConfig

from data.tasktrove.build_swe_verifier_isolation_v49 import (
    OPENSWE_PATCH_BLOCK,
    OPENSWE_SUCCESS_BLOCK,
    SWE_REBENCH_PATCH_BLOCK,
    REVISIONS,
    patch_paths,
    patch_openswe_verifier,
    patch_calendar_verifier,
    patch_swe_rebench_verifier,
    read_task,
    transform_files,
    write_task,
)
from data.nemotron_gym.verifiers import CALENDAR_VERIFIER_PY


BASE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_swe_rebench_verifier_rejects_partial_hidden_patch_application() -> None:
    transformed = patch_swe_rebench_verifier(
        b"#!/bin/bash\n" + SWE_REBENCH_PATCH_BLOCK + b"echo test-run\n",
        BASE_COMMIT,
    )

    assert b"install_trusted_test_patch.sh" in transformed
    assert b"--reject" not in transformed
    assert b"exit 1" in transformed


def test_openswe_setup_failure_does_not_emit_reward() -> None:
    source = (
        b"#!/bin/bash\n"
        b"# OpenSWE v6 verifier: provision dependencies and score only executed tests.\n"
        b"logs_dir=/logs/verifier\n"
        b'mkdir -p "$logs_dir"\n'
        b'echo 0 > "$logs_dir/reward.txt"\n'
        b"bash /tests/setup_files/setup.sh\n"
        b"setup_rc=$?\n"
        b'if [ "$setup_rc" -ne 0 ]; then\n'
        b'    echo "OPENSWE_SETUP_EXIT_CODE=$setup_rc"\n'
        b"    exit 0\n"
        b"fi\n"
        b"source /tmp/openswe-setup-environment.sh || exit 0\n"
        + OPENSWE_PATCH_BLOCK
        + b'echo ">>>>> Start Test Output"\n'
        + OPENSWE_SUCCESS_BLOCK
    )

    transformed = patch_openswe_verifier(source, BASE_COMMIT)
    before_tests = transformed.split(b'echo ">>>>> Start Test Output"', 1)[0]

    assert b'rm -f "$logs_dir/reward.txt"' in before_tests
    assert b'echo 0 > "$logs_dir/reward.txt"' not in before_tests
    assert b'exit "$setup_rc"' in before_tests
    assert b"install_trusted_test_patch.sh" in before_tests
    assert b'echo 0 > "$logs_dir/reward.txt"' in transformed


def test_verifier_transform_rejects_unknown_source_contract() -> None:
    with pytest.raises(ValueError, match="does not match"):
        patch_swe_rebench_verifier(b"#!/bin/bash\necho changed\n", BASE_COMMIT)


def test_patch_paths_preserves_unquoted_spaces() -> None:
    patch = (
        b"diff --git a/Sources/Parsable Types/Value.swift "
        b"b/Sources/Parsable Types/Value.swift\n"
    )

    assert patch_paths(patch) == {"Sources/Parsable Types/Value.swift"}


def test_swe_rebench_successor_requests_explicit_memory_and_storage() -> None:
    files = {
        "task.toml": b'version = "1.0"\n',
        "tests/config.json": json.dumps({"base_commit": BASE_COMMIT}).encode(),
        "tests/test.sh": b"#!/bin/bash\n" + SWE_REBENCH_PATCH_BLOCK,
        "tests/test_patch.diff": b"",
        "solution/solve.sh": b"#!/bin/bash\n",
    }

    transformed = transform_files(files, REVISIONS[0])
    config = TaskConfig.model_validate_toml(transformed["task.toml"].decode())

    assert config.environment.memory_mb == 4096
    assert config.environment.storage_mb == 8192


def test_openswe_successor_requests_four_gibibytes_storage() -> None:
    files = {
        "task.toml": b'version = "1.0"\n',
        "tests/config.json": json.dumps({"base_commit": BASE_COMMIT}).encode(),
        "tests/test.sh": (
            b"#!/bin/bash\n"
            b"# OpenSWE v6 verifier: provision dependencies and score only executed tests.\n"
            b'logs_dir=/logs/verifier\nmkdir -p "$logs_dir"\n'
            b'echo 0 > "$logs_dir/reward.txt"\n'
            b"bash /tests/setup_files/setup.sh\nsetup_rc=$?\n"
            b'if [ "$setup_rc" -ne 0 ]; then\n'
            b'    echo "OPENSWE_SETUP_EXIT_CODE=$setup_rc"\n    exit 0\nfi\n'
            b"source /tmp/openswe-setup-environment.sh || exit 0\n"
            + OPENSWE_PATCH_BLOCK
            + b'echo ">>>>> Start Test Output"\n'
            + OPENSWE_SUCCESS_BLOCK
        ),
        "tests/test_patch.diff": b"",
        "solution/solve.sh": b"#!/bin/bash\n",
    }

    transformed = transform_files(files, REVISIONS[1])
    config = TaskConfig.model_validate_toml(transformed["task.toml"].decode())

    assert config.environment.storage_mb == 4096


def test_calendar_successor_replaces_stale_packaged_verifier() -> None:
    stale = b"def evaluate_calendar(expected, events):\n    return True, []\n"

    transformed = patch_calendar_verifier(
        {
            "instruction.md": b"Ensure that there are no conflicts (overlapping events).",
            "tests/verifier.py": stale,
            "tests/verifier_data.json": json.dumps(
                {
                    "expected_events": {
                        "0": {
                            "event_id": 0,
                            "event_name": "A",
                            "duration": 30,
                            "min_time": "10:00",
                            "max_time": "12:00",
                            "constraint": None,
                        },
                        "1": {
                            "event_id": 1,
                            "event_name": "B",
                            "duration": 30,
                            "min_time": "10:00",
                            "max_time": "12:00",
                            "constraint": None,
                        },
                    }
                }
            ).encode(),
        }
    )

    packaged = read_task(write_task(transformed))
    namespace = {"__name__": "packaged_calendar_test"}
    exec(packaged["tests/verifier.py"], namespace)
    expected = {
        "0": {
            "event_id": 0,
            "event_name": "A",
            "duration": 30,
            "min_time": "10:00",
            "max_time": "12:00",
            "constraint": None,
        },
        "1": {
            "event_id": 1,
            "event_name": "B",
            "duration": 30,
            "min_time": "10:00",
            "max_time": "12:00",
            "constraint": None,
        },
    }
    overlap = [
        {"event_id": 0, "event_name": "A", "start_time": "10:00", "duration": 30},
        {"event_id": 1, "event_name": "B", "start_time": "10:15", "duration": 30},
    ]
    back_to_back = [overlap[0], {**overlap[1], "start_time": "10:30"}]

    assert packaged["tests/verifier.py"] == CALENDAR_VERIFIER_PY.encode()
    assert packaged["instruction.md"].count(b"overlap") == 1
    oracle = json.loads(packaged["solution/answer.json"])
    assert namespace["evaluate_calendar"](expected, oracle) == (True, [])
    assert b"/solution/answer.json /app/answer.txt" in packaged["solution/solve.sh"]
    assert namespace["evaluate_calendar"](expected, overlap)[0] is False
    assert namespace["evaluate_calendar"](expected, back_to_back) == (True, [])
