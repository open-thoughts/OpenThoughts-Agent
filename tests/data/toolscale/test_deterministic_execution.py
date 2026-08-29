from __future__ import annotations

import json
import os
import subprocess
import sys

from data.toolscale.deterministic_execution import (
    build_domain_catalog,
    expected_fixture,
    render_check,
    render_runtime,
)


def _source_row() -> dict:
    return {
        "id": "calendar-1",
        "user_scenario": {
            "instructions": {
                "domain": "calendar",
                "task_instructions": "Find the design review.",
                "known_info": "The attendee is user-7.",
            }
        },
        "evaluation_criteria": {
            "actions": [
                {
                    "name": "find_events",
                    "arguments": {"attendee_id": "user-7", "day": "2026-09-01"},
                }
            ],
            "nl_assertions": ["The design review starts at 10:00."],
        },
    }


def _runtime(tmp_path):
    row = _source_row()
    fixture = expected_fixture(row, "toolscale-v4-0000")
    catalog = build_domain_catalog([row])["calendar"]
    runtime_path = tmp_path / "toolscale_runtime.py"
    runtime_path.write_text(render_runtime(fixture, catalog))
    return fixture, runtime_path


def test_fixture_backed_runtime_and_verifier_accept_executed_call(tmp_path) -> None:
    fixture, runtime_path = _runtime(tmp_path)
    app_dir = tmp_path / "app"
    tests_dir = tmp_path / "tests"
    app_dir.mkdir()
    tests_dir.mkdir()
    log_path = app_dir / "toolscale_calls.jsonl"
    arguments = json.dumps(fixture["calls"][0]["arguments"])

    result = subprocess.run(
        [
            sys.executable,
            runtime_path,
            "call",
            "--task-id",
            fixture["task_id"],
            "--tool",
            fixture["calls"][0]["name"],
            "--arguments",
            arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "TOOL_SCALE_LOG": str(log_path)},
    )
    call_result = json.loads(result.stdout)
    assert call_result["evidence_id"] == fixture["calls"][0]["evidence_id"]
    assert json.loads(log_path.read_text()) == call_result

    (tests_dir / "expected.json").write_text(json.dumps(fixture))
    check_path = tests_dir / "check.py"
    check_path.write_text(render_check())
    (app_dir / "response.md").write_text(
        f"{fixture['assertions'][0]}\n{fixture['calls'][0]['evidence_id']}\n"
    )
    verifier = subprocess.run(
        [sys.executable, check_path],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "APP_DIR": str(app_dir),
            "TOOL_SCALE_TESTS_DIR": str(tests_dir),
        },
    )
    assert verifier.returncode == 0, verifier.stderr


def test_runtime_rejects_non_fixture_arguments_without_logging(tmp_path) -> None:
    fixture, runtime_path = _runtime(tmp_path)
    log_path = tmp_path / "toolscale_calls.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            runtime_path,
            "call",
            "--task-id",
            fixture["task_id"],
            "--tool",
            fixture["calls"][0]["name"],
            "--arguments",
            json.dumps({"attendee_id": "another-user", "day": "2026-09-01"}),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "TOOL_SCALE_LOG": str(log_path)},
    )
    assert result.returncode != 0
    assert "no fixture-backed result" in result.stderr
    assert not log_path.exists()
