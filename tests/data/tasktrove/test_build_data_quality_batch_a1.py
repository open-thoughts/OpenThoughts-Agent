from __future__ import annotations

import io
import json
import tarfile

import pytest

from data.tasktrove.build_data_quality_batch_a1 import (
    _feasible_calendar,
    build,
    read_task,
)


def _archive(members: list[tuple[tarfile.TarInfo, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for member, content in members:
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def _file(name: str) -> tarfile.TarInfo:
    return tarfile.TarInfo(name)


@pytest.mark.parametrize("name", ["/absolute", "../escape", "tests/../escape"])
def test_read_task_rejects_unsafe_paths(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe archive member path"):
        read_task(_archive([(_file(name), b"bad")]))


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_read_task_rejects_links(link_type: bytes) -> None:
    link = tarfile.TarInfo("tests/link")
    link.type = link_type
    link.linkname = "../instruction.md"
    with pytest.raises(ValueError, match="archive links are forbidden"):
        read_task(_archive([(link, b"")]))


def test_read_task_rejects_duplicate_files() -> None:
    with pytest.raises(ValueError, match="duplicate archive member"):
        read_task(
            _archive(
                [
                    (_file("instruction.md"), b"first"),
                    (_file("instruction.md"), b"second"),
                ]
            )
        )


def test_rspec_build_is_blocked_without_output(tmp_path) -> None:
    stage = tmp_path / "rspec"
    with pytest.raises(ValueError, match="no packaged oracle"):
        build("rspec", stage, None)

    report = json.loads((stage / "report.json").read_text())
    assert report["status"] == "blocked"
    assert report["probe_false_positives"] == 3
    assert not list(stage.rglob("*.parquet"))


def test_calendar_solver_supports_five_minute_constraints() -> None:
    expected = {
        "0": {
            "event_id": 0,
            "event_name": "Five-minute boundary",
            "duration": 10,
            "min_time": "10:05",
            "max_time": "10:15",
            "constraint": "at 10:05am",
        }
    }

    assert _feasible_calendar(expected) == [
        {
            "event_id": 0,
            "event_name": "Five-minute boundary",
            "start_time": "10:05",
            "duration": 10,
        }
    ]
