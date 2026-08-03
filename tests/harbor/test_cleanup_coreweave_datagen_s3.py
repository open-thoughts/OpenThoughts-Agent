import json
from pathlib import Path

import pytest

from scripts.harbor.cleanup_coreweave_datagen_s3 import (
    CleanupTarget,
    direct_trial_directory,
    download_prefix,
    local_object_path,
    merge_trial_directories,
    parse_target,
    realness_summary,
)


class FakePaginator:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def paginate(self, *, Bucket: str, Prefix: str):
        del Bucket
        yield {
            "Contents": [
                {"Key": key, "Size": len(value)}
                for key, value in self.objects.items()
                if key.startswith(Prefix)
            ]
        }


class FakeS3Client:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def get_paginator(self, operation: str):
        assert operation == "list_objects_v2"
        return FakePaginator(self.objects)

    def download_file(self, bucket: str, key: str, destination: str):
        del bucket
        Path(destination).write_bytes(self.objects[key])


def write_trial(
    run_dir: Path, name: str, turns: int, exception_type: str | None = None
) -> None:
    trial_dir = run_dir / name
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "exception_info": {"exception_type": exception_type}
                if exception_type
                else None
            }
        )
    )
    (trial_dir / "agent" / "trajectory.json").write_text(
        json.dumps({"steps": [{"index": index} for index in range(turns)]})
    )


@pytest.mark.parametrize(
    "relative_run",
    [Path("job-name"), Path("trace_jobs") / "job-name"],
)
def test_direct_trial_directory_finds_run_from_trial_results(
    tmp_path: Path, relative_run: Path
) -> None:
    run_dir = tmp_path / relative_run
    (run_dir / "result.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text("{}")
    write_trial(run_dir, "task__abc", turns=3)

    assert direct_trial_directory(tmp_path) == run_dir


def test_direct_trial_directory_rejects_bundle_with_multiple_runs(
    tmp_path: Path,
) -> None:
    write_trial(tmp_path / "first", "task__abc", turns=2)
    write_trial(tmp_path / "second", "task__def", turns=2)

    with pytest.raises(RuntimeError, match="exactly one Harbor run"):
        direct_trial_directory(tmp_path)


def test_download_prefix_materializes_relative_s3_tree(
    tmp_path: Path, monkeypatch
) -> None:
    prefix = "iris/runs/job/"
    objects = {
        f"{prefix}job/config.json": b"{}",
        f"{prefix}job/task__abc/result.json": b'{"reward": 1}',
        "iris/runs/other/ignored.json": b"ignored",
    }
    fake_client = FakeS3Client(objects)
    monkeypatch.setattr(
        "scripts.harbor.cleanup_coreweave_datagen_s3.coreweave_s3_client",
        lambda: fake_client,
    )
    target = CleanupTarget("job", ("s3://bucket/iris/runs/job",), "penfever/job")

    copied = download_prefix(target, tmp_path)

    assert copied == 2
    assert (tmp_path / "job" / "config.json").read_bytes() == b"{}"
    assert (
        tmp_path / "job" / "task__abc" / "result.json"
    ).read_bytes() == b'{"reward": 1}'
    assert not (tmp_path / "ignored.json").exists()


def test_local_object_path_rejects_parent_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe relative S3 object key"):
        local_object_path(tmp_path, "../outside")


def test_realness_summary_reports_trials_turns_and_exceptions(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text('{"stats": {}}')
    write_trial(tmp_path, "task__abc", turns=2)
    write_trial(tmp_path, "task__def", turns=4, exception_type="AgentTimeoutError")

    assert realness_summary(tmp_path) == (2, 3.0, 1)


def test_parse_target_accepts_multiple_retry_prefixes() -> None:
    target = parse_target(
        "job|s3://bucket/iris/first,s3://bucket/iris/retry|penfever/job"
    )

    assert target == CleanupTarget(
        "job",
        ("s3://bucket/iris/first", "s3://bucket/iris/retry"),
        "penfever/job",
    )


def test_merge_trial_directories_keeps_all_retry_trials_and_run_metadata(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first-run"
    second = tmp_path / "second-run"
    first.mkdir()
    second.mkdir()
    (first / "config.json").write_text('{"run": "first"}')
    (second / "config.json").write_text('{"run": "second"}')
    write_trial(first, "same-task__attempt", turns=2)
    write_trial(second, "same-task__attempt", turns=4)

    merged = merge_trial_directories([first, second], tmp_path / "merged")

    assert realness_summary(merged) == (2, 3.0, 0)
    merged_runs = sorted(merged.iterdir())
    assert len(merged_runs) == 2
    assert [
        json.loads(run.joinpath("config.json").read_text())["run"]
        for run in merged_runs
    ] == [
        "first",
        "second",
    ]
    assert all(
        (run / "same-task__attempt" / "result.json").is_file() for run in merged_runs
    )
