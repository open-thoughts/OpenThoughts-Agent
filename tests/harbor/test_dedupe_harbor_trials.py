import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from harbor.models.trial.result import TrialResult

from scripts.harbor.dedupe_harbor_trials import deduplicate_job


def _trial_result(
    source: TrialResult, *, started_at: datetime, reward: float | None
) -> TrialResult:
    data = source.model_dump(mode="json")
    data["id"] = str(data["id"])
    data["trial_name"] = f"{source.task_name}__{started_at.timestamp()}"
    data["started_at"] = started_at.isoformat()
    data["finished_at"] = (started_at + timedelta(seconds=1)).isoformat()
    if reward is None:
        data["verifier_result"] = None
    else:
        data["verifier_result"] = {"rewards": {"reward": reward}}
    return TrialResult.model_validate(data)


def _write_trial(job_dir: Path, result: TrialResult) -> Path:
    trial_dir = job_dir / result.trial_name
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(result.model_dump_json())
    return trial_dir


def test_deduplicate_job_prefers_earliest_scored_attempts_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source = TrialResult.model_validate(
        {
            "task_name": "task",
            "trial_name": "task__source",
            "trial_uri": "file:///tmp/task__source",
            "task_id": {"path": "/task"},
            "task_checksum": "checksum",
            "config": {"task": {"path": "/task"}},
            "agent_info": {
                "name": "agent",
                "version": "1",
                "model_info": {"name": "model"},
            },
            "source": "benchmark",
        }
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    attempts = [
        _trial_result(source, started_at=now, reward=None),
        _trial_result(source, started_at=now + timedelta(seconds=1), reward=0.0),
        _trial_result(source, started_at=now + timedelta(seconds=2), reward=1.0),
        _trial_result(source, started_at=now + timedelta(seconds=3), reward=1.0),
    ]
    for result in attempts:
        _write_trial(tmp_path, result)

    dry_run = deduplicate_job(tmp_path, 2)
    assert (dry_run.before, dry_run.after, dry_run.removed) == (4, 2, 2)
    assert len(list(tmp_path.glob("*/result.json"))) == 4

    applied = deduplicate_job(tmp_path, 2, apply=True)
    assert applied == dry_run
    kept_rewards = []
    for path in tmp_path.glob("*/result.json"):
        result = TrialResult.model_validate_json(path.read_text())
        kept_rewards.append(result.verifier_result.rewards["reward"])
    assert sorted(kept_rewards) == [0.0, 1.0]
    assert json.loads((tmp_path / "result.json").read_text())["n_total_trials"] == 2
    manifest = (tmp_path / "dedupe_manifest.json").read_text()

    second = deduplicate_job(tmp_path, 2, apply=True)
    assert (second.before, second.after, second.removed) == (2, 2, 0)
    assert (tmp_path / "dedupe_manifest.json").read_text() == manifest


def test_deduplicate_job_preserves_kept_result_nested_below_removed_result(
    tmp_path: Path,
) -> None:
    source = TrialResult.model_validate(
        {
            "task_name": "task",
            "trial_name": "task__source",
            "trial_uri": "file:///tmp/task__source",
            "task_id": {"path": "/task"},
            "task_checksum": "checksum",
            "config": {"task": {"path": "/task"}},
            "agent_info": {
                "name": "agent",
                "version": "1",
                "model_info": {"name": "model"},
            },
            "source": "benchmark",
        }
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    outer = _trial_result(source, started_at=now, reward=None)
    inner = _trial_result(source, started_at=now + timedelta(seconds=1), reward=1.0)
    outer_dir = _write_trial(tmp_path, outer)
    inner_path = outer_dir / inner.trial_name / "result.json"
    inner_path.parent.mkdir()
    inner_path.write_text(inner.model_dump_json())

    summary = deduplicate_job(tmp_path, 1, apply=True)

    assert (summary.before, summary.after, summary.removed) == (2, 1, 1)
    assert not (outer_dir / "result.json").exists()
    assert inner_path.exists()
    [removed] = json.loads((tmp_path / "dedupe_manifest.json").read_text())["removed"]
    assert removed["removal_action"] == "file"


def test_deduplicate_job_removes_duplicate_nested_below_kept_result(
    tmp_path: Path,
) -> None:
    source = TrialResult.model_validate(
        {
            "task_name": "task",
            "trial_name": "task__source",
            "trial_uri": "file:///tmp/task__source",
            "task_id": {"path": "/task"},
            "task_checksum": "checksum",
            "config": {"task": {"path": "/task"}},
            "agent_info": {
                "name": "agent",
                "version": "1",
                "model_info": {"name": "model"},
            },
            "source": "benchmark",
        }
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    kept = _trial_result(source, started_at=now, reward=1.0)
    duplicate = kept.model_copy(deep=True)
    kept_dir = _write_trial(tmp_path, kept)
    nested_path = kept_dir / "copied-results" / kept.trial_name / "result.json"
    nested_path.parent.mkdir(parents=True)
    nested_path.write_text(duplicate.model_dump_json())

    summary = deduplicate_job(tmp_path, 1, apply=True)

    assert (summary.before, summary.after, summary.removed) == (2, 1, 1)
    assert (kept_dir / "result.json").exists()
    assert not nested_path.exists()
    [removed] = json.loads((tmp_path / "dedupe_manifest.json").read_text())["removed"]
    assert removed["removal_action"] == "directory"


def test_deduplicate_job_collapses_copied_trial_names_before_counting_replicas(
    tmp_path: Path,
) -> None:
    source = TrialResult.model_validate(
        {
            "task_name": "task",
            "trial_name": "task__source",
            "trial_uri": "file:///tmp/task__source",
            "task_id": {"path": "/task"},
            "task_checksum": "checksum",
            "config": {"task": {"path": "/task"}},
            "agent_info": {
                "name": "agent",
                "version": "1",
                "model_info": {"name": "model"},
            },
            "source": "benchmark",
        }
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    copied = _trial_result(source, started_at=now, reward=0.0)
    distinct = _trial_result(source, started_at=now + timedelta(seconds=1), reward=1.0)
    _write_trial(tmp_path, copied)
    copied_dir = tmp_path / "copied-results" / copied.trial_name
    copied_dir.mkdir(parents=True)
    (copied_dir / "result.json").write_text(copied.model_dump_json())
    _write_trial(tmp_path, distinct)

    summary = deduplicate_job(tmp_path, 2, apply=True)

    assert (summary.before, summary.after, summary.removed) == (3, 2, 1)
    kept = [
        TrialResult.model_validate_json(path.read_text())
        for path in tmp_path.rglob("result.json")
        if path.parent != tmp_path
    ]
    assert len({result.trial_name for result in kept}) == 2


def test_deduplicate_job_collapses_trial_name_across_inconsistent_task_keys(
    tmp_path: Path,
) -> None:
    source = TrialResult.model_validate(
        {
            "task_name": "task",
            "trial_name": "task__source",
            "trial_uri": "file:///tmp/task__source",
            "task_id": {"path": "/task"},
            "task_checksum": "checksum",
            "config": {"task": {"path": "/task"}},
            "agent_info": {
                "name": "agent",
                "version": "1",
                "model_info": {"name": "model"},
            },
            "source": "benchmark",
        }
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    original = _trial_result(source, started_at=now, reward=0.0)
    copied_data = original.model_dump(mode="json")
    copied_data["task_name"] = "stale-task-key"
    copied = TrialResult.model_validate(copied_data)
    _write_trial(tmp_path, original)
    copied_dir = tmp_path / "copied-results" / copied.trial_name
    copied_dir.mkdir(parents=True)
    (copied_dir / "result.json").write_text(copied.model_dump_json())

    summary = deduplicate_job(tmp_path, 3, apply=True)

    assert (summary.before, summary.after, summary.removed) == (2, 1, 1)
    kept = [
        TrialResult.model_validate_json(path.read_text())
        for path in tmp_path.rglob("result.json")
        if path.parent != tmp_path
    ]
    assert [result.trial_name for result in kept] == [original.trial_name]
