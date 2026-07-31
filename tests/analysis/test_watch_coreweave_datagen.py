from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from scripts.iris import watch_iris_harbor as monitor
from scripts.iris import iris_ops


def test_run_iris_retries_transient_dns_failure(monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess(
                [],
                1,
                stderr="client error (Connect): dns error: no records found for Query",
            ),
            subprocess.CompletedProcess([], 0, stdout="job_id,state\n"),
        ]
    )
    calls: list[list[str]] = []
    delays: list[int] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return next(results)

    monkeypatch.setattr(iris_ops.subprocess, "run", fake_run)
    monkeypatch.setattr(iris_ops.time, "sleep", delays.append)

    result = iris_ops.run_iris_command(
        ["query", "SELECT 1"], cluster="marin", iris_bin="/fake/iris"
    )

    assert result.returncode == 0
    assert len(calls) == 2
    assert delays == [iris_ops.DNS_INITIAL_BACKOFF]


def test_run_iris_does_not_retry_non_dns_failure(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess([], 1, stderr="permission denied")

    monkeypatch.setattr(iris_ops.subprocess, "run", fake_run)
    monkeypatch.setattr(iris_ops.time, "sleep", lambda _delay: None)

    result = iris_ops.run_iris_command(
        ["query", "SELECT 1"], cluster="marin", iris_bin="/fake/iris"
    )

    assert result.returncode == 1
    assert len(calls) == 1


def test_progress_from_harbor_aggregate_includes_mean_and_typed_errors():
    progress = monitor.progress_from_harbor_aggregate(
        {
            "n_total_trials": 12,
            "stats": {
                "n_completed_trials": 8,
                "n_errored_trials": 4,
                "evals": {
                    "first": {
                        "n_trials": 3,
                        "metrics": [{"mean": 0.5}],
                        "exception_stats": {"AgentTimeoutError": ["a", "b"]},
                    },
                    "second": {
                        "n_trials": 5,
                        "metrics": [{"mean": 0.8}],
                        "exception_stats": {"APIConnectionError": ["c"]},
                    },
                },
            },
        },
        "Harbor aggregate",
    )

    assert progress.completed == 8
    assert progress.total == 12
    assert progress.mean_reward == 0.6875
    assert progress.error_counts == {
        "AgentTimeoutError": 2,
        "APIConnectionError": 1,
        "other Harbor errors": 1,
    }


def test_dataset_from_command_reads_tasks_input_path():
    command = "python run_tracegen.py --tasks_input_path DCAgent/code-contests-noblock --job_name pilot"

    assert monitor.dataset_from_command(command) == "DCAgent/code-contests-noblock"


def test_harbor_job_from_row_classifies_datagen_and_keeps_durable_identity():
    cluster = monitor.Cluster("test", monitor.Path("/fake/iris"), {})
    row = {
        "job_id": "/benjaminfeuer/tracegen-test",
        "state": "3",
        "submitted_at_ms": "1",
        "entrypoint_json": (
            "python run_tracegen.py --tasks_input_path DCAgent/tasks --job_name tracegen-test "
            "--harbor_extra_arg=--jobs-dir=s3://bucket/runs"
        ),
    }

    job = monitor.harbor_job_from_row(cluster, row)

    assert job is not None
    assert (job.kind, job.dataset, job.jobs_dir, job.harbor_job_name) == (
        "datagen",
        "DCAgent/tasks",
        "s3://bucket/runs",
        "tracegen-test",
    )


def test_harbor_job_from_row_classifies_eval_without_hiding_legacy_log_only_job():
    cluster = monitor.Cluster("test", monitor.Path("/fake/iris"), {})
    row = {
        "job_id": "/benjaminfeuer/eval-test",
        "state": "3",
        "submitted_at_ms": "1",
        "entrypoint_json": (
            "exec python -m eval.local.run_eval --dataset_path DCAgent/dev_set "
            "--harbor_config hpc/harbor_yaml/eval.yaml"
        ),
    }

    job = monitor.harbor_job_from_row(cluster, row)

    assert job is not None
    assert (job.kind, job.dataset, job.jobs_dir, job.harbor_job_name) == (
        "eval",
        "DCAgent/dev_set",
        None,
        None,
    )


def _queued_harbor_job():
    cluster = monitor.Cluster("cw-rno2a", monitor.Path("/fake/iris"), {})
    job = monitor.harbor_job_from_row(
        cluster,
        {
            "job_id": "/benjaminfeuer/tracegen-queued",
            "state": "3",
            "task_state": "2",
            "submitted_at_ms": "1",
            "entrypoint_json": (
                "python run_tracegen.py --tasks_input_path DCAgent/tasks "
                "--job_name tracegen-queued"
            ),
        },
    )
    assert job is not None
    return job


def test_harbor_job_uses_task_state_while_root_job_waits_for_placement():
    job = _queued_harbor_job()

    assert job.state == "running"
    assert job.task_state == "building"
    assert monitor.display_state(job) == "awaiting placement"


def test_queued_harbor_job_does_not_require_a_running_worker_pod(monkeypatch, tmp_path):
    job = _queued_harbor_job()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("queued jobs must not be probed for running worker logs")

    monkeypatch.setattr(monitor, "find_pod", fail_if_called)

    assert monitor.fetch_ray_vllm_logs(job, tmp_path) == ("awaiting placement", None)


def test_queued_harbor_job_health_is_not_an_output_failure():
    job = _queued_harbor_job()

    health, _ = monitor.health_label(
        job,
        monitor.Progress(None, None, "unavailable", error="no output yet"),
        {},
        datetime(2026, 7, 31, tzinfo=UTC),
        120,
    )

    assert health == "awaiting placement"


def test_finelog_activity_reports_visible_eval_trials(tmp_path):
    log_path = tmp_path / "finelog.log"
    log_path.write_text(
        "alpha__Ab12: starting environment...\n"
        "beta__Cd34: running agent\n"
        "alpha__Ab12: starting environment...\n"
    )

    assert monitor.finelog_activity(log_path) == "finelog (2 recent trial IDs)"
