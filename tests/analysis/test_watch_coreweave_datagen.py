from __future__ import annotations

import subprocess
from types import SimpleNamespace

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


def test_run_iris_retries_transient_finelog_stats_error(monkeypatch):
    results = iter(
        [
            subprocess.CompletedProcess(
                [],
                1,
                stderr=(
                    "raise _translate_connect_error(exc) from exc "
                    "finelog.errors.StatsError"
                ),
            ),
            subprocess.CompletedProcess([], 0, stdout="logs\n"),
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
        ["job", "logs", "/benjaminfeuer/glm52-pilot-codecontests-r10", "--no-tail"],
        cluster="cw-rno2a",
        iris_bin="/fake/iris",
    )

    assert result.returncode == 0
    assert len(calls) == 2
    assert delays == [iris_ops.DNS_INITIAL_BACKOFF]


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


def test_finelog_activity_reports_visible_eval_trials(tmp_path):
    log_path = tmp_path / "finelog.log"
    log_path.write_text(
        "alpha__Ab12: starting environment...\n"
        "beta__Cd34: running agent\n"
        "alpha__Ab12: starting environment...\n"
    )

    assert monitor.finelog_activity(log_path) == "finelog (2 recent trial IDs)"


def test_harbor_report_row_keeps_monitor_exception_text_out_of_table():
    cluster = monitor.Cluster("cw-rno2a", monitor.Path("/fake/iris"), {})
    job = monitor.HarborJob(
        cluster,
        "/benjaminfeuer/tracegen-test",
        "running",
        1,
        "datagen",
        "s3://bucket/runs",
        "tracegen-test",
        "DCAgent/tasks",
    )
    progress = monitor.Progress(
        3,
        10,
        "unavailable",
        "ClientError: raw proxy exception body",
    )

    row = monitor.report_row(job, progress, "output-unavailable", None, "unavailable")

    assert "raw proxy exception body" not in repr(row)
    assert len(row) == 9
    assert row[-1].value == "output-unavailable"


def test_harbor_main_degrades_job_failures_and_writes_separate_error_report(
    monkeypatch, tmp_path, capsys
):
    cluster = monitor.Cluster("cw-rno2a", monitor.Path("/fake/iris"), {})
    job = monitor.HarborJob(
        cluster,
        "/benjaminfeuer/tracegen-test",
        "running",
        1,
        "datagen",
        "s3://bucket/runs",
        "tracegen-test",
        "DCAgent/tasks",
    )
    monkeypatch.setattr(
        monitor,
        "parse_args",
        lambda: SimpleNamespace(
            bundle_root=tmp_path,
            stalled_after_minutes=120,
            hours=24.0,
            job=None,
            filter=[],
            notify=False,
        ),
    )
    monkeypatch.setattr(monitor, "CLUSTERS", (cluster,))
    monkeypatch.setattr(
        monitor, "discover_harbor_jobs", lambda *_args, **_kwargs: ([job], [])
    )
    monkeypatch.setattr(
        monitor,
        "fetch_finelog",
        lambda *_args, **_kwargs: (None, "finelog raw exception", None),
    )
    monkeypatch.setattr(
        monitor,
        "fetch_ray_vllm_logs",
        lambda *_args, **_kwargs: ("unavailable", "ray raw exception"),
    )
    monkeypatch.setattr(monitor, "coreweave_client", lambda _cluster: object())
    monkeypatch.setattr(
        monitor,
        "read_s3_progress",
        lambda *_args, **_kwargs: monitor.Progress(
            None, None, "unavailable", "progress raw exception"
        ),
    )
    monkeypatch.setattr(
        monitor, "write_bundle_manifest", lambda *_args, **_kwargs: None
    )

    assert monitor.main() == 0

    stdout = capsys.readouterr().out
    assert "raw exception" not in stdout
    errors = (tmp_path / "reports/harbor/latest-errors.md").read_text()
    assert "finelog raw exception" in errors
    assert "ray raw exception" in errors
    assert "progress raw exception" in errors
