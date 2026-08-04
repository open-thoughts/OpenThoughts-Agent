from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from scripts.iris import watch_iris_harbor as watcher


NOW = datetime(2026, 7, 26, 16, tzinfo=UTC)
CUTOFF = NOW - timedelta(hours=watcher.TRACE_TREND_HOURS)


def _controller_row(
    job_id: str, command: str, *, task_state: str = "3"
) -> dict[str, str]:
    return {
        "job_id": job_id,
        "state": "3",
        "task_state": task_state,
        "submitted_at_ms": str(int(NOW.timestamp() * 1000)),
        "entrypoint_json": command,
    }


def test_harbor_job_from_row_discovers_callable_eval_job():
    row = _controller_row(
        "/owner/eval-20260729-model-abcd",
        "exec $IRIS_PYTHON -u $IRIS_WORKDIR/_callable_runner.py",
    )

    job = watcher.harbor_job_from_row(
        watcher.Cluster("test", Path("/tmp/iris"), {}), row
    )

    assert job is not None
    assert job.kind == "eval"
    assert job.jobs_dir is None


def test_harbor_job_from_row_ignores_child_eval_worker():
    row = _controller_row(
        "/owner/eval-20260729-model-abcd/inference-1234",
        "exec $IRIS_PYTHON -u $IRIS_WORKDIR/_callable_runner.py",
    )

    assert (
        watcher.harbor_job_from_row(watcher.Cluster("test", Path("/tmp/iris"), {}), row)
        is None
    )


def test_discover_harbor_jobs_queries_only_root_jobs(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_run_iris(_cluster, args):
        captured["args"] = args
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "job_id,state,submitted_at_ms,entrypoint_json,task_state\n"
                "/owner/eval-20260729-model,3,1780000000000,"
                "exec $IRIS_PYTHON -u $IRIS_WORKDIR/_callable_runner.py,3\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(watcher, "run_iris", fake_run_iris)

    jobs, errors = watcher.discover_harbor_jobs(
        watcher.Cluster("test", Path("/tmp/iris"), {})
    )

    assert len(jobs) == 1
    assert errors == []
    assert "j.root_job_id = j.job_id" in captured["args"][1]
    assert "AS task_state" in captured["args"][1]


def test_discover_harbor_jobs_rolls_up_preemption_across_job_tree(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_run_iris(_cluster, args):
        captured["args"] = args
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "job_id,state,submitted_at_ms,entrypoint_json,task_state\n"
                "/owner/eval-20260729-model,3,1780000000000,"
                "exec $IRIS_PYTHON -u $IRIS_WORKDIR/_callable_runner.py,10\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(watcher, "run_iris", fake_run_iris)

    jobs, errors = watcher.discover_harbor_jobs(
        watcher.Cluster("test", Path("/tmp/iris"), {})
    )

    assert errors == []
    assert len(jobs) == 1
    assert jobs[0].task_state == "preempted"
    assert watcher.effective_state(jobs[0]) == "preempted"
    query = captured["args"][1]
    assert "tree_job.root_job_id = j.job_id" in query
    assert "task_attempts" in query
    assert "state=10" in query


def test_harbor_job_uses_task_state_while_root_job_waits_for_placement():
    row = _controller_row(
        "/owner/tracegen-queued",
        "python run_tracegen.py --tasks_input_path DCAgent/tasks "
        "--job_name tracegen-queued",
        task_state="2",
    )

    job = watcher.harbor_job_from_row(
        watcher.Cluster("cw-rno2a", Path("/tmp/iris"), {}), row
    )

    assert job is not None
    assert job.state == "running"
    assert job.task_state == "building"
    assert watcher.effective_state(job) == "awaiting placement"


def test_harbor_job_from_row_does_not_guess_non_eval_callable_job():
    row = _controller_row(
        "/owner/train-20260729-model-abcd",
        "exec $IRIS_PYTHON -u $IRIS_WORKDIR/_callable_runner.py",
    )

    job = watcher.harbor_job_from_row(
        watcher.Cluster("test", Path("/tmp/iris"), {}), row
    )

    assert job is None


def test_harbor_job_from_row_preserves_command_style_eval_detection():
    row = _controller_row(
        "/owner/custom-name",
        "python -m eval.local.run_eval --model test-model",
    )

    job = watcher.harbor_job_from_row(
        watcher.Cluster("test", Path("/tmp/iris"), {}), row
    )

    assert job is not None
    assert job.kind == "eval"
    assert job.model == "test-model"


def test_progress_from_eval_finelog_recovers_terminal_trials_and_errors(tmp_path):
    log = tmp_path / "finelog.log"
    log.write_text(
        "\n".join(
            [
                "[10:00:00] task=/owner/eval/0 | Trial alpha__a1: started",
                "[10:01:00] task=/owner/eval/0 | Trial alpha__a1: completed",
                "[10:02:00] task=/owner/eval/0 | Trial beta__b2: failed (AgentTimeoutError)",
                "[10:03:00] task=/owner/eval/0 | Trial gamma__c3: failed (VerifierRuntimeError)",
                "[10:04:00] task=/owner/eval/0 | Trial gamma__c3: started",
            ]
        )
        + "\n"
    )
    job = _job(age_hours=1)
    progress = watcher.progress_from_eval_finelog(
        job, log, NOW.replace(hour=10, minute=30)
    )

    assert progress.completion_source == "finelog lifecycle fallback"
    assert progress.completed == 2
    assert progress.total is None
    assert progress.error_counts == {"AgentTimeoutError": 1}
    assert progress.recent_completed == 2
    assert progress.recent_errored == 0
    assert progress.recent_benign_timeouts == 1


def test_resumed_eval_reads_cumulative_harbor_progress_from_finelog_identity(
    monkeypatch, tmp_path
):
    log = tmp_path / "finelog.log"
    log.write_text(
        "\n".join(
            [
                "starting Harbor job old-run (dataset=old jobs_dir=s3://bucket/results/old-run)",
                "starting Harbor job resumed-run "
                "(dataset=current jobs_dir=s3://bucket/results/resumed-run)",
            ]
        )
        + "\n"
    )
    job = _job(
        kind="eval",
        jobs_dir=None,
        harbor_job_name=None,
    )

    resolved = watcher.eval_job_with_finelog_harbor_identity(job, log)

    assert resolved.jobs_dir == "s3://bucket/results"
    assert resolved.harbor_job_name == "resumed-run"

    monkeypatch.setattr(
        watcher,
        "iter_objects",
        lambda *_args: [
            {
                "Key": "results/resumed-run/old-trial/result.json",
                "LastModified": NOW - timedelta(days=1),
            },
            {
                "Key": "results/resumed-run/new-trial/result.json",
                "LastModified": NOW - timedelta(minutes=10),
            },
        ],
    )
    aggregate = {
        "n_total_trials": 300,
        "stats": {
            "n_completed_trials": 2,
            "n_errored_trials": 0,
            "evals": {
                "reward": {
                    "n_trials": 2,
                    "metrics": [{"mean": 0.625}],
                }
            },
        },
    }

    class Client:
        def get_object(self, *, Bucket, Key):  # noqa: N803
            assert Bucket == "bucket"
            assert Key == "results/resumed-run/result.json"
            return {"Body": io.BytesIO(json.dumps(aggregate).encode())}

    progress = watcher.read_s3_progress(resolved, Client(), tmp_path, CUTOFF)

    assert progress.completed == 2
    assert progress.total == 300
    assert progress.mean_reward == 0.625


def _job(
    *,
    state: str = "running",
    task_state: str | None = "running",
    age_hours: int = 4,
    job_id: str = "/owner/job",
    n_concurrent: int | None = None,
    gpu_count: int | None = None,
    kind: str = "datagen",
    jobs_dir: str | None = "s3://bucket/jobs",
    harbor_job_name: str | None = "run",
) -> watcher.HarborJob:
    return watcher.HarborJob(
        cluster=watcher.Cluster("test", Path("/tmp/iris"), {}),
        job_id=job_id,
        state=state,
        task_state=task_state,
        submitted_at_ms=int((NOW - timedelta(hours=age_hours)).timestamp() * 1000),
        kind=kind,
        jobs_dir=jobs_dir,
        harbor_job_name=harbor_job_name,
        dataset="dataset",
        n_concurrent=n_concurrent,
        gpu_count=gpu_count,
    )


def test_queued_harbor_job_does_not_require_a_running_worker_pod(monkeypatch, tmp_path):
    job = _job(task_state="building")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("queued jobs must not be probed for running worker logs")

    monkeypatch.setattr(watcher, "find_pod", fail_if_called)

    assert watcher.fetch_ray_vllm_logs(job, tmp_path) == (
        "awaiting placement",
        None,
    )


def test_queued_harbor_job_health_is_not_an_output_failure():
    progress = watcher.Progress(
        None, None, "unavailable", error="no output while waiting for placement"
    )

    health, _ = watcher.health_label(
        _job(task_state="building"), progress, {}, NOW, 120
    )

    assert health == "awaiting placement"


def test_preempted_job_tree_is_stalled_when_trace_progress_stops():
    progress = watcher.Progress(
        10,
        20,
        "test",
        recent_completed=0,
        recent_errored=0,
    )

    health, _ = watcher.health_label(
        _job(task_state="preempted"), progress, {}, NOW, 120
    )

    assert health == "stalled / preempted"


def test_recently_preempted_job_tree_is_not_prematurely_stalled():
    progress = watcher.Progress(
        10,
        20,
        "test",
        recent_completed=0,
        recent_errored=0,
    )

    health, _ = watcher.health_label(
        _job(task_state="preempted", age_hours=1), progress, {}, NOW, 120
    )

    assert health == "preempted"


def test_preempted_job_tree_does_not_require_a_running_worker_pod(
    monkeypatch, tmp_path
):
    job = _job(task_state="preempted")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError(
            "preempted jobs must not be probed for running worker logs"
        )

    monkeypatch.setattr(watcher, "find_pod", fail_if_called)

    assert watcher.fetch_ray_vllm_logs(job, tmp_path) == ("preempted", None)


def test_trial_artifacts_counts_recent_completed_traces_and_their_errors():
    root = "jobs/run"
    artifacts = watcher._trial_artifacts(
        [
            (f"{root}/recent-ok/result.json", NOW - timedelta(minutes=15)),
            (f"{root}/recent-error/result.json", NOW - timedelta(minutes=30)),
            (f"{root}/recent-error/exception.txt", NOW - timedelta(minutes=29)),
            (f"{root}/old-error/result.json", NOW - timedelta(hours=3)),
            (f"{root}/old-error/exception.txt", NOW - timedelta(minutes=5)),
            (f"{root}/incomplete/exception.txt", NOW - timedelta(minutes=5)),
            (f"{root}/nested/agent/result.json", NOW - timedelta(minutes=5)),
        ],
        root,
        CUTOFF,
    )

    assert artifacts.completed_names == ["old-error", "recent-error", "recent-ok"]
    assert artifacts.exception_file_count == 3
    assert artifacts.recent_completed == 2
    assert artifacts.recent_errored == 1


def test_trial_artifacts_excludes_agent_timeouts_from_failure_signal():
    root = "jobs/run"
    artifacts = watcher._trial_artifacts(
        [
            (f"{root}/timeout/result.json", NOW - timedelta(minutes=15)),
            (f"{root}/timeout/exception.txt", NOW - timedelta(minutes=14)),
            (f"{root}/error/result.json", NOW - timedelta(minutes=15)),
            (f"{root}/error/exception.txt", NOW - timedelta(minutes=14)),
        ],
        root,
        CUTOFF,
        benign_timeout_trials={"timeout"},
    )

    assert artifacts.recent_completed == 2
    assert artifacts.recent_errored == 1
    assert artifacts.recent_benign_timeouts == 1


def test_gcs_trial_artifacts_uses_server_filtered_blob_listings():
    blobs = {
        "result.json": [
            SimpleNamespace(
                name="jobs/run/ok/result.json",
                time_created=NOW - timedelta(hours=1),
            ),
            SimpleNamespace(
                name="jobs/run/bad/result.json",
                time_created=NOW - timedelta(minutes=30),
            ),
        ],
        "exception.txt": [
            SimpleNamespace(
                name="jobs/run/bad/exception.txt",
                time_created=NOW - timedelta(minutes=29),
            )
        ],
    }

    class FakeClient:
        def list_blobs(self, _bucket, *, match_glob, **_kwargs):
            return blobs[match_glob.rsplit("/", 1)[-1]]

    artifacts = watcher.gcs_trial_artifacts(
        FakeClient(), "gs://bucket/jobs/run", CUTOFF
    )

    assert artifacts.recent_completed == 2
    assert artifacts.recent_errored == 1


def test_s3_trial_artifacts_classifies_recent_agent_timeout_as_benign(monkeypatch):
    root = "jobs/run"
    objects = [
        {
            "Key": f"{root}/timeout/result.json",
            "LastModified": NOW - timedelta(minutes=15),
        },
        {
            "Key": f"{root}/timeout/exception.txt",
            "LastModified": NOW - timedelta(minutes=14),
        },
        {
            "Key": f"{root}/error/result.json",
            "LastModified": NOW - timedelta(minutes=15),
        },
        {
            "Key": f"{root}/error/exception.txt",
            "LastModified": NOW - timedelta(minutes=14),
        },
    ]
    messages = {
        f"{root}/timeout/exception.txt": b"AgentTimeoutError: trial deadline exceeded",
        f"{root}/error/exception.txt": b"RuntimeError: verifier crashed",
    }

    monkeypatch.setattr(watcher, "iter_objects", lambda *_args: objects)

    class Client:
        def get_object(self, *, Bucket, Key):  # noqa: N803
            assert Bucket == "bucket"
            return {"Body": io.BytesIO(messages[Key])}

    artifacts = watcher.s3_trial_artifacts(Client(), "bucket", root, CUTOFF)

    assert artifacts.recent_completed == 2
    assert artifacts.recent_errored == 1
    assert artifacts.recent_benign_timeouts == 1


def test_health_uses_two_hour_trace_and_error_window():
    healthy = watcher.Progress(10, 20, "test", recent_completed=4, recent_errored=0)
    degraded = watcher.Progress(10, 20, "test", recent_completed=4, recent_errored=1)
    failing = watcher.Progress(10, 20, "test", recent_completed=4, recent_errored=4)
    stalled = watcher.Progress(10, 20, "test", recent_completed=0, recent_errored=0)

    assert watcher.health_label(_job(), healthy, {}, NOW, 120)[0] == (
        "advancing (+4/2h; 0 errors)"
    )
    assert watcher.health_label(_job(), degraded, {}, NOW, 120)[0] == (
        "degraded (+4/2h; 1 errors)"
    )
    assert watcher.health_label(_job(), failing, {}, NOW, 120)[0] == (
        "failing (+4/2h; 4 errors)"
    )
    assert watcher.health_label(_job(), stalled, {}, NOW, 120)[0] == (
        "stalled (+0 traces/2h)"
    )
    assert watcher.health_label(_job(age_hours=1), stalled, {}, NOW, 120)[0] == (
        "warming up (+0 traces/2h)"
    )


def test_health_treats_agent_timeouts_as_benign_in_the_two_hour_signal():
    progress = watcher.Progress(
        10,
        20,
        "test",
        recent_completed=4,
        recent_errored=0,
        recent_benign_timeouts=4,
        error_counts={"AgentTimeoutError": 4},
    )

    assert watcher.health_label(_job(), progress, {}, NOW, 120)[0] == (
        "advancing (+4/2h; 0 errors; 4 benign timeouts)"
    )
    assert watcher.recent_trend_cell(_job(), progress).tone == "success"
    assert (
        watcher.report_row(_job(), progress, "advancing", None, "saved")[6].tone
        == "muted"
    )


def test_terminal_states_render_retryable_and_nonretryable_failures():
    assert watcher.display_state("worker_failed") == "FAILED (RETRYABLE)"
    assert watcher.display_state("unschedulable") == "FAILED (RETRYABLE)"
    assert watcher.display_state("failed") == "FAILED (NON-RETRYABLE)"
    assert watcher.display_state("killed") == "FAILED (NON-RETRYABLE)"
    assert watcher.health_label(
        _job(state="worker_failed"), watcher.Progress(None, None, "test"), {}, NOW, 120
    )[0] == ("terminal (FAILED (RETRYABLE))")
    assert watcher._health_cell("terminal (FAILED (RETRYABLE))").tone == "error"


def test_report_row_exposes_two_hour_trace_delta_and_error_rate():
    progress = watcher.Progress(10, 20, "test", recent_completed=4, recent_errored=1)

    row = watcher.report_row(_job(), progress, "degraded", None, "unavailable")

    assert len(row) == 10
    assert row[7].value == "+4 traces; 1 errors (25%)"


def test_glm52_capacity_floor_keeps_normal_single_node_progress_healthy():
    job = _job(
        job_id="/owner/glm52-datagen-r11-14-unitsyn-large",
        n_concurrent=8,
        gpu_count=8,
    )
    progress = watcher.Progress(
        100, 200, "test", recent_completed=55, recent_errored=10
    )

    assert watcher.capacity_floor_2h(job) == 41
    assert watcher.health_label(job, progress, {}, NOW, 120)[0] == (
        "healthy (+55/2h; 10 errors; FLOPS floor 41/2h)"
    )
    assert watcher.recent_trend_cell(job, progress).tone == "success"
