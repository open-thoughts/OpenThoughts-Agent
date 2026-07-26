from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from scripts.iris import watch_iris_harbor as watcher


NOW = datetime(2026, 7, 26, 16, tzinfo=UTC)
CUTOFF = NOW - timedelta(hours=watcher.TRACE_TREND_HOURS)


def _job(
    *,
    state: str = "running",
    age_hours: int = 4,
    job_id: str = "/owner/job",
    n_concurrent: int | None = None,
    gpu_count: int | None = None,
) -> watcher.HarborJob:
    return watcher.HarborJob(
        cluster=watcher.Cluster("test", Path("/tmp/iris"), {}),
        job_id=job_id,
        state=state,
        submitted_at_ms=int((NOW - timedelta(hours=age_hours)).timestamp() * 1000),
        kind="datagen",
        jobs_dir="s3://bucket/jobs",
        harbor_job_name="run",
        dataset="dataset",
        n_concurrent=n_concurrent,
        gpu_count=gpu_count,
    )


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


def test_report_row_exposes_two_hour_trace_delta_and_error_rate():
    progress = watcher.Progress(10, 20, "test", recent_completed=4, recent_errored=1)

    row = watcher.report_row(_job(), progress, "degraded", None, "unavailable")

    assert len(row) == 10
    assert row[7].value == "+4 traces; 1 errors (25%)"


def test_glm52_capacity_floor_keeps_normal_single_node_progress_healthy():
    job = _job(
        job_id="/owner/glm52-datagen-r11-14-unitsyn-large",
        n_concurrent=4,
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
