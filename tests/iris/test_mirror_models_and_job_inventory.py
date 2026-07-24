from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.iris import list_iris_jobs, mirror_models
from scripts.iris.launch_mirror import GcsToS3Launcher, HfMirrorIrisLauncher


def test_mirror_router_dispatches_hf_to_gcs_without_changing_route_arguments(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        mirror_models.mirror_hf_to_gcs,
        "mirror",
        lambda repo, prefixes, **kwargs: calls.append((repo, prefixes, kwargs)),
    )
    assert (
        mirror_models.main(
            [
                "hf-to-gcs",
                "--repo",
                "org/model",
                "--gcs-prefix",
                "gs://models",
                "--quiet",
            ]
        )
        == 0
    )
    assert calls == [
        ("org/model", ["gs://models"], {"verbose": False, "iris_job_id": None})
    ]


def test_mirror_router_rejects_wrong_source_scheme_before_any_transfer(monkeypatch):
    monkeypatch.setattr(mirror_models.mirror_gcs_to_s3, "mirror_repo", pytest.fail)
    with pytest.raises(SystemExit, match="must start with gs://"):
        mirror_models.main(
            [
                "gcs-to-s3",
                "--repo",
                "org/model",
                "--gcs-prefix",
                "bad",
                "--s3-bucket",
                "bucket",
                "--s3-prefix",
                "models",
            ]
        )


def test_mirror_launchers_issue_the_canonical_mirror_command():
    hf_args = SimpleNamespace(gcs_prefix=["gs://one", "gs://two"], repo=["org/model"])
    gcs_args = SimpleNamespace(
        gcs_prefix="gs://models",
        s3_bucket="bucket",
        s3_prefix="models",
        s3_endpoint=None,
        repo=["org/model"],
    )
    assert HfMirrorIrisLauncher(".").build_task_command(hf_args, "unused") == [
        "python",
        "-m",
        "scripts.iris.mirror_models",
        "hf-to-gcs",
        "--gcs-prefix",
        "gs://one",
        "--gcs-prefix",
        "gs://two",
        "--repo",
        "org/model",
    ]
    assert GcsToS3Launcher(".").build_task_command(gcs_args, "unused") == [
        "python",
        "-m",
        "scripts.iris.mirror_models",
        "gcs-to-s3",
        "--gcs-prefix",
        "gs://models",
        "--s3-bucket",
        "bucket",
        "--s3-prefix",
        "models",
        "--repo",
        "org/model",
    ]


def test_job_inventory_classifies_and_orders_controller_rows(monkeypatch):
    output = """info line
job_id,state,submitted_at_ms,started_at_ms,finished_at_ms,error,exit_code
/benjaminfeuer/tracegen-a,4,1000,1100,1200,,0
/benjaminfeuer/rl-run,3,2000,2100,,,
/benjaminfeuer/eval-b,6,3000,3100,3200,manual stop,1
"""
    monkeypatch.setattr(
        list_iris_jobs,
        "run_iris_command",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=output, stderr=""
        ),
    )
    rows = list_iris_jobs.query_jobs(
        user="benjaminfeuer", hours=24, cluster="cw", now_ms=86_400_000
    )
    table = list_iris_jobs.render_table(rows, now_ms=3_800_000)
    assert [row["job_id"] for row in rows] == [
        "/benjaminfeuer/tracegen-a",
        "/benjaminfeuer/rl-run",
        "/benjaminfeuer/eval-b",
    ]
    assert (
        "datagen" in table
        and "RL" in table
        and "eval" in table
        and "terminated" in table
    )
    assert "Duration" in table and "0m" in table


def test_job_inventory_formats_running_and_finished_durations():
    finished = {
        "submitted_at_ms": "0",
        "started_at_ms": "60_000",
        "finished_at_ms": "7_320_000",
    }
    pending = {"submitted_at_ms": "0", "started_at_ms": "", "finished_at_ms": ""}

    assert list_iris_jobs.job_filter_values(finished)["duration"] == "2h 1m"
    assert list_iris_jobs.job_filter_values(pending, now_ms=90_000)["duration"] == "1m"


def test_job_inventory_filters_regex_fields():
    rows = [
        {
            "job_id": "/benjaminfeuer/glm52-running",
            "state": "3",
            "cluster": "cw-rno2a",
        },
        {
            "job_id": "/benjaminfeuer/other-running",
            "state": "3",
            "cluster": "marin",
        },
        {
            "job_id": "/benjaminfeuer/glm52-failed",
            "state": "5",
            "cluster": "cw-rno2a",
        },
    ]
    filters = list_iris_jobs.parse_regex_filters(
        ["state=RUNNING", "name=^glm52", "cluster=^cw-"],
        {
            "cluster",
            "submitted",
            "job",
            "name",
            "type",
            "state",
            "duration",
            "exit",
            "error",
        },
    )

    filtered = list_iris_jobs.filter_records(
        rows, filters, list_iris_jobs.job_filter_values
    )

    assert [row["job_id"] for row in filtered] == ["/benjaminfeuer/glm52-running"]


def test_job_inventory_queries_all_default_clusters_and_labels_rows(
    monkeypatch, capsys
):
    queried_clusters = []

    def fake_query_jobs(*, user, hours, cluster):
        queried_clusters.append((user, hours, cluster))
        return [{"job_id": f"/{user}/{cluster}", "state": "3", "submitted_at_ms": "1"}]

    monkeypatch.setattr(list_iris_jobs, "query_jobs", fake_query_jobs)

    assert (
        list_iris_jobs.main(
            ["--user", "benjaminfeuer", "--hours", "6", "--filter", "state=running"]
        )
        == 0
    )

    assert [cluster for _user, _hours, cluster in queried_clusters] == list(
        list_iris_jobs.DEFAULT_CLUSTERS
    )
    output = capsys.readouterr().out
    assert "clusters=cw-rno2a,cw-us-east-02a,marin" in output
    assert "Cluster" in output and "cw-rno2a" in output and "marin" in output


def test_job_inventory_overrides_an_inherited_non_coreweave_kubeconfig(monkeypatch):
    captured = {}
    monkeypatch.setenv("KUBECONFIG", "/Users/benjaminfeuer/.kube/lambdaconfig")
    monkeypatch.setattr(
        list_iris_jobs,
        "run_iris_command",
        lambda *_args, **kwargs: (
            captured.update(kwargs)
            or SimpleNamespace(
                returncode=0,
                stdout="job_id,state,submitted_at_ms,started_at_ms,finished_at_ms,error,exit_code\n",
                stderr="",
            )
        ),
    )

    list_iris_jobs.query_jobs(user="benjaminfeuer", hours=24, cluster="cw-us-east-02a")

    assert (
        captured["environment"]["KUBECONFIG"]
        == "/Users/benjaminfeuer/.kube/coreweave-iris-gpu"
    )


def test_job_inventory_hours_zero_queries_all_history(monkeypatch):
    commands = []
    monkeypatch.setattr(
        list_iris_jobs,
        "run_iris_command",
        lambda arguments, **_kwargs: (
            commands.append(arguments)
            or SimpleNamespace(
                returncode=0,
                stdout="job_id,state,submitted_at_ms,started_at_ms,finished_at_ms,error,exit_code\n",
                stderr="",
            )
        ),
    )

    list_iris_jobs.query_jobs(
        user="benjaminfeuer", hours=0, cluster="marin", now_ms=86_400_000
    )

    assert "submitted_at_ms >=" not in commands[0][1]


def test_job_inventory_rejects_an_invalid_user_before_query():
    with pytest.raises(ValueError, match="Invalid Iris user"):
        list_iris_jobs.query_jobs(user="x' OR 1=1", hours=24, cluster="cw")
