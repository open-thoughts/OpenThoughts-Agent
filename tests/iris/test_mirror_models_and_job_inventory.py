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
    table = list_iris_jobs.render_table(rows)
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


def test_job_inventory_rejects_an_invalid_user_before_query():
    with pytest.raises(ValueError, match="Invalid Iris user"):
        list_iris_jobs.query_jobs(user="x' OR 1=1", hours=24, cluster="cw")
