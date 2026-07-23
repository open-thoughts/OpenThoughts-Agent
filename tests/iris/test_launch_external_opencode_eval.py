import argparse
import importlib.util
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/iris/launch_external_opencode_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("launch_external_opencode_eval", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_mint_requires_one_capability_url(monkeypatch):
    class Result:
        returncode = 0
        stderr = ""
        stdout = "Capability URL: https://iris.oa.dev/proxy/t/token/serve.example/\n"

    monkeypatch.setattr(_MODULE.subprocess, "run", lambda *a, **k: Result())
    assert (
        _MODULE.mint_capability_api_base(
            iris_bin="iris",
            parent_cluster="marin",
            parent_ingress_host="https://iris.oa.dev",
            endpoint_name="/serve/example",
            ttl_hours=24,
        )
        == "https://iris.oa.dev/proxy/t/token/serve.example/v1"
    )


def test_mint_rejects_missing_or_ambiguous_urls(monkeypatch):
    class Result:
        returncode = 0
        stderr = ""
        stdout = "not a URL\n"

    monkeypatch.setattr(_MODULE.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match="unambiguous"):
        _MODULE.mint_capability_api_base(
            iris_bin="iris",
            parent_cluster="marin",
            parent_ingress_host="https://iris.oa.dev",
            endpoint_name="/serve/example",
            ttl_hours=24,
        )


def test_submit_uses_env_for_url_and_fails_fast_when_missing():
    args = argparse.Namespace(
        iris_bin="iris",
        cluster="cw-us-east-02a",
        task_image="image@sha256:abc",
        cpu=32,
        memory="128GB",
        disk="128GB",
        priority="batch",
        job_name="eval-v2",
        harbor_config="config.yaml",
        datagen_config="external.yaml",
        model="vllm/model",
        dataset_path="DCAgent/dev_set_v2",
        n_concurrent=256,
        n_attempts=3,
        s3_output_dir="s3://marin-us-east-02a/iris",
        upload_hf_repo="laion/traces",
    )
    command = _MODULE.build_submit_command(
        args,
        {
            "DAYTONA_API_KEY": "daytona",
            "HF_TOKEN": "hf",
            "OPENAI_API_KEY": "judge",
        },
        "https://iris.oa.dev/proxy/t/token/serve.example/v1",
    )
    shell = command[-1]
    assert "set -eu" in shell
    assert "${EXTERNAL_AGENT_API_BASE:?missing minted endpoint URL}" in shell
    assert "api_base=${EXTERNAL_AGENT_API_BASE}" in shell
    assert "https://iris.oa.dev" not in shell
    assert "--n_concurrent 256" in shell
    assert "--experiments_dir /tmp/ot-agent-runs/eval-v2" in shell
    assert (
        "--harbor_extra_arg=--jobs-dir=s3://marin-us-east-02a/iris/eval-v2/trace_jobs"
        in shell
    )


def test_durable_harbor_jobs_dir_isolated_per_iris_job():
    assert (
        _MODULE.durable_harbor_jobs_dir(
            s3_output_root="s3://marin-us-east-02a/iris/",
            iris_job_name="eval-v2",
        )
        == "s3://marin-us-east-02a/iris/eval-v2/trace_jobs"
    )


@pytest.mark.parametrize("root", ["", "/tmp/jobs", "gs://marin/jobs"])
def test_durable_harbor_jobs_dir_rejects_non_s3_roots(root):
    with pytest.raises(ValueError, match="s3-output-dir"):
        _MODULE.durable_harbor_jobs_dir(s3_output_root=root, iris_job_name="eval-v2")


def test_parent_mirror_requires_a_peer_row(monkeypatch):
    class Result:
        returncode = 0
        stderr = ""
        stdout = "NAME ACCESS PEER ADDRESS TASK\n/serve/example link cw-us-east-02a 10.0.0.1 /user/job\n"

    monkeypatch.setattr(_MODULE.subprocess, "run", lambda *a, **k: Result())
    _MODULE.wait_for_parent_endpoint_mirror(
        iris_bin="iris",
        parent_cluster="marin",
        endpoint_name="/serve/example",
        timeout_seconds=0,
        sleep=lambda _: None,
        monotonic=lambda: 0,
    )


def test_parent_mirror_rejects_a_missing_or_local_endpoint(monkeypatch):
    class Result:
        returncode = 0
        stderr = ""
        stdout = "NAME ACCESS PEER ADDRESS TASK\n/serve/example link local 10.0.0.1 /user/job\n"

    monkeypatch.setattr(_MODULE.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match="did not mirror"):
        _MODULE.wait_for_parent_endpoint_mirror(
            iris_bin="iris",
            parent_cluster="marin",
            endpoint_name="/serve/example",
            timeout_seconds=0,
            sleep=lambda _: None,
            monotonic=lambda: 0,
        )


def test_mint_rejects_a_peer_ingress_url(monkeypatch):
    class Result:
        returncode = 0
        stderr = ""
        stdout = "Capability URL: https://iris-cw-us-east-02a.oa.dev/proxy/t/token/serve.example/\n"

    monkeypatch.setattr(_MODULE.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match="unexpected ingress host"):
        _MODULE.mint_capability_api_base(
            iris_bin="iris",
            parent_cluster="marin",
            parent_ingress_host="https://iris.oa.dev",
            endpoint_name="/serve/example",
            ttl_hours=24,
        )


def test_selected_secrets_file_replaces_stale_inherited_values(tmp_path):
    secret_path = tmp_path / "secrets.env"
    secret_path.write_text(
        "# current credentials\nexport DAYTONA_API_KEY=current\nHF_TOKEN='hf'\n"
    )
    environ = {"DAYTONA_API_KEY": "stale", "UNCHANGED": "value"}

    _MODULE.load_secrets_env(secret_path, environ)

    assert environ == {
        "DAYTONA_API_KEY": "current",
        "HF_TOKEN": "hf",
        "UNCHANGED": "value",
    }
