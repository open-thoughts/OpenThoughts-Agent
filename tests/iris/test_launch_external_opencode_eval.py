import argparse
import importlib.util
import sys
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
            ttl_hours=168,
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
            ttl_hours=168,
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
        timeout=604800,
        capability_token_duration_policy_json='{"token_required": true}',
        job_name="eval-v2",
        harbor_config="config.yaml",
        datagen_config="external.yaml",
        model="vllm/model",
        dataset_path="DCAgent/dev_set_v2",
        n_concurrent=256,
        n_attempts=3,
        agent_kwarg=["thinking_format=chat-template"],
        harbor_extra_arg=["--n-tasks=1"],
        s3_output_dir="s3://marin-us-east-02a/iris",
        upload_hf_repo="laion/traces",
        skip_hf_upload=True,
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
    assert "thinking_format=chat-template" in shell
    assert "--harbor_extra_arg=--n-tasks=1" in shell
    assert "--n_concurrent 256" in shell
    assert "--job_name eval-v2" in shell
    assert "--experiments_dir /tmp/ot-agent-runs/eval-v2" in shell
    assert command[command.index("--timeout") + 1] == "604800"
    assert (
        "--harbor_extra_arg=--jobs-dir=s3://marin-us-east-02a/iris/eval-v2/trace_jobs"
        in shell
    )
    assert "--upload_hf_repo" not in shell
    assert command[command.index("EXTERNAL_AGENT_API_KEY") + 1] == _MODULE.DUMMY_API_KEY


def test_s3_harbor_jobs_dir_isolated_per_iris_job():
    assert (
        _MODULE.s3_harbor_jobs_dir("s3://marin-us-east-02a/iris/", "eval-v2")
        == "s3://marin-us-east-02a/iris/eval-v2/trace_jobs"
    )


@pytest.mark.parametrize("root", ["", "/tmp/jobs", "gs://marin/jobs"])
def test_s3_harbor_jobs_dir_rejects_non_s3_roots(root):
    with pytest.raises(ValueError, match="s3-output-dir"):
        _MODULE.s3_harbor_jobs_dir(root, "eval-v2")


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
            ttl_hours=168,
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


def test_main_submits_federated_serve_then_parent_minted_durable_eval(
    tmp_path, monkeypatch
):
    """The default profile must run the full parent→peer→eval launch sequence."""
    secret_path = tmp_path / "secrets.env"
    secret_path.write_text(
        "DAYTONA_API_KEY=daytona\nHF_TOKEN=hf\nOPENAI_API_KEY=judge\n"
    )
    marin_repo = tmp_path / "marin"
    marin_repo.mkdir()
    calls: list[tuple[list[str], dict]] = []

    class Result:
        def __init__(self, stdout="", returncode=0):
            self.stdout = stdout
            self.returncode = returncode
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "endpoints" in command and "list" in command:
            return Result(
                "NAME ACCESS PEER ADDRESS TASK\n/serve/grug-r7-serve link cw-us-east-02a 10.0.0.1 task\n"
            )
        if "endpoints" in command and "mint" in command:
            return Result(
                "Capability URL: https://iris.oa.dev/proxy/t/token/serve.grug-r7-serve/\n"
            )
        return Result()

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--job-name",
            "grug-r7",
            "--secrets-env",
            str(secret_path),
            "--marin-repo",
            str(marin_repo),
        ],
    )

    assert _MODULE.main() == 0

    serve_command, serve_kwargs = calls[0]
    assert serve_command[:8] == [
        "uv",
        "run",
        "--project",
        "lib/marin",
        "marin-serve",
        "iris",
        _MODULE.DEFAULT_SERVE_MODEL,
        "--cluster",
    ]
    assert "--target-cluster" in serve_command
    assert (
        serve_command[serve_command.index("--target-cluster") + 1] == "cw-us-east-02a"
    )
    assert serve_command[serve_command.index("--idle-timeout-hours") + 1] == "1.0"
    assert "--model-loader-extra-config" not in " ".join(serve_command)
    assert serve_kwargs["env"]["KUBECONFIG"] == _MODULE.DEFAULT_KUBECONFIG

    eval_command, eval_kwargs = calls[-1]
    assert eval_command[eval_command.index("--job-name") + 1] == "grug-r7"
    assert (
        eval_command[eval_command.index("--task-image") + 1]
        == _MODULE.DEFAULT_TASK_IMAGE
    )
    assert eval_kwargs["env"]["KUBECONFIG"] == _MODULE.DEFAULT_KUBECONFIG
    assert "https://iris.oa.dev" not in eval_command[-1]
    assert (
        "--jobs-dir=s3://marin-us-east-02a/iris/grug-r7/trace_jobs" in eval_command[-1]
    )
    assert eval_command[eval_command.index("--timeout") + 1] == "604800"
