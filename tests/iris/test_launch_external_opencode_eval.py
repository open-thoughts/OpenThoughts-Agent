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
            cluster="cw-us-east-02a",
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
            cluster="cw-us-east-02a",
            endpoint_name="/serve/example",
            ttl_hours=24,
        )


def test_submit_uses_env_for_url_and_fails_fast_when_missing():
    args = argparse.Namespace(
        iris_bin="iris",
        cluster="cw-us-east-02a",
        task_image="image@sha256:abc",
        cpu=8,
        memory="64GB",
        disk="64GB",
        priority="batch",
        job_name="eval-v2",
        harbor_config="config.yaml",
        datagen_config="external.yaml",
        model="vllm/model",
        dataset_path="DCAgent/dev_set_v2",
        n_concurrent=6,
        n_attempts=3,
        upload_hf_repo="laion/traces",
    )
    command = _MODULE.build_submit_command(
        args,
        {
            "DAYTONA_API_KEY": "daytona",
            "HF_TOKEN": "hf",
            "OPENAI_API_KEY": "judge",
            "IRIS_INGRESS_API_KEY": "sidecar",
        },
        "https://iris.oa.dev/proxy/t/token/serve.example/v1",
    )
    shell = command[-1]
    assert "set -eu" in shell
    assert "${EXTERNAL_AGENT_API_BASE:?missing minted endpoint URL}" in shell
    assert "api_base=${EXTERNAL_AGENT_API_BASE}" in shell
    assert "https://iris.oa.dev" not in shell
