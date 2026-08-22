"""Dry-run tests for the Iris eval offline pre-cache (hpc/iris/precache.py + wiring).

Proves, WITHOUT a real GCS/HF/iris round-trip:
  (a) cache-HIT path issues ZERO HuggingFace calls and returns the region cloud URIs;
  (b) the eval launcher, on a cache-HIT, emits HF_HUB_OFFLINE=1 + the mirror URIs in
      the job spec (serve URI via --vllm_model_uri, dataset rewritten to gs://) — and
      leaves --model as the HF id so worker model_config resolution is unaffected;
  (c) a cache-MISS in the default 'auto' mode does NOT enable offline (byte-identical
      online behavior) and does NOT abort the launch; 'strict' mode fails loud.

Run:
    /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python -m pytest \
        tests/eval/test_iris_precache_offline.py -q
"""

from __future__ import annotations

import argparse
import os

import pytest

from eval.local.run_eval import _configure_gcs_model_credentials
from hpc.iris import precache
from hpc.iris.outputs import IrisOutputPaths
from hpc.launch_utils import PROJECT_ROOT


class _FakeGCS:
    """Minimal fsspec-like stand-in. ``layout`` maps a gs:// dir -> [basenames]."""

    def __init__(self, layout):
        self.layout = {k.rstrip("/"): v for k, v in layout.items()}
        self.puts = []

    def ls(self, path, detail=True):
        names = self.layout.get(path.rstrip("/"))
        if names is None:
            raise FileNotFoundError(path)
        return [{"name": f"{path.rstrip('/')}/{n}"} for n in names]

    def glob(self, pattern):
        return []

    def info(self, uri):
        raise FileNotFoundError(uri)

    def put(self, *a, **k):
        self.puts.append((a, k))


HIT_LAYOUT = {
    "gs://marin-models-us/ot-agent/models/Qwen/Qwen3-8B": [
        "config.json",
        "model.safetensors",
    ],
    "gs://marin-models-us/ot-agent/datasets/DCAgent/foo": ["train.parquet"],
}


def _gcs_output_paths(remote_output_dir: str) -> IrisOutputPaths:
    return IrisOutputPaths(
        remote_output_dir=remote_output_dir,
        work_output_dir=remote_output_dir,
        harbor_jobs_dir=remote_output_dir,
    )


@pytest.fixture
def no_hf(monkeypatch):
    """Make ANY HuggingFace call an immediate hard failure (proves zero-HF on hit)."""
    import huggingface_hub

    def _boom(*a, **k):
        raise AssertionError("HuggingFace was called on a cache-HIT path")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _boom, raising=False)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom, raising=False)
    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda *a, **k: _boom(), raising=False
    )


def _use_layout(monkeypatch, layout):
    fake = _FakeGCS(layout)
    monkeypatch.setattr(precache, "_gcs_fs", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# (a) helper: cache-HIT => zero HF calls + region cloud URIs + offline env
# ---------------------------------------------------------------------------


def test_helper_cache_hit_zero_hf_and_offline(monkeypatch, no_hf):
    _use_layout(monkeypatch, HIT_LAYOUT)

    result = precache.precache_for_eval(
        "Qwen/Qwen3-8B", ["DCAgent/foo"], region=None, mode="auto", verbose=False
    )

    assert result.offline_ok is True
    # Model served from the region-local mirror as an s3:// (runai_streamer) URI.
    assert (
        result.model_serve_uri == "s3://marin-models-us/ot-agent/models/Qwen/Qwen3-8B/"
    )
    # Dataset routed through the region-local GCS mirror.
    assert result.dataset_uris == ["gs://marin-models-us/ot-agent/datasets/DCAgent/foo"]
    # Offline flags + the GCS S3-compat endpoint for runai_streamer.
    assert result.env["HF_HUB_OFFLINE"] == "1"
    assert result.env["TRANSFORMERS_OFFLINE"] == "1"
    assert result.env["AWS_ENDPOINT_URL"] == "https://storage.googleapis.com"


def test_helper_region_selects_eu_bucket(monkeypatch, no_hf):
    layout = {
        "gs://marin-models-eu/ot-agent/models/Qwen/Qwen3-8B": [
            "config.json",
            "model.safetensors",
        ],
    }
    _use_layout(monkeypatch, layout)
    result = precache.precache_for_eval(
        "Qwen/Qwen3-8B", [], region="europe-west4", mode="auto", verbose=False
    )
    assert result.offline_ok is True
    assert result.model_serve_uri.startswith("s3://marin-models-eu/")


def test_helper_rejects_incomplete_sharded_model(monkeypatch):
    _use_layout(
        monkeypatch,
        {
            "gs://marin-models-us/ot-agent/models/Qwen/Qwen3-8B": [
                "config.json",
                "model-00001-of-00002.safetensors",
            ],
        },
    )

    result = precache.precache_for_eval(
        "Qwen/Qwen3-8B", [], region=None, mode="auto", verbose=False
    )

    assert result.offline_ok is False


# ---------------------------------------------------------------------------
# (c) helper: cache-MISS behavior (auto = online fallback; strict = fail loud)
# ---------------------------------------------------------------------------


def test_helper_model_miss_auto_falls_back_online(monkeypatch):
    # Only the dataset is present; the model dir is missing -> not offline-eligible.
    _use_layout(
        monkeypatch,
        {
            "gs://marin-models-us/ot-agent/datasets/DCAgent/foo": ["train.parquet"],
        },
    )
    result = precache.precache_for_eval(
        "Qwen/Qwen3-8B", ["DCAgent/foo"], region=None, mode="auto", verbose=False
    )
    assert result.offline_ok is False
    assert result.env == {}
    assert any("model Qwen/Qwen3-8B" in n for n in result.notes)


def test_helper_model_miss_strict_fails_loud(monkeypatch):
    _use_layout(monkeypatch, {})  # nothing present
    with pytest.raises(SystemExit) as ei:
        precache.precache_for_eval(
            "Qwen/Qwen3-8B", [], region=None, mode="strict", verbose=False
        )
    assert "launch_mirror hf-to-gcs" in str(ei.value)


def test_helper_mode_off_is_noop(monkeypatch):
    # mode=off must not even touch GCS.
    def _explode():
        raise AssertionError("GCS touched in mode=off")

    monkeypatch.setattr(precache, "_gcs_fs", _explode)
    result = precache.precache_for_eval(
        "Qwen/Qwen3-8B", ["DCAgent/foo"], region=None, mode="off"
    )
    assert result.offline_ok is False
    assert result.env == {}


# ---------------------------------------------------------------------------
# (b) launcher wiring: cache-HIT => offline env + mirror URIs in the job spec
# ---------------------------------------------------------------------------


def _eval_args(**over):
    """A namespace covering everything build_task_command / pre_submit read."""
    base = dict(
        hf_offline_mode="auto",
        model="Qwen/Qwen3-8B",
        _orig_dataset_repo="DCAgent/foo",
        _pinned_region=None,
        # build_task_command surface:
        harbor_config="hpc/harbor_yaml/eval/eval_ctx32k.yaml",
        datagen_config=None,
        dataset=None,
        dataset_path="DCAgent/foo",
        agent="terminus-2",
        n_concurrent=16,
        n_attempts=3,
        gpus=4,
        harbor_env="daytona",
        job_name="eval-precache-smoke",
        dry_run=False,
        ray_object_store_gb=None,
        agent_kwarg=[],
        harbor_extra_arg=[],
        upload_to_database=False,
        upload_username=None,
        upload_error_mode=None,
        upload_hf_repo=None,
        upload_hf_token=None,
        upload_hf_private=False,
        upload_hf_episodes=None,
        upload_forced_update=False,
    )
    base.update(over)
    return argparse.Namespace(**base)


def test_launcher_hit_emits_offline_and_mirror_uris(monkeypatch, no_hf):
    from eval.cloud.launch_eval_iris import EvalIrisLauncher

    _use_layout(monkeypatch, HIT_LAYOUT)
    launcher = EvalIrisLauncher(PROJECT_ROOT)
    args = _eval_args()

    env = launcher.pre_submit_precache(
        args, remote_output_dir="gs://marin-models-us/ot-agent/eval-x"
    )

    # Offline env is emitted...
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    # ...the serve URI is the mirror (runai_streamer), while --model stays the HF id...
    assert args._vllm_model_uri == "s3://marin-models-us/ot-agent/models/Qwen/Qwen3-8B/"
    assert args.model == "Qwen/Qwen3-8B"
    # ...and the dataset is rewritten to the region GCS mirror.
    assert args.dataset_path == "gs://marin-models-us/ot-agent/datasets/DCAgent/foo"

    # The built job command carries the serve URI + the offline dataset path,
    # and still passes --model as the HF id.
    cmd = launcher.build_task_command(
        args, _gcs_output_paths("gs://marin-models-us/ot-agent/eval-x")
    )
    assert "--vllm_model_uri" in cmd
    i = cmd.index("--vllm_model_uri")
    assert cmd[i + 1] == "s3://marin-models-us/ot-agent/models/Qwen/Qwen3-8B/"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "Qwen/Qwen3-8B"
    assert "gs://marin-models-us/ot-agent/datasets/DCAgent/foo" in cmd


def test_gcs_model_credentials_are_scoped_to_vllm(monkeypatch):
    monkeypatch.setenv("MARIN_HMAC_ACCESS_ID", "gcs-access")
    monkeypatch.setenv("MARIN_HMAC_SECRET", "gcs-secret")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://cwobject.com")
    args = argparse.Namespace(
        vllm_model_uri="s3://marin-models-us/ot-agent/models/Qwen/Qwen3-8B/",
        _vllm_env_vars={"KEEP": "value"},
    )

    _configure_gcs_model_credentials(args)

    assert os.environ["AWS_ENDPOINT_URL"] == "https://cwobject.com"
    assert args._vllm_env_vars["AWS_ENDPOINT_URL"] == "https://storage.googleapis.com"
    assert args._vllm_env_vars["AWS_ACCESS_KEY_ID"] == "gcs-access"
    assert args._vllm_env_vars["AWS_SECRET_ACCESS_KEY"] == "gcs-secret"
    assert args._vllm_env_vars["HF_HUB_OFFLINE"] == "1"
    assert args._vllm_env_vars["TRANSFORMERS_OFFLINE"] == "1"
    copied = args._vllm_env_vars["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"]
    assert "HF_HUB_OFFLINE" in copied
    assert "TRANSFORMERS_OFFLINE" in copied
    assert args._vllm_env_vars["KEEP"] == "value"


def test_launcher_selector_keeps_dataset_online_while_serving_cached_model(
    monkeypatch, no_hf
):
    from eval.cloud.launch_eval_iris import EvalIrisLauncher

    model_only = {
        "gs://marin-models-us/ot-agent/models/Qwen/Qwen3-8B": [
            "config.json",
            "model.safetensors",
        ],
    }
    _use_layout(monkeypatch, model_only)
    launcher = EvalIrisLauncher(PROJECT_ROOT)
    selector = "open-thoughts/TaskTrove@abc123::DCAgent__foo"
    args = _eval_args(
        dataset_path=selector,
        _orig_dataset_repo=None,
        _dataset_uses_online_selector=True,
    )

    env = launcher.pre_submit_precache(args, remote_output_dir="gs://x/y")

    assert "HF_HUB_OFFLINE" not in env
    assert "TRANSFORMERS_OFFLINE" not in env
    assert args._vllm_model_uri == "s3://marin-models-us/ot-agent/models/Qwen/Qwen3-8B/"
    assert args.dataset_path == selector


def test_launcher_miss_auto_no_offline_no_serve_uri(monkeypatch):
    from eval.cloud.launch_eval_iris import EvalIrisLauncher

    _use_layout(monkeypatch, {})  # nothing mirrored
    # dataset inline-mirror would try HF; force it to fail so we stay online.
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no HF in test")),
        raising=False,
    )
    launcher = EvalIrisLauncher(PROJECT_ROOT)
    args = _eval_args()

    env = launcher.pre_submit_precache(args, remote_output_dir="gs://x/y")

    assert env == {}
    assert getattr(args, "_vllm_model_uri", None) is None
    # dataset_path left as the HF id (online worker snapshot_download, unchanged).
    assert args.dataset_path == "DCAgent/foo"
    cmd = launcher.build_task_command(args, _gcs_output_paths("gs://x/y"))
    assert "--vllm_model_uri" not in cmd
    assert "HF_HUB_OFFLINE" not in " ".join(cmd)
