"""Tests for datagen preempt-resume identity determinism.

A preemptible iris datagen job is re-run (same baked command) on every retry.
For harbor's GCS auto-resume to CONTINUE (skip done tasks) instead of re-running
the whole dataset from task 1 in a fresh timestamped serve dir, the harbor
identity must be DETERMINISTIC per job:
  - the served-model-name (``generate_served_model_id(job_name)``), and
  - the harbor jobs_dir/<job_name> run dir.

Both derive from ``args.job_name``. The bug: ``IrisLauncher.run()`` derived a
stable name into a LOCAL var but never wrote it to ``args.job_name``, so
``build_task_command`` skipped ``--job_name`` and the worker fell back to
per-serve time-based values (``default_job_name()`` / the timestamp served-id).
The fix persists ``args.job_name = self._derive_job_name(args)``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hpc import launch_utils  # noqa: E402
from hpc.launch_utils import generate_served_model_id  # noqa: E402
from hpc.harbor_utils import default_job_name  # noqa: E402


# --------------------------------------------------------------------------- #
# served-model-name determinism (given a stable job_name)
# --------------------------------------------------------------------------- #
def test_served_model_id_deterministic_for_same_job_name(monkeypatch):
    j = "tracegen-iris-20260704-070416"
    # two "serves" of the same job -> identical synthetic served-model id
    assert generate_served_model_id(job_name=j) == generate_served_model_id(job_name=j)
    # different job -> different id
    assert generate_served_model_id(job_name=j) != generate_served_model_id(
        job_name=j + "-b"
    )
    # no job_name -> time-based fallback (NOT stable) — the pre-fix worker behavior
    timestamps = iter((1_700_000_000.0, 1_700_000_001.0))
    monkeypatch.setattr(launch_utils.time, "time", lambda: next(timestamps))
    assert generate_served_model_id(job_name=None) != generate_served_model_id(
        job_name=None
    )


def test_harbor_run_dir_name_stable_when_job_name_set():
    # local_runner_utils computes: job_name = args.job_name or default_job_name(...).
    # With a stable args.job_name the run dir is that name (deterministic); without
    # it, default_job_name() mints a fresh per-serve timestamp.
    stable = "tracegen-iris-20260704-070416"
    assert (stable or default_job_name("tracegen", "ds", "m")) == stable
    # default_job_name IS per-serve (drifts) — what the worker fell back to:
    assert default_job_name("tracegen", "ds", "m") != "" and "__" in default_job_name(
        "tracegen", "ds", "m"
    )


# --------------------------------------------------------------------------- #
# launcher bakes --job_name (the fix's lever)
# --------------------------------------------------------------------------- #
def _tracegen_args(job_name):
    return argparse.Namespace(
        harbor_config="hpc/harbor_yaml/datagen/opencode_ctx131k.yaml",
        datagen_config="hpc/datagen_yaml/x.yaml",
        tasks_input_path="DCAgent/inferredbugs-sandboxes-verifier",
        model=None,
        agent="opencode",
        n_concurrent=32,
        n_attempts=1,
        gpus=8,
        health_max_attempts=600,
        health_retry_delay=None,
        harbor_env="daytona",
        record_literal=True,
        ingress_mode="controller",
        ingress_host="https://h",
        job_name=job_name,
        dry_run=False,
        agent_kwarg=[],
        harbor_extra_arg=[],
        upload_hf_repo=None,
        upload_hf_token=None,
        upload_hf_private=False,
    )


def _launcher():
    from data.cloud.launch_tracegen_iris import TracegenIrisLauncher
    from hpc.launch_utils import PROJECT_ROOT

    return TracegenIrisLauncher(PROJECT_ROOT)


def test_build_task_command_bakes_job_name_when_set():
    launcher = _launcher()
    cmd = launcher.build_task_command(
        _tracegen_args("tracegen-iris-20260704-070416"),
        "gs://b/ot-agent/tracegen-iris-20260704-070416",
    )
    assert "--job_name" in cmd
    assert cmd[cmd.index("--job_name") + 1] == "tracegen-iris-20260704-070416"


def test_build_task_command_omits_job_name_when_none():
    # Regression baseline: with args.job_name=None (and no resume override) the
    # command carries NO --job_name -> the worker drifts. run()'s persist fixes this.
    launcher = _launcher()
    args = _tracegen_args(None)
    cmd = launcher.build_task_command(args, "gs://b/ot-agent/j")
    assert "--job_name" not in cmd


def test_baked_venv_skips_iris_setup_scripts():
    """A baked model image must not have its venv replaced by Iris uv sync."""
    launcher = _launcher()
    args = _tracegen_args("glm52-pilot-codecontests-r9")
    args.baked_venv = True

    assert launcher.setup_scripts(args) == []


def test_derive_and_persist_yields_stable_baked_name_across_serves():
    # Mirror the run() fix: derive once, persist to args.job_name, then the SAME
    # command (re-run each preempt-retry) carries the SAME --job_name -> harbor
    # resumes the same jobs_dir/served-model instead of re-running from task 1.
    launcher = _launcher()
    args = _tracegen_args(None)
    args.job_name = launcher._derive_job_name(args)  # the fix
    assert args.job_name and args.job_name.startswith("tracegen-iris-")
    out = "gs://b/ot-agent/" + args.job_name
    cmd_serve0 = launcher.build_task_command(args, out)
    cmd_serve1 = launcher.build_task_command(args, out)  # iris re-runs the baked cmd
    n0 = cmd_serve0[cmd_serve0.index("--job_name") + 1]
    n1 = cmd_serve1[cmd_serve1.index("--job_name") + 1]
    assert n0 == n1 == args.job_name
    # same job_name -> same served-model id both serves
    assert generate_served_model_id(job_name=n0) == generate_served_model_id(
        job_name=n1
    )
