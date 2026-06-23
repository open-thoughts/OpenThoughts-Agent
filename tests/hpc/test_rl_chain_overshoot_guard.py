"""Unit tests for the RL resume-overshoot CHAIN guard.

``hpc/rl_launch_utils.py:RLJobRunner`` runs as each link of the ``afterany``
auto-restart chain. The chain guard (``_already_complete_on_disk`` + the early
return in ``run()``) short-circuits a link whose canonical checkpoint already
recorded a completed ``global_step >= trainer.max_steps``: it returns 0 BEFORE
bringing up Ray, so every queued ``afterany`` successor immediately no-ops
instead of spinning up a 14/16-node cluster just to resume-and-exit.

This is the launcher half of the two-part "resume-overshoot trap" fix; the
trainer half (SkyRL ``_handle_resume_at_max_steps``) prevents the spurious
gs N+1 step within a running link. The two guards key off the SAME on-disk
marker (``latest_ckpt_global_step.txt``) so they agree on "complete".

Run from the OT-Agent repo root with:
    .venv/bin/python -m pytest tests/hpc/test_rl_chain_overshoot_guard.py -v
"""

from __future__ import annotations

import types

import pytest

from hpc.rl_launch_utils import RLJobRunner


# ---------------------------------------------------------------------------
# _hydra_arg_value: last-wins dotted lookup with +/++ marker + quote handling
# ---------------------------------------------------------------------------


def test_hydra_arg_value_basic():
    args = ["trainer.max_steps=80", "trainer.epochs=2"]
    assert RLJobRunner._hydra_arg_value(args, "trainer.max_steps") == "80"


def test_hydra_arg_value_missing_returns_none():
    assert RLJobRunner._hydra_arg_value(["trainer.epochs=2"], "trainer.max_steps") is None


def test_hydra_arg_value_last_wins():
    # The launcher's auto-resume guard appends a second trainer.ckpt_path pinned
    # to the canonical dir; last-wins must pick the canonical one.
    args = [
        "trainer.ckpt_path=/redirected/_dryrun/checkpoints",
        "trainer.ckpt_path=/canonical/checkpoints",
    ]
    assert (
        RLJobRunner._hydra_arg_value(args, "trainer.ckpt_path")
        == "/canonical/checkpoints"
    )


def test_hydra_arg_value_strips_override_markers():
    assert RLJobRunner._hydra_arg_value(["++trainer.max_steps=80"], "trainer.max_steps") == "80"
    assert RLJobRunner._hydra_arg_value(["+trainer.max_steps=80"], "trainer.max_steps") == "80"


def test_hydra_arg_value_strips_surrounding_quotes():
    assert (
        RLJobRunner._hydra_arg_value(["trainer.ckpt_path='/a path/ckpts'"], "trainer.ckpt_path")
        == "/a path/ckpts"
    )
    assert (
        RLJobRunner._hydra_arg_value(['trainer.ckpt_path="/a:b/ckpts"'], "trainer.ckpt_path")
        == "/a:b/ckpts"
    )


# ---------------------------------------------------------------------------
# _already_complete_on_disk: the chain-guard predicate
# ---------------------------------------------------------------------------


def _runner_with(args):
    """Bare RLJobRunner carrying only skyrl_hydra_args (no heavy __init__)."""
    runner = RLJobRunner.__new__(RLJobRunner)
    runner.config = types.SimpleNamespace(skyrl_hydra_args=list(args))
    return runner


def _write_marker(ckpt_dir, step):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "latest_ckpt_global_step.txt").write_text(str(step))


def test_complete_when_marker_at_max(tmp_path):
    ckpt = tmp_path / "checkpoints"
    _write_marker(ckpt, 80)
    runner = _runner_with([f"trainer.ckpt_path={ckpt}", "trainer.max_steps=80"])
    assert runner._already_complete_on_disk() is True


def test_complete_when_marker_past_max(tmp_path):
    ckpt = tmp_path / "checkpoints"
    _write_marker(ckpt, 81)
    runner = _runner_with([f"trainer.ckpt_path={ckpt}", "trainer.max_steps=80"])
    assert runner._already_complete_on_disk() is True


def test_not_complete_when_marker_below_max(tmp_path):
    ckpt = tmp_path / "checkpoints"
    _write_marker(ckpt, 79)
    runner = _runner_with([f"trainer.ckpt_path={ckpt}", "trainer.max_steps=80"])
    assert runner._already_complete_on_disk() is False


def test_not_complete_when_no_marker(tmp_path):
    # Fresh run: ckpt dir exists but no marker yet -> not complete.
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    runner = _runner_with([f"trainer.ckpt_path={ckpt}", "trainer.max_steps=80"])
    assert runner._already_complete_on_disk() is False


def test_not_complete_when_ckpt_path_missing_arg(tmp_path):
    runner = _runner_with(["trainer.max_steps=80"])
    assert runner._already_complete_on_disk() is False


def test_not_complete_when_max_steps_missing_arg(tmp_path):
    ckpt = tmp_path / "checkpoints"
    _write_marker(ckpt, 80)
    runner = _runner_with([f"trainer.ckpt_path={ckpt}"])
    assert runner._already_complete_on_disk() is False


def test_not_complete_when_max_steps_unset_or_zero(tmp_path):
    # max_steps<=0 means "no cap"; never treat as complete on it.
    ckpt = tmp_path / "checkpoints"
    _write_marker(ckpt, 80)
    runner = _runner_with([f"trainer.ckpt_path={ckpt}", "trainer.max_steps=0"])
    assert runner._already_complete_on_disk() is False


def test_not_complete_when_marker_unparseable(tmp_path):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "latest_ckpt_global_step.txt").write_text("not-an-int")
    runner = _runner_with([f"trainer.ckpt_path={ckpt}", "trainer.max_steps=80"])
    assert runner._already_complete_on_disk() is False


# ---------------------------------------------------------------------------
# run(): completed link returns 0 WITHOUT Ray bring-up; incomplete falls through
# ---------------------------------------------------------------------------


def test_run_short_circuits_completed_link(tmp_path, monkeypatch):
    ckpt = tmp_path / "checkpoints"
    _write_marker(ckpt, 80)
    runner = _runner_with([f"trainer.ckpt_path={ckpt}", "trainer.max_steps=80"])
    runner.config.job_name = "done-run"

    # If the guard fails to short-circuit, these would be hit -> fail loudly.
    def _boom(*a, **k):
        raise AssertionError("Ray bring-up / setup must NOT run on a completed link")

    runner._setup_environment = _boom
    runner._run_with_ray = _boom
    runner._launch_trace_upload = lambda *a, **k: None

    assert runner.run() == 0


def test_run_proceeds_for_incomplete_link(tmp_path):
    ckpt = tmp_path / "checkpoints"
    _write_marker(ckpt, 40)
    runner = _runner_with([f"trainer.ckpt_path={ckpt}", "trainer.max_steps=80"])
    runner.config.job_name = "mid-run"

    calls = {"setup": 0, "ray": 0}

    def _setup():
        calls["setup"] += 1

    def _ray():
        calls["ray"] += 1
        return 0

    runner._setup_environment = _setup
    runner._run_with_ray = _ray
    runner._launch_trace_upload = lambda *a, **k: None

    assert runner.run() == 0
    assert calls["setup"] == 1 and calls["ray"] == 1, "incomplete link must train normally"
