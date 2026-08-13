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
from hpc.rl_paths import hydra_override_values


# ---------------------------------------------------------------------------
# Hydra override parsing: last-wins dotted lookup with marker and quote handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "key", "expected"),
    [
        (["trainer.max_steps=80", "trainer.epochs=2"], "trainer.max_steps", "80"),
        (["trainer.epochs=2"], "trainer.max_steps", None),
        (
            [
                "trainer.ckpt_path=/first/checkpoints",
                "trainer.ckpt_path=/latest/checkpoints",
            ],
            "trainer.ckpt_path",
            "/latest/checkpoints",
        ),
        (["++trainer.max_steps=80"], "trainer.max_steps", "80"),
        (["+trainer.max_steps=80"], "trainer.max_steps", "80"),
        (["trainer.ckpt_path='/a path/ckpts'"], "trainer.ckpt_path", "/a path/ckpts"),
        (['trainer.ckpt_path="/a:b/ckpts"'], "trainer.ckpt_path", "/a:b/ckpts"),
    ],
)
def test_hydra_override_values_normalize_and_use_last_value(arguments, key, expected):
    assert hydra_override_values(arguments).get(key) == expected


# ---------------------------------------------------------------------------
# _already_complete_on_disk: the chain-guard predicate
# ---------------------------------------------------------------------------


def _runner_with(args):
    """Bare RLJobRunner carrying only skyrl_hydra_args (no heavy __init__)."""
    runner = RLJobRunner.__new__(RLJobRunner)
    runner.config = types.SimpleNamespace(
        job_name="chain-arm",
        experiments_dir="/unused",
        trials_dir="/unused/trials",
        resume_policy="fixed",
        skyrl_hydra_args=list(args),
    )
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
    assert calls["setup"] == 1 and calls["ray"] == 1, (
        "incomplete link must train normally"
    )


def test_auto_resume_is_re_evaluated_for_each_chain_link(tmp_path):
    ckpt = tmp_path / "checkpoints"
    runner = _runner_with(
        [
            f"trainer.ckpt_path={ckpt}",
            f"trainer.export_path={tmp_path / 'exports'}",
            "trainer.resume_mode=none",
            "trainer.resume_path=null",
            "trainer.max_steps=80",
        ]
    )
    runner.config.resume_policy = "at_link_start"
    runner.config.trials_dir = str(tmp_path / "trials")
    runner._setup_environment = lambda: None
    runner._run_with_ray = lambda: 0
    runner._launch_trace_upload = lambda *args, **kwargs: None

    assert runner.run() == 0
    first_link = hydra_override_values(runner.config.skyrl_hydra_args)
    assert first_link["trainer.resume_mode"] == "none"
    assert first_link["trainer.resume_path"] == "null"

    checkpoint = ckpt / "global_step_6"
    checkpoint.mkdir(parents=True)
    _write_marker(ckpt, 6)

    assert runner.run() == 0
    second_link = hydra_override_values(runner.config.skyrl_hydra_args)
    assert second_link["trainer.resume_mode"] == "latest"
    assert second_link["trainer.resume_path"] == str(checkpoint)
