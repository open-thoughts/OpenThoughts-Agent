"""Unit tests for the ``include_literal_tokens`` threading through the trace exporter.

Covers ``scripts/harbor/make_and_upload_trace_dataset.py``:

  * FLAG-OFF PARITY: ``_install_harbor_patches()`` (default) installs the
    inline-subagent-merger (the terminus-2 path is unchanged), and
    ``_collect_trial_rows`` forwards ``include_literal_tokens=False`` — the
    Harbor default, so the emitted rows are byte-identical to today.
  * FLAG-ON: ``_install_harbor_patches(include_literal_tokens=True)`` SKIPS the
    inline-subagent-merger (which would drop the token columns) so Harbor's
    native literal-aware extraction runs; and the flag is forwarded to
    ``collect_conversations_from_trial``.

Hermetic — no HF, no real Harbor job. ``_collect_trial_rows`` takes ``traces_utils``
as a parameter, so we pass a fake that records the kwargs it receives.

Run:
    .venv/bin/python -m pytest tests/hpc/test_export_literal_threading.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from harbor.utils import traces_utils

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.harbor import make_and_upload_trace_dataset as mkupload  # noqa: E402


# --------------------------------------------------------------------------- #
# _install_harbor_patches gating
# --------------------------------------------------------------------------- #
def _patch_installers(monkeypatch) -> list[str]:
    """Replace the three _install_* helpers with call recorders."""
    called: list[str] = []
    monkeypatch.setattr(
        mkupload, "_install_safe_episode_guard", lambda: called.append("safe")
    )
    monkeypatch.setattr(
        mkupload, "_install_dataset_sanitizer", lambda: called.append("sanitize")
    )
    monkeypatch.setattr(
        mkupload, "_install_inline_subagent_merger", lambda: called.append("inline")
    )
    return called


def test_patches_default_installs_inline_merger(monkeypatch):
    called = _patch_installers(monkeypatch)
    mkupload._install_harbor_patches()  # default include_literal_tokens=False
    assert called == ["safe", "sanitize", "inline"]


def test_patches_literal_skips_inline_merger(monkeypatch):
    called = _patch_installers(monkeypatch)
    mkupload._install_harbor_patches(include_literal_tokens=True)
    # inline-subagent-merger is skipped so Harbor's native literal-aware extraction runs
    assert called == ["safe", "sanitize"]
    assert "inline" not in called


def test_inline_merger_accepts_current_harbor_extractor_keywords(tmp_path):
    """Keep the replacement callable compatible with Harbor's extractor API."""
    original = traces_utils.extract_conversations_from_trajectory
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text('{"steps": []}')
    try:
        mkupload._install_inline_subagent_merger()
        assert (
            traces_utils.extract_conversations_from_trajectory(
                trajectory,
                {"agent_name": "terminus-2", "model_name": "m", "start_time": "now"},
                sft_format=False,
            )
            == []
        )
    finally:
        traces_utils.extract_conversations_from_trajectory = original


# --------------------------------------------------------------------------- #
# _collect_trial_rows forwards include_literal_tokens to Harbor's collector
# --------------------------------------------------------------------------- #
class _FakeTracesUtils:
    def __init__(self):
        self.received: dict = {}

    def load_run_metadata(self, trial_dir):
        return {
            "agent_name": "terminus-2",
            "model_name": "m",
            "trial_name": trial_dir.name,
            "task_name": "t",
        }

    def collect_conversations_from_trial(self, trial_dir, **kwargs):
        self.received = kwargs
        return [{"conversations": []}]


def test_collect_trial_rows_defaults_literal_false(tmp_path):
    fake = _FakeTracesUtils()
    trial = tmp_path / "trial-1"
    trial.mkdir()
    rows = mkupload._collect_trial_rows(
        fake,
        trial,
        episodes="last",
        success_filter=None,
        include_instruction=False,
        include_verifier_output=False,
        verbose=False,
    )
    assert rows == [{"conversations": []}]
    # default = False (Harbor's default) -> byte-identical text-only rows
    assert fake.received["include_literal_tokens"] is False


def test_collect_trial_rows_forwards_literal_true(tmp_path):
    fake = _FakeTracesUtils()
    trial = tmp_path / "trial-1"
    trial.mkdir()
    mkupload._collect_trial_rows(
        fake,
        trial,
        episodes="last",
        success_filter=None,
        include_instruction=False,
        include_verifier_output=False,
        verbose=False,
        include_literal_tokens=True,
    )
    assert fake.received["include_literal_tokens"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
