"""The shared SLURM preset catalog must not override per-model thinking."""

import pytest

from eval.presets import load_presets


@pytest.mark.parametrize("preset_name", ["swebench", "tb2", "aider", "v2"])
def test_catalog_presets_do_not_carry_thinking(preset_name):
    """Thinking is PER-MODEL authoritative — sourced from the baseline model
    config, NOT the preset. Presets must carry no thinking kwarg (and no stale
    enable_thinking key); otherwise a preset would force thinking on a
    non-thinking model regardless of its baseline config."""
    preset = load_presets()[preset_name]
    assert "enable_thinking" not in preset, (
        f"{preset_name}: stale enable_thinking key — thinking is per-model now"
    )
    assert not any(
        "enable_thinking" in kw for kw in preset.get("agent_kwargs", []) or []
    ), (
        f"{preset_name}: preset still injects thinking; it must be per-model: {preset.get('agent_kwargs')}"
    )
