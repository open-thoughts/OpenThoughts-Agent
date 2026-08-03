"""Selection rules shared by the HF-sourced model mirrors.

The regression these guard: an allowlist of known suffixes silently DROPPED
``chat_template.jinja``, so a mirrored repo produced a model with no chat template
and the mirror still reported success.
"""

from __future__ import annotations

from scripts.iris.hf_model_files import select_model_files

SHARDS = [f"model-{i:05d}-of-00003.safetensors" for i in range(1, 4)]


def test_chat_template_jinja_is_mirrored():
    """The file whose omission produced a silently-templateless model."""
    kept = select_model_files(["config.json", "chat_template.jinja", *SHARDS])
    assert "chat_template.jinja" in kept


def test_unrecognized_extensions_are_mirrored_rather_than_dropped():
    """The denylist property: a file nobody anticipated is copied, not silently lost.

    An allowlist fails these by construction, which is the whole reason for the
    inversion — the cost of an extra small file is bytes, the cost of a missing one
    is a broken model discovered much later.
    """
    novel = [
        "chat_template.jinja2",
        "preprocessor_config.xml",
        "inference.toml",
        "adapter_model.tensors",
        "merges.bpe",
        "spiece.vocab",
    ]
    kept = select_model_files([*novel, "config.json", *SHARDS])
    assert set(novel) <= set(kept)


def test_duplicate_weight_formats_are_dropped_when_safetensors_exist():
    """A repo shipping both formats holds the same tensors twice; copying both
    doubles a 125 GiB mirror for nothing."""
    kept = select_model_files(["pytorch_model.bin", "model.h5", "config.json", *SHARDS])
    assert "pytorch_model.bin" not in kept
    assert "model.h5" not in kept
    assert set(SHARDS) <= set(kept)


def test_duplicate_weight_formats_are_KEPT_when_the_repo_has_no_safetensors():
    """The exclusion is conditional. Dropping .bin from a .bin-only repo would
    mirror a model with no weights at all."""
    kept = select_model_files(["pytorch_model.bin", "config.json", "tokenizer.json"])
    assert "pytorch_model.bin" in kept


def test_repository_plumbing_and_media_are_dropped():
    kept = select_model_files(
        [
            ".gitattributes",
            ".git/config",
            "figure.png",
            "demo.mp4",
            "config.json",
            *SHARDS,
        ]
    )
    assert kept == ["config.json", *SHARDS]


def test_metadata_is_ordered_before_shards_so_a_partial_mirror_leaves_usable_config():
    kept = select_model_files(
        [*SHARDS, "tokenizer.json", "config.json", "chat_template.jinja"]
    )
    shard_positions = [kept.index(s) for s in SHARDS]
    metadata_positions = [
        kept.index(f) for f in ("config.json", "tokenizer.json", "chat_template.jinja")
    ]
    assert max(metadata_positions) < min(shard_positions)


def test_readme_is_mirrored_because_model_cards_carry_licence_and_usage_terms():
    """README.md was dropped by the old allowlist. It is small and it is the only
    place a repo's licence and intended-use terms live, so a mirror without it is a
    redistribution with the terms stripped off."""
    assert "README.md" in select_model_files(["README.md", "config.json", *SHARDS])
