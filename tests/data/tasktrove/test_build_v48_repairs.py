import tomllib

from data.tasktrove.build_storage_repair import task_toml_with_memory
from data.tasktrove.build_v48_repairs import (
    PHP_NEW_EXECUTION_GATE,
    PHP_OLD_EXECUTION_GATE,
    patch_php_verifier,
)
from data.tasktrove.publish_v48_release import updated_readme


def test_memory_requirement_is_added_to_existing_environment() -> None:
    source = b'version = "1.0"\n\n[environment]\nstorage_mb = 4096\n'
    transformed = task_toml_with_memory(source, None, 4096)
    parsed = tomllib.loads(transformed.decode())

    assert parsed["environment"] == {"memory_mb": 4096, "storage_mb": 4096}


def test_memory_requirement_is_added_with_environment_table() -> None:
    source = b'version = "1.0"\n'
    transformed = task_toml_with_memory(source, None, 4096)
    parsed = tomllib.loads(transformed.decode())

    assert parsed["environment"]["memory_mb"] == 4096


def test_php_success_output_counts_as_executed_tests() -> None:
    source = b"header\n" + PHP_OLD_EXECUTION_GATE + b"\nfooter\n"
    transformed = patch_php_verifier(source)

    assert PHP_OLD_EXECUTION_GATE not in transformed
    assert PHP_NEW_EXECUTION_GATE in transformed


def test_v48_readme_replaces_current_marker() -> None:
    manifest = {
        "release_unique_images": 6,
        "datasets": [
            {
                "output": "DCAgent__exp_rle_adversarial-v6",
                "source_rows": 2731,
                "output_rows": 2730,
            }
        ],
    }
    result = updated_readme("> **v4.7 (current)** — previous\n", manifest)

    assert result.startswith("> **v4.8 (current)**")
    assert "> **v4.7** — previous" in result
