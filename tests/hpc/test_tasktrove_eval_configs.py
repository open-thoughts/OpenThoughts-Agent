from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EVAL_CONFIG_DIR = REPOSITORY_ROOT / "hpc/harbor_yaml/eval"
MAX_VERIFIER_TIMEOUT = 600


def tasktrove_eval_configs() -> list[Path]:
    return sorted(
        {
            *EVAL_CONFIG_DIR.glob("tasktrove*.yaml"),
            *EVAL_CONFIG_DIR.glob("snowball_tasktrove*.yaml"),
        }
    )


def test_tasktrove_verifiers_are_capped_at_ten_minutes() -> None:
    paths = tasktrove_eval_configs()
    assert paths

    for path in paths:
        config = yaml.safe_load(path.read_text())
        verifier = config["verifier"]
        assert verifier["max_timeout_sec"] <= MAX_VERIFIER_TIMEOUT, path
        override = verifier.get("override_timeout_sec")
        assert override is None or override <= MAX_VERIFIER_TIMEOUT, path
