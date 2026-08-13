from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_30b_kl_reference_does_not_share_policy_host_memory() -> None:
    config_path = (
        REPOSITORY_ROOT / "hpc/skyrl_yaml/jupiter/24GPU_qwen3_30b_a3b_thinking.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    assert config["trainer"]["placement"]["colocate_policy_ref"] is True
    assert config["trainer"]["ref"]["fsdp_config"]["cpu_offload"] is False
