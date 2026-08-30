import pytest

from hpc.rl_launch_utils import (
    RLJobConfig,
    RLJobRunner,
    _parse_artifact_store,
    validate_trace_upload_environment,
)


def test_trace_upload_rejects_offline_hub_environment() -> None:
    terminal_bench = {"trace_upload": {"enabled": True}}
    container = {
        "extra_env": {
            "HF_HUB_OFFLINE": 1,
            "APPTAINERENV_HF_HUB_OFFLINE": 1,
        }
    }

    with pytest.raises(
        ValueError,
        match=r"trace_upload\.enabled=true conflicts with .*HF_HUB_OFFLINE",
    ):
        validate_trace_upload_environment(terminal_bench, container)


def test_disabled_trace_upload_allows_offline_hub_environment() -> None:
    terminal_bench = {"trace_upload": {"enabled": False}}
    container = {"extra_env": {"HF_HUB_OFFLINE": 1}}

    validate_trace_upload_environment(terminal_bench, container)


def test_artifact_store_config_defaults_to_the_large_sparse_geometry() -> None:
    assert _parse_artifact_store({"artifact_store": {"enabled": True}}) == (
        True,
        "1T",
        50_000_000,
    )


def test_artifact_store_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown container.artifact_store fields"):
        _parse_artifact_store(
            {"artifact_store": {"enabled": True, "surprise": "value"}}
        )


def test_artifact_authority_override_replaces_a_stale_value() -> None:
    config = RLJobConfig(
        job_name="run",
        experiments_dir="/tmp",
        cluster_name="test",
        skyrl_entrypoint="skyrl_train.entrypoints.main_base",
        trials_dir="/tmp/trials",
        skyrl_hydra_args=[
            "trainer.max_steps=10",
            "++trainer.entrypoint_node_ip=10.0.0.1",
        ],
    )
    runner = RLJobRunner(config)

    runner._set_hydra_override("trainer.entrypoint_node_ip", "10.0.0.2", optional=True)

    assert config.skyrl_hydra_args == [
        "trainer.max_steps=10",
        "++trainer.entrypoint_node_ip=10.0.0.2",
    ]


def test_runner_forwards_term_to_the_active_trainer() -> None:
    class Process:
        pid = 42
        signal = None

        def poll(self):
            return None

        def send_signal(self, signum):
            self.signal = signum

    config = RLJobConfig(
        job_name="run",
        experiments_dir="/tmp",
        cluster_name="test",
        skyrl_entrypoint="skyrl_train.entrypoints.main_base",
        trials_dir="/tmp/trials",
    )
    runner = RLJobRunner(config)
    process = Process()
    runner._active_process = process

    runner.handle_termination(15, None)

    assert process.signal == 15
    assert runner._termination_requested is True
