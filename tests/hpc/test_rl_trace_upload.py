import pytest

from hpc.rl_launch_utils import validate_trace_upload_environment


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
