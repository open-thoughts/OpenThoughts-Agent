from hpc.rl_launch_utils import build_apptainer_prefix


def _container_environment(prefix: list[str]) -> dict[str, str]:
    return {
        value.split("=", 1)[0]: value.split("=", 1)[1]
        for index, value in enumerate(prefix)
        if index > 0 and prefix[index - 1] == "--env"
    }


def test_apptainer_runtime_bounds_both_raylet_startup_phases(monkeypatch) -> None:
    monkeypatch.delenv("RAY_raylet_start_wait_time_s", raising=False)
    monkeypatch.delenv("RAY_raylet_client_num_connect_attempts", raising=False)

    environment = _container_environment(
        build_apptainer_prefix("runtime.sif", binds=[])
    )

    assert environment["RAY_raylet_start_wait_time_s"] == "120"
    assert environment["RAY_raylet_client_num_connect_attempts"] == "120"


def test_apptainer_runtime_honors_explicit_raylet_startup_overrides(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RAY_raylet_start_wait_time_s", "181")
    monkeypatch.setenv("RAY_raylet_client_num_connect_attempts", "182")

    environment = _container_environment(
        build_apptainer_prefix("runtime.sif", binds=[])
    )

    assert environment["RAY_raylet_start_wait_time_s"] == "181"
    assert environment["RAY_raylet_client_num_connect_attempts"] == "182"
