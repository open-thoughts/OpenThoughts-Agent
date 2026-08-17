from unittest.mock import Mock

from hpc import rl_launch_utils


def test_hub_model_prefetch_keeps_replayable_repository_reference(monkeypatch) -> None:
    download = Mock(return_value=Mock(local_path="/cache/hub/models--Qwen--Model/snapshots/abc123"))
    monkeypatch.setattr(rl_launch_utils, "pre_download_model", download)

    model_path = rl_launch_utils.prefetch_rl_model("Qwen/Model")

    assert model_path == "Qwen/Model"
    download.assert_called_once_with("Qwen/Model")


def test_explicit_local_model_path_is_unchanged(monkeypatch) -> None:
    download = Mock()
    monkeypatch.setattr(rl_launch_utils, "pre_download_model", download)

    model_path = rl_launch_utils.prefetch_rl_model("/shared/models/model")

    assert model_path == "/shared/models/model"
    download.assert_not_called()
