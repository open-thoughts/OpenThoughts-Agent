from pathlib import Path

import pytest

from hpc.rl_paths import (
    AmbiguousCheckpointError,
    CheckpointLayoutError,
    RLLaunchIntent,
    RLPathManager,
    RLResumePolicy,
    RLResumeMode,
    hydra_override_values,
)
from hpc.rl_config_utils import ParsedRLConfig, build_skyrl_hydra_args
from hpc.rl_launch_utils import RLJobConfig, RLJobRunner


JOB_NAME = "tasktrove-arm"


def _write_checkpoint(state_root: Path, step: int) -> Path:
    checkpoint_root = state_root / JOB_NAME / "checkpoints"
    checkpoint_path = checkpoint_root / f"global_step_{step}"
    checkpoint_path.mkdir(parents=True)
    (checkpoint_root / "latest_ckpt_global_step.txt").write_text(str(step))
    return checkpoint_path


def _hydra_value(arguments: list[str], key: str) -> str | None:
    return hydra_override_values(arguments).get(key)


def test_forked_checkpoint_becomes_the_single_durable_run_root(tmp_path: Path) -> None:
    canonical_root = tmp_path / JOB_NAME
    launch_root = tmp_path / f"{JOB_NAME}_4"
    checkpoint_root = tmp_path / f"{JOB_NAME}_3"
    checkpoint_path = _write_checkpoint(checkpoint_root, 10)

    resolved = RLPathManager(JOB_NAME, canonical_root, launch_root).resolve()

    assert resolved.resume_mode is RLResumeMode.LATEST
    assert resolved.resume_path == checkpoint_path
    assert resolved.checkpoint_dir == checkpoint_root / JOB_NAME / "checkpoints"
    assert resolved.export_dir == checkpoint_root / JOB_NAME / "exports"
    assert resolved.trials_dir == checkpoint_root / JOB_NAME / "trace_jobs"


def test_highest_checkpoint_wins_across_canonical_and_forked_roots(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / JOB_NAME
    launch_root = tmp_path / f"{JOB_NAME}_4"
    _write_checkpoint(canonical_root, 6)
    checkpoint_path = _write_checkpoint(tmp_path / f"{JOB_NAME}_3", 10)

    resolved = RLPathManager(JOB_NAME, canonical_root, launch_root).resolve()

    assert resolved.resume_path == checkpoint_path
    assert resolved.checkpoint_dir == checkpoint_path.parent


def test_new_run_uses_canonical_state_root_after_artifact_collision(
    tmp_path: Path,
) -> None:
    canonical_root = tmp_path / JOB_NAME
    launch_root = tmp_path / f"{JOB_NAME}_2"

    resolved = RLPathManager(JOB_NAME, canonical_root, launch_root).resolve()

    assert resolved.resume_mode is RLResumeMode.NONE
    assert resolved.resume_policy is RLResumePolicy.AT_LINK_START
    assert resolved.resume_path is None
    assert resolved.checkpoint_dir == canonical_root / JOB_NAME / "checkpoints"


def test_explicit_fresh_launch_uses_the_launch_root(tmp_path: Path) -> None:
    canonical_root = tmp_path / JOB_NAME
    launch_root = tmp_path / f"{JOB_NAME}_2"
    _write_checkpoint(canonical_root, 8)

    resolved = RLPathManager(JOB_NAME, canonical_root, launch_root).resolve(
        launch_intent=RLLaunchIntent.FRESH,
    )

    assert resolved.resume_mode is RLResumeMode.NONE
    assert resolved.resume_policy is RLResumePolicy.FIXED
    assert resolved.resume_path is None
    assert resolved.checkpoint_dir == launch_root / JOB_NAME / "checkpoints"


def test_explicit_latest_rejects_empty_checkpoint_directory(tmp_path: Path) -> None:
    root = tmp_path / JOB_NAME
    checkpoint_dir = root / JOB_NAME / "checkpoints"

    with pytest.raises(CheckpointLayoutError, match="resume_mode=latest"):
        RLPathManager(JOB_NAME, root, root).resolve(
            skyrl_overrides=(
                f"trainer.ckpt_path={checkpoint_dir}",
                "trainer.resume_mode=latest",
            )
        )


def test_explicit_resume_path_becomes_the_checkpoint_write_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / JOB_NAME
    checkpoint_path = _write_checkpoint(tmp_path / f"{JOB_NAME}_3", 10)

    resolved = RLPathManager(JOB_NAME, root, root).resolve(
        skyrl_overrides=(
            "trainer.resume_mode=from_path",
            f"trainer.resume_path={checkpoint_path}",
        )
    )

    assert resolved.resume_path == checkpoint_path
    assert resolved.checkpoint_dir == checkpoint_path.parent
    assert resolved.export_dir == tmp_path / f"{JOB_NAME}_3" / JOB_NAME / "exports"


def test_explicit_resume_path_rejects_a_different_checkpoint_write_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / JOB_NAME
    checkpoint_path = _write_checkpoint(tmp_path / f"{JOB_NAME}_3", 10)

    with pytest.raises(CheckpointLayoutError, match="is not under trainer.ckpt_path"):
        RLPathManager(JOB_NAME, root, root).resolve(
            skyrl_overrides=(
                f"trainer.ckpt_path={root / JOB_NAME / 'checkpoints'}",
                "trainer.resume_mode=from_path",
                f"trainer.resume_path={checkpoint_path}",
            )
        )


def test_nonstandard_checkpoint_path_requires_explicit_sibling_destinations(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "standalone-checkpoints"

    with pytest.raises(CheckpointLayoutError, match="requires explicit values"):
        RLPathManager(JOB_NAME, tmp_path, tmp_path).resolve(
            skyrl_overrides=(f"trainer.ckpt_path={checkpoint_dir}",),
        )


def test_checkpoint_marker_must_name_an_existing_step(tmp_path: Path) -> None:
    root = tmp_path / JOB_NAME
    checkpoint_root = root / JOB_NAME / "checkpoints"
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "global_step_4").mkdir()
    (checkpoint_root / "latest_ckpt_global_step.txt").write_text("5")

    with pytest.raises(CheckpointLayoutError, match="global_step_5"):
        RLPathManager(JOB_NAME, root, root).resolve()


def test_duplicate_highest_steps_require_explicit_resume_path(tmp_path: Path) -> None:
    canonical_root = tmp_path / JOB_NAME
    launch_root = tmp_path / f"{JOB_NAME}_4"
    _write_checkpoint(canonical_root, 10)
    _write_checkpoint(tmp_path / f"{JOB_NAME}_3", 10)

    with pytest.raises(AmbiguousCheckpointError, match="global_step_10"):
        RLPathManager(JOB_NAME, canonical_root, launch_root).resolve()


def test_yaml_path_settings_feed_the_manager_contract(tmp_path: Path) -> None:
    root = tmp_path / JOB_NAME
    configured_root = tmp_path / "configured"
    checkpoint_path = _write_checkpoint(configured_root, 7)
    parsed = ParsedRLConfig(
        config_path=tmp_path / "config.yaml",
        raw={},
        entrypoint="skyrl_train.entrypoints.main_base",
        trainer={
            "resume_mode": "latest",
            "ckpt_path": str(checkpoint_path.parent),
            "export_path": str(configured_root / JOB_NAME / "model_exports"),
        },
        terminal_bench={"trials_dir": str(configured_root / JOB_NAME / "trials")},
    )
    run_paths = RLPathManager(JOB_NAME, root, root).resolve(
        trainer_config=parsed.trainer,
        terminal_bench_config=parsed.terminal_bench,
    )

    assert run_paths.checkpoint_dir == checkpoint_path.parent
    assert run_paths.export_dir == configured_root / JOB_NAME / "model_exports"
    assert run_paths.trials_dir == configured_root / JOB_NAME / "trials"
    assert run_paths.resume_path == checkpoint_path


def test_hydra_builder_consumes_the_resolved_path_contract(tmp_path: Path) -> None:
    root = tmp_path / JOB_NAME
    checkpoint_path = _write_checkpoint(root, 7)
    run_paths = RLPathManager(JOB_NAME, root, root).resolve()
    parsed = ParsedRLConfig(
        config_path=tmp_path / "config.yaml",
        raw={},
        entrypoint="skyrl_train.entrypoints.main_base",
        trainer={
            "resume_mode": "latest",
            "ckpt_path": "/stale/checkpoints",
            "export_path": "/stale/exports",
        },
        terminal_bench={"trials_dir": "/stale/trials"},
    )
    hpc = type("HPC", (), {"gpus_per_node": 4})()

    arguments = build_skyrl_hydra_args(
        parsed, {"job_name": JOB_NAME}, hpc, run_paths=run_paths
    )

    assert _hydra_value(arguments, "trainer.ckpt_path") == str(run_paths.checkpoint_dir)
    assert _hydra_value(arguments, "trainer.export_path") == str(run_paths.export_dir)
    assert _hydra_value(arguments, "trainer.resume_mode") == "latest"
    assert _hydra_value(arguments, "trainer.resume_path") == str(checkpoint_path)
    assert _hydra_value(arguments, "terminal_bench_config.trials_dir") == str(
        run_paths.trials_dir
    )


def test_hydra_builder_does_not_enable_hf_export_without_a_repository(
    tmp_path: Path,
) -> None:
    root = tmp_path / JOB_NAME
    run_paths = RLPathManager(JOB_NAME, root, root).resolve()
    parsed = ParsedRLConfig(
        config_path=tmp_path / "config.yaml",
        raw={},
        entrypoint="skyrl_train.entrypoints.main_base",
        trainer={"hf_save_interval": 6},
    )
    hpc = type("HPC", (), {"gpus_per_node": 4})()

    arguments = build_skyrl_hydra_args(
        parsed, {"job_name": JOB_NAME}, hpc, run_paths=run_paths
    )

    assert _hydra_value(arguments, "trainer.hf_save_interval") == "6"
    assert _hydra_value(arguments, "trainer.hf_hub_repo_id") is None


def test_hydra_builder_enables_hf_export_for_an_explicit_repository(
    tmp_path: Path,
) -> None:
    root = tmp_path / JOB_NAME
    run_paths = RLPathManager(JOB_NAME, root, root).resolve()
    parsed = ParsedRLConfig(
        config_path=tmp_path / "config.yaml",
        raw={},
        entrypoint="skyrl_train.entrypoints.main_base",
        trainer={"hf_save_interval": 6},
    )
    hpc = type("HPC", (), {"gpus_per_node": 4})()

    arguments = build_skyrl_hydra_args(
        parsed,
        {"job_name": JOB_NAME, "hf_hub_repo_id": "open-thoughts/explicit-run"},
        hpc,
        run_paths=run_paths,
    )

    assert (
        _hydra_value(arguments, "trainer.hf_hub_repo_id")
        == "open-thoughts/explicit-run"
    )


def test_yaml_latest_without_a_checkpoint_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / JOB_NAME

    with pytest.raises(CheckpointLayoutError, match="resume_mode=latest"):
        RLPathManager(JOB_NAME, root, root).resolve(
            trainer_config={"resume_mode": "latest"},
        )


def test_trace_upload_consumes_the_manager_resolved_trials_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trials_dir = tmp_path / "durable" / "custom-trials"
    trials_dir.mkdir(parents=True)
    launch_root = tmp_path / "launch-artifacts"
    config = RLJobConfig(
        job_name=JOB_NAME,
        experiments_dir=str(launch_root),
        cluster_name="test",
        skyrl_entrypoint="skyrl_train.entrypoints.main_base",
        trials_dir=str(trials_dir),
        trace_upload_enabled=True,
    )
    captured: dict[str, list[str]] = {}

    def fake_popen(command: list[str], *, stdout, stderr):
        captured["command"] = command
        stdout.close()
        return object()

    monkeypatch.setattr("hpc.rl_launch_utils.subprocess.Popen", fake_popen)

    process = RLJobRunner(config)._launch_trace_upload(training_exit_code=0)

    assert process is not None
    command = captured["command"]
    assert command[command.index("--job_dir") + 1] == str(trials_dir)
