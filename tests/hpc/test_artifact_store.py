from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import json
import subprocess

import pytest

from hpc.artifact_store import (
    ArtifactStoreBusyError,
    ensure_image,
    mount_lease_path,
    mount_path_for_image,
    mounted,
    resolve_trials_root,
    write_authority_record,
)


def test_resolve_trials_root_supports_legacy_and_image_runs(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    (legacy / "trace_jobs").mkdir(parents=True)
    assert resolve_trials_root(legacy) == ("bare", legacy / "trace_jobs")

    image_run = tmp_path / "image-run"
    image_run.mkdir()
    image = image_run / "artifact_store.img"
    image.touch()
    assert resolve_trials_root(image_run) == ("image", image)


def test_ensure_image_formats_a_sparse_temporary_then_publishes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool):
        calls.append(command)
        if command[0] == "truncate":
            Path(command[-1]).touch()
        return SimpleNamespace(returncode=0, args=command)

    monkeypatch.setattr("hpc.artifact_store.subprocess.run", fake_run)
    image = tmp_path / "run" / "artifact_store.img"

    ensure_image(image, size="2T", inode_count=123)

    assert image.is_file()
    assert calls[0][:3] == ["truncate", "-s", "2T"]
    assert calls[1][:4] == ["mkfs.ext4", "-F", "-N", "123"]


def test_mounted_read_only_unmounts_after_the_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "artifact_store.img"
    image.touch()
    mount_path = tmp_path / "mount"
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool):
        calls.append(command)
        return SimpleNamespace(returncode=0, args=command)

    monkeypatch.setattr("hpc.artifact_store.subprocess.run", fake_run)

    with mounted(image, mount_path=mount_path) as root:
        assert root == mount_path

    assert calls == [
        ["fuse2fs", "-o", "ro", str(image), str(mount_path)],
        ["mountpoint", "-q", str(mount_path)],
        ["fusermount3", "-u", str(mount_path)],
    ]
    assert not mount_lease_path(image).exists()


def test_mounted_read_write_uses_fakeroot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "artifact_store.img"
    image.touch()
    mount_path = tmp_path / "mount"
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool):
        calls.append(command)
        return SimpleNamespace(returncode=0, args=command)

    monkeypatch.setattr("hpc.artifact_store.subprocess.run", fake_run)

    with mounted(image, mode="rw", mount_path=mount_path):
        pass

    assert ["fuse2fs", "-o", "rw,fakeroot", str(image), str(mount_path)] in calls


def test_mounted_refuses_an_image_with_an_active_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "artifact_store.img"
    image.touch()

    mount_lease_path(image).mkdir()

    with pytest.raises(ArtifactStoreBusyError, match="active mount lease"):
        with mounted(image):
            pass


def test_mount_path_is_stable_and_node_local(tmp_path: Path) -> None:
    image = tmp_path / "run" / "artifact_store.img"

    first = mount_path_for_image(image)
    second = mount_path_for_image(image)

    assert first == second
    assert first.parent == Path("/tmp/otagent-artifact-stores")


def test_mounted_fails_closed_when_fuse2fs_did_not_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "artifact_store.img"
    image.touch()
    calls: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool):
        calls.append(command)
        return SimpleNamespace(
            returncode=1 if command[0] == "mountpoint" else 0, args=command
        )

    monkeypatch.setattr("hpc.artifact_store.subprocess.run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        with mounted(image, mount_path=tmp_path / "mount"):
            pass

    assert ["fusermount3", "-u", str(tmp_path / "mount")] in calls
    assert not mount_lease_path(image).exists()


def test_authority_record_names_the_live_node_and_mount(tmp_path: Path) -> None:
    image = tmp_path / "artifact_store.img"
    mount_path = tmp_path / "artifact_mnt"

    record = write_authority_record(
        image,
        job_id="123",
        node="jwb0123",
        node_ip="10.0.0.2",
        mount_path=mount_path,
    )

    assert json.loads(record.read_text()) == {
        "image": str(image),
        "job_id": "123",
        "mount": str(mount_path),
        "node": "jwb0123",
        "node_ip": "10.0.0.2",
        "trials": str(mount_path / "trace_jobs"),
    }


def test_rl_template_mounts_before_container_setup_and_forwards_term() -> None:
    template = Path("hpc/sbatch_rl/universal_rl.sbatch").read_text()

    assert "#SBATCH --signal=B:USR1@300" in template
    assert template.index("fuse2fs -o rw,fakeroot") < template.index(
        "setup_container_runtime"
    )
    assert "allow_other" not in template
    assert 'mountpoint -q "$ARTIFACT_STORE_MOUNT"' in template
    assert 'mkdir "$ARTIFACT_STORE_LEASE"' in template
    assert "trap forward_termination TERM USR1" in template
    assert 'kill -TERM "$RL_RUNNER_PID"' in template
    assert template.index("sync") < template.index(
        'fusermount3 -u "$ARTIFACT_STORE_MOUNT"'
    )


def test_trace_exporter_mounts_an_image_backed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.harbor import make_and_upload_trace_dataset as exporter

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    image = run_dir / "artifact_store.img"
    image.touch()
    mounted_root = tmp_path / "mounted"
    mounted_root.mkdir()
    args = SimpleNamespace(job_dir=str(run_dir))
    observed = {}

    @contextmanager
    def fake_mounted(image_path, mode):
        observed["mount"] = (image_path, mode)
        yield mounted_root

    def fake_run_export(actual_args):
        observed["job_dir"] = actual_args.job_dir
        observed["literal_root"] = actual_args.literal_discovery_root

    monkeypatch.setattr(exporter, "_parse_args", lambda: args)
    monkeypatch.setattr(exporter, "mounted", fake_mounted)
    monkeypatch.setattr(exporter, "_run_export", fake_run_export)

    exporter.main()

    assert observed == {
        "mount": (image, "ro"),
        "job_dir": str(mounted_root),
        "literal_root": str(run_dir),
    }
