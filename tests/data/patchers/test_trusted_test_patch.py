from __future__ import annotations

import subprocess
from pathlib import Path

from data.patchers.trusted_test_patch import (
    TRUSTED_TEST_PATCH_INSTALLER,
    write_trusted_test_patch_installer,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _base_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@example.com")
    _git(repo, "config", "user.name", "Test Author")
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "feature.py").write_text("VALUE = 'base'\n")
    (repo / "tests" / "test_feature.py").write_text("EXPECTED = 'old'\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD").stdout.strip()


def _trusted_patch(repo: Path, patch_path: Path) -> None:
    (repo / "tests" / "test_feature.py").write_text("EXPECTED = 'trusted'\n")
    (repo / "tests" / "test_hidden.py").write_text("HIDDEN = True\n")
    _git(repo, "add", "--intent-to-add", "tests/test_hidden.py")
    patch_path.write_text(_git(repo, "diff", "--binary").stdout)
    _git(repo, "reset", "--hard", "--quiet", "HEAD")
    _git(repo, "clean", "-fd", "--quiet")


def test_installer_replaces_agent_test_edits_and_preserves_source_edits(
    tmp_path: Path,
) -> None:
    repo, base_commit = _base_repository(tmp_path)
    patch_path = tmp_path / "test_patch.diff"
    _trusted_patch(repo, patch_path)

    (repo / "src" / "feature.py").write_text("VALUE = 'agent fix'\n")
    (repo / "tests" / "test_feature.py").write_text("EXPECTED = 'agent bypass'\n")
    (repo / "tests" / "test_hidden.py").write_text("HIDDEN = False\n")

    installer = tmp_path / "install_trusted_test_patch.sh"
    write_trusted_test_patch_installer(installer)
    subprocess.run(
        ["bash", str(installer), str(repo), str(patch_path), base_commit],
        check=True,
    )

    assert (repo / "src" / "feature.py").read_text() == "VALUE = 'agent fix'\n"
    assert (repo / "tests" / "test_feature.py").read_text() == "EXPECTED = 'trusted'\n"
    assert (repo / "tests" / "test_hidden.py").read_text() == "HIDDEN = True\n"


def test_installer_fails_when_trusted_patch_cannot_be_installed(tmp_path: Path) -> None:
    repo, base_commit = _base_repository(tmp_path)
    patch_path = tmp_path / "test_patch.diff"
    patch_path.write_text(
        "diff --git a/tests/missing.py b/tests/missing.py\n"
        "--- a/tests/missing.py\n"
        "+++ b/tests/missing.py\n"
        "@@ -1 +1 @@\n"
        "-missing\n"
        "+trusted\n"
    )
    installer = tmp_path / "install_trusted_test_patch.sh"
    write_trusted_test_patch_installer(installer)

    result = subprocess.run(
        ["bash", str(installer), str(repo), str(patch_path), base_commit],
        check=False,
    )

    assert result.returncode != 0


def test_installer_does_not_require_git_apply_allow_empty() -> None:
    assert "--allow-empty" not in TRUSTED_TEST_PATCH_INSTALLER


def test_installer_marks_only_the_target_repository_as_safe() -> None:
    assert 'git -c safe.directory="$repository"' in TRUSTED_TEST_PATCH_INSTALLER
    assert "git config --global" not in TRUSTED_TEST_PATCH_INSTALLER
