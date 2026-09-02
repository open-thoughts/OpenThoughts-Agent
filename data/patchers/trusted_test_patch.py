"""Install hidden test patches independently of agent workspace edits."""

from __future__ import annotations

import stat
import shlex
from pathlib import Path


TRUSTED_TEST_PATCH_INSTALLER = r"""#!/bin/bash
set -Eeuo pipefail

repository=${1:?repository path is required}
patch_path=${2:?test patch path is required}
base_commit=${3:?base commit is required}

cd "$repository"
git -c safe.directory="$repository" cat-file -e "${base_commit}^{commit}"

if [ ! -s "$patch_path" ]; then
    exit 0
fi

# Validate the patch before changing the workspace. --numstat -z makes path
# records unambiguous even when a filename contains spaces.
git -c safe.directory="$repository" -c core.quotePath=false \
    apply --numstat -z "$patch_path" >/dev/null

while IFS= read -r -d '' record; do
    path=${record#*$'\t'}
    path=${path#*$'\t'}
    if [ "$path" = "$record" ]; then
        echo "Malformed git-apply path record" >&2
        exit 1
    fi
    case "$path" in
        ""|/*|.|..|../*|*/..|*/../*)
            echo "Unsafe hidden-test path: $path" >&2
            exit 1
            ;;
    esac

    # The hidden patch owns these exact paths. Remove untracked/ignored agent
    # replacements and restore tracked files from the immutable task base.
    git -c safe.directory="$repository" clean -ffdx -- "$path"
    if git -c safe.directory="$repository" \
        cat-file -e "${base_commit}:${path}" 2>/dev/null; then
        git -c safe.directory="$repository" \
            restore --source="$base_commit" --staged --worktree -- "$path"
    fi
done < <(git -c safe.directory="$repository" -c core.quotePath=false \
    apply --numstat -z "$patch_path")

git -c safe.directory="$repository" apply --check "$patch_path"
git -c safe.directory="$repository" apply --verbose "$patch_path"
"""


def write_trusted_test_patch_installer(path: Path) -> None:
    """Write the executable hidden-test installer used by packaged verifiers."""
    path.write_text(TRUSTED_TEST_PATCH_INSTALLER, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def trusted_test_patch_command(base_commit: str) -> str:
    """Return the verifier command anchored to the task's immutable base commit."""
    if not base_commit:
        raise ValueError("base commit is required for trusted hidden-test installation")
    return (
        "bash /tests/install_trusted_test_patch.sh "
        f"/testbed /tests/test_patch.diff {shlex.quote(base_commit)}"
    )
