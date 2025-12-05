#!/usr/bin/env python3
"""
Clone a Hugging Face repository, strip all history, remove checkpoint directories,
and force push the cleaned snapshot back to the original location.

Steps:
1. Clone the target repository locally.
2. Delete the git history and reinitialize a fresh repository.
3. Remove any directories matching the pattern ``checkpoints*``.
4. Force push the sanitized tree back to Hugging Face using the same repo address.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfFolder, Repository


def _run(cmd: list[str], cwd: Path, redact: list[str] | None = None) -> None:
    """Run a shell command in ``cwd`` while printing a redacted log line."""
    display = " ".join(cmd)
    if redact:
        for secret in redact:
            if secret:
                display = display.replace(secret, "***")
    print(f"[reset] $ {display}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _resolve_token(explicit_token: str | None) -> str:
    if explicit_token:
        return explicit_token
    env_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token
    token = HfFolder.get_token()
    if token:
        return token
    raise RuntimeError(
        "Hugging Face access token not found. "
        "Pass --token or set HF_TOKEN / HUGGINGFACE_HUB_TOKEN."
    )


def _repo_base_url(repo_id: str, repo_type: str) -> str:
    base = "https://huggingface.co"
    if repo_type == "dataset":
        base = f"{base}/datasets"
    elif repo_type == "space":
        base = f"{base}/spaces"
    return f"{base}/{repo_id}.git"


def _with_auth(url: str, token: str) -> str:
    scheme, rest = url.split("://", 1)
    return f"{scheme}://user:{token}@{rest}"


def _remove_checkpoint_dirs(repo_dir: Path) -> list[Path]:
    candidates = [
        path
        for path in repo_dir.rglob("*")
        if path.is_dir() and path.name.startswith("checkpoints")
    ]
    # Remove deeper paths first so parents survive until children are gone.
    candidates.sort(key=lambda p: len(p.parts), reverse=True)

    removed = []
    for path in candidates:
        shutil.rmtree(path)
        print(f"[reset] Removed checkpoint directory: {path.relative_to(repo_dir)}")
        removed.append(path)
    if not removed:
        print("[reset] No checkpoint directories found.")
    return removed


def reset_repo(
    repo_id: str,
    repo_type: str,
    branch: str,
    commit_message: str,
    token: str,
    git_user_name: str,
    git_user_email: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="hf-reset-") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        repo_dir = tmpdir / "repo"
        print(f"[reset] Cloning {repo_type} repo '{repo_id}' into {repo_dir}")
        Repository(
            local_dir=str(repo_dir),
            clone_from=repo_id,
            repo_type=repo_type,
            token=token,
            revision=branch,
            skip_lfs_files=False,
        )

        _run(["git", "checkout", branch], cwd=repo_dir)
        _run(["git", "pull", "--ff-only"], cwd=repo_dir)

        remote_url = _with_auth(_repo_base_url(repo_id, repo_type), token)

        git_dir = repo_dir / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)
            print(f"[reset] Removed existing git metadata at {git_dir}")
        else:
            print("[reset] No existing .git directory found; continuing.")

        _remove_checkpoint_dirs(repo_dir)

        _run(["git", "init"], cwd=repo_dir)
        _run(["git", "config", "user.name", git_user_name], cwd=repo_dir)
        _run(["git", "config", "user.email", git_user_email], cwd=repo_dir)
        _run(["git", "add", "-A"], cwd=repo_dir)
        _run(["git", "commit", "--allow-empty", "-m", commit_message], cwd=repo_dir)
        _run(["git", "branch", "-M", branch], cwd=repo_dir)
        _run(["git", "remote", "add", "origin", remote_url], cwd=repo_dir, redact=[token])
        _run(["git", "push", "--force", "origin", branch], cwd=repo_dir, redact=[token])
        print("[reset] Force push complete; repository history replaced.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a Hugging Face repository from a clean snapshot "
            "after removing checkpoint directories."
        )
    )
    parser.add_argument("repo_id", help="Target repo in the form <namespace>/<name>.")
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="dataset",
        help="Hugging Face repo type (default: %(default)s).",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to recreate and force push (default: %(default)s).",
    )
    parser.add_argument(
        "--commit-message",
        default="Reset repository without checkpoints directories",
        help="Commit message for the new snapshot commit.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (otherwise taken from environment).",
    )
    parser.add_argument(
        "--git-user-name",
        default="hf-reset",
        help="Git user.name to use for the replacement commit.",
    )
    parser.add_argument(
        "--git-user-email",
        default="hf-reset@example.com",
        help="Git user.email to use for the replacement commit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        token = _resolve_token(args.token)
        reset_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            branch=args.branch,
            commit_message=args.commit_message,
            token=token,
            git_user_name=args.git_user_name,
            git_user_email=args.git_user_email,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[reset] Command failed: {exc}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:  # noqa: BLE001
        print(f"[reset] Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
