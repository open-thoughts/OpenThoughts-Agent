#!/usr/bin/env python3
"""
Download task-binary datasets listed in a markdown file and upload them as
subdirectories of a single HF dataset repo — without loading or altering
the underlying files.

Each source repo `org/name` becomes a subdirectory `org__name/` in the target
repo, preserving every file exactly as downloaded (Parquet shards, README, etc.).

Example:
  python -m scripts.datagen.collect_task_binaries \\
    --source-file /path/to/task_binary_interesting.md \\
    --target mlfoundations-dev/all-task-binaries \\
    --private

  # Preview which repos would be processed:
  python -m scripts.datagen.collect_task_binaries \\
    --source-file /path/to/task_binary_interesting.md \\
    --target x/y --dry-run

Requires: HF_TOKEN env var (or --token).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download, create_repo


# ── parsing ───────────────────────────────────────────────────────────────────

_REPO_RE = re.compile(r"\b([A-Za-z0-9_\-\.]+/[A-Za-z0-9_\-\.]+)\s*$")


def _parse_repo_ids(md_path: Path) -> list[str]:
    """Extract HF repo IDs from a numbered markdown list."""
    seen: set[str] = set()
    result: list[str] = []
    for line in md_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _REPO_RE.search(line)
        if m:
            repo_id = m.group(1)
            if repo_id not in seen:
                seen.add(repo_id)
                result.append(repo_id)
    return result


# ── upload ────────────────────────────────────────────────────────────────────

def _subdir_name(repo_id: str) -> str:
    """Convert org/name → org__name for use as a subdirectory."""
    return repo_id.replace("/", "__")


def _download_and_upload(
    repo_id: str,
    target: str,
    token: str | None,
    api: HfApi,
    private: bool,
    cache_dir: str | None,
) -> None:
    """Snapshot-download one dataset repo and upload its files as a subdirectory."""
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        cache_dir=cache_dir,
        ignore_patterns=["*.lock"],
    )
    subdir = _subdir_name(repo_id)
    api.upload_folder(
        folder_path=local_dir,
        path_in_repo=subdir,
        repo_id=target,
        repo_type="dataset",
        commit_message=f"Add {repo_id}",
        ignore_patterns=["*.lock", ".cache/**"],
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect task-binary datasets into a single HF repo as subdirectories",
    )
    p.add_argument("--source-file", required=True,
                   help="Markdown file listing HF dataset repo IDs (one per line)")
    p.add_argument("--target", required=True, help="Target HF dataset repo (org/name)")
    p.add_argument("--private", action="store_true", help="Create target repo as private")
    p.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env var)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print repo IDs that would be processed without downloading")
    p.add_argument("--skip-on-error", action="store_true",
                   help="Skip repos that fail instead of aborting")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N repos (useful for smoke-testing)")
    p.add_argument("--cache-dir", default=None,
                   help="HF cache directory for snapshot_download (default: HF_HOME)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    source = Path(args.source_file)
    if not source.exists():
        print(f"Error: source file not found: {source}", file=sys.stderr)
        sys.exit(1)

    repos = _parse_repo_ids(source)
    if args.limit:
        repos = repos[: args.limit]
    print(f"[collect] Found {len(repos)} repos in {source.name}")

    if args.dry_run:
        for repo_id in repos:
            print(f"  {repo_id}  →  {args.target}/{_subdir_name(repo_id)}/")
        return

    api = HfApi(token=token)
    create_repo(repo_id=args.target, repo_type="dataset", private=args.private,
                token=token, exist_ok=True)

    failed: list[str] = []

    for i, repo_id in enumerate(repos, 1):
        print(f"[collect] [{i}/{len(repos)}] {repo_id} ...", flush=True)
        try:
            _download_and_upload(
                repo_id=repo_id,
                target=args.target,
                token=token,
                api=api,
                private=args.private,
                cache_dir=args.cache_dir,
            )
            print(f"[collect] [{i}/{len(repos)}] {repo_id} ✓")
        except Exception as exc:
            print(f"[collect] [{i}/{len(repos)}] {repo_id} FAILED: {exc}")
            failed.append(repo_id)
            if not args.skip_on_error:
                print("[collect] Aborting. Use --skip-on-error to continue past failures.",
                      file=sys.stderr)
                sys.exit(1)

    print(f"\n[collect] Done. https://huggingface.co/datasets/{args.target}")

    if failed:
        print(f"\n[collect] {len(failed)} repo(s) failed:")
        for repo_id in failed:
            print(f"  {repo_id}")


if __name__ == "__main__":
    main()
