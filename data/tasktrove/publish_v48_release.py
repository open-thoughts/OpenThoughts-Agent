#!/usr/bin/env python3
"""Publish, verify, tag, and retire standalones for TaskTrove v4.8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    hf_hub_download,
)

REPO_ID = "open-thoughts/TaskTrove"
VERSION = "4.8"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_files(api: HfApi, repo: str, revision: str) -> dict[str, object]:
    return {
        item.path: item
        for item in api.list_repo_tree(
            repo, repo_type="dataset", revision=revision, recursive=True, expand=True
        )
        if hasattr(item, "blob_id")
    }


def identity(item: object) -> tuple[str, int]:
    lfs = getattr(item, "lfs", None)
    digest = lfs.sha256 if lfs is not None else str(getattr(item, "blob_id"))
    return digest, int(getattr(item, "size"))


def release_note(manifest: dict[str, object]) -> str:
    rows = {item["output"]: item for item in manifest["datasets"]}
    adversarial = rows["DCAgent__exp_rle_adversarial-v6"]
    return (
        "> **v4.8 (current)** — verifier and sandbox-memory remediation — "
        "replaces four sources. `DCAgent/exp_rle_adversarial-v6` records target "
        "exceptions and target import failures as scoreable zeros and removes the "
        "one statically invalid verifier task "
        f"({int(adversarial['source_rows']):,} → {int(adversarial['output_rows']):,}). "
        "`laion/exp_rpt_stack-php-large-v9` recognizes PHPUnit 10 successful-test "
        "output instead of treating it as a verifier crash. The C++, PHP-large, "
        "and PHP-v2 sources now explicitly request 4 GiB RAM, double the previous "
        "2 GiB default. Every active source retains at least 300 tasks, the release "
        f"uses {int(manifest['release_unique_images'])} unique images, and versioned "
        "sources are hosted only inside TaskTrove. Superseded source versions remain "
        "available through earlier TaskTrove tags.\n>\n"
        "> - `DCAgent/exp_rle_adversarial-v5` → "
        "`DCAgent/exp_rle_adversarial-v6`\n"
        "> - `laion/exp_rpt_stack-cpp-v3` → `laion/exp_rpt_stack-cpp-v4`\n"
        "> - `laion/exp_rpt_stack-php-large-v8` → "
        "`laion/exp_rpt_stack-php-large-v9`\n"
        "> - `laion/exp_rpt_stack-php-v2-v7` → "
        "`laion/exp_rpt_stack-php-v2-v8`\n>\n"
    )


def updated_readme(source: str, manifest: dict[str, object]) -> str:
    marker = "> **v4.7 (current)**"
    if marker not in source:
        raise ValueError("README does not identify v4.7 as current")
    return source.replace(marker, release_note(manifest) + "> **v4.7**", 1)


def publish(stage: Path) -> str:
    manifest = json.loads((stage / "manifest.json").read_text())
    source_revision = str(manifest["source_revision"])
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.repo_info(REPO_ID, repo_type="dataset").sha != source_revision:
        raise ValueError("TaskTrove changed after the v4.8 build")
    if any(
        ref.name == f"v{VERSION}"
        for ref in api.list_repo_refs(REPO_ID, repo_type="dataset").tags
    ):
        raise ValueError(f"v{VERSION} already exists")
    current = repo_files(api, REPO_ID, source_revision)
    for item in manifest["datasets"]:
        source = f"{item['source']}/tasks.parquet"
        target = f"{item['output']}/tasks.parquet"
        if source not in current:
            raise ValueError(f"missing source: {source}")
        if identity(current[source])[0] != item["source_sha256"]:
            raise ValueError(f"source hash mismatch: {source}")
        if target in current:
            raise ValueError(f"target already exists: {target}")

    readme_source = Path(
        hf_hub_download(
            REPO_ID,
            "README.md",
            repo_type="dataset",
            revision=source_revision,
            token=api.token,
        )
    ).read_text()
    readme = stage / "README-v4.8.md"
    readme.write_text(updated_readme(readme_source, manifest))
    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd("README.md", readme)
    ]
    for item in manifest["datasets"]:
        operations.extend(
            (
                CommitOperationDelete(str(item["source"])),
                CommitOperationAdd(
                    f"{item['output']}/tasks.parquet", stage / str(item["parquet"])
                ),
            )
        )
    commit = api.create_commit(
        REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message="TaskTrove v4.8: repair verifiers and memory limits",
        parent_commit=source_revision,
        num_threads=1,
    )
    return commit.oid


def verify_and_retire(stage: Path, commit: str) -> None:
    manifest = json.loads((stage / "manifest.json").read_text())
    source_revision = str(manifest["source_revision"])
    api = HfApi(token=os.environ["HF_TOKEN"])
    before = repo_files(api, REPO_ID, source_revision)
    after = repo_files(api, REPO_ID, commit)
    removed = tuple(f"{item['source']}/" for item in manifest["datasets"])
    expected = {
        path for path in before if path != "README.md" and not path.startswith(removed)
    }
    expected.add("README.md")
    expected.update(f"{item['output']}/tasks.parquet" for item in manifest["datasets"])
    if set(after) != expected:
        raise ValueError(
            f"unexpected tree: missing={sorted(expected - set(after))}, "
            f"extra={sorted(set(after) - expected)}"
        )
    for path, item in before.items():
        if path == "README.md" or path.startswith(removed):
            continue
        if identity(after[path]) != identity(item):
            raise ValueError(f"untouched file changed: {path}")
    for item in manifest["datasets"]:
        target = f"{item['output']}/tasks.parquet"
        if identity(after[target])[0] != item["output_sha256"]:
            raise ValueError(f"output hash mismatch: {target}")
    readme = Path(
        hf_hub_download(
            REPO_ID, "README.md", repo_type="dataset", revision=commit, token=api.token
        )
    ).read_text()
    if not re.search(r"^> \*\*v4\.8 \(current\)\*\*", readme, re.MULTILINE):
        raise ValueError("README does not identify v4.8 as current")
    api.create_tag(REPO_ID, tag="v4.8", repo_type="dataset", revision=commit)
    repositories = {
        str(item[field]).replace("__", "/", 1)
        for item in manifest["datasets"]
        for field in ("source", "output")
    }
    for repo in sorted(repositories):
        if api.repo_exists(repo, repo_type="dataset"):
            api.delete_repo(repo, repo_type="dataset")
        if api.repo_exists(repo, repo_type="dataset"):
            raise ValueError(f"standalone remains: {repo}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--commit")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--verify-and-retire", action="store_true")
    args = parser.parse_args()
    commit = args.commit
    if args.publish:
        commit = publish(args.stage)
        print(commit)
    if args.verify_and_retire:
        if commit is None:
            raise ValueError("--commit is required")
        verify_and_retire(args.stage, commit)
        print(f"verified {commit}, tagged v4.8, and retired standalones")


if __name__ == "__main__":
    main()
