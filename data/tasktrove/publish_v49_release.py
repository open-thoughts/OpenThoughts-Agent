#!/usr/bin/env python3
"""Publish, verify, tag, and retire standalones for TaskTrove v4.9."""

from __future__ import annotations

import argparse
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
VERSION = "4.9"


def _repo_files(api: HfApi, revision: str) -> dict[str, object]:
    return {
        item.path: item
        for item in api.list_repo_tree(
            REPO_ID,
            repo_type="dataset",
            revision=revision,
            recursive=True,
            expand=True,
        )
        if hasattr(item, "blob_id")
    }


def _identity(item: object) -> tuple[str, int]:
    lfs = getattr(item, "lfs", None)
    digest = lfs.sha256 if lfs is not None else str(getattr(item, "blob_id"))
    return digest, int(getattr(item, "size"))


def _release_note(manifest: dict[str, object]) -> str:
    return (
        "> **v4.9 (current)** — trusted-test and calendar-verifier remediation — "
        "replaces three sources. The SWE-ReBench and OpenSWE verifiers restore "
        "hidden-test paths from the immutable base commit before applying the "
        "trusted patch, so agent edits cannot suppress or replace target tests; "
        "patch and setup failures now remain infrastructure failures rather than "
        "scoreable zeros. SWE-ReBench explicitly requests 4 GiB memory and "
        "8 GiB storage; "
        "OpenSWE requests 4 GiB. The "
        "instruction-following calendar verifier now rejects pairwise overlaps "
        "using half-open intervals and reports both event IDs and intervals. The "
        "agent-calendar sibling was audited and already contained this check. "
        f"The release uses {int(manifest['release_unique_images'])} unique images, "
        "and versioned sources are hosted only inside TaskTrove. Superseded source "
        "versions remain available through earlier TaskTrove tags.\n>\n"
        "> - `DCAgent/swe_rebench_v2_patched_oracle` → "
        "`DCAgent/swe_rebench_v2_patched_oracle-v2`\n"
        "> - `laion/openswe-tasks-patched-v6-oracle-success` → "
        "`laion/openswe-tasks-patched-v7-oracle-success`\n"
        "> - `laion/nemotron-gym-instruction-following-calendar-v2` → "
        "`laion/nemotron-gym-instruction-following-calendar-v3` (5,673 retained; "
        "2,714 without recoverable exact event names removed)\n>\n"
    )


def _updated_readme(source: str, manifest: dict[str, object]) -> str:
    marker = "> **v4.8 (current)**"
    if marker not in source:
        raise ValueError("README does not identify v4.8 as current")
    return source.replace(marker, _release_note(manifest) + "> **v4.8**", 1)


def publish(stage: Path) -> str:
    manifest = json.loads((stage / "manifest.json").read_text())
    source_revision = str(manifest["source_revision"])
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.repo_info(REPO_ID, repo_type="dataset").sha != source_revision:
        raise ValueError("TaskTrove changed after the v4.9 build")
    if any(
        ref.name == f"v{VERSION}"
        for ref in api.list_repo_refs(REPO_ID, repo_type="dataset").tags
    ):
        raise ValueError(f"v{VERSION} already exists")
    current = _repo_files(api, source_revision)
    for item in manifest["datasets"]:
        source = f"{item['source']}/tasks.parquet"
        target = f"{item['output']}/tasks.parquet"
        if (
            source not in current
            or _identity(current[source])[0] != item["source_sha256"]
        ):
            raise ValueError(f"source mismatch: {source}")
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
    readme = stage / "README-v4.9.md"
    readme.write_text(_updated_readme(readme_source, manifest))
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
        commit_message="TaskTrove v4.9: repair trusted tests and calendar scoring",
        parent_commit=source_revision,
        num_threads=1,
    )
    return commit.oid


def verify_and_retire(stage: Path, commit: str) -> None:
    manifest = json.loads((stage / "manifest.json").read_text())
    source_revision = str(manifest["source_revision"])
    api = HfApi(token=os.environ["HF_TOKEN"])
    before = _repo_files(api, source_revision)
    after = _repo_files(api, commit)
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
        if _identity(after[path]) != _identity(item):
            raise ValueError(f"untouched file changed: {path}")
    for item in manifest["datasets"]:
        target = f"{item['output']}/tasks.parquet"
        if _identity(after[target])[0] != item["output_sha256"]:
            raise ValueError(f"output hash mismatch: {target}")
    readme = Path(
        hf_hub_download(
            REPO_ID, "README.md", repo_type="dataset", revision=commit, token=api.token
        )
    ).read_text()
    if not re.search(r"^> \*\*v4\.9 \(current\)\*\*", readme, re.MULTILINE):
        raise ValueError("README does not identify v4.9 as current")
    api.create_tag(REPO_ID, tag="v4.9", repo_type="dataset", revision=commit)
    repositories = {
        str(item[field]).replace("__", "/", 1)
        for item in manifest["datasets"]
        for field in ("source", "output")
    }
    for repository in sorted(repositories):
        if api.repo_exists(repository, repo_type="dataset"):
            api.delete_repo(repository, repo_type="dataset")
        if api.repo_exists(repository, repo_type="dataset"):
            raise ValueError(f"standalone remains: {repository}")


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
        print(f"verified {commit}, tagged v4.9, and retired standalones")


if __name__ == "__main__":
    main()
