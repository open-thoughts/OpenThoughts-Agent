#!/usr/bin/env python3
"""Publish, verify, and tag the TaskTrove v4.4 low-ceiling repairs."""

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
VERSION = "4.4"


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


def label(name: str) -> str:
    return name.replace("__", "/", 1)


def storage_name(name: str) -> str:
    return name.replace("/", "__", 1)


def release_note(manifest: dict[str, object]) -> str:
    replacements = manifest["datasets"]
    rows = "\n".join(
        f"> - `{label(item['source_dataset'])}` → `{label(item['output_dataset'])}` "
        f"({int(item['source_rows']):,} → {int(item['output_rows']):,} tasks)"
        for item in replacements
    )
    return (
        "> **v4.4 (current)** — low-ceiling source remediation — repairs four "
        "sources identified by the GLM 5.2 failure-trace audit. Scaffold tasks now "
        "provide their actual starter code before the agent runs; scaffold, multifile, "
        "and curriculum tasks expose the generated verifier test as an explicit task "
        "contract and share one dependency-complete Python image. Math uses typed, "
        "fail-closed answer comparison and excludes free-form or multiple-valid-answer "
        "prompts that cannot be scored soundly by deterministic equality. Two shared "
        "images cover the release. Every retained source remains above the 300-task "
        "floor; superseded rewards are excluded.\n>\n"
        f"{rows}\n>\n"
        "> - `DCAgent/exp_rpt_curriculum-hard` is retired: it was an independent "
        "rewrite of the same source pool as the retained medium curriculum, and its "
        "sample had more task/verifier defects and substantially less evaluation "
        "coverage.\n>\n"
    )


def updated_readme(source: str, manifest: dict[str, object]) -> str:
    marker = "> **v4.3 (current)**"
    if marker not in source:
        raise ValueError("README does not identify v4.3 as current")
    return source.replace(marker, release_note(manifest) + "> **v4.3**", 1)


def all_sources(manifest: dict[str, object]) -> list[dict[str, object]]:
    return [*manifest["datasets"], *manifest["retired_datasets"]]


def prepare_standalones(
    api: HfApi, stage: Path, manifest: dict[str, object]
) -> tuple[dict[str, Path], list[str]]:
    deprecated: dict[str, Path] = {}
    existing: list[str] = []
    for item in all_sources(manifest):
        dataset = str(item["source_dataset"])
        repo = label(dataset)
        if not api.repo_exists(repo, repo_type="dataset"):
            continue
        existing.append(repo)
        files = repo_files(api, repo, "main")
        remote = files.get("tasks.parquet")
        if remote is None:
            raise ValueError(f"standalone lacks tasks.parquet: {repo}")
        if identity(remote)[0] == item["source_sha256"]:
            continue
        downloaded = Path(
            hf_hub_download(
                repo,
                "tasks.parquet",
                repo_type="dataset",
                token=api.token,
                local_dir=stage / "standalone-provenance" / dataset,
            )
        )
        deprecated[dataset] = downloaded
    return deprecated, existing


def publish(stage: Path) -> str:
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_revision = str(manifest["source_revision"])
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.repo_info(REPO_ID, repo_type="dataset").sha != source_revision:
        raise ValueError("TaskTrove changed after the v4.4 build")
    if any(
        ref.name == f"v{VERSION}"
        for ref in api.list_repo_refs(REPO_ID, repo_type="dataset").tags
    ):
        raise ValueError(f"v{VERSION} already exists")
    current = repo_files(api, REPO_ID, source_revision)
    for item in all_sources(manifest):
        source_path = f"{storage_name(str(item['source_dataset']))}/tasks.parquet"
        if source_path not in current:
            raise ValueError(f"missing source: {source_path}")
        if identity(current[source_path])[0] != item["source_sha256"]:
            raise ValueError(f"source hash mismatch: {source_path}")
    for item in manifest["datasets"]:
        target = f"{storage_name(str(item['output_dataset']))}/tasks.parquet"
        if target in current:
            raise ValueError(f"target already exists: {target}")

    readme_source = Path(
        hf_hub_download(
            REPO_ID, "README.md", repo_type="dataset", revision=source_revision
        )
    ).read_text()
    readme = stage / "README-v4.4.md"
    readme.write_text(updated_readme(readme_source, manifest))
    deprecated, standalone_repositories = prepare_standalones(api, stage, manifest)
    manifest["standalone_repositories"] = standalone_repositories
    manifest["deprecated_standalones"] = {
        dataset: {
            "output": f"deprecated/{storage_name(dataset)}/tasks.parquet",
            "sha256": file_sha256(path),
        }
        for dataset, path in deprecated.items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd("README.md", readme)
    ]
    for item in all_sources(manifest):
        operations.append(
            CommitOperationDelete(storage_name(str(item["source_dataset"])))
        )
    for item in manifest["datasets"]:
        operations.append(
            CommitOperationAdd(
                f"{storage_name(str(item['output_dataset']))}/tasks.parquet",
                Path(item["output"]),
            )
        )
    for dataset, path in deprecated.items():
        operations.append(
            CommitOperationAdd(
                f"deprecated/{storage_name(dataset)}/tasks.parquet", path
            )
        )
    commit = api.create_commit(
        REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message="TaskTrove v4.4: repair low-ceiling sources",
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
    removed = tuple(
        f"{storage_name(str(item['source_dataset']))}/"
        for item in all_sources(manifest)
    )
    expected = {
        path for path in before if path != "README.md" and not path.startswith(removed)
    }
    expected.add("README.md")
    expected.update(
        f"{storage_name(str(item['output_dataset']))}/tasks.parquet"
        for item in manifest["datasets"]
    )
    expected.update(
        item["output"] for item in manifest["deprecated_standalones"].values()
    )
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
        target = f"{storage_name(str(item['output_dataset']))}/tasks.parquet"
        if identity(after[target])[0] != item["output_sha256"]:
            raise ValueError(f"output hash mismatch: {target}")
    readme = Path(
        hf_hub_download(REPO_ID, "README.md", repo_type="dataset", revision=commit)
    ).read_text()
    if not re.search(r"^> \*\*v4\.4 \(current\)\*\*", readme, re.MULTILINE):
        raise ValueError("README does not identify v4.4 as current")
    api.create_tag(REPO_ID, tag="v4.4", repo_type="dataset", revision=commit)
    for repo in manifest["standalone_repositories"]:
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
        print(f"verified {commit}, tagged v4.4, and retired standalones")


if __name__ == "__main__":
    main()
