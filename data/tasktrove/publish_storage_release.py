#!/usr/bin/env python3
"""Publish, verify, tag, and retire standalones for a TaskTrove storage release."""

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
UPLOAD_THREADS = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_label(name: str) -> str:
    return name.replace("__", "/", 1)


def repo_files(api: HfApi, repo: str, revision: str) -> dict[str, object]:
    return {
        item.path: item
        for item in api.list_repo_tree(
            repo, repo_type="dataset", revision=revision, recursive=True, expand=True
        )
        if hasattr(item, "blob_id")
    }


def file_identity(item: object) -> tuple[str, int]:
    lfs = getattr(item, "lfs", None)
    digest = lfs.sha256 if lfs is not None else str(getattr(item, "blob_id"))
    return digest, int(getattr(item, "size"))


def updated_readme(source: str, manifest: dict[str, object]) -> str:
    source_version = str(manifest["source_version"])
    target_version = str(manifest["target_version"])
    datasets = manifest["datasets"]
    lines = "\n".join(
        f"> - `{dataset_label(dataset['source_dataset'])}` → "
        f"`{dataset_label(dataset['output_dataset'])}`: "
        f"{int(dataset['storage_mb']) // 1024} GiB storage "
        f"({int(dataset['output_rows']):,} tasks)"
        for dataset in datasets
    )
    note = (
        f"> **v{target_version} (current)** — task storage remediation — "
        f"replaces {len(datasets)} source{'s' if len(datasets) != 1 else ''} with "
        "explicit task-level storage requirements. All other packaged task files are "
        "preserved byte-for-byte, every source remains above the 300-task floor, and "
        "the new versions are hosted only inside TaskTrove. Historical rewards from "
        "the superseded versions are excluded.\n>\n"
        f"{lines}\n>\n"
    )
    marker = f"> **v{source_version} (current)**"
    if marker not in source:
        raise ValueError(f"README does not identify v{source_version} as current")
    return source.replace(marker, note + f"> **v{source_version}**", 1)


def prepare_standalones(
    api: HfApi, stage: Path, manifest: dict[str, object]
) -> tuple[dict[str, Path], list[str]]:
    deprecated: dict[str, Path] = {}
    existing = []
    for dataset in manifest["datasets"]:
        source_name = dataset["source_dataset"]
        repo = dataset_label(source_name)
        if not api.repo_exists(repo, repo_type="dataset"):
            continue
        existing.append(repo)
        files = repo_files(api, repo, "main")
        remote = files.get("tasks.parquet")
        if remote is None:
            raise ValueError(f"standalone lacks tasks.parquet: {repo}")
        if file_identity(remote)[0] == dataset["source_sha256"]:
            continue
        downloaded = Path(
            hf_hub_download(
                repo,
                "tasks.parquet",
                repo_type="dataset",
                token=api.token,
                local_dir=stage / "standalone-provenance" / source_name,
            )
        )
        if file_sha256(downloaded) == dataset["source_sha256"]:
            raise ValueError(f"standalone metadata mismatch: {repo}")
        deprecated[source_name] = downloaded
    return deprecated, existing


def publish(stage: Path) -> str:
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_revision = manifest["source_revision"]
    target_version = manifest["target_version"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    if api.repo_info(REPO_ID, repo_type="dataset").sha != source_revision:
        raise ValueError("TaskTrove changed after the release build")
    tag = f"v{target_version}"
    if any(
        ref.name == tag for ref in api.list_repo_refs(REPO_ID, repo_type="dataset").tags
    ):
        raise ValueError(f"tag already exists: {tag}")
    current = repo_files(api, REPO_ID, source_revision)
    for dataset in manifest["datasets"]:
        source_path = f"{dataset['source_dataset']}/tasks.parquet"
        output_path = f"{dataset['output_dataset']}/tasks.parquet"
        if source_path not in current:
            raise ValueError(f"missing source: {source_path}")
        if file_identity(current[source_path])[0] != dataset["source_sha256"]:
            raise ValueError(f"source hash mismatch: {source_path}")
        if output_path in current:
            raise ValueError(f"target already exists: {output_path}")

    readme_source = Path(
        hf_hub_download(
            REPO_ID, "README.md", repo_type="dataset", revision=source_revision
        )
    ).read_text()
    readme = stage / f"README-v{target_version}.md"
    readme.write_text(updated_readme(readme_source, manifest))
    deprecated, standalone_repos = prepare_standalones(api, stage, manifest)
    manifest["standalone_repositories"] = standalone_repos
    manifest["deprecated_standalones"] = {
        dataset: {
            "output": f"deprecated/{dataset}/tasks.parquet",
            "sha256": file_sha256(path),
        }
        for dataset, path in deprecated.items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd("README.md", readme)
    ]
    for dataset in manifest["datasets"]:
        operations.append(CommitOperationDelete(dataset["source_dataset"]))
        operations.append(
            CommitOperationAdd(
                f"{dataset['output_dataset']}/tasks.parquet", stage / dataset["output"]
            )
        )
    for dataset, path in deprecated.items():
        operations.append(
            CommitOperationAdd(f"deprecated/{dataset}/tasks.parquet", path)
        )
    commit = api.create_commit(
        REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message=f"TaskTrove v{target_version}: increase task storage",
        parent_commit=source_revision,
        num_threads=UPLOAD_THREADS,
    )
    return commit.oid


def verify_and_retire(stage: Path, commit: str) -> None:
    manifest = json.loads((stage / "manifest.json").read_text())
    source_revision = manifest["source_revision"]
    target_version = manifest["target_version"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    before = repo_files(api, REPO_ID, source_revision)
    after = repo_files(api, REPO_ID, commit)
    replaced = tuple(
        f"{dataset['source_dataset']}/" for dataset in manifest["datasets"]
    )
    expected = {
        path for path in before if path != "README.md" and not path.startswith(replaced)
    }
    expected.add("README.md")
    expected.update(
        f"{dataset['output_dataset']}/tasks.parquet" for dataset in manifest["datasets"]
    )
    expected.update(
        item["output"] for item in manifest["deprecated_standalones"].values()
    )
    if set(after) != expected:
        raise ValueError(
            f"unexpected remote tree: missing={sorted(expected - set(after))}, "
            f"extra={sorted(set(after) - expected)}"
        )
    for path, item in before.items():
        if path == "README.md" or path.startswith(replaced):
            continue
        if file_identity(after[path]) != file_identity(item):
            raise ValueError(f"untouched file changed: {path}")
    for dataset in manifest["datasets"]:
        output = f"{dataset['output_dataset']}/tasks.parquet"
        if file_identity(after[output])[0] != dataset["output_sha256"]:
            raise ValueError(f"remote output hash mismatch: {output}")
    readme = Path(
        hf_hub_download(REPO_ID, "README.md", repo_type="dataset", revision=commit)
    ).read_text()
    if not re.search(
        rf"^> \*\*v{re.escape(target_version)} \(current\)\*\*", readme, re.MULTILINE
    ):
        raise ValueError("remote README has the wrong current version")
    api.create_tag(
        REPO_ID, tag=f"v{target_version}", repo_type="dataset", revision=commit
    )
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
            raise ValueError("--commit is required without --publish")
        verify_and_retire(args.stage, commit)
        print(f"verified {commit}, tagged, and retired standalones")


if __name__ == "__main__":
    main()
