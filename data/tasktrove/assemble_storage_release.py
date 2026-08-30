#!/usr/bin/env python3
"""Assemble validated storage-repair outputs into one TaskTrove release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.tasktrove.build_storage_repair import file_sha256, validate_output


def assemble(args: argparse.Namespace) -> None:
    if args.stage.exists() and any(args.stage.iterdir()):
        raise ValueError(f"release stage must be absent or empty: {args.stage}")
    args.stage.mkdir(parents=True, exist_ok=True)
    datasets = []
    for manifest_path in args.manifests:
        manifest = json.loads(manifest_path.read_text())
        source = manifest_path.parent / manifest["output"]
        if file_sha256(source) != manifest["output_sha256"]:
            raise ValueError(f"output hash mismatch: {source}")
        validate_output(
            source, int(manifest["output_rows"]), int(manifest["storage_mb"])
        )
        relative = Path("datasets") / manifest["output_dataset"] / "tasks.parquet"
        destination = args.stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.hardlink_to(source)
        datasets.append({**manifest, "output": str(relative)})
    source_names = [dataset["source_dataset"] for dataset in datasets]
    output_names = [dataset["output_dataset"] for dataset in datasets]
    if len(source_names) != len(set(source_names)):
        raise ValueError("duplicate source dataset")
    if len(output_names) != len(set(output_names)):
        raise ValueError("duplicate output dataset")
    source_revisions = {dataset["source_revision"] for dataset in datasets}
    if source_revisions != {args.source_revision}:
        raise ValueError(f"wrong source revisions: {sorted(source_revisions)}")
    release = {
        "source_repo": "open-thoughts/TaskTrove",
        "source_revision": args.source_revision,
        "source_version": args.source_version,
        "target_version": args.target_version,
        "datasets": datasets,
    }
    (args.stage / "manifest.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps(release, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--target-version", required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    assemble(parser.parse_args())


if __name__ == "__main__":
    main()
